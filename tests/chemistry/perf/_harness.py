"""Shared machinery for the Phase 4.5 benchmarks.

Three pieces:

1. `time_best` -- warm-up-then-best-of-N timing.
2. `build_strip` / `REGIMES` -- strip states whose per-cell composition
   spans the ISM regimes the solver branches on.
3. `classify_regime` -- the benchmark-local regime tag.

The regime tag lives here and not on `ChemState`. `regime_tag` was one
of the named scratch fields removed when `state.scratch` became a dict
(see the Phase 3 entry in `pyathena/chemistry/CHANGES.md`), and whether
it earns a place back in the schema is exactly what these benchmarks
measure. Re-adding it before the measurement would assume the answer.
"""
from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Dict, Tuple

import numpy as np

from pyathena.chemistry.config import ChemistryConfig
from pyathena.chemistry.cooling.factories import make_ncr_default_cooling
from pyathena.chemistry.networks.ncr3 import NCRNetwork3
from pyathena.chemistry.solvers.explicit_subcycling import (
    ExplicitSubcyclingSolver,
)
from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import ChemState
from pyathena.chemistry.thermo.ncr import NCRThermo


# ---- Regime classes -----------------------------------------------------
# The tags track the two branches `_do_one_substep` actually takes, not
# a general ISM phase taxonomy:
#
#   - the `temp_hot1` gate in `_estimate_dt_sub` / `_update_chemistry`,
#     which zeroes the H2 rows above 35000 K;
#   - the `x_H2 < 0.25` low/high split in the 2x2 Cramer solve.
#
# Branch coherence is what sorting could buy, so cells are grouped by
# which side of those two gates they fall on. The physical names are
# labels for the resulting groups, not independent definitions.
COLD_MOL = 0    # T < 300 K,  x_H2 > 0.25   -> high-H2 branch, cold gate
CNM = 1         # 300-3000 K, atomic HI     -> low-H2 branch,  cold gate
WNM = 2         # 3000 K - temp_hot0        -> low-H2 branch,  cold gate
HII_REG = 3     # x_HII > 0.5, T ~ 1e4 K    -> low-H2 branch,  cold gate
HOT = 4         # T >= temp_hot1            -> low-H2 branch,  hot gate

REGIME_NAMES: Dict[int, str] = {
    COLD_MOL: 'cold_molecular',
    CNM: 'CNM',
    WNM: 'WNM',
    HII_REG: 'HII',
    HOT: 'hot',
}

# (nH [cm^-3], T [K], x_H2, x_HII) per regime. x_HI follows from the
# hydrogen closure x_HI = 1 - 2 x_H2 - x_HII. x_H2 is floored at
# `_TINY_NUMBER` rather than set to 0: the H2 heating channels form
# `1e6 / ncrit` with an ncrit that vanishes with x_H2, and an exact
# zero raises a divide-by-zero warning on every substep. The solver
# floors species at the same value.
_TINY_NUMBER = 1.0e-20

REGIMES: Dict[int, Tuple[float, float, float, float]] = {
    COLD_MOL: (1.0e3, 50.0, 0.49, 1.0e-8),
    CNM: (5.0e1, 100.0, 1.0e-3, 1.0e-4),
    WNM: (5.0e-1, 8.0e3, 1.0e-8, 1.0e-2),
    HII_REG: (1.0e1, 1.0e4, 1.0e-10, 0.99),
    HOT: (1.0e-2, 1.0e6, _TINY_NUMBER, 1.0 - _TINY_NUMBER),
}


def classify_regime(state: ChemState,
                    temp_hot1: float = 35000.0) -> np.ndarray:
    """Return an int8 regime tag per cell.

    Mirrors the gate structure in `ExplicitSubcyclingSolver`; see the
    module comment above for why this is not a `ChemState` field.
    """
    idx = state.species.idx
    x_H2 = state.x[idx['H2']]
    x_HII = state.x[idx['HII']]
    T = state.T

    tag = np.empty(state.ncell, dtype=np.int8)
    tag.fill(CNM)
    np.copyto(tag, np.int8(COLD_MOL), where=(T < 300.0) & (x_H2 > 0.25))
    np.copyto(tag, np.int8(WNM),
              where=(T >= 3.0e3) & (T < temp_hot1) & (x_HII <= 0.5))
    np.copyto(tag, np.int8(HII_REG), where=(x_HII > 0.5) & (T < temp_hot1))
    np.copyto(tag, np.int8(HOT), where=T >= temp_hot1)
    return tag


# ---- Strip construction -------------------------------------------------
def build_strip(regime_seq: np.ndarray) -> ChemState:
    """Build a strip whose cell `i` sits in regime `regime_seq[i]`.

    Radiation slots are pre-populated with zeros so `NCRNetwork3` reads
    stable buffers instead of allocating per call -- the same idiom
    `tests/chemistry/test_solvers_explicit_subcycling.py` uses.
    """
    regime_seq = np.asarray(regime_seq, dtype=np.int8)
    ncell = regime_seq.size
    species = SpeciesSet.ncr3_with_ghosts()

    nH = np.empty(ncell)
    T = np.empty(ncell)
    x_H2 = np.empty(ncell)
    x_HII = np.empty(ncell)
    for tag, (n_r, T_r, xH2_r, xHII_r) in REGIMES.items():
        sel = regime_seq == tag
        nH[sel], T[sel], x_H2[sel], x_HII[sel] = n_r, T_r, xH2_r, xHII_r

    r = np.linspace(0.1, 1.0, ncell)
    state = ChemState.from_grid(r, nH, T, species, xi_CR=2.0e-16, nfreq=3)

    idx = species.idx
    state.x[idx['H2']] = x_H2
    state.x[idx['HII']] = x_HII
    state.x[idx['HI']] = 1.0 - 2.0 * x_H2 - x_HII
    state.x[idx['electron']] = x_HII

    state.chi_FUV = np.zeros(ncell, dtype=np.float64)
    state.zeta_pi = {'HI': np.zeros(ncell, dtype=np.float64),
                     'H2': np.zeros(ncell, dtype=np.float64)}
    state.zeta_diss = {'H2': np.zeros(ncell, dtype=np.float64)}
    return state


def build_solver(state: ChemState,
                 cfl_cool_sub: float = 0.1,
                 with_cooling: bool = True) -> ExplicitSubcyclingSolver:
    """Build the production solver against `state`.

    Cooling is on by default. With `cooling=None` the solver skips the
    temperature step entirely, and the ~20 cooling channels are the
    bulk of the per-substep work, so a strip-width curve measured
    without them would not be the curve the C++ port needs.
    """
    config = ChemistryConfig(cfl_cool_sub=cfl_cool_sub)
    cooling = (make_ncr_default_cooling(state.species)
               if with_cooling else None)
    solver = ExplicitSubcyclingSolver(config, NCRNetwork3(), NCRThermo(),
                                      cooling=cooling)
    solver.allocate_scratch(state)
    if cooling is not None and hasattr(cooling, 'allocate_scratch'):
        cooling.allocate_scratch(state)
    return solver


def uniform_strip(ncell: int, regime: int = CNM) -> ChemState:
    return build_strip(np.full(ncell, regime, dtype=np.int8))


def mixed_strip(ncell: int, seed: int = 0) -> ChemState:
    """Strip with cells drawn from all five regimes in shuffled order."""
    rng = np.random.default_rng(seed)
    tags = np.array(sorted(REGIMES), dtype=np.int8)
    seq = tags[rng.integers(0, tags.size, size=ncell)]
    return build_strip(seq)


# ---- Timing -------------------------------------------------------------
def time_best(fn: Callable[[], Any], repeat: int = 7, warmup: int = 2,
              setup: Callable[[], Any] | None = None) -> float:
    """Best-of-`repeat` wall time in seconds, after `warmup` calls.

    `setup` runs before each call and is not timed. The solver mutates
    the strip in place and a relaxed strip takes a different substep
    count than a fresh one, so every repeat has to start from the same
    state to be comparable.

    Best-of rather than mean: the quantity of interest is the machine's
    capability, and every source of noise on a shared laptop adds time.
    """
    for _ in range(warmup):
        if setup is not None:
            setup()
        fn()
    best = float('inf')
    for _ in range(repeat):
        if setup is not None:
            setup()
        t0 = perf_counter()
        fn()
        best = min(best, perf_counter() - t0)
    return best


def snapshot(state: ChemState) -> Tuple[np.ndarray, np.ndarray]:
    """Copy the fields `step` mutates, for restore between repeats."""
    return state.x.copy(), state.T.copy()


def restore(state: ChemState,
            snap: Tuple[np.ndarray, np.ndarray],
            solver: Any = None) -> None:
    """Restore the strip, and optionally the solver's carried substep.

    `ExplicitSubcyclingSolver` carries `_dt_sub_next` from one `step`
    to the next (`explicit_subcycling.py:301,349`), so restoring only
    the state leaves the controller warmed from the previous call and
    the next `step` reports a different substep count than a cold
    solve. Pass `solver` to put both back.
    """
    x0, T0 = snap
    np.copyto(state.x, x0)
    np.copyto(state.T, T0)
    if solver is not None:
        solver._dt_sub_next = float('inf')


def nsub_converged(state: ChemState, solver: Any, dt: float,
                   nwarm: int = 6) -> int:
    """Substep count once the adaptive substep controller has settled.

    A cold solver starts at `_dt_sub_next = inf` and takes whatever
    first substep the pre-step estimate allows; the count only reaches
    its steady value after a few calls. Production steps a strip
    repeatedly, so the converged count is the representative one.
    """
    snap = snapshot(state)
    nsub = 0
    for _ in range(nwarm):
        restore(state, snap)   # state only -- let the controller warm
        nsub = int(solver.step(dt, state))
    return nsub
