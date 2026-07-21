"""ChemistryDriver -- the top-level orchestrator over a single ChemState.

The driver wires concrete network / solver / thermo (plus optional
cooling / opacity / radiation) policies together and exposes a single
`step(dt)` entry point. The substep semantics live in the solver; the
driver's job is to manage scratch lifetime, call the radiation /
opacity / cooling refresh hooks before each accepted substep, and
record per-step diagnostics.

The C++ analogue is `PhotochemistryNCR::UpdateSourceTermsOperatorSplit`
(`tigris-ncr/src/photchem/photchem_ncr.cpp:284`), which iterates over
cells (or strips, once Phase B lands) and dispatches each to
`solver_->SolveCell(var, dt)`. The Python driver collapses the cell
loop to one call per `ChemState` because the state already represents a
strip; multi-strip orchestration belongs to the host (Athena++ /
AthenaK) on top of this driver.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from .config import ChemistryConfig, SolverSpec, SOLVER_REGISTRY
from .networks.base import NetworkBase
from .state import ChemState
from .thermo.base import ThermoPolicy

# Import the solvers package for its registration side effect so the
# `_build_solver` lookup succeeds even when callers have not imported
# `pyathena.chemistry.solvers` themselves.
from . import solvers  # noqa: F401


class ChemistryDriver:
    """Top-level orchestrator over a single `ChemState`.

    Parameters
    ----------
    config : ChemistryConfig
        Runtime parameters; the solver name is read from
        `config.solver.name`.
    network : NetworkBase
        Concrete chemistry network instance.
    solver : object, optional
        Concrete solver instance. When `None`, the driver builds one
        from `config.solver` via `SOLVER_REGISTRY`.
    thermo : ThermoPolicy
        Concrete thermo policy (NCRThermo for NCRNetwork3).
    cooling : object, optional
        Phase 4 cooling policy. May be `None` until Phase 4 lands.
    opacity : object, optional
        Phase 5 opacity policy. May be `None` until Phase 5 lands.
    radiation : object, optional
        Radiation policy (or a stand-in). May be `None` -- in that
        case the chemistry runs in the dark regime defined by the
        network's optional radiation-field defaults.

    Notes
    -----
    The driver does NOT own the `ChemState`. Callers pass the state
    into `setup(state)` once at construction time so the driver can
    register scratch buffers via `network.allocate_scratch(state)`
    followed by `solver.allocate_scratch(state)`. Subsequent
    `step(dt, state)` calls operate on the same state in place.
    """

    __version__: str = '0.1'

    def __init__(
        self,
        config: ChemistryConfig,
        network: NetworkBase,
        thermo: ThermoPolicy,
        *,
        solver: Optional[Any] = None,
        cooling: Optional[Any] = None,
        opacity: Optional[Any] = None,
        radiation: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.network = network
        self.thermo = thermo
        self.cooling = cooling
        self.opacity = opacity
        self.radiation = radiation
        if solver is None:
            solver = self._build_solver(config, network, thermo, cooling)
        self.solver = solver

    @staticmethod
    def _build_solver(
        config: ChemistryConfig,
        network: NetworkBase,
        thermo: ThermoPolicy,
        cooling: Any,
    ) -> Any:
        """Resolve `config.solver.name` via `SOLVER_REGISTRY` and
        construct the named class.
        """
        spec: SolverSpec = config.solver
        name = spec.name if spec is not None else 'explicit_subcycling'
        try:
            cls = SOLVER_REGISTRY[name]
        except KeyError as exc:
            raise KeyError(
                f'solver name {name!r} is not registered; '
                f'known = {sorted(SOLVER_REGISTRY)}'
            ) from exc
        return cls(config, network, thermo, cooling=cooling)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def setup(self, state: ChemState) -> None:
        """Register network, solver, and cooling scratch on `state`.

        Idempotent: repeated calls re-register the same buffer names
        without leaking memory because `ChemState.alloc_scratch`
        overwrites existing entries.
        """
        self.network.allocate_scratch(state)
        alloc = getattr(self.solver, 'allocate_scratch', None)
        if callable(alloc):
            alloc(state)
        # CoolingChannels and similar policies own per-channel scratch
        # buffers (e.g. cooling:Lambda:CII, heating:Gamma:CR). Allocate
        # them here so `cooling.update(state)` runs allocation-free.
        # Also let individual channels register any internal scratch
        # they need by calling their `allocate_scratch(state)` hooks
        # when present (e.g. the metal-line channels need T2 / lnT2 /
        # mask buffers).
        for slot in (self.cooling, self.opacity, self.radiation):
            alloc = getattr(slot, 'allocate_scratch', None)
            if callable(alloc):
                alloc(state)

    # ------------------------------------------------------------------
    # Per-step API
    # ------------------------------------------------------------------
    def step(self, dt: float, state: ChemState) -> int:
        """Advance `state` over `[t, t + dt]`.

        Order of operations per step:

        1. `radiation.update(state)` -- refresh chi / xi_ph fields.
        2. `opacity.update(state)` -- refresh absorption coefficients.
        3. `cooling.update(state)` -- refresh cooling table cache.
        4. `solver.step(dt, state)` -- run the substep loop.

        Returns the number of substeps the solver used.
        """
        if self.radiation is not None:
            self.radiation.update(state)
        if self.opacity is not None:
            self.opacity.update(state)
        if self.cooling is not None:
            self.cooling.update(state)
        return self.solver.step(dt, state)

    def reset_step(self, dt: float, t: float, state: ChemState) -> None:
        """Forward to `state.reset_step`. Drivers typically call this
        at the head of every hydro step before `step(dt, state)`.
        """
        state.reset_step(dt, t)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def diagnostics(state: ChemState) -> Mapping[str, int]:
        """Snapshot of the per-step diagnostic counters on `state`."""
        return state.diag.as_dict()
