"""Deliverable 4 -- timescale sort and strip splitting.

`ExplicitSubcyclingSolver` takes a strip-MIN `dt_sub`, so every cell in
a strip walks the substep count set by the stiffest cell in it (see the
module docstring of `solvers/explicit_subcycling.py`). A strip mixing a
hot cell with cold molecular cells makes the cold cells repeat work
they did not need.

This benchmark counts that waste and asks whether sorting cells by
stiffness and splitting into sub-strips recovers it. The unit is
cell-substeps, not seconds: the strip-MIN rule is a property of the
sweep, so the counts here hold for the C++ port as well as for the
NumPy implementation. Nothing in this file is timed.

Three quantities:

- `ideal`     -- each cell integrated in its own 1-cell strip, so no
                 cell pays for another. The lower bound.
- `one_strip` -- the whole mixed strip in one piece. What the solver
                 does today.
- `split`     -- the strip cut into `nchunk` pieces, unsorted and then
                 sorted by stiffness.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from . import _harness as H


DT_STEP = 3.15e13  # 1 Myr
NCELL = 256
NCHUNKS = (2, 4, 8, 16)


def _nsub(regime_seq: np.ndarray) -> int:
    """Converged substep count for a strip holding `regime_seq`.

    Converged rather than cold: the solver carries `_dt_sub_next`
    between calls, and production steps a strip many times. A cold
    count understates the steady cost (CNM reads 1 cold, 25 settled).
    """
    state = H.build_strip(regime_seq)
    solver = H.build_solver(state)
    return H.nsub_converged(state, solver, DT_STEP)


def _cell_substeps(regime_seq: np.ndarray) -> int:
    """Cell-substeps charged to one strip: width x its substep count."""
    return regime_seq.size * _nsub(regime_seq)


def _split_cost(regime_seq: np.ndarray, nchunk: int) -> int:
    """Total cell-substeps when the strip is cut into `nchunk` pieces."""
    return sum(_cell_substeps(chunk)
               for chunk in np.array_split(regime_seq, nchunk))


@pytest.fixture(scope='module')
def per_regime_nsub():
    """Substep count for a 1-cell strip of each regime.

    Cells within a regime are identical, so this is the per-cell cost
    and `ideal` is a weighted sum over it -- 5 solves instead of 256.
    """
    return {tag: _nsub(np.array([tag], dtype=np.int8))
            for tag in sorted(H.REGIMES)}


def test_strip_split_load_balance(per_regime_nsub, results_dir):
    rng = np.random.default_rng(20260729)
    tags = np.array(sorted(H.REGIMES), dtype=np.int8)
    seq = tags[rng.integers(0, tags.size, size=NCELL)]

    counts = {int(t): int((seq == t).sum()) for t in tags}
    ideal = sum(counts[t] * per_regime_nsub[t] for t in counts)
    one_strip = _cell_substeps(seq)

    print("\n--- per-cell substep cost by regime ---")
    for t in sorted(per_regime_nsub):
        print(f"  {H.REGIME_NAMES[t]:>15}: nsub={per_regime_nsub[t]:>4} "
              f"  cells={counts[t]:>4}")

    print(f"\nstrip width            : {NCELL}")
    print(f"ideal  (1-cell strips) : {ideal:>9} cell-substeps")
    print(f"one strip (today)      : {one_strip:>9} cell-substeps "
          f"  {one_strip / ideal:.2f}x ideal")

    rows = []
    print(f"\n{'nchunk':>7} {'unsorted':>12} {'sorted':>12} "
          f"{'unsorted/ideal':>15} {'sorted/ideal':>13}")
    order = np.argsort([per_regime_nsub[int(t)] for t in seq], kind='stable')
    seq_sorted = seq[order]
    for nchunk in NCHUNKS:
        unsorted_cost = _split_cost(seq, nchunk)
        sorted_cost = _split_cost(seq_sorted, nchunk)
        rows.append({'nchunk': nchunk, 'unsorted': unsorted_cost,
                     'sorted': sorted_cost, 'ideal': ideal,
                     'one_strip': one_strip})
        print(f"{nchunk:>7} {unsorted_cost:>12} {sorted_cost:>12} "
              f"{unsorted_cost / ideal:>14.2f}x {sorted_cost / ideal:>12.2f}x")

    out = results_dir / "strip_split_load_balance.json"
    out.write_text(json.dumps(
        {'ncell': NCELL, 'ideal': ideal, 'one_strip': one_strip,
         'per_regime_nsub': {H.REGIME_NAMES[k]: v
                             for k, v in per_regime_nsub.items()},
         'counts': {H.REGIME_NAMES[int(k)]: v for k, v in counts.items()},
         'splits': rows}, indent=2))

    # `ideal` is a lower bound by construction: no split can charge a
    # cell fewer substeps than it needs alone.
    assert one_strip >= ideal
    for r in rows:
        assert r['sorted'] >= ideal
