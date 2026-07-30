"""Deliverable 1 -- strip-size sweet spot.

Times `ExplicitSubcyclingSolver.step` on uniform strips at increasing
`ncell` and reports throughput in cell-substeps per second. The knee of
that curve is the width at which vectorised work outgrows the fixed
per-strip cost (scratch refresh, the `bool(np.any(...))` reductions in
the rejection path, the Python-level `while` in `step`). That width is
the recommended `CHUNK_SIZE` for the C++ port.

Strips are uniform in regime so every cell does the same work and the
substep count does not drift with width; a mixed strip would confound
width against the strip-MIN `dt_sub` picking up whichever stiff cell
the sampler happened to place.

Kill criterion from the plan: a knee below 8 cells means the SoA strip
layout is not worth carrying into C++.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from . import _harness as H


# The plan lists {1, 8, 16, 32, 64, 256}. 2 and 4 are added to resolve
# the overhead-amortisation knee against the "sweet spot < 8 cells"
# kill criterion, and 512-2048 to find where throughput saturates.
NCELLS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
DT_STEP = 3.15e13  # 1 Myr -- long enough to exercise the subcycle loop


def _throughput(ncell: int, regime: int) -> dict:
    """Cell-substeps per second for one (width, regime) pair."""
    state = H.uniform_strip(ncell, regime=regime)
    solver = H.build_solver(state)
    snap = H.snapshot(state)
    nsub_seen = []

    def run():
        nsub_seen.append(solver.step(DT_STEP, state))

    wall = H.time_best(run, repeat=5, warmup=2,
                       setup=lambda: H.restore(state, snap))
    nsub = int(np.median(nsub_seen))
    return {
        'ncell': ncell,
        'regime': H.REGIME_NAMES[regime],
        'nsub': nsub,
        'wall_s': wall,
        'cell_substeps_per_s': (ncell * nsub) / wall,
    }


@pytest.mark.parametrize(
    'regime', sorted(H.REGIMES),
    ids=[H.REGIME_NAMES[k] for k in sorted(H.REGIMES)])
def test_strip_size_sweep(regime, results_dir):
    rows = [_throughput(n, regime) for n in NCELLS]

    name = H.REGIME_NAMES[regime]
    print(f"\n--- strip-size sweep: {name} ---")
    print(f"{'ncell':>6} {'nsub':>6} {'wall [s]':>10} "
          f"{'cell-substeps/s':>18} {'vs ncell=1':>11} {'marginal':>9}")
    base = rows[0]['cell_substeps_per_s']
    for i, r in enumerate(rows):
        # Marginal scaling: throughput gain per doubling of width.
        # 1.0 is perfect linear scaling; below 1.0 the strip has
        # saturated and extra width buys nothing.
        if i == 0:
            marg = '--'
        else:
            prev = rows[i - 1]
            gain = r['cell_substeps_per_s'] / prev['cell_substeps_per_s']
            width = r['ncell'] / prev['ncell']
            marg = f"{gain / width:.2f}"
        print(f"{r['ncell']:>6} {r['nsub']:>6} {r['wall_s']:>10.4f} "
              f"{r['cell_substeps_per_s']:>18.3e} "
              f"{r['cell_substeps_per_s'] / base:>11.2f}x {marg:>9}")

    out = results_dir / f"strip_size_{name}.json"
    out.write_text(json.dumps(rows, indent=2))

    # The benchmark records a curve; it does not pin a performance
    # number. The only assertion is that the sweep produced usable
    # data for every width.
    assert all(r['cell_substeps_per_s'] > 0 for r in rows)
