"""Deliverables 3 and 2 -- SIMD ceiling and sorted-strip payoff.

Both questions are asked of numba kernels, not of the NumPy solver.
The NumPy hot path is written branch-free with `np.where`, which
evaluates every branch for every cell unconditionally, so sorting a
NumPy strip by regime cannot change the work done and the sorted /
mixed ratio would come out at 1.0 as an artifact of the formulation.
Numba compiles a real loop with real branches, which is the structure
the C++ port has.

The kernel mirrors the two binary gates `_do_one_substep` actually
takes -- the `temp_hot1` temperature gate and the `x_H2 < 0.25` split
in the Cramer solve -- rather than a general five-way regime switch.
Each gate selects between two rate-coefficient-shaped expressions.

Two forms of the same arithmetic:

- *branch-free*: evaluate both sides of each gate for every cell and
  blend. More flops, no data-dependent control flow, vectorisable.
- *branched*: a per-cell `if` per gate. Fewer flops, control flow that
  depends on the data.

`transcendental=False` variants replace exp/log/sqrt with polynomials
of similar flop count. That isolates whether the limit on vectorising
this kernel is the branching or the lack of a vectorised libm: numba
has SVML on x86 but nothing equivalent on arm64, so a scalar `exp`
call in the loop blocks vectorisation regardless of how the branches
are written.
"""
from __future__ import annotations

import json
import platform

import numpy as np
import pytest

from . import _harness as H

numba = pytest.importorskip('numba')
njit = numba.njit


TEMP_HOT1 = 35000.0
NCELL = 4096
NREP = 200


# ---- Kernels ------------------------------------------------------------
def _make_kernels(fastmath: bool, transcendental: bool):
    """Build the four kernels for one (fastmath, transcendental) pair.

    Compiled inside a factory so `fastmath` and the arithmetic form are
    compile-time constants rather than runtime branches.
    """
    if transcendental:
        @njit(inline='always')
        def path_a(T, n, x):
            return np.exp(-0.5 * np.log(T)) * n / (1.0 + np.sqrt(x))

        @njit(inline='always')
        def path_b(T, n, x):
            return np.exp(-1.2e4 / T) * n * (1.0 + x * x)
    else:
        @njit(inline='always')
        def path_a(T, n, x):
            u = 1.0 / T
            return (((0.31 * u + 0.72) * u + 1.4) * u + 2.1) * n / (1.0 + x)

        @njit(inline='always')
        def path_b(T, n, x):
            u = T * 1.0e-4
            poly = ((0.11 * u + 0.53) * u + 1.9) * u + 0.7
            return poly * n * (1.0 + x * x)

    @njit(fastmath=fastmath, cache=False)
    def branch_free(T, nH, xH2, out):
        for i in range(T.shape[0]):
            t = T[i]
            n = nH[i]
            x = xH2[i]
            # Both sides of both gates, always. `hot` / `low` are 0/1
            # weights, the scalar analogue of the strip-wide np.where
            # the NumPy solver uses.
            hot = 1.0 if t >= TEMP_HOT1 else 0.0
            low = 1.0 if x < 0.25 else 0.0
            a = path_a(t, n, x)
            b = path_b(t, n, x)
            c = path_a(t * 1.01, n, x)
            d = path_b(t * 1.01, n, x)
            out[i] = (hot * a + (1.0 - hot) * b
                      + low * c + (1.0 - low) * d)

    @njit(fastmath=fastmath, cache=False)
    def branched(T, nH, xH2, out):
        for i in range(T.shape[0]):
            t = T[i]
            n = nH[i]
            x = xH2[i]
            if t >= TEMP_HOT1:
                r1 = path_a(t, n, x)
            else:
                r1 = path_b(t, n, x)
            if x < 0.25:
                r2 = path_a(t * 1.01, n, x)
            else:
                r2 = path_b(t * 1.01, n, x)
            out[i] = r1 + r2

    return branch_free, branched


# ---- Inputs -------------------------------------------------------------
def _strip_arrays(regime_seq: np.ndarray):
    state = H.build_strip(regime_seq)
    idx = state.species.idx
    return (np.ascontiguousarray(state.T),
            np.ascontiguousarray(state.nH),
            np.ascontiguousarray(state.x[idx['H2']]))


def _mixed_seq(ncell: int, seed: int = 20260729) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tags = np.array(sorted(H.REGIMES), dtype=np.int8)
    return tags[rng.integers(0, tags.size, size=ncell)]


def _bench(kernel, arrays, out) -> float:
    T, nH, xH2 = arrays
    kernel(T, nH, xH2, out)          # compile + warm

    def run():
        for _ in range(NREP):
            kernel(T, nH, xH2, out)

    return H.time_best(run, repeat=5, warmup=1) / NREP


@pytest.mark.parametrize('transcendental', [True, False],
                         ids=['exp_log', 'poly'])
@pytest.mark.parametrize('fastmath', [False, True],
                         ids=['nofast', 'fastmath'])
def test_simd_ceiling_and_sort_payoff(transcendental, fastmath, results_dir):
    branch_free, branched = _make_kernels(fastmath, transcendental)

    seq_mixed = _mixed_seq(NCELL)
    # Sorted by regime tag: runs of identical branch outcomes.
    seq_sorted = np.sort(seq_mixed)
    # Uniform: every cell takes the same branch, the coherent limit.
    seq_uniform = np.full(NCELL, H.CNM, dtype=np.int8)

    out = np.empty(NCELL)
    res = {}
    for name, seq in (('mixed', seq_mixed), ('sorted', seq_sorted),
                      ('uniform', seq_uniform)):
        arrays = _strip_arrays(seq)
        res[f'branch_free_{name}'] = _bench(branch_free, arrays, out)
        res[f'branched_{name}'] = _bench(branched, arrays, out)

    tag = f"{'fastmath' if fastmath else 'nofast'}/" \
          f"{'exp_log' if transcendental else 'poly'}"
    print(f"\n--- numba kernels: {tag} "
          f"(ncell={NCELL}, {platform.machine()}) ---")
    print(f"{'order':>9} {'branch-free [us]':>18} {'branched [us]':>15} "
          f"{'branched/free':>14}")
    for name in ('mixed', 'sorted', 'uniform'):
        bf = res[f'branch_free_{name}'] * 1e6
        br = res[f'branched_{name}'] * 1e6
        print(f"{name:>9} {bf:>18.2f} {br:>15.2f} {br / bf:>14.2f}x")

    # D3 -- SIMD ceiling: branch-free vs branched on a coherent strip.
    ceiling = res['branched_uniform'] / res['branch_free_uniform']
    # D2 -- sort payoff: branched kernel, mixed order vs sorted order.
    sort_payoff = res['branched_mixed'] / res['branched_sorted']
    print(f"\n  D3 SIMD ceiling (branched/branch-free, uniform): "
          f"{ceiling:.2f}x   [plan threshold > 3x]")
    print(f"  D2 sort payoff  (branched mixed/sorted)         : "
          f"{sort_payoff:.2f}x   [plan threshold > 2x]")

    res.update({'ceiling': ceiling, 'sort_payoff': sort_payoff,
                'fastmath': fastmath, 'transcendental': transcendental,
                'ncell': NCELL, 'machine': platform.machine()})
    out_path = results_dir / (
        f"simd_{'fastmath' if fastmath else 'nofast'}_"
        f"{'exp_log' if transcendental else 'poly'}.json")
    out_path.write_text(json.dumps(res, indent=2))

    assert ceiling > 0 and sort_payoff > 0
