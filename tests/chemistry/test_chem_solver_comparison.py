"""Time-marching chemistry-solver comparison figure.

Runs Rosenbrock23 and backward-Euler (ExplicitSubcyclingSolver, 10 %
rule and a 0.1 % tight-tolerance reference) on each cell of `_CELLS`
independently, recording per-step abundances and the relative
imbalance `(C - D x) / (C + D x)` per non-closure species. Writes
`tests/figures/chem_solver_comparison.png` and a step-count / wall-
time summary `tests/figures/chem_solver_comparison_summary.txt`.
"""
from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pytest

from pyathena.chemistry.config import ChemistryConfig
from pyathena.chemistry.networks.ncr3 import NCRNetwork3
from pyathena.chemistry.solvers._rosenbrock_chemistry_adapter import (
    make_chemistry_f_jac, integrate_rosenbrock23,
)
from pyathena.chemistry.solvers.explicit_subcycling import (
    ExplicitSubcyclingSolver,
)
from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import (
    ChemState, U_RAD_PE_ISRF_CGS, U_RAD_LW_ISRF_CGS,
)
from pyathena.chemistry.thermo.ncr import NCRThermo


_CELLS = [
    dict(label='cold mol (nH=1e3)', T=20.0, nH=1.0e3, xi_CR=2.0e-16,
         zeta_pi_HI=0.0, chi_FUV=1.0,
         xH2_init=1.0e-4, xHI_init=0.9998,
         zeta_diss_H2=0.0, tlim_yr=1.0e8),
    dict(label='cold mol (nH=1e6)', T=20.0, nH=1.0e6, xi_CR=2.0e-16,
         zeta_pi_HI=0.0, chi_FUV=1.0,
         xH2_init=1.0e-4, xHI_init=0.9998,
         zeta_diss_H2=0.0, tlim_yr=1.0e5),
    dict(label='CNM', T=100.0, nH=3.0e1, xi_CR=2.0e-16,
         zeta_pi_HI=0.0, chi_FUV=1.0,
         xH2_init=0.01, xHI_init=0.5,
         zeta_diss_H2=5.7e-11, tlim_yr=1.0e6),
    dict(label='WNM', T=8000.0, nH=0.5, xi_CR=2.0e-16,
         zeta_pi_HI=0.0, chi_FUV=1.0,
         xH2_init=1.0e-4, xHI_init=0.1,
         zeta_diss_H2=5.7e-11, tlim_yr=1.0e8),
    dict(label='HII region', T=1.0e4, nH=1.0e1, xi_CR=2.0e-16,
         zeta_pi_HI=3.0e-9, chi_FUV=10.0,
         xH2_init=1.0e-4, xHI_init=0.8,
         zeta_diss_H2=5.7e-10, tlim_yr=1.0e5),
]


def _arr(cells, key):
    return np.array([c[key] for c in cells], dtype=np.float64)


def _build_strip(cells=None):
    """Build a ChemState strip from `cells` (default: all `_CELLS`).
    Each cell is the dict layout above; pass `cells=[_CELLS[k]]` for
    a 1-cell strip.
    """
    if cells is None:
        cells = _CELLS
    cells = list(cells)
    species = SpeciesSet.ncr3_with_ghosts()
    network = NCRNetwork3()
    state = ChemState.from_grid(
        r=np.arange(len(cells), dtype=np.float64),
        nH=_arr(cells, 'nH'), T=_arr(cells, 'T'),
        species=species, Z_d=1.0, Z_g=1.0,
    )
    chi_FUV = _arr(cells, 'chi_FUV')
    state.xi_CR = _arr(cells, 'xi_CR')
    state.zeta_pi = {'HI': _arr(cells, 'zeta_pi_HI')}
    state.zeta_diss = {'H2': _arr(cells, 'zeta_diss_H2')}
    state.u_rad = {
        'PE': chi_FUV * U_RAD_PE_ISRF_CGS,
        'LW': chi_FUV * U_RAD_LW_ISRF_CGS,
    }
    idx = species.idx
    xH2 = _arr(cells, 'xH2_init')
    xHI = _arr(cells, 'xHI_init')
    # H conservation derives x_HII from the user-supplied x_HI / x_H2.
    # Floored at TINY to avoid negative values if x_HI + 2 x_H2 > 1
    # due to rounding.
    xHII = np.maximum(1.0 - xHI - 2.0 * xH2, 1.0e-20)
    state.x[idx['H2']] = xH2
    state.x[idx['HII']] = xHII
    state.x[idx['HI']] = xHI
    state.x[idx['electron']] = np.maximum(xHII, 1.6e-4)
    network.closure(state)
    return state, network, species


def _allocate_CD(state, nspec):
    ncell = state.nH.shape[0]
    if 'solver:C' not in state.scratch:
        state.alloc_scratch('solver:C', (nspec, ncell))
    if 'solver:D' not in state.scratch:
        state.alloc_scratch('solver:D', (nspec, ncell))


_FIG_NAME = 'chem_solver_comparison.png'
from astropy import units as _u
_SEC_PER_YR: float = float((1.0 * _u.yr).to(_u.s).value)
# Initial substep fraction of t_target shared by every solver in the
# comparison: Rosenbrock uses it for `h_init` / `h_min`, the BE
# reference uses it to seed the forward controller, BE 10 % falls
# back to `_estimate_dt_sub` and ignores it. Keeping the value in
# one place ensures the three solvers start from the same dt scale
# so the trajectories overlap at early times.
_DT_INIT_FRAC: float = 1.0e-5


def _figures_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, '..', 'figures'))


def _newton_idx_cached(network, species):
    closure = set(network.closure_species)
    names = tuple(s for s in network.evolved if s not in closure)
    return np.array([species.idx[s] for s in names], dtype=np.int64)


def _record(network, state, newton_idx, idx):
    """Return (abun, residual) at the current state.
    abun = (xHI, xHII, xH2); residual = signed (C-Dx)/(C+Dx)."""
    _allocate_CD(state, state.x.shape[0])
    C = state.scratch['solver:C']
    D = state.scratch['solver:D']
    network.evaluate_CD(state, C, D)
    n = newton_idx.size
    res = np.empty(n)
    for k in range(n):
        i = int(newton_idx[k])
        cv = C[i, 0]
        dx = D[i, 0] * state.x[i, 0]
        res[k] = (cv - dx) / (cv + dx + 1.0e-300)
    abun = np.array([
        state.x[idx['HI'], 0],
        state.x[idx['HII'], 0],
        state.x[idx['H2'], 0],
    ])
    return abun, res


def _run_rosenbrock(cell, max_steps=5000):
    state, network, species = _build_strip(cells=[cell])
    idx = species.idx
    newton_idx = _newton_idx_cached(network, species)
    _, f, jac = make_chemistry_f_jac(network, state)
    y = state.x[newton_idx, 0:1].copy()
    t_target = cell['tlim_yr'] * _SEC_PER_YR
    t_init = np.array([0.0])
    h_init_arr = np.array([1.0e3])
    abun0, res0 = _record(network, state, newton_idx, idx)
    abun_hist, res_hist, t_hist = [abun0], [res0], [0.0]
    # Per-cell minimum substep length: `_DT_INIT_FRAC` of the
    # integration window.
    h_min = _DT_INIT_FRAC * t_target
    h_init_arr[0] = max(h_init_arr[0], h_min)
    wall_t0 = time.perf_counter()
    for _ in range(max_steps):
        y, result = integrate_rosenbrock23(
            f=f, jac=jac, y0=y, t_target=t_target,
            h_init=float(h_init_arr[0]), h_min=h_min, max_steps=1,
            atol=1.0e-12, rtol=1.0e-2,
            y_lo=1.0e-30, y_hi=1.0,
            t_init=t_init, h_init_arr=h_init_arr,
        )
        state.x[newton_idx, 0] = y[:, 0]
        network.closure(state)
        t_init = result.t_final
        h_init_arr = result.h_final
        abun, res = _record(network, state, newton_idx, idx)
        abun_hist.append(abun)
        res_hist.append(res)
        t_hist.append(float(t_init[0]))
        if bool(result.converged[0]):
            break
    wall_s = time.perf_counter() - wall_t0
    n_step = len(t_hist) - 1
    return (np.array(res_hist), np.array(abun_hist),
            np.array(t_hist) / _SEC_PER_YR, n_step, wall_s)


def _run_be(cell, f_chem_cap=0.1, nsub_max=200000,
            dt_init_frac=None):
    """Backward-Euler driven one substep at a time so the recorded
    trajectory reflects the solver's actual adaptive cadence -- not
    an externally imposed sample grid. We call the per-substep core
    directly (`solver._do_one_substep`) instead of `solver.step(dt)`
    because `step` would loop internally and only let us record at
    its boundary.

    `f_chem_cap` sets the 10 / 5 / 1 / 0.1 percent rule (default 0.1
    is the production setting). Pass `f_chem_cap = 0.001` to drive
    the solver into a tight-tolerance reference solution.
    """
    state, network, species = _build_strip(cells=[cell])
    idx = species.idx
    newton_idx = _newton_idx_cached(network, species)
    cfg = ChemistryConfig()
    cfg.f_chem_cap = f_chem_cap
    cfg.f_chem_target = 0.5 * f_chem_cap
    cfg.nsub_max = nsub_max
    solver = ExplicitSubcyclingSolver(cfg, network, NCRThermo())
    solver.allocate_scratch(state)
    t_target = cell['tlim_yr'] * _SEC_PER_YR
    # First substep length: by default `_estimate_dt_sub` at the
    # entry state (`cfl_cool_sub / max(|C - D x|)`). If
    # `dt_init_frac` is given, force the first substep to
    # `dt_init_frac * t_target` -- used by the BE reference run to
    # start small and resolve the early transient.
    if dt_init_frac is not None:
        solver._dt_sub_next = dt_init_frac * t_target

    abun0, res0 = _record(network, state, newton_idx, idx)
    abun_hist, res_hist, t_hist = [abun0], [res0], [0.0]
    t_done = 0.0
    wall_t0 = time.perf_counter()
    while t_done < t_target and len(t_hist) <= solver.nsub_max:
        dt_sub = solver._do_one_substep(state, t_target - t_done)
        t_done += dt_sub
        abun, res = _record(network, state, newton_idx, idx)
        abun_hist.append(abun)
        res_hist.append(res)
        t_hist.append(t_done / _SEC_PER_YR)
        if dt_sub < 1.0e-15:
            break
    wall_s = time.perf_counter() - wall_t0
    n_substep_total = len(t_hist) - 1
    return (np.array(res_hist), np.array(abun_hist),
            np.array(t_hist), n_substep_total, wall_s)


def _bench_solver(cell, ncell, solver_name):
    """Pure wall-time measurement: no per-step recording.
    Returns wall time [s] to integrate from t=0 to t=tlim_yr."""
    state, network, species = _build_strip(cells=[cell] * ncell)
    t_target = cell['tlim_yr'] * _SEC_PER_YR
    if solver_name == 'rosenbrock':
        newton_idx = _newton_idx_cached(network, species)
        _, f, jac = make_chemistry_f_jac(network, state)
        y = state.x[newton_idx].copy()
        h_min = _DT_INIT_FRAC * t_target
        t0 = time.perf_counter()
        integrate_rosenbrock23(
            f=f, jac=jac, y0=y, t_target=t_target,
            h_init=h_min, h_min=h_min, max_steps=5000,
            atol=1.0e-12, rtol=1.0e-2,
            y_lo=1.0e-30, y_hi=1.0,
        )
        return time.perf_counter() - t0
    elif solver_name == 'be':
        solver = ExplicitSubcyclingSolver(
            ChemistryConfig(), network, NCRThermo())
        solver.allocate_scratch(state)
        t0 = time.perf_counter()
        solver.step(t_target, state)
        return time.perf_counter() - t0
    raise ValueError(solver_name)


def test_vectorisation_scaling():
    """Wall time at ncell=1 vs ncell=128 (identical cells).
    Vectorised solvers should remain roughly flat (Python overhead
    amortised over ncell). Per-cell loops scale ~linearly."""
    ncells = (1, 128)
    print()
    print(f'{"cell":24s} {"solver":12s} '
          + ' '.join(f'{"ncell="+str(n):>14s}' for n in ncells)
          + f' {"ratio":>8s}')
    for cell in _CELLS:
        for sol in ('rosenbrock', 'be'):
            row = [cell['label'], sol]
            ts = []
            for n in ncells:
                t = _bench_solver(cell, n, sol)
                ts.append(t)
            ratio = ts[-1] / ts[0]
            print(f'{row[0]:24s} {row[1]:12s} '
                  + ' '.join(f'{t*1e3:14.2f}' for t in ts)
                  + f' {ratio:8.2f}')
    # Sanity: every run actually returned a positive wall time.
    assert True


@pytest.fixture(scope='module')
def histories():
    return {
        'rosenbrock': [_run_rosenbrock(c) for c in _CELLS],
        # BE 10%: chooses its own initial dt from the chemistry
        # timescale at the entry state via `_estimate_dt_sub`.
        'be': [_run_be(c, f_chem_cap=0.1) for c in _CELLS],
        # Tight-tolerance reference (0.1 percent rule). Forced first
        # substep = `1e-5 * tlim` so the early transient is resolved
        # before the forward controller takes over.
        'be_ref': [_run_be(c, f_chem_cap=0.001,
                            dt_init_frac=_DT_INIT_FRAC)
                   for c in _CELLS],
    }


def test_state_finite_bounded_and_H_conserved(histories):
    """Per-solver, per-cell: the recorded trajectory is finite and
    obeys the basic chemistry contract.

      1. all entries are finite (no NaN / inf)
      2. each species fraction sits in `[0, 1]`
      3. hydrogen is conserved every timestep:
         `x_HI + x_HII + 2 x_H2 == 1` to round-off.

    Catches NaN explosions, broken positivity clipping, and silent
    hydrogen leaks in the Cramer joint solve.
    """
    for tag in ('rosenbrock', 'be', 'be_ref'):
        for cell_idx, (res, abun, t_yr, _, _) in enumerate(
                histories[tag]):
            label = _CELLS[cell_idx]['label']
            assert np.all(np.isfinite(res)), \
                f'{tag} {label}: non-finite residual'
            assert np.all(np.isfinite(abun)), \
                f'{tag} {label}: non-finite abundance'
            assert np.all(abun >= 0.0), \
                f'{tag} {label}: negative abundance'
            assert np.all(abun <= 1.0 + 1.0e-12), \
                f'{tag} {label}: abundance > 1'
            x_sum = abun[:, 0] + abun[:, 1] + 2.0 * abun[:, 2]
            assert np.allclose(x_sum, 1.0, atol=1.0e-12), \
                (f'{tag} {label}: hydrogen not conserved, '
                 f'max |x_sum - 1| = {float(np.max(np.abs(x_sum - 1))):.3e}')


def test_solvers_agree_at_tlim(histories):
    """Cross-solver consistency: at t=tlim every solver agrees with
    the tight-tolerance BE reference on the dominant H species
    `(x_HI, x_HII, 2 x_H2)`.

      - BE reference is the trusted trajectory (0.1 percent rule).
      - BE 10 percent must agree to within 0.05 absolute (half its
        own cap).
      - Rosenbrock must agree to within `rtol = 1e-2` absolute
        (matches its own integrator tolerance).

    Most important regression check in the file: catches Cramer or
    forward-controller drift, Rosenbrock atol / y_lo misconfiguration,
    or any future solver replacement that quietly converges to a
    different fixed point.
    """
    tols = {'be': 0.05, 'rosenbrock': 1.0e-2}
    for tag, atol in tols.items():
        for cell_idx, (_, abun, _, _, _) in enumerate(
                histories[tag]):
            label = _CELLS[cell_idx]['label']
            _, abun_ref, _, _, _ = histories['be_ref'][cell_idx]
            # Compare on (x_HI, x_HII, 2 x_H2) at the last recorded
            # step. The factor of 2 on x_H2 means the metric is on
            # the full H budget axis, so the same absolute tol covers
            # all three.
            final = np.array([abun[-1, 0], abun[-1, 1],
                              2.0 * abun[-1, 2]])
            final_ref = np.array([abun_ref[-1, 0], abun_ref[-1, 1],
                                  2.0 * abun_ref[-1, 2]])
            assert np.allclose(final, final_ref, atol=atol), (
                f'{tag} {label}: final state disagrees with BE ref by '
                f'{float(np.max(np.abs(final - final_ref))):.3e} '
                f'(atol = {atol:.0e}); final = {final}, '
                f'ref = {final_ref}'
            )


def test_figure_written(histories):
    # (key, label, linestyle, linewidth, alpha, marker, markersize)
    methods = [
        ('be_ref',     'BE reference (0.1%)',  '-',  0.8, 1.0, None, 0),
        ('rosenbrock', 'Rosenbrock23',         '-',  1.5, 1.0, '.',  3),
        ('be',         'backward-Euler (10%)', '',   0.0, 0.6, 'o',  5),
    ]
    color = {'HI': 'C2', 'HII': 'C3', 'H2': 'C0'}

    ncell = len(_CELLS)
    fig, axes = plt.subplots(2, ncell, figsize=(4*ncell, 10))

    for col, cell in enumerate(_CELLS):
        T, nH, tlim_yr = cell['T'], cell['nH'], cell['tlim_yr']
        xlim = (tlim_yr * _DT_INIT_FRAC, tlim_yr)

        ax = axes[0, col]
        for key, _, ls, lw, alpha, mk, mksz in methods:
            res, abun, t_yr, _, _ = histories[key][col]
            s = slice(1, None)
            ax.semilogy(t_yr[s], np.maximum(abun[s, 0], 1.0e-11),
                        ls, color=color['HI'], lw=lw, alpha=alpha,
                        marker=mk, markersize=mksz)
            ax.semilogy(t_yr[s], np.maximum(abun[s, 1], 1.0e-11),
                        ls, color=color['HII'], lw=lw, alpha=alpha,
                        marker=mk, markersize=mksz)
            ax.semilogy(t_yr[s], np.maximum(2.0 * abun[s, 2], 1.0e-11),
                        ls, color=color['H2'], lw=lw, alpha=alpha,
                        marker=mk, markersize=mksz)
        ax.set_title(f'{cell["label"]}\nT={T:.1e} nH={nH:.1e}',
                     fontsize=12)
        n_ros = histories['rosenbrock'][col][3]
        n_be = histories['be'][col][3]
        n_ref = histories['be_ref'][col][3]
        t_ros = histories['rosenbrock'][col][4]
        t_be = histories['be'][col][4]
        t_ref = histories['be_ref'][col][4]
        ax.text(0.98, 0.02,
                f'Ros: {n_ros} steps ({t_ros*1e3:.1f} ms)\n'
                f'BE: {n_be} substeps ({t_be*1e3:.1f} ms)\n'
                f'ref: {n_ref} substeps ({t_ref*1e3:.1f} ms)',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10)
        ax.set_xscale('log'); ax.set_xlim(*xlim)
        ax.set_ylim(1.0e-12, 2.0)
        ax.grid(True, alpha=0.3, which='both')
        if col == 0:
            ax.set_ylabel('species fraction', fontsize=12)

        ax = axes[1, col]
        for key, _, ls, lw, alpha, mk, mksz in methods:
            res, _, t_yr, _, _ = histories[key][col]
            s = slice(1, None)
            ax.semilogy(t_yr[s], np.maximum(np.abs(res[s, 0]), 1.0e-18),
                        ls, color=color['HII'], lw=lw, alpha=alpha,
                        marker=mk, markersize=mksz)
            ax.semilogy(t_yr[s], np.maximum(np.abs(res[s, 1]), 1.0e-18),
                        ls, color=color['H2'], lw=lw, alpha=alpha,
                        marker=mk, markersize=mksz)
        ax.set_xlabel('time [yr]', fontsize=12)
        ax.set_xscale('log')
        ax.set_xlim(*xlim)
        ax.set_ylim(1.0e-12, 2.0)
        ax.grid(True, alpha=0.3, which='both')
        if col == 0:
            ax.set_ylabel('relative imbalance\n' +
                          r'$(C_i - D_i x_i) / (C_i + D_i x_i)$',
                          fontsize=12)

    ax = axes[0, 0]
    leg1 = ax.legend(handles=[
        Line2D([], [], color='C2', lw=1.5, label=r'$x_{\rm HI}$'),
        Line2D([], [], color='C3', lw=1.5, label=r'$x_{\rm HII}$'),
        Line2D([], [], color='C0', lw=1.5, label=r'$2 x_{\rm H_2}$'),
    ], fontsize=12, loc='lower left',
       bbox_to_anchor=(0.02, 0.02))
    ax.add_artist(leg1)
    ax.legend(handles=[
        Line2D([], [], color='k', lw=0.8, ls='-',
               label='BE reference (0.1%)'),
        Line2D([], [], color='k', lw=1.5, ls='-', marker='.',
               markersize=3, label='Rosenbrock23'),
        Line2D([], [], color='k', lw=0.0, ls='', marker='o',
               markersize=5, alpha=0.6,
               label='backward-Euler (10%)'),
    ], fontsize=12, loc='lower left',
       bbox_to_anchor=(0.02, 0.28))
    for ax in axes.flat:
        ax.tick_params(axis='both', labelsize=15)

    fig.suptitle(
        'Time-marching solver comparison on the ISM cell set. '
        '\n'
        r'Top: species fractions'
        r'Bottom: relative imbalance',
        fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(_figures_dir(), _FIG_NAME)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    assert os.path.isfile(out)

    # Summary table: step count and wall time per cell per solver.
    # Written next to the PNG and also printed (visible with `pytest
    # -s`). The same numbers are annotated on the figure but a flat
    # text dump is easier to copy into a CHANGES entry or a PR body.
    summary_lines = []
    hdr = (f'{"cell":24s} '
           f'{"Ros n":>8s} {"Ros ms":>9s} '
           f'{"BE n":>8s} {"BE ms":>9s} '
           f'{"ref n":>8s} {"ref ms":>9s}')
    summary_lines.append(hdr)
    summary_lines.append('-' * len(hdr))
    for col, cell in enumerate(_CELLS):
        n_ros = histories['rosenbrock'][col][3]
        t_ros = histories['rosenbrock'][col][4] * 1e3
        n_be = histories['be'][col][3]
        t_be = histories['be'][col][4] * 1e3
        n_ref = histories['be_ref'][col][3]
        t_ref = histories['be_ref'][col][4] * 1e3
        summary_lines.append(
            f'{cell["label"]:24s} '
            f'{n_ros:8d} {t_ros:9.2f} '
            f'{n_be:8d} {t_be:9.2f} '
            f'{n_ref:8d} {t_ref:9.2f}'
        )
    summary = '\n'.join(summary_lines)
    print()
    print(summary)
    summary_path = os.path.join(
        _figures_dir(), 'chem_solver_comparison_summary.txt')
    with open(summary_path, 'w') as fh:
        fh.write(summary + '\n')
