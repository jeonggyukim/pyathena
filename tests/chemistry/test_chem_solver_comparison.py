"""Time-marching equilibrium-solver comparison figure.

Runs Rosenbrock23 and BE-marched (ExplicitSubcyclingSolver) on each
cell of `_CELLS` independently, recording per-step abundances and
relative imbalance. Writes
`tests/figures/equilibrium_solver_convergence.png`.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pytest

from pyathena.chemistry.config import ChemistryConfig
from pyathena.chemistry.equilibrium_solver_v2 import _allocate_CD
from pyathena.chemistry.solvers._rosenbrock_chemistry_adapter import (
    make_chemistry_f_jac, integrate_rosenbrock23,
)
from pyathena.chemistry.solvers.explicit_subcycling import (
    ExplicitSubcyclingSolver,
)
from pyathena.chemistry.thermo.ncr import NCRThermo

from .test_equilibrium_solver import _build_strip, _CELLS


_FIG_NAME = 'equilibrium_solver_convergence.png'
_SEC_PER_YR = 3.155693e7


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


def _run_rosenbrock(cell, max_steps=400):
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
    for _ in range(max_steps):
        y, result = integrate_rosenbrock23(
            f=f, jac=jac, y0=y, t_target=t_target,
            h_init=float(h_init_arr[0]), max_steps=1,
            atol=1.0e-14, rtol=1.0e-6,
            y_lo=1.0e-20, y_hi=1.0 - 1.0e-12,
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
    return (np.array(res_hist), np.array(abun_hist),
            np.array(t_hist) / _SEC_PER_YR)


def _run_be(cell, n_samples=100):
    """BE-marched with log-spaced sample times. Each call advances
    `dt = t_next - t_prev`; ExplicitSubcyclingSolver subcycles
    internally with cfl_cool_sub * 1/|C-Dx|."""
    state, network, species = _build_strip(cells=[cell])
    idx = species.idx
    newton_idx = _newton_idx_cached(network, species)
    solver = ExplicitSubcyclingSolver(
        ChemistryConfig(), network, NCRThermo())
    solver.allocate_scratch(state)
    tlim_yr = cell['tlim_yr']
    t_samples_yr = np.geomspace(tlim_yr * 1.0e-4, tlim_yr, n_samples)
    abun0, res0 = _record(network, state, newton_idx, idx)
    abun_hist, res_hist, t_hist = [abun0], [res0], [0.0]
    t_prev = 0.0
    for t_next in t_samples_yr:
        solver.step((t_next - t_prev) * _SEC_PER_YR, state)
        abun, res = _record(network, state, newton_idx, idx)
        abun_hist.append(abun)
        res_hist.append(res)
        t_hist.append(float(t_next))
        t_prev = t_next
    return (np.array(res_hist), np.array(abun_hist), np.array(t_hist))


@pytest.fixture(scope='module')
def histories():
    return {
        'rosenbrock': [_run_rosenbrock(c) for c in _CELLS],
        'be': [_run_be(c) for c in _CELLS],
    }


def test_rosenbrock_runs(histories):
    for res, _, _ in histories['rosenbrock']:
        assert res.shape[0] >= 2


def test_be_runs(histories):
    for res, _, _ in histories['be']:
        assert res.shape[0] >= 2


def test_figure_written(histories):
    methods = [
        ('rosenbrock', 'Rosenbrock23', '-'),
        ('be', 'BE-marched', '--'),
    ]
    color = {'HI': 'C2', 'HII': 'C3', 'H2': 'C0'}

    ncell = len(_CELLS)
    fig, axes = plt.subplots(2, ncell, figsize=(3.4 * ncell, 7))
    for col, cell in enumerate(_CELLS):
        T, nH, tlim_yr = cell['T'], cell['nH'], cell['tlim_yr']
        xlim = (tlim_yr * 1.0e-4, tlim_yr)

        ax = axes[0, col]
        for key, _, ls in methods:
            res, abun, t_yr = histories[key][col]
            s = slice(1, None)
            ax.semilogy(t_yr[s], np.maximum(abun[s, 0], 1.0e-9),
                        ls, color=color['HI'], lw=1.5)
            ax.semilogy(t_yr[s], np.maximum(abun[s, 1], 1.0e-9),
                        ls, color=color['HII'], lw=1.5)
            ax.semilogy(t_yr[s], np.maximum(abun[s, 2], 1.0e-9),
                        ls, color=color['H2'], lw=1.5)
        ax.set_title(f'{cell["label"]}\nT={T:.1e} nH={nH:.1e}',
                     fontsize=10)
        ax.set_xscale('log'); ax.set_xlim(*xlim)
        ax.set_ylim(1.0e-8, 2.0)
        ax.grid(True, alpha=0.3, which='both')
        if col == 0:
            ax.set_ylabel('species fraction')

        ax = axes[1, col]
        for key, _, ls in methods:
            res, _, t_yr = histories[key][col]
            s = slice(1, None)
            ax.semilogy(t_yr[s], np.maximum(np.abs(res[s, 0]), 1.0e-18),
                        ls, color=color['HII'], lw=1.5)
            ax.semilogy(t_yr[s], np.maximum(np.abs(res[s, 1]), 1.0e-18),
                        ls, color=color['H2'], lw=1.5)
        ax.set_xlabel('t [yr]')
        ax.set_xscale('log'); ax.set_xlim(*xlim)
        ax.set_ylim(1.0e-8, 2.0)
        ax.grid(True, alpha=0.3, which='both')
        if col == 0:
            ax.set_ylabel('relative imbalance')

    axes[0, 0].add_artist(axes[0, 0].legend(handles=[
        Line2D([], [], color='C2', lw=1.5, label=r'$x_{\rm HI}$'),
        Line2D([], [], color='C3', lw=1.5, label=r'$x_{\rm HII}$'),
        Line2D([], [], color='C0', lw=1.5, label=r'$x_{\rm H_2}$'),
    ], fontsize=9, loc='lower left'))
    axes[0, 0].legend(handles=[
        Line2D([], [], color='k', lw=1.5, ls='-', label='Rosenbrock23'),
        Line2D([], [], color='k', lw=1.5, ls='--', label='BE-marched'),
    ], fontsize=9, loc='lower right')

    fig.suptitle(
        'Time-marching solver comparison on the ISM cell set. '
        r'Top: species fractions vs $t$. '
        r'Bottom: $(C_i - D_i x_i) / (C_i + D_i x_i)$ per Newton species.',
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = os.path.join(_figures_dir(), _FIG_NAME)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    assert os.path.isfile(out)
