"""Time-dependent chemistry against the CIE cooling curve.

With photoionization and cosmic-ray ionization switched off, the
time-dependent network has no non-thermal driver, so it relaxes to
collisional ionization equilibrium. Its cooling should then agree with
a CIE table computed independently, and should join smoothly across the
temperature where the cold-ISM channels fade out.

That makes this the end-to-end acceptance check for the cooling path.
A disagreement here is a defect, not physics, and the shape of the
disagreement says which one:

- a constant factor offset -> the three normalization conventions
  (`Lambda_e` per H per e, `Lambda_NCR` per H^2, per-H channel output)
  crossed wrong somewhere;
- a step or kink at a fade window -> `T_FADE` in the wrong place, or a
  channel switching without a ramp;
- a slow drift with temperature -> the ionization balance not actually
  relaxing to CIE.

`NCRNetwork3` tracks hydrogen only, with metals as ghosts, so today
this compares against the hydrogen CIE curve. The multi-ion version of
the same figure is the acceptance test for the CIE pool design in
`ai-notes/docs-claude/pyathena/cie-pool-cooling-design.md`, and it
needs Phase 6.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry.config import ChemistryConfig
from pyathena.chemistry.cooling.factories import make_ncr_default_cooling
from pyathena.chemistry.networks.ncr3 import NCRNetwork3
from pyathena.chemistry.solvers.explicit_subcycling import (
    ExplicitSubcyclingSolver,
)
from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import ChemState
from pyathena.chemistry.thermo.ncr import NCRThermo


_NH = 1.0e-2          # cm^-3, low enough to stay in the low-density limit
_T = np.logspace(3.7, 8.0, 60)


def _relaxed_state(T):
    """Evolve the chemistry to equilibrium at fixed temperature.

    `cooling=None` leaves the temperature untouched, so the solver
    advances composition only. With no radiation and no cosmic rays the
    fixed point is collisional ionization equilibrium.
    """
    species = SpeciesSet.ncr3_with_ghosts()
    n = T.size
    state = ChemState.from_grid(r=np.arange(n, dtype=np.float64),
                                nH=np.full(n, _NH), T=T.copy(),
                                species=species, xi_CR=0.0, nfreq=3)
    idx = species.idx
    state.x[idx['HI']] = 1.0
    state.x[idx['HII']] = 0.0
    state.x[idx['H2']] = 0.0
    state.x[idx['electron']] = 0.0
    state.chi_FUV = np.zeros(n)
    state.zeta_pi = {'HI': np.zeros(n), 'H2': np.zeros(n)}
    state.zeta_diss = {'H2': np.zeros(n)}

    solver = ExplicitSubcyclingSolver(ChemistryConfig(), NCRNetwork3(),
                                      NCRThermo(), cooling=None)
    solver.allocate_scratch(state)
    # Long enough for the slowest recombination at this density.
    for _ in range(40):
        solver.step(3.15e15, state)     # 100 Myr per call
    return state, species


def _cooling_lambda_e(state, species):
    """Net cooling as Lambda_e, the per-H-per-e convention."""
    cooling = make_ncr_default_cooling(species)
    n = state.T.size
    state.alloc_scratch('solver:net_cool', (n,))
    state.alloc_scratch('solver:d_net_cool_d_temp_mu', (n,))
    state.alloc_scratch('solver:mu_at_entry', (n,))
    state.scratch['solver:mu_at_entry'][:] = 1.0
    cooling.allocate_scratch(state)
    cooling.update(state)
    net = state.get_scratch('solver:net_cool')     # erg / s / cm^3
    ne = state.nH * state.x[species.idx['electron']]
    return net / np.maximum(state.nH * ne, 1.0e-300), cooling


@pytest.fixture(scope='module')
def relaxed():
    state, species = _relaxed_state(_T)
    lam, cooling = _cooling_lambda_e(state, species)
    return state, species, lam, cooling


def test_relaxes_to_a_sensible_ionization_state(relaxed):
    """Hydrogen is neutral in cold gas and ionized in hot gas.

    A weak check, but it fails loudly if the run has not relaxed, which
    would make every other comparison here meaningless.
    """
    state, species, _, _ = relaxed
    xHII = state.x[species.idx['HII']]
    assert xHII[0] < 0.5, f'not neutral at {_T[0]:.3g} K: {xHII[0]:.3g}'
    assert xHII[-1] > 0.95, f'not ionized at {_T[-1]:.3g} K: {xHII[-1]:.3g}'
    assert np.all(np.diff(xHII) >= -1.0e-6), 'ionization is not monotone'


def test_cooling_is_smooth_across_the_fade_window(relaxed):
    """No step or kink where the dust channels switch off.

    The fade runs from 2e4 to 3.5e4 K. In log-log the curve should have
    no jump in slope there beyond what the physics already produces, so
    the check is that successive log-slopes stay bounded.
    """
    _, _, lam, _ = relaxed
    good = lam > 0
    ln_l = np.log(lam[good])
    ln_T = np.log(_T[good])
    slope = np.diff(ln_l) / np.diff(ln_T)
    # A hard cutoff of a channel carrying a tenth of the total would
    # show as a slope spike of order 10 over one bin.
    assert np.abs(slope).max() < 8.0, (
        f'log-slope spike {np.abs(slope).max():.2f}; '
        'a channel is probably switching without a ramp')


def test_hot_gas_matches_hydrogen_cie(relaxed):
    """Above 1e6 K the only hydrogen coolant left is free-free.

    The network carries no helium and no metals, so the total must sit
    near the hydrogen-only CIE value rather than the full curve. This
    pins the normalization: a factor of n_H or x_e in the wrong place
    shows up here immediately.
    """
    _, _, lam, _ = relaxed
    hot = _T > 3.0e6
    # Draine 2011 10.11 free-free with gff ~ 1.2, per H per e.
    ff = 1.422e-25 * 1.2 * np.sqrt(_T[hot] / 1.0e4)
    np.testing.assert_allclose(lam[hot], ff, rtol=0.35)


def test_cie_consistency_figure(relaxed, figures_dir, save_figures):
    """Element-by-element comparison, hydrogen for now.

    Overlays the relaxed time-dependent cooling on the CHIANTI CIE
    curves from `cool_gnat12`, per channel and in total.
    """
    if not save_figures:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    state, species, lam, cooling = relaxed
    ne = state.nH * state.x[species.idx['electron']]
    norm = np.maximum(state.nH * ne, 1.0e-300)

    fig, ax = plt.subplots(2, 1, figsize=(7.5, 8.5), sharex=True,
                           height_ratios=(2, 1))

    for ch in cooling._channels:
        v = state.get_scratch(f'cooling:Lambda:{ch.name}') * state.nH / norm
        if np.nanmax(v) > 1.0e-30:
            ax[0].loglog(_T, np.abs(v), lw=1, alpha=0.75, label=ch.name)
    ax[0].loglog(_T, lam, 'k-', lw=2.5, label='net (time-dependent)')

    try:
        from pyathena.microphysics.cool_gnat12 import CoolGnat12
        g = CoolGnat12(read_all=True)
        g.get_cool_cie_total()
        ax[0].loglog(g.temp, g.cool['H'], 'r--', lw=2,
                     label='CIE H only (GF12)')
        ax[0].loglog(g.temp, g.cool_tot, '0.5', ls='--', lw=2,
                     label='CIE all elements (GF12)')
        ref = np.interp(_T, g.temp, g.cool['H'])
        ax[1].semilogx(_T, lam / ref, 'k-')
        ax[1].axhline(1.0, color='r', ls='--', lw=1)
        ax[1].set_ylabel('net / CIE(H)')
        ax[1].set_ylim(0, 3)
    except Exception as exc:                       # noqa: BLE001
        ax[1].text(0.5, 0.5, f'GF12 unavailable: {exc}',
                   ha='center', transform=ax[1].transAxes)

    for a in ax:
        a.axvspan(2.0e4, 3.5e4, color='0.9', zorder=0)
    ax[0].set_ylim(1e-27, 1e-21)
    ax[0].set_ylabel(r'$\Lambda_e$ [erg cm$^3$ s$^{-1}$] (per H per e)')
    ax[0].set_title('Time-dependent chemistry relaxed with no radiation\n'
                    f'and no cosmic rays, $n_H$ = {_NH:g} cm$^{{-3}}$; '
                    'shaded band is the dust fade window')
    ax[0].legend(fontsize=7, ncol=2, loc='lower right')
    ax[1].set_xlabel('$T$ [K]')
    fig.tight_layout()
    fig.savefig(figures_dir / 'cie_consistency_hydrogen.png', dpi=120)
    plt.close(fig)
