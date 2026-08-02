"""Per-channel temperature fade in `CoolingChannels`.

A channel declares `T_FADE = (T_full, T_off)` and the aggregator ramps
its contribution to zero over that window with a C2 smootherstep. This
is the mechanism for both the hot-gas gate (cold-ISM fits must not run
at 1e6 K) and the per-ion caps in `cie-pool-cooling-design.md`.

The property that matters most here is the derivative. With
`L_eff = w(T) L(T)` the product rule contributes `L dw/dT`, and
dropping it would leave `d_net_cool_d_temp_mu` wrong exactly inside the
ramp, which is where the solver most needs it right.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry.cooling.base import CoolingChannel, CoolingChannels
from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import ChemState


_MU = 1.2


class _LinearChannel(CoolingChannel):
    """Lambda = T, with the exact derivative. Any error in the fade
    shows up undiluted by a channel fit.
    """

    name = 'Linear'
    SCRATCH_NAMES = ()

    def evaluate(self, state, out, d_out=None):
        np.copyto(out, state.T)
        if d_out is not None:
            # dLambda/dT = 1, converted to temp_mu space by * mu.
            d_out[...] = state.get_scratch('solver:mu_at_entry')


class _FadedChannel(_LinearChannel):
    name = 'Faded'
    T_FADE = (1.0e4, 1.0e5)          # full at 1e4, off by 1e5


def _build(T):
    species = SpeciesSet.ncr3_with_ghosts()
    n = T.size
    state = ChemState.from_grid(r=np.arange(n, dtype=np.float64),
                                nH=np.ones(n), T=T.copy(), species=species)
    state.alloc_scratch('solver:net_cool', (n,))
    state.alloc_scratch('solver:d_net_cool_d_temp_mu', (n,))
    state.alloc_scratch('solver:mu_at_entry', (n,))
    state.scratch['solver:mu_at_entry'][:] = _MU
    return state


def _net(channel_cls, T):
    state = _build(T)
    cooling = CoolingChannels(channels=(channel_cls(),))
    cooling.allocate_scratch(state)
    cooling.update(state)
    return (state.get_scratch('solver:net_cool').copy(),
            state.get_scratch('solver:d_net_cool_d_temp_mu').copy())


def test_no_fade_declared_leaves_the_channel_untouched():
    T = np.logspace(2.0, 7.0, 40)
    plain, _ = _net(_LinearChannel, T)
    np.testing.assert_allclose(plain, T * 1.0, rtol=1.0e-14)


def test_weight_is_one_below_and_zero_above():
    T = np.array([1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7])
    faded, _ = _net(_FadedChannel, T)
    plain, _ = _net(_LinearChannel, T)
    # Full strength at and below T_full.
    np.testing.assert_allclose(faded[:3], plain[:3], rtol=1.0e-14)
    # Exactly off at and above T_off -- not merely small.
    assert np.all(faded[3:] == 0.0)


def test_weight_is_monotone_inside_the_ramp():
    T = np.linspace(1.0e4, 1.0e5, 200)
    faded, _ = _net(_FadedChannel, T)
    plain, _ = _net(_LinearChannel, T)
    w = faded / plain
    assert np.all(np.diff(w) <= 1.0e-14)
    assert w[0] == pytest.approx(1.0, abs=1.0e-14)
    assert w[-1] == pytest.approx(0.0, abs=1.0e-14)


def test_faded_derivative_matches_finite_difference():
    """The product-rule check. Scaling d_out alone would fail this."""
    T = np.linspace(1.2e4, 9.0e4, 120)      # strictly inside the ramp
    dT = 1.0
    lam_hi, _ = _net(_FadedChannel, T + dT)
    lam_lo, _ = _net(_FadedChannel, T - dT)
    fd = (lam_hi - lam_lo) / (2.0 * dT)      # dLambda/dT

    _, d_analytic = _net(_FadedChannel, T)
    # Channels report d/d(temp_mu) = mu * d/dT; net_cool carries the
    # per-H to per-volume factor of nH, which is 1 here.
    np.testing.assert_allclose(d_analytic / _MU, fd, rtol=2.0e-6)


def test_derivative_is_continuous_at_the_ramp_edges():
    """C2 in the weight means the reported derivative is C1.

    Continuity is checked by closing in on each edge: the gap between
    the value just inside and just outside must shrink with the sample
    spacing. A hard cutoff would leave a fixed jump, so the gap would
    stop shrinking.
    """
    for edge in (1.0e4, 1.0e5):
        gaps = []
        for eps in (1.0e-3, 1.0e-4, 1.0e-5):
            _, d = _net(_FadedChannel,
                        np.array([edge * (1.0 - eps), edge * (1.0 + eps)]))
            gaps.append(abs(d[1] - d[0]))
        # Each tenfold refinement must shrink the gap by at least 3x.
        # For a genuine step the gap would be flat.
        assert gaps[1] < gaps[0] / 3.0, f'edge {edge:g}: {gaps}'
        assert gaps[2] < gaps[1] / 3.0, f'edge {edge:g}: {gaps}'


def test_fade_figure(figures_dir, save_figures):
    """Diagnostic plot of the fade windows actually in use.

    Top panel shows every production `T_FADE` on one axis, so the
    windows can be checked against where each ion or grain population
    is expected to survive. Lower panels verify the machinery on one
    channel: the faded value, and the reported derivative against a
    finite difference.
    """
    if not save_figures:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from pyathena.chemistry.cooling.factories import (
        make_ncr_default_cooling,
    )
    from pyathena.chemistry.species import SpeciesSet

    cooling = make_ncr_default_cooling(SpeciesSet.ncr3_with_ghosts())
    faded_channels = [
        ch for ch in tuple(cooling._channels) + tuple(cooling._heating)
        if ch.T_FADE is not None
    ]

    T = np.logspace(3.0, 8.0, 2000)
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 9.5), sharex=True)

    w_buf, dw_buf, tmp = (np.empty(T.size) for _ in range(3))
    for ch in faded_channels:
        CoolingChannels._fade_weight(T, ch.T_FADE, w_buf, dw_buf, tmp)
        ax[0].semilogx(T, w_buf.copy(), lw=1.6,
                       label=f'{ch.name}  {ch.T_FADE}')
    ax[0].set_ylabel('fade weight $w(T)$')
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_title('Production T_FADE windows (C2 smootherstep)')
    ax[0].legend(fontsize=7, loc='center left')

    # Machinery check on one channel, on its own window.
    Tm = np.logspace(3.5, 6.5, 1500)
    plain, _ = _net(_LinearChannel, Tm)
    faded, d_analytic = _net(_FadedChannel, Tm)
    dT = Tm * 1.0e-6
    hi, _ = _net(_FadedChannel, Tm + dT)
    lo, _ = _net(_FadedChannel, Tm - dT)
    fd = (hi - lo) / (2.0 * dT)

    ax[1].loglog(Tm, plain, '0.6', lw=3, label='unfaded $\\Lambda = T$')
    ax[1].loglog(Tm, faded, 'k-', label='faded')
    ax[1].set_ylabel('$\\Lambda$  (test channel)')
    ax[1].set_ylim(1e2, 2e6)
    ax[1].legend(fontsize=8)

    ax[2].semilogx(Tm, fd, '0.6', lw=3, label='finite difference')
    ax[2].semilogx(Tm, d_analytic / _MU, 'k--',
                   label='analytic (product rule)')
    ax[2].set_ylabel('$d\\Lambda/dT$  (test channel)')
    ax[2].set_xlabel('$T$ [K]')
    ax[2].legend(fontsize=8)
    for e in _FadedChannel.T_FADE:
        for a in ax[1:]:
            a.axvline(e, color='0.7', ls=':', lw=1)

    fig.tight_layout()
    fig.savefig(figures_dir / 'cooling_fade_weight.png', dpi=120)
    plt.close(fig)


def test_fade_direction_can_be_reversed():
    """`T_full > T_off` fades the channel out towards low temperature."""

    class _LowFade(_LinearChannel):
        name = 'LowFade'
        T_FADE = (1.0e5, 1.0e4)      # full at 1e5, off by 1e4

    T = np.array([1.0e3, 1.0e4, 1.0e5, 1.0e6])
    faded, _ = _net(_LowFade, T)
    plain, _ = _net(_LinearChannel, T)
    assert np.all(faded[:2] == 0.0)
    np.testing.assert_allclose(faded[2:], plain[2:], rtol=1.0e-14)
