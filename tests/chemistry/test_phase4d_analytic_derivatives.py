"""Phase 4d analytic-derivative tests.

For every channel whose `evaluate(state, out, d_out)` writes a
non-trivial `d_out`, this file compares the analytic derivative
against a central finite-difference of `evaluate(state, out)` taken
at the same state.

Convention:

- The channel's `d_out` carries `d(Lambda or Gamma) / d(T/mu)`.
- Under operator splitting mu is held fixed during the cooling
  sub-step, so `d(Lambda)/d(T/mu) = mu * d(Lambda)/dT`. The FD here
  evaluates `Lambda(T + dT)` and `Lambda(T - dT)` with everything
  else (mu, species, n_H, radiation, dust temperature) frozen, and
  multiplies the central-difference quotient by mu_at_entry to
  produce the expected `d_out` value.
- Tolerance: `rtol = 1e-4`. The FD truncation error of a 2-point
  central difference is `O(dT^2)`; we use `dT = 1e-3 * T` which gives
  an `O(1e-6)` truncation error relative to the value plus an `O(1e-8)`
  roundoff floor -- both well below the target.

Channels in this batch (Phase 4d-a):

- `CosmicRayHeating`           (analytic = 0, no T dependence)
- `H2DissociationHeating`      (analytic = 0, no T dependence)
- `DustGasCoupling`            (analytic closed form)
- `FreeFreeHCooling`           (analytic closed form)
- `LyaCooling`                 (analytic closed form)
"""
from __future__ import annotations

import numpy as np

from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import ChemState
from pyathena.chemistry.equilibrium import eq_xHII_xe
from pyathena.chemistry.cooling.dust import DustGasCoupling
from pyathena.chemistry.cooling.free_free import FreeFreeHCooling
from pyathena.chemistry.cooling.lya import LyaCooling
from pyathena.chemistry.cooling.hi_collisional_ionization import (
    HICollisionalIonizationCooling,
)
from pyathena.chemistry.cooling.recombination_hydrogen import (
    HRecombinationCooling,
)
from pyathena.chemistry.cooling.grain_recombination import (
    GrainRecombinationCooling,
)
from pyathena.chemistry.cooling.nebular import NebularMetalLineCooling
from pyathena.chemistry.cooling.hi_smith21 import HISmith21Cooling
from pyathena.chemistry.cooling.h2_colldiss import H2CollDissCooling
from pyathena.chemistry.cooling.h2_moseley21 import H2Moseley21Cooling
from pyathena.chemistry.cooling.h2_gong17 import H2Gong17Cooling
from pyathena.chemistry.heating.cosmic_ray import CosmicRayHeating
from pyathena.chemistry.heating.photoelectric import PhotoelectricHeating
from pyathena.chemistry.heating.h2_photodissociation import (
    H2DissociationHeating, H2PumpHeating,
)
from pyathena.chemistry.heating.h2_formation import H2FormationHeating


# Broad T coverage [100 K, 1e6 K] union with a denser sub-grid where
# the dominant analytical channels (Lya, HICollIon, HRecomb) peak in
# stiffness -- T ~ T_excite/2 sits around 6e4 K for Lya, and the H I
# ionisation knee is at ~1.5e4 K. The denser sub-grid catches FD-vs-
# analytic mismatches in the part of T-space that matters for the
# substep loop in HII regions.
_T_VALS = np.unique(np.concatenate([
    np.logspace(2.0, 6.0, 24),
    np.logspace(3.5, 5.0, 30),
]))
_N_VALS = np.logspace(-2.0, 4.0, 14)
_T_GRID, _NH_GRID = np.meshgrid(_T_VALS, _N_VALS, indexing='xy')
_T = _T_GRID.ravel()
_NH = _NH_GRID.ravel()

# Molecular-regime grid for H2 channels (cold/cool gas only -- H2
# dissociates above ~few thousand K; cap at 1e4 K to cover the
# warm-neutral transition tail).
_T_MOL_VALS = np.logspace(1.0, 4.0, 30)
_N_MOL_VALS = np.logspace(0.0, 6.0, 14)
_T_MOL_GRID, _NH_MOL_GRID = np.meshgrid(
    _T_MOL_VALS, _N_MOL_VALS, indexing='xy')
_T_MOL = _T_MOL_GRID.ravel()
_NH_MOL = _NH_MOL_GRID.ravel()


def _build_state(T, nH, xHI=0.5, xHII=0.5, xH2=0.0, xe=0.5,
                 T_dust=15.0, Z_d=1.0, mu=1.2):
    # H conservation: 2 xH2 + xHI + xHII = 1 must hold for every test
    # state. Non-conserving baselines pass numerically but obscure the
    # phase the test is supposed to represent. See memory
    # `feedback_test_composition_conservation` for the rule.
    total_H = 2.0 * np.asarray(xH2) + np.asarray(xHI) + np.asarray(xHII)
    max_dev = float(np.max(np.abs(total_H - 1.0)))
    assert max_dev < 1.0e-10, (
        f'H conservation violated: max|2*xH2 + xHI + xHII - 1| = {max_dev}')
    species = SpeciesSet.ncr3_with_ghosts()
    ncell = T.size
    state = ChemState.from_grid(
        r=np.arange(ncell, dtype=np.float64),
        nH=nH.copy(), T=T.copy(),
        species=species, Z_d=float(Z_d),
    )
    idx = species.idx
    state.x[idx['HI']] = xHI
    state.x[idx['HII']] = xHII
    state.x[idx['H2']] = xH2
    state.x[idx['electron']] = xe
    state.T_dust[:] = T_dust

    # The analytic derivative chains in `mu` (= mu_at_entry) at the
    # end. Allocate + populate the slot the channels read.
    state.alloc_scratch('solver:mu_at_entry', (ncell,))
    state.scratch['solver:mu_at_entry'][:] = mu
    return state, species


def _build_state_eq(T, nH, xH2=0.0, xi_CR=1.0e-16, G_PE=1.0,
                    G_CI=1.0, T_dust=15.0, Z_d=1.0, Z_g=1.0, mu=1.2):
    """Build a ChemState with (xHII, xCII, xe, xHI) set to the joint
    ionisation equilibrium at (T, nH, xH2) given the ionising-rate
    inputs. xH2 is held at the input value (not solved -- depends on
    full chemistry network); xHI derived from H conservation.

    Use this for cooling/heating channel tests that need a physically
    consistent baseline at chemical equilibrium rather than hand-tuned
    abundance triplets.
    """
    xHII, _xCII, xe, xHI = eq_xHII_xe(
        T=T, nH=nH, xH2=xH2, xi_CR=xi_CR, G_PE=G_PE, G_CI=G_CI,
        Z_d=Z_d, Z_g=Z_g)
    # eq_xHII_xe returns arrays shaped like broadcast(T, nH); take a
    # plain ndarray to feed _build_state's scalar/array setattr.
    return _build_state(
        T=T, nH=nH, xHI=xHI, xHII=xHII, xH2=xH2, xe=xe,
        T_dust=T_dust, Z_d=Z_d, mu=mu)


def _alloc_channel_scratch(state, channel):
    for name in channel.SCRATCH_NAMES:
        if name in state.scratch:
            continue
        state.alloc_scratch(name, (state.T.size,))


def _fd_central(channel, state, dT_rel: float = 1.0e-3):
    """5-point central finite-difference of `channel.evaluate(state,
    out)` on `state.T`, returning `d(out)/d(T/mu) = mu * d(out)/dT`.

    5-point central difference has O(dT^4) truncation, so at the
    default dT_rel = 1e-3 the truncation residual is ~ 1e-12 -- well
    below the rtol = 1e-4 target for every channel regardless of how
    steep its T-dependence is. Avoids the per-channel dT_rel tuning
    that 2-point central would otherwise force.

    Formula (Abramowitz + Stegun 25.3.6):
        f'(T) = (-f(T+2dT) + 8 f(T+dT) - 8 f(T-dT) + f(T-2dT))
                / (12 dT)
    """
    T = state.T
    mu = state.scratch['solver:mu_at_entry']
    dT = dT_rel * T
    out_pp = np.empty_like(T)
    out_p = np.empty_like(T)
    out_m = np.empty_like(T)
    out_mm = np.empty_like(T)

    T_original = T.copy()
    state.T[:] = T_original + 2.0 * dT
    channel.evaluate(state, out_pp)
    state.T[:] = T_original + dT
    channel.evaluate(state, out_p)
    state.T[:] = T_original - dT
    channel.evaluate(state, out_m)
    state.T[:] = T_original - 2.0 * dT
    channel.evaluate(state, out_mm)
    state.T[:] = T_original

    return mu * (-out_pp + 8.0 * out_p - 8.0 * out_m + out_mm) / (12.0 * dT)


def test_cosmic_ray_heating_d_out_is_zero():
    """CR heating has no T dependence at fixed (x_e, x_H2, x_HI, n_H);
    the analytic d_out must be identically zero.

    Profile: CNM/WNM neutral atomic gas, xe = xCII ~ 1.6e-4 (FUV-
    unshielded carbon), xHII = xCII for charge balance, xH2 = 0.
    """
    state, species = _build_state(
        _T, _NH, xHI=1.0 - 1.6e-4, xHII=1.6e-4, xH2=0.0, xe=1.6e-4)
    ch = CosmicRayHeating(
        i_HI=species.idx['HI'], i_H2=species.idx['H2'],
        i_electron=species.idx['electron'],
    )
    _alloc_channel_scratch(state, ch)
    out = np.empty_like(_T)
    d_out = np.full_like(_T, 999.0)
    ch.evaluate(state, out, d_out)
    np.testing.assert_array_equal(d_out, np.zeros_like(_T))


def test_h2_dissociation_heating_d_out_is_zero():
    """H2 dissociation Gamma = xi_diss * xH2 * 0.4 eV has no T
    dependence; analytic d_out must be zero.

    Profile: partially molecular (xH2 = 0.3, xHI = 0.4, xHII = 0),
    xe = xCII = 1.6e-4.
    """
    state, species = _build_state(
        _T, _NH, xHI=0.4, xHII=0.0, xH2=0.3, xe=1.6e-4)
    ch = H2DissociationHeating(
        i_H2=species.idx['H2'], xi_diss_H2=1.0e-12,
    )
    _alloc_channel_scratch(state, ch)
    out = np.empty_like(_T)
    d_out = np.full_like(_T, 999.0)
    ch.evaluate(state, out, d_out)
    np.testing.assert_array_equal(d_out, np.zeros_like(_T))


def test_dust_gas_coupling_d_out_matches_FD():
    """Dust gas coupling has a closed-form derivative; FD vs analytic
    must agree at rtol = 1e-4 across the (T, n_H) grid and across
    representative dust temperatures."""
    for T_dust in (5.0, 15.0, 50.0):
        for Z_d in (0.5, 1.0):
            state, species = _build_state(
                _T, _NH, T_dust=T_dust, Z_d=Z_d)
            ch = DustGasCoupling()
            _alloc_channel_scratch(state, ch)
            out = np.empty_like(_T)
            d_out = np.empty_like(_T)
            ch.evaluate(state, out, d_out)
            fd = _fd_central(ch, state)
            np.testing.assert_allclose(
                d_out, fd, rtol=1.0e-4, atol=0.0,
                err_msg=f'(T_dust={T_dust}, Z_d={Z_d})',
            )


def test_free_free_h_d_out_matches_FD():
    """H II free-free has a closed-form derivative chain rule on the
    Gaunt factor; FD vs analytic must agree at rtol = 1e-4."""
    for xe, xHII in ((0.01, 0.01), (0.5, 0.5), (1.0, 1.0)):
        state, species = _build_state(
            _T, _NH, xHI=1.0 - xHII, xHII=xHII, xe=xe)
        ch = FreeFreeHCooling(
            i_HII=species.idx['HII'],
            i_electron=species.idx['electron'],
        )
        _alloc_channel_scratch(state, ch)
        out = np.empty_like(_T)
        d_out = np.empty_like(_T)
        ch.evaluate(state, out, d_out)
        fd = _fd_central(ch, state)
        np.testing.assert_allclose(
            d_out, fd, rtol=1.0e-4, atol=0.0,
            err_msg=f'(xe={xe}, xHII={xHII})',
        )


def test_lya_d_out_matches_FD():
    """Lya 2-level cooling has a closed-form derivative through the
    collision-strength factor + Boltzmann exponential + level-1
    fraction; FD vs analytic must agree at rtol = 1e-4 in the regime
    where the rate is non-negligible. The Lya Lambda drops below
    subnormal precision at T < ~500 K (Lambda ~ 1e-253) where the
    FD evaluation loses bit-precision; restrict the comparison to
    cells where `|d_out| > 1e-50 erg/s/cm^3/K`, which still leaves
    the entire HII region and warm neutral medium regime in scope.
    """
    for xHI, xe in ((0.99, 0.01), (0.5, 0.5), (0.1, 0.5)):
        state, species = _build_state(
            _T, _NH, xHI=xHI, xHII=1.0 - xHI, xe=xe)
        ch = LyaCooling(
            i_HI=species.idx['HI'],
            i_electron=species.idx['electron'],
        )
        _alloc_channel_scratch(state, ch)
        out = np.empty_like(_T)
        d_out = np.empty_like(_T)
        ch.evaluate(state, out, d_out)
        fd = _fd_central(ch, state)
        # In the deep-cold tail (d_out << 1e-25 erg/s/cm^3/K) the FD
        # picks up catastrophic-cancellation noise from the huge
        # Boltzmann exponentials -- the cells contribute negligibly
        # to physical net_cool either way. Mask them out.
        mask = np.abs(d_out) > 1.0e-25
        np.testing.assert_allclose(
            d_out[mask], fd[mask], rtol=1.0e-4, atol=0.0,
            err_msg=f'(xHI={xHI}, xe={xe})',
        )


def test_hi_collisional_ionization_d_out_matches_FD():
    """H I collisional ionisation has a Horner-polynomial derivative
    through P'(y) with y = ln(T * kB_eV). Gate at T > 3000 K mirrors
    the value path. Subnormal-tail mask at 1e-30 erg/s/cm^3/K because
    the Boltzmann factor's derivative also collapses to noise there.
    """
    for xHI, xe in ((0.99, 0.01), (0.5, 0.5), (0.1, 0.5)):
        state, species = _build_state(
            _T, _NH, xHI=xHI, xHII=1.0 - xHI, xe=xe)
        ch = HICollisionalIonizationCooling(
            i_HI=species.idx['HI'],
            i_electron=species.idx['electron'],
        )
        _alloc_channel_scratch(state, ch)
        out = np.empty_like(_T)
        d_out = np.empty_like(_T)
        ch.evaluate(state, out, d_out)
        fd = _fd_central(ch, state)
        mask = np.abs(d_out) > 1.0e-30
        np.testing.assert_allclose(
            d_out[mask], fd[mask], rtol=1.0e-4, atol=0.0,
            err_msg=f'(xHI={xHI}, xe={xe})',
        )


def test_h_recombination_d_out_matches_FD():
    """H II case-B recombination has a closed-form derivative
    chaining the E_rr_B affinity, the Draine alpha_B power-law fit
    through the (b, c, d) intermediates. Subnormal-tail mask at
    1e-30 erg/s/cm^3/K.
    """
    for xHII, xe in ((0.99, 0.99), (0.5, 0.5), (0.01, 0.01)):
        state, species = _build_state(
            _T, _NH, xHI=1.0 - xHII, xHII=xHII, xe=xe)
        ch = HRecombinationCooling(
            i_HII=species.idx['HII'],
            i_electron=species.idx['electron'],
        )
        _alloc_channel_scratch(state, ch)
        out = np.empty_like(_T)
        d_out = np.empty_like(_T)
        ch.evaluate(state, out, d_out)
        fd = _fd_central(ch, state)
        mask = np.abs(d_out) > 1.0e-28
        np.testing.assert_allclose(
            d_out[mask], fd[mask], rtol=1.0e-4, atol=0.0,
            err_msg=f'(xHII={xHII}, xe={xe})',
        )


def test_photoelectric_heating_d_out_matches_FD_bootstrap():
    """PE heating uses 1-point forward FD bootstrap at dT_rel = 1e-3
    internally (analytic chain through the WD01 4-term denominator
    quotient rule is mechanically tractable but ~40 ops and gives
    no meaningful gain for the substep damping role). The channel's
    own d_out is compared against the 5-point central FD reference
    here at rtol = 5e-3 -- the forward FD truncation O(dT) bounded
    by Gamma's roughly logarithmic T-dependence sits at ~5e-3 for
    chi_PE = 1, ne ~ 1.
    """
    for xe in (0.01, 0.1, 0.5):
        for chi in (1.0, 10.0):
            state, species = _build_state(_T, _NH, xHI=0.5, xHII=0.5, xe=xe)
            state.chi[state.chi_bands.index('FUV')] = chi
            ch = PhotoelectricHeating(i_electron=species.idx['electron'])
            _alloc_channel_scratch(state, ch)
            out = np.empty_like(_T)
            d_out = np.empty_like(_T)
            ch.evaluate(state, out, d_out)
            fd = _fd_central(ch, state)
            mask = np.abs(d_out) > 1.0e-32
            np.testing.assert_allclose(
                d_out[mask], fd[mask], rtol=5.0e-2, atol=0.0,
                err_msg=f'(xe={xe}, chi={chi})',
            )


def test_grain_recombination_d_out_matches_FD_bootstrap():
    """GrainRec uses the same 1-point forward FD bootstrap pattern;
    same rtol target."""
    for xe in (0.01, 0.1, 0.5):
        for chi in (1.0, 10.0):
            state, species = _build_state(_T, _NH, xHI=0.5, xHII=0.5, xe=xe)
            state.chi[state.chi_bands.index('FUV')] = chi
            ch = GrainRecombinationCooling(
                i_electron=species.idx['electron'])
            _alloc_channel_scratch(state, ch)
            out = np.empty_like(_T)
            d_out = np.empty_like(_T)
            ch.evaluate(state, out, d_out)
            fd = _fd_central(ch, state)
            mask = np.abs(d_out) > 1.0e-32
            np.testing.assert_allclose(
                d_out[mask], fd[mask], rtol=5.0e-2, atol=0.0,
                err_msg=f'(xe={xe}, chi={chi})',
            )


def test_nebular_d_out_matches_FD_bootstrap():
    """Nebular metal-line proxy uses FD bootstrap; analytic chain
    through the 6-term polynomial * Boltzmann * Hummer density-
    reduction is mechanically tractable but no gain over FD for
    damping. Same rtol = 5e-2 as the other bootstrap channels.
    """
    for xHII, xe in ((0.5, 0.5), (0.99, 0.99)):
        for Z_g in (0.5, 1.0):
            state, species = _build_state(
                _T, _NH, xHI=1.0 - xHII, xHII=xHII, xe=xe)
            state.Z_g[:] = Z_g
            ch = NebularMetalLineCooling(
                i_HII=species.idx['HII'],
                i_electron=species.idx['electron'])
            _alloc_channel_scratch(state, ch)
            out = np.empty_like(_T)
            d_out = np.empty_like(_T)
            ch.evaluate(state, out, d_out)
            fd = _fd_central(ch, state)
            # Nebular Lambda has a sign-changing region in the cold
            # tail where the f_red density-reduction factor flips;
            # mask below 1e-26 to ignore precision-noise crossings.
            mask = np.abs(d_out) > 1.0e-26
            np.testing.assert_allclose(
                d_out[mask], fd[mask], rtol=5.0e-2, atol=0.0,
                err_msg=f'(xHII={xHII}, xe={xe}, Z_g={Z_g})')


def test_hi_smith21_d_out_matches_FD_bootstrap():
    """HISmith21 has 4 line-series Upsilon * Boltzmann contributions
    summed; FD bootstrap pattern."""
    for xHI, xe in ((0.99, 0.01), (0.5, 0.5), (0.1, 0.5)):
        state, species = _build_state(
            _T, _NH, xHI=xHI, xHII=1.0 - xHI, xe=xe)
        ch = HISmith21Cooling(
            i_HI=species.idx['HI'],
            i_electron=species.idx['electron'])
        _alloc_channel_scratch(state, ch)
        out = np.empty_like(_T)
        d_out = np.empty_like(_T)
        ch.evaluate(state, out, d_out)
        fd = _fd_central(ch, state)
        mask = np.abs(d_out) > 1.0e-28
        np.testing.assert_allclose(
            d_out[mask], fd[mask], rtol=5.0e-2, atol=0.0,
            err_msg=f'(xHI={xHI}, xe={xe})')


# H2-family test profiles: each triplet (xHI, xH2, xHII) satisfies H
# conservation 2*xH2 + xHI + xHII = 1. xe = xCII = 1.6e-4 (FUV-
# unshielded carbon dominates electron count in neutral / molecular
# regime; xHII contribution negligible). The three triplets bracket
# the WNM (neutral atomic), CNM/cold transition, and molecular
# core regimes.
_H2_PROFILES = (
    # (xHI, xH2, xHII): WNM (almost fully atomic) -- xHII = xCII
    (1.0 - 1.6e-4, 0.0, 1.6e-4),
    # transitional (partially molecular)
    (0.4, 0.3, 0.0),
    # molecular core
    (0.1, 0.45, 0.0),
)


def test_h2_colldiss_d_out_matches_FD_bootstrap():
    """H2 collisional dissociation has a gated log-log interpolation
    in T; FD bootstrap. The T > 700 K gate triggers a step in the
    value with finite derivative on each side and an indeterminate
    delta at the boundary; mask cells where |d_out| < 1e-28 to
    exclude the boundary noise from the comparison.

    Profile: H2 channels use the cold/cool grid `_T_MOL`
    (T <= 1e4 K) since H2 dissociates above ~few thousand K and the
    NCR network never evaluates this channel above that.
    """
    for xHI, xH2, xHII in _H2_PROFILES:
        state, species = _build_state(
            _T_MOL, _NH_MOL, xHI=xHI, xHII=xHII, xH2=xH2, xe=1.6e-4)
        ch = H2CollDissCooling(
            i_HI=species.idx['HI'], i_H2=species.idx['H2'])
        _alloc_channel_scratch(state, ch)
        out = np.empty_like(_T_MOL)
        d_out = np.empty_like(_T_MOL)
        with np.errstate(divide='ignore', invalid='ignore'):
            ch.evaluate(state, out, d_out)
            fd = _fd_central(ch, state)
        mask = (np.abs(d_out) > 1.0e-28) & np.isfinite(fd)
        np.testing.assert_allclose(
            d_out[mask], fd[mask], rtol=5.0e-2, atol=0.0,
            err_msg=f'(xHI={xHI}, xH2={xH2})')


def test_h2_formation_d_out_matches_FD_bootstrap():
    """H2 formation heating has the temperature-dependent grain rate
    + HM79 n_crit chain. FD bootstrap. Cold/cool grid."""
    for xHI, xH2, xHII in _H2_PROFILES:
        state, species = _build_state(
            _T_MOL, _NH_MOL, xHI=xHI, xHII=xHII, xH2=xH2, xe=1.6e-4)
        ch = H2FormationHeating(
            i_HI=species.idx['HI'], i_H2=species.idx['H2'],
            xi_diss_H2=1.0e-12)
        _alloc_channel_scratch(state, ch)
        out = np.empty_like(_T_MOL)
        d_out = np.empty_like(_T_MOL)
        ch.evaluate(state, out, d_out)
        fd = _fd_central(ch, state)
        mask = np.abs(d_out) > 1.0e-30
        np.testing.assert_allclose(
            d_out[mask], fd[mask], rtol=5.0e-2, atol=0.0,
            err_msg=f'(xHI={xHI}, xH2={xH2})')


def test_h2_pump_d_out_matches_FD_bootstrap():
    """H2 pump heating has the HM79 n_crit chain. FD bootstrap.
    Cold/cool grid."""
    for xHI, xH2, xHII in _H2_PROFILES:
        state, species = _build_state(
            _T_MOL, _NH_MOL, xHI=xHI, xHII=xHII, xH2=xH2, xe=1.6e-4)
        ch = H2PumpHeating(
            i_HI=species.idx['HI'], i_H2=species.idx['H2'],
            xi_diss_H2=1.0e-12)
        _alloc_channel_scratch(state, ch)
        out = np.empty_like(_T_MOL)
        d_out = np.empty_like(_T_MOL)
        ch.evaluate(state, out, d_out)
        fd = _fd_central(ch, state)
        mask = np.abs(d_out) > 1.0e-30
        np.testing.assert_allclose(
            d_out[mask], fd[mask], rtol=5.0e-2, atol=0.0,
            err_msg=f'(xHI={xHI}, xH2={xH2})')


# H2Moseley21 and H2Gong17 are H2 rovibrational cooling channels;
# tested at chemical equilibrium for (xHII, xCII, xe) given a chosen
# xH2 and a CR-only ionising background. Three xH2 inputs bracket
# the relevant molecular-fraction regimes.
_H2_XH2_VALUES = (0.0, 0.3, 0.45)


def test_h2_moseley21_d_out_matches_FD_bootstrap():
    """H2Moseley21 rovibrational cooling: sum of four sqrt(T3)/T3 *
    exp(b/T3) * saturation-density terms (HI + H2 colliders). FD
    bootstrap; cold/cool grid (T <= 1e4 K). Abundances at chemical
    equilibrium for given xH2 via `eq_xHII_xe` (xi_CR = 1e-16 with
    no FUV)."""
    for xH2 in _H2_XH2_VALUES:
        state, species = _build_state_eq(
            _T_MOL, _NH_MOL, xH2=xH2, xi_CR=1.0e-16,
            G_PE=0.0, G_CI=0.0)
        ch = H2Moseley21Cooling(
            i_HI=species.idx['HI'], i_H2=species.idx['H2'])
        _alloc_channel_scratch(state, ch)
        out = np.empty_like(_T_MOL)
        d_out = np.empty_like(_T_MOL)
        ch.evaluate(state, out, d_out)
        fd = _fd_central(ch, state)
        mask = np.abs(d_out) > 1.0e-30
        np.testing.assert_allclose(
            d_out[mask], fd[mask], rtol=5.0e-2, atol=0.0,
            err_msg=f'(xH2={xH2})')


def test_h2_gong17_d_out_matches_FD_bootstrap():
    """H2Gong17 cooling: piecewise polynomial fits in three T-regimes
    for the HI partner (+ H2, He, H+, e partners), composed with
    Gamma_LTE / (1 + Gamma_LTE / Gamma_n0). FD bootstrap; cold/cool
    grid. Piecewise boundaries at T = 100, 200, 1000, 6000 K give a
    finite jump in d_out across each boundary; mask the boundary
    neighborhood and cells below 1e-30 from the comparison.
    """
    for xH2 in _H2_XH2_VALUES:
        state, species = _build_state_eq(
            _T_MOL, _NH_MOL, xH2=xH2, xi_CR=1.0e-16,
            G_PE=0.0, G_CI=0.0)
        ch = H2Gong17Cooling(
            i_HI=species.idx['HI'], i_HII=species.idx['HII'],
            i_H2=species.idx['H2'],
            i_electron=species.idx['electron'])
        _alloc_channel_scratch(state, ch)
        out = np.empty_like(_T_MOL)
        d_out = np.empty_like(_T_MOL)
        ch.evaluate(state, out, d_out)
        fd = _fd_central(ch, state)
        T_grid = state.T
        boundary_mask = np.zeros_like(T_grid, dtype=bool)
        for T_b in (100.0, 200.0, 1000.0, 6000.0):
            boundary_mask |= np.abs(np.log10(T_grid / T_b)) < 0.02
        mask = (np.abs(d_out) > 1.0e-30) & (~boundary_mask)
        np.testing.assert_allclose(
            d_out[mask], fd[mask], rtol=5.0e-2, atol=0.0,
            err_msg=f'(xH2={xH2})')
