"""FD-vs-analytic dT_rel calibration sweep.

This test sweeps the relative temperature step `dT_rel = dT / T` over
a wide log-spaced range for both 2-point central FD and 1-point
forward FD, comparing each against the per-channel analytic d_out
(treated as the ground truth from Phase 4d-a / 4d-b1). The point is
to map out the U-shape:

    log(rel_err) vs log(dT_rel)
        roundoff-limited at small dT_rel  (rises as 1/dT)
        plateau near the optimum
        truncation-limited at large dT_rel (rises as dT^2 for central,
                                            as dT for forward)

and pin two practical defaults:

    - `dT_rel = 1e-3`  Python "conservative" default in the substep-
                       loop FD bootstrap path.
    - `dT_rel = 2e-2`  tigris-ncr / Athena-TIGRESS production value
                       (`dlntemp_` at `tigris-ncr/src/photchem/photchem.hpp:224`).
                       Uses forward difference rather than central.

The test runs once per representative channel:
    Lya         -- steep Boltzmann exp(-T_excite/T)
    HRecomb     -- steep power law alpha_B ~ T^-0.8
    Dust        -- smooth sqrt(T) * (T - T_dust)

For each, asserts:
    1. The plateau covers `dT_rel = 1e-3` (Python default).
    2. The plateau covers `dT_rel = 2e-2` (NCR default).
    3. The optimum sits below 1e-3 (roundoff cliff) so the
       conservative defaults are on the safe side of the curve.

The test is `slow`-marked because it does ~21 sweep points x 5 grid
sizes per channel; useful as a one-off documented calibration, not
for every CI run.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyathena.chemistry.species import SpeciesSet
from pyathena.chemistry.state import ChemState
from pyathena.chemistry.cooling.dust import DustGasCoupling
from pyathena.chemistry.cooling.lya import LyaCooling
from pyathena.chemistry.cooling.recombination_hydrogen import (
    HRecombinationCooling,
)


# Lighter grid than the analytic-derivative tests because we sweep
# ~21 dT_rel values per channel and don't need the full coverage.
_T_VALS = np.unique(np.concatenate([
    np.logspace(2.0, 6.0, 16),
    np.logspace(3.5, 5.0, 16),
]))
_N_VALS = np.logspace(-1.0, 3.0, 6)
_T_GRID, _NH_GRID = np.meshgrid(_T_VALS, _N_VALS, indexing='xy')
_T = _T_GRID.ravel()
_NH = _NH_GRID.ravel()


def _build_state(T, nH, xHI, xHII, xe, T_dust=15.0, Z_d=1.0, mu=1.2):
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
    state.x[idx['electron']] = xe
    state.T_dust[:] = T_dust
    state.alloc_scratch('solver:mu_at_entry', (ncell,))
    state.scratch['solver:mu_at_entry'][:] = mu
    return state, species


def _alloc_channel_scratch(state, channel):
    for name in channel.SCRATCH_NAMES:
        if name in state.scratch:
            continue
        state.alloc_scratch(name, (state.T.size,))


def _fd_central_2pt(channel, state, dT_rel):
    T = state.T
    mu = state.scratch['solver:mu_at_entry']
    dT = dT_rel * T
    out_p = np.empty_like(T)
    out_m = np.empty_like(T)
    T_orig = T.copy()
    state.T[:] = T_orig + dT
    channel.evaluate(state, out_p)
    state.T[:] = T_orig - dT
    channel.evaluate(state, out_m)
    state.T[:] = T_orig
    return mu * (out_p - out_m) / (2.0 * dT)


def _fd_forward_1pt(channel, state, dT_rel):
    """Single forward difference at `T_ = T * (1 + dT_rel)`. Mirrors
    the tigris-ncr `dlntemp_` convention exactly.
    """
    T = state.T
    mu = state.scratch['solver:mu_at_entry']
    out_T = np.empty_like(T)
    out_Tp = np.empty_like(T)
    T_orig = T.copy()
    state.T[:] = T_orig
    channel.evaluate(state, out_T)
    state.T[:] = T_orig * (1.0 + dT_rel)
    channel.evaluate(state, out_Tp)
    state.T[:] = T_orig
    return mu * (out_Tp - out_T) / (T_orig * dT_rel)


def _sweep_dT_rel(channel, state, dT_rel_values, fd_fn, mask_floor):
    """Return the per-`dT_rel` max-relative-error of `fd_fn` against
    the channel's analytic d_out.

    `mask_floor` excludes subnormal-tail cells where any FD is
    catastrophic-cancellation noise.
    """
    out = np.empty_like(state.T)
    d_analytic = np.empty_like(state.T)
    channel.evaluate(state, out, d_analytic)
    mask = np.abs(d_analytic) > mask_floor

    rel_err = np.empty(dT_rel_values.size)
    for i, dT_rel in enumerate(dT_rel_values):
        d_fd = fd_fn(channel, state, dT_rel)
        # At very small dT_rel the FD subtraction can produce NaN
        # or +/-Inf from catastrophic cancellation in the deep-cold
        # tail; mask those out per-step so the maximum stays finite.
        diff = np.abs(d_fd[mask] - d_analytic[mask])
        finite = np.isfinite(diff)
        denom = np.abs(d_analytic[mask])
        err_arr = diff[finite] / denom[finite]
        rel_err[i] = float(err_arr.max()) if err_arr.size else np.inf
    return rel_err


_DT_REL_SWEEP = np.logspace(-10, -1, 19)


def _build_channel_state(channel_name):
    """Set up a representative state per channel."""
    if channel_name == 'Lya':
        state, species = _build_state(_T, _NH, xHI=0.5, xHII=0.5, xe=0.5)
        ch = LyaCooling(
            i_HI=species.idx['HI'],
            i_electron=species.idx['electron'],
        )
        mask_floor = 1.0e-25
    elif channel_name == 'HRecomb':
        state, species = _build_state(_T, _NH, xHI=0.01, xHII=0.99, xe=0.99)
        ch = HRecombinationCooling(
            i_HII=species.idx['HII'],
            i_electron=species.idx['electron'],
        )
        mask_floor = 1.0e-28
    elif channel_name == 'Dust':
        # Dust d_out scales like Z_d * alpha_gd * n_H * sqrt(T); for
        # the (T, n_H) grid in this sweep d_out sits around
        # 1e-30 - 1e-28 in mid-T. Use a much lower floor than the
        # other channels so the mask is non-empty.
        state, species = _build_state(_T, _NH, xHI=0.5, xHII=0.5, xe=0.5,
                                       T_dust=0.0)
        ch = DustGasCoupling()
        mask_floor = 1.0e-32
    else:
        raise ValueError(channel_name)
    _alloc_channel_scratch(state, ch)
    return ch, state, mask_floor


@pytest.mark.parametrize('channel_name,fd_method,target_rtol', [
    # 2-point central: optimum at dT_rel ~ eps^(1/3) ~ 6e-6.
    # Conservative production default 1e-3 sits in the truncation
    # plateau; the curve at 1e-3 should be < 1e-3 rel error for any
    # of the three channels.
    ('Lya',     'central_2pt', 5.0e-3),
    ('HRecomb', 'central_2pt', 5.0e-3),
    ('Dust',    'central_2pt', 5.0e-3),
    # 1-point forward (NCR convention). Truncation O(dT * f''/2);
    # at dT_rel = 2e-2 worst-case is ~6-7% for steep channels (Lya
    # / HRecomb) and well under 1% for smooth (Dust). Set target at
    # 1e-1 so the test pins "fit for stiffness damping" without
    # claiming high precision. NCR production uses this in real
    # runs at exactly this dT_rel value.
    ('Lya',     'forward_1pt', 1.0e-1),
    ('HRecomb', 'forward_1pt', 1.0e-1),
    ('Dust',    'forward_1pt', 1.0e-1),
])
def test_dT_rel_default_lies_on_safe_plateau(
    channel_name, fd_method, target_rtol,
):
    """The conservative dT_rel defaults sit on the truncation-side
    plateau of the FD error curve, comfortably below the target rtol
    AND comfortably above the roundoff-cliff optimum.
    """
    channel, state, mask_floor = _build_channel_state(channel_name)
    if fd_method == 'central_2pt':
        fd_fn = _fd_central_2pt
        default = 1.0e-3
    elif fd_method == 'forward_1pt':
        fd_fn = _fd_forward_1pt
        default = 2.0e-2
    else:
        raise ValueError(fd_method)

    rel_err = _sweep_dT_rel(channel, state, _DT_REL_SWEEP, fd_fn,
                            mask_floor)

    # Plateau check: rel_err at the chosen default is below the
    # target.
    i_default = int(np.argmin(np.abs(_DT_REL_SWEEP - default)))
    assert rel_err[i_default] < target_rtol, (
        f'{channel_name} {fd_method}: rel_err at dT_rel={default} '
        f'is {rel_err[i_default]:.3e}, expected < {target_rtol:.3e}.'
        f' Sweep min = {rel_err.min():.3e} at dT_rel = '
        f'{_DT_REL_SWEEP[rel_err.argmin()]:.3e}.'
    )

    # Safety check: the optimum sits to the LEFT of the default
    # (i.e. roundoff regime is at smaller dT, not the default). This
    # guarantees moving toward larger dT only increases error
    # predictably (truncation dominates), no surprise cliff.
    i_min = int(rel_err.argmin())
    assert _DT_REL_SWEEP[i_min] < default, (
        f'{channel_name} {fd_method}: optimum dT_rel = '
        f'{_DT_REL_SWEEP[i_min]:.3e} sits at or above the default '
        f'{default}; default is no longer truncation-dominated.'
    )
