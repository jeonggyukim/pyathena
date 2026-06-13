"""Grain recombination cooling, Weingartner & Draine 2001 Table 3.

Port of `pyathena.microphysics.cool.coolRec`. When a charged grain
captures an electron from the gas, the work done against the grain's
electrostatic potential cools the gas. The WD01 fit:

    Lambda = 1e-28 * Z_d * n_e *
             T**(D[0] + D[1] / ln(x)) *
             exp(D[2] + (D[3] - D[4] * ln(x)) * ln(x))

with `D = (0.4535, 2.234, -6.266, 1.442, 0.05089)` (Rv = 3.1, bC = 4,
distribution A, ISRF) and the WD01 charge parameter `x = 1.7 * chi_PE
* sqrt(T) / (n_e * phi) + 50.0` -- the same `x` the PE heating
channel uses, since both rates come from the same grain-charge
balance.

This is the natural cooling companion to PhotoelectricHeating. The
two channels share the WD01 fit family and should be enabled or
disabled together in production runs.

Stays hand-coded indefinitely; gas-grain charge exchange has no
CHIANTI / Cloudy table form.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from .base import CoolingChannel

_D0: float = 0.4535
_D1: float = 2.234
_D2: float = -6.266
_D3: float = 1.442
_D4: float = 0.05089
_PREFACTOR: float = 1.0e-28
_NE_FLOOR: float = 1.0e-10


class GrainRecombinationCooling(CoolingChannel):
    """Cooling by electron capture on dust grains (WD01)."""

    name: ClassVar[str] = 'GrainRecombination'
    SCRATCH_NAMES: ClassVar[tuple] = (
        'cooling:grain_rec:tmp',
        'cooling:grain_rec:ne_floor',
        'cooling:grain_rec:lnx',
        # Phase 4d FD-bootstrap derivative slots: same pattern as
        # PhotoelectricHeating (which shares the WD01 charge-parameter
        # chain). The full analytic chain rule on
        #   Lambda ~ Z_d * ne * T^(D0 + D1/ln x) * exp(D2 + (D3 - D4 ln x) ln x)
        # is mechanically tractable but tedious through the
        # log-exponent + power-of-T product; FD bootstrap at
        # dT_rel = 1e-3 forward (the project convention; see
        # feedback_fd_bootstrap_convention) matches the accuracy
        # the substep-damping role needs.
        'cooling:grain_rec:T_orig',
        'cooling:grain_rec:out_tp',
    )
    __version__: ClassVar[str] = '0.1@phase4b'

    def __init__(
        self,
        *,
        i_electron: int,
        chi_band: str = 'FUV',
        phi: float = 1.0,
    ) -> None:
        self._i_electron = int(i_electron)
        self._chi_band = chi_band
        self._phi = float(phi)

    def _compute_lambda(self, state: Any, out: np.ndarray) -> None:
        """Write Lambda into `out`. Reads `state.T` directly so the
        FD bootstrap can perturb T transiently."""
        T = state.T
        nH = state.nH
        xe = state.x[self._i_electron]
        Z_d = state.Z_d
        chi_PE = state.chi_for(self._chi_band)

        tmp = state.get_scratch('cooling:grain_rec:tmp')
        ne_floor = state.get_scratch('cooling:grain_rec:ne_floor')
        lnx = state.get_scratch('cooling:grain_rec:lnx')

        # ne = xe * nH; floor against zero.
        np.multiply(xe, nH, out=ne_floor)
        np.maximum(ne_floor, _NE_FLOOR, out=ne_floor)

        # x = 1.7 * chi_PE * sqrt(T) / (ne * phi) + 50.0  (WD01 charge
        # parameter; same formula as the PE heating channel)
        np.multiply(ne_floor, self._phi, out=tmp)
        np.sqrt(T, out=lnx)
        np.multiply(lnx, chi_PE, out=lnx)
        np.multiply(lnx, 1.7, out=lnx)
        np.divide(lnx, tmp, out=lnx)
        np.add(lnx, 50.0, out=lnx)
        # lnx = ln(x)
        np.log(lnx, out=lnx)

        # exponent of T: D0 + D1 / lnx
        np.divide(_D1, lnx, out=tmp)
        np.add(tmp, _D0, out=tmp)
        np.power(T, tmp, out=out)

        # exp term: exp(D2 + (D3 - D4*lnx) * lnx)
        # = exp(D2 + D3*lnx - D4*lnx^2)
        np.multiply(lnx, _D4, out=tmp)
        np.subtract(_D3, tmp, out=tmp)
        np.multiply(tmp, lnx, out=tmp)
        np.add(tmp, _D2, out=tmp)
        np.exp(tmp, out=tmp)

        # Lambda = prefactor * Z_d * ne * (T**exp_term) * exp_term
        np.multiply(out, tmp, out=out)
        np.multiply(out, ne_floor, out=out)  # ne (or floor)
        np.multiply(out, Z_d, out=out)
        np.multiply(out, _PREFACTOR, out=out)

    def evaluate(
        self,
        state: Any,
        out: np.ndarray,
        d_out: Optional[np.ndarray] = None,
    ) -> None:
        # Lambda at current state.T.
        self._compute_lambda(state, out)

        if d_out is not None:
            # FD bootstrap at dT_rel = 1e-3 forward (project
            # convention; see CoolingChannel.evaluate docstring and
            # tests/chemistry/test_fd_calibration.py). Same pattern as
            # the PE heating channel which shares the WD01 charge-
            # parameter chain.
            _DT_REL = 1.0e-3
            T_orig = state.get_scratch('cooling:grain_rec:T_orig')
            out_tp = state.get_scratch('cooling:grain_rec:out_tp')
            np.copyto(T_orig, state.T)
            np.multiply(state.T, 1.0 + _DT_REL, out=state.T)
            self._compute_lambda(state, out_tp)
            np.copyto(state.T, T_orig)
            np.subtract(out_tp, out, out=d_out)
            np.divide(d_out, T_orig, out=d_out)
            np.multiply(d_out, 1.0 / _DT_REL, out=d_out)
            mu = state.get_scratch('solver:mu_at_entry')
            np.multiply(d_out, mu, out=d_out)
