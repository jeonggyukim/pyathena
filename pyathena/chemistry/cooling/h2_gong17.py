"""H2 rovibrational cooling, Gong + Ostriker + Wolfire 2017 fits.

Port of `pyathena.microphysics.cool.coolH2G17`. Combines a
low-density limit `Gamma_n0 = sum_partner L_partner * x_partner * n_H`
with a Hollenbach + McKee 1979 LTE limit `Gamma_LTE = Gamma_LTE_HR +
Gamma_LTE_HV` via

    Gamma_tot = Gamma_LTE / (1 + Gamma_LTE / Gamma_n0)   (Gamma_n0 > 0)
              = 0                                        (otherwise)

and gates the result to T >= 10 K. The legacy implementation clips T
at 6000 K before evaluating the polynomial fits, then computes
Gamma_n0 at that capped T but the final Lambda uses the original
T-cap for the LTE evaluation too. This port matches that behaviour
byte-for-byte.

H2 cooling does not swap to CHIANTI (which is purely atomic). The
Cloudy molecular database may eventually serve as a fallback, but
this channel stays hand-coded for the Phase 4 horizon.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from .base import CoolingChannel

_TMIN: float = 10.0
_TMAX: float = 6000.0
_XHE: float = 0.1  # He / H ratio used by the legacy coolH2G17.

# Polynomial coefficients are stored as (c0, c1, c2, c3, c4, c5) and
# evaluated as
#     log10(L) = c0 + c1*y + c2*y^2 + c3*y^3 + c4*y^4 + c5*y^5
# with y = log10(T * 1e-3).
_LHI_COLD = (-16.818342, 37.383713,  58.145166,  48.656103,
             20.159831,   3.8479610)
_LHI_WARM = (-24.311209,  3.5692468, -11.332860, -27.850082,
             -21.328264, -4.2519023)
_LHI_HOT  = (-24.311209,  4.6450521,  -3.7209846,  5.9369081,
              -5.5108049, 1.5538288)
_LH2  = (-23.962112, 2.0943374, -0.77151436, 0.43693353,
         -0.14913216, -0.033638326)
_LHE  = (-23.689237, 2.1892372, -0.81520438, 0.29036281,
         -0.16596184, 0.19191375)
_LHPLUS = (-21.716699, 1.3865783, -0.37915285, 0.11453688,
           -0.23214154, 0.058538864)
_LE_COLD = (-34.286155, -48.537163, -77.121176, -51.352459,
            -15.169150, -0.98120322)
_LE_HOT  = (-22.190316, 1.5728955, -0.213351, 0.96149759,
            -0.91023195, 0.13749749)


def _horner5(y, c, out, scratch):
    """log10(L) = c0 + c1 y + ... + c5 y^5 via Horner. Writes 10**(...)
    into `out`; uses `scratch` for the polynomial accumulation.
    """
    np.multiply(y, c[5], out=scratch)
    np.add(scratch, c[4], out=scratch)
    np.multiply(scratch, y, out=scratch)
    np.add(scratch, c[3], out=scratch)
    np.multiply(scratch, y, out=scratch)
    np.add(scratch, c[2], out=scratch)
    np.multiply(scratch, y, out=scratch)
    np.add(scratch, c[1], out=scratch)
    np.multiply(scratch, y, out=scratch)
    np.add(scratch, c[0], out=out)
    np.power(10.0, out, out=out)


class H2Gong17Cooling(CoolingChannel):
    """H2 cooling via Gong+17 (Gong, Ostriker, Wolfire 2017) fits."""

    name: ClassVar[str] = 'H2Gong17'
    SCRATCH_NAMES: ClassVar[tuple] = (
        'cooling:h2_g17:T_eff',
        'cooling:h2_g17:T3',
        'cooling:h2_g17:y',
        'cooling:h2_g17:tmp',
        'cooling:h2_g17:tmp_b',
        'cooling:h2_g17:Lpartner',
        'cooling:h2_g17:Gamma_n0',
        'cooling:h2_g17:Gamma_LTE',
        'cooling:h2_g17:mask_cold',
        'cooling:h2_g17:mask_warm',
        'cooling:h2_g17:mask_hot',
        'cooling:h2_g17:mask_T_floor',
        # Phase 4d FD-bootstrap derivative slots. The Gong17 form has
        # piecewise polynomial fits in three T-regimes for the HI
        # partner plus a min(T, 6000) cap and a T >= 10 K gate; the
        # analytic chain rule is mechanically tractable but the
        # piecewise branches make it tedious. FD bootstrap at
        # dT_rel = 1e-3 forward (project convention; see
        # feedback_fd_bootstrap_convention) matches substep-damping
        # accuracy. At T near the piecewise boundaries (100 K, 200 K,
        # 1000 K, 6000 K) the FD has a known O(1) jump in d_out across
        # the boundary; the substep damping role tolerates this
        # because the boundary is crossed on a single cell-step
        # boundary, not within the cell.
        'cooling:h2_g17:T_orig',
        'cooling:h2_g17:out_tp',
    )
    __version__: ClassVar[str] = '0.1@phase4b'

    def __init__(
        self,
        *,
        i_HI: int,
        i_HII: int,
        i_H2: int,
        i_electron: int,
    ) -> None:
        self._i_HI = int(i_HI)
        self._i_HII = int(i_HII)
        self._i_H2 = int(i_H2)
        self._i_electron = int(i_electron)

    def _compute_lambda(self, state: Any, out: np.ndarray) -> None:
        """Write Lambda into `out`. Reads `state.T` directly so the FD
        bootstrap can perturb T transiently."""
        T = state.T
        nH = state.nH
        xHI = state.x[self._i_HI]
        xHII = state.x[self._i_HII]
        xH2 = state.x[self._i_H2]
        xe = state.x[self._i_electron]

        T_eff = state.get_scratch('cooling:h2_g17:T_eff')
        T3 = state.get_scratch('cooling:h2_g17:T3')
        y = state.get_scratch('cooling:h2_g17:y')
        scratch = state.get_scratch('cooling:h2_g17:tmp')
        scratch_b = state.get_scratch('cooling:h2_g17:tmp_b')
        Lpartner = state.get_scratch('cooling:h2_g17:Lpartner')
        Gamma_n0 = state.get_scratch('cooling:h2_g17:Gamma_n0')
        Gamma_LTE = state.get_scratch('cooling:h2_g17:Gamma_LTE')
        mask_cold = state.get_scratch('cooling:h2_g17:mask_cold')
        mask_warm = state.get_scratch('cooling:h2_g17:mask_warm')
        mask_hot = state.get_scratch('cooling:h2_g17:mask_hot')
        mask_T_floor = state.get_scratch('cooling:h2_g17:mask_T_floor')

        # T_eff = min(T, Tmax)
        np.minimum(T, _TMAX, out=T_eff)
        # T3 = T_eff * 1e-3; y = log10(T3)
        np.multiply(T_eff, 1.0e-3, out=T3)
        np.log10(T3, out=y)

        # HI piece: piecewise in T < 100 / 100 <= T < 1000 / T >= 1000
        np.less(T_eff, 100.0, out=mask_cold)
        np.logical_and.reduce((
            np.greater_equal(T_eff, 100.0, out=scratch_b),
            np.less(T_eff, 1000.0, out=scratch),
        ), out=mask_warm)
        np.greater_equal(T_eff, 1000.0, out=mask_hot)

        # cold
        _horner5(y, _LHI_COLD, scratch, scratch_b)
        np.multiply(scratch, mask_cold, out=Lpartner)
        # warm
        _horner5(y, _LHI_WARM, scratch, scratch_b)
        np.multiply(scratch, mask_warm, out=scratch)
        np.add(Lpartner, scratch, out=Lpartner)
        # hot
        _horner5(y, _LHI_HOT, scratch, scratch_b)
        np.multiply(scratch, mask_hot, out=scratch)
        np.add(Lpartner, scratch, out=Lpartner)
        # Gamma_n0 = LHI * xHI * nH
        np.multiply(Lpartner, xHI, out=Gamma_n0)
        np.multiply(Gamma_n0, nH, out=Gamma_n0)

        # H2 partner
        _horner5(y, _LH2, Lpartner, scratch_b)
        np.multiply(Lpartner, xH2, out=scratch)
        np.multiply(scratch, nH, out=scratch)
        np.add(Gamma_n0, scratch, out=Gamma_n0)

        # He partner (xHe constant)
        _horner5(y, _LHE, Lpartner, scratch_b)
        np.multiply(Lpartner, _XHE, out=scratch)
        np.multiply(scratch, nH, out=scratch)
        np.add(Gamma_n0, scratch, out=Gamma_n0)

        # H+ partner
        _horner5(y, _LHPLUS, Lpartner, scratch_b)
        np.multiply(Lpartner, xHII, out=scratch)
        np.multiply(scratch, nH, out=scratch)
        np.add(Gamma_n0, scratch, out=Gamma_n0)

        # e partner: piecewise T < 200 K
        np.less(T_eff, 200.0, out=mask_cold)
        np.greater_equal(T_eff, 200.0, out=mask_warm)
        _horner5(y, _LE_COLD, scratch, scratch_b)
        np.multiply(scratch, mask_cold, out=Lpartner)
        _horner5(y, _LE_HOT, scratch, scratch_b)
        np.multiply(scratch, mask_warm, out=scratch)
        np.add(Lpartner, scratch, out=Lpartner)
        np.multiply(Lpartner, xe, out=scratch)
        np.multiply(scratch, nH, out=scratch)
        np.add(Gamma_n0, scratch, out=Gamma_n0)

        # Gamma_LTE_HR = (9.5e-22 * T3**3.76 / (1 + 0.12 * T3**2.1))
        #                * exp(-(0.13/T3)**3)
        #              + 3e-24 * exp(-0.51/T3)
        np.power(T3, 3.76, out=scratch)
        np.multiply(scratch, 9.5e-22, out=scratch)
        np.power(T3, 2.1, out=scratch_b)
        np.multiply(scratch_b, 0.12, out=scratch_b)
        np.add(scratch_b, 1.0, out=scratch_b)
        np.divide(scratch, scratch_b, out=Gamma_LTE)
        np.divide(0.13, T3, out=scratch_b)
        np.power(scratch_b, 3.0, out=scratch_b)
        np.negative(scratch_b, out=scratch_b)
        np.exp(scratch_b, out=scratch_b)
        np.multiply(Gamma_LTE, scratch_b, out=Gamma_LTE)
        np.divide(0.51, T3, out=scratch_b)
        np.negative(scratch_b, out=scratch_b)
        np.exp(scratch_b, out=scratch_b)
        np.multiply(scratch_b, 3.0e-24, out=scratch_b)
        np.add(Gamma_LTE, scratch_b, out=Gamma_LTE)

        # Gamma_LTE_HV = 6.7e-19 exp(-5.86/T3) + 1.6e-18 exp(-11.7/T3)
        np.divide(5.86, T3, out=scratch)
        np.negative(scratch, out=scratch)
        np.exp(scratch, out=scratch)
        np.multiply(scratch, 6.7e-19, out=scratch)
        np.divide(11.7, T3, out=scratch_b)
        np.negative(scratch_b, out=scratch_b)
        np.exp(scratch_b, out=scratch_b)
        np.multiply(scratch_b, 1.6e-18, out=scratch_b)
        np.add(scratch, scratch_b, out=scratch)
        np.add(Gamma_LTE, scratch, out=Gamma_LTE)

        # Gamma_tot = Gamma_LTE / (1 + Gamma_LTE / Gamma_n0) when
        # Gamma_n0 > 1e-100, else 0.
        # Use the masked-multiply pattern to keep this branch-free.
        # ratio = Gamma_LTE / max(Gamma_n0, 1e-100); result = Gamma_LTE
        # / (1 + ratio).
        np.maximum(Gamma_n0, 1.0e-100, out=scratch)
        np.divide(Gamma_LTE, scratch, out=scratch_b)
        np.add(scratch_b, 1.0, out=scratch_b)
        np.divide(Gamma_LTE, scratch_b, out=out)
        # Mask: Gamma_n0 > 1e-100
        np.greater(Gamma_n0, 1.0e-100, out=scratch_b)
        np.multiply(out, scratch_b, out=out)

        # Gate at T >= T_min (use ORIGINAL T, not T_eff -- the legacy
        # code applies the gate after the T_eff capping).
        np.greater_equal(T, _TMIN, out=mask_T_floor)
        np.multiply(out, mask_T_floor, out=out)

        # Lambda = Gamma_tot * xH2
        np.multiply(out, xH2, out=out)

    def evaluate(
        self,
        state: Any,
        out: np.ndarray,
        d_out: Optional[np.ndarray] = None,
    ) -> None:
        # Lambda at current state.T.
        self._compute_lambda(state, out)
        if d_out is not None:
            # FD bootstrap at dT_rel = 1e-3 forward.
            _DT_REL = 1.0e-3
            T_orig = state.get_scratch('cooling:h2_g17:T_orig')
            out_tp = state.get_scratch('cooling:h2_g17:out_tp')
            np.copyto(T_orig, state.T)
            np.multiply(state.T, 1.0 + _DT_REL, out=state.T)
            self._compute_lambda(state, out_tp)
            np.copyto(state.T, T_orig)
            np.subtract(out_tp, out, out=d_out)
            np.divide(d_out, T_orig, out=d_out)
            np.multiply(d_out, 1.0 / _DT_REL, out=d_out)
            mu = state.get_scratch('solver:mu_at_entry')
            np.multiply(d_out, mu, out=d_out)
