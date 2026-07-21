"""C I 3-level fine-structure cooling: [C I] 370 + 609 um.

Port of `pyathena.microphysics.cool.coolCI`. The C I 3-P ground term:

    Level 0 = 3P_0 (J=0, g=1, ground)
    Level 1 = 3P_1 (J=1, g=3, 24 K above ground; [C I] 609 um)
    Level 2 = 3P_2 (J=2, g=5, 62 K above ground; [C I] 370 um)

Electron collision strengths from Johnson, Burke & Kingston 1987
(JPhysB 20, 2553), with a 4-coefficient Horner polynomial in ln T
piecewise at T = 1000 K. HI and H2 partners from Draine 2011 F.6.
The CHIANTI swap at Phase 7 replaces the e and HI partner fits; the
H2 partners stay hand-coded.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from .base import CoolingChannel
from ._level_helpers import cool_3level

_K_B_CGS: float = 1.380649e-16

_G0: float = 1.0
_G1: float = 3.0
_G2: float = 5.0
_A10: float = 7.880e-8
_A20: float = 1.810e-14
_A21: float = 2.650e-7
_E10: float = 3.261e-15
_E20: float = 8.624e-15
_E21: float = 5.363e-15

_F_PARA: float = 0.25
_F_ORTHO: float = 0.75


def _horner4(lnT, c0, c1, c2, c3, c4, out, scratch):
    """In-place evaluation of c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4
    (x = lnT) via Horner; writes the result into `out`.
    """
    np.multiply(lnT, c4, out=scratch)
    np.add(scratch, c3, out=scratch)
    np.multiply(scratch, lnT, out=scratch)
    np.add(scratch, c2, out=scratch)
    np.multiply(scratch, lnT, out=scratch)
    np.add(scratch, c1, out=scratch)
    np.multiply(scratch, lnT, out=scratch)
    np.add(scratch, c0, out=out)


def _piecewise_e_strength(
    T, lnT, mask_cold, mask_warm,
    cold_coeffs, warm_coeffs,
    out, tmp_a, tmp_b,
):
    """Compute log Upsilon(T) as a 5-term piecewise-Horner polynomial
    in ln T and write its `exp` into `out`. `cold_coeffs` and
    `warm_coeffs` are 5-tuples `(c0, c1, c2, c3, c4)`.

    Needs three scratches: `out` receives the result, `tmp_a` and
    `tmp_b` are caller-owned `(ncell,)` buffers. The cold piece is
    held in `tmp_b` while the warm piece is computed into `out`, so
    the two halves do not clobber each other. `tmp_a` is the Horner
    workspace, mangled inside the polynomial evaluation.
    """
    _horner4(lnT, *cold_coeffs, tmp_b, tmp_a)
    np.multiply(tmp_b, mask_cold, out=tmp_b)
    _horner4(lnT, *warm_coeffs, out, tmp_a)
    np.multiply(out, mask_warm, out=out)
    np.add(out, tmp_b, out=out)
    np.exp(out, out=out)


class CIFineStructureCooling(CoolingChannel):
    """C I 370 + 609 um fine-structure cooling. Phase 4b literal port."""

    name: ClassVar[str] = 'CIFineStructure'
    SCRATCH_NAMES: ClassVar[tuple] = (
        'cooling:ci:T2',
        'cooling:ci:lnT',
        'cooling:ci:lnT2',
        'cooling:ci:tmp_a',
        'cooling:ci:tmp_b',
        'cooling:ci:mask_cold',
        'cooling:ci:mask_warm',
        'cooling:ci:gamma10',
        'cooling:ci:gamma20',
        'cooling:ci:gamma21',
        'cooling:ci:k_e_10',
        'cooling:ci:k_e_20',
        'cooling:ci:k_e_21',
        'cooling:ci:k_HI_10',
        'cooling:ci:k_HI_20',
        'cooling:ci:k_HI_21',
        'cooling:ci:k_H2_10',
        'cooling:ci:k_H2_20',
        'cooling:ci:k_H2_21',
        'cooling:ci:q10',
        'cooling:ci:q20',
        'cooling:ci:q21',
        'cooling:ci:q01',
        'cooling:ci:q02',
        'cooling:ci:q12',
        'cooling:ci:tmp0',
        'cooling:ci:tmp1',
        'cooling:ci:tmp2',
        'cooling:ci:fac',
    )
    __version__: ClassVar[str] = '0.1@phase4b'

    def __init__(
        self,
        *,
        i_HI: int,
        i_H2: int,
        i_CI: int,
        i_electron: int,
    ) -> None:
        self._i_HI = int(i_HI)
        self._i_H2 = int(i_H2)
        self._i_CI = int(i_CI)
        self._i_electron = int(i_electron)

    def evaluate(
        self,
        state: Any,
        out: np.ndarray,
        d_out: Optional[np.ndarray] = None,
    ) -> None:
        T = state.T
        nH = state.nH
        xHI = state.x[self._i_HI]
        xH2 = state.x[self._i_H2]
        xCI = state.x[self._i_CI]
        xe = state.x[self._i_electron]

        T2 = state.get_scratch('cooling:ci:T2')
        lnT = state.get_scratch('cooling:ci:lnT')
        lnT2 = state.get_scratch('cooling:ci:lnT2')
        tmp_a = state.get_scratch('cooling:ci:tmp_a')
        tmp_b = state.get_scratch('cooling:ci:tmp_b')
        mask_cold = state.get_scratch('cooling:ci:mask_cold')
        mask_warm = state.get_scratch('cooling:ci:mask_warm')
        gamma10 = state.get_scratch('cooling:ci:gamma10')
        gamma20 = state.get_scratch('cooling:ci:gamma20')
        gamma21 = state.get_scratch('cooling:ci:gamma21')
        k_e_10 = state.get_scratch('cooling:ci:k_e_10')
        k_e_20 = state.get_scratch('cooling:ci:k_e_20')
        k_e_21 = state.get_scratch('cooling:ci:k_e_21')
        k_HI_10 = state.get_scratch('cooling:ci:k_HI_10')
        k_HI_20 = state.get_scratch('cooling:ci:k_HI_20')
        k_HI_21 = state.get_scratch('cooling:ci:k_HI_21')
        k_H2_10 = state.get_scratch('cooling:ci:k_H2_10')
        k_H2_20 = state.get_scratch('cooling:ci:k_H2_20')
        k_H2_21 = state.get_scratch('cooling:ci:k_H2_21')
        q10 = state.get_scratch('cooling:ci:q10')
        q20 = state.get_scratch('cooling:ci:q20')
        q21 = state.get_scratch('cooling:ci:q21')
        q01 = state.get_scratch('cooling:ci:q01')
        q02 = state.get_scratch('cooling:ci:q02')
        q12 = state.get_scratch('cooling:ci:q12')
        tmp0 = state.get_scratch('cooling:ci:tmp0')
        tmp1 = state.get_scratch('cooling:ci:tmp1')
        tmp2 = state.get_scratch('cooling:ci:tmp2')
        fac = state.get_scratch('cooling:ci:fac')

        np.multiply(T, 1.0e-2, out=T2)
        np.log(T, out=lnT)
        np.log(T2, out=lnT2)
        np.less(T, 1.0e3, out=mask_cold)
        np.greater_equal(T, 1.0e3, out=mask_warm)

        # Electron collision strengths (piecewise quartic in lnT).
        _piecewise_e_strength(
            T, lnT, mask_cold, mask_warm,
            (-9.25141, -7.73782e-1, 3.61184e-1, -1.50892e-2, -6.56325e-4),
            (4.446e2, -2.27913e2, 4.2595e1, -3.47620, 1.0508e-1),
            gamma10, tmp_a, tmp_b,
        )
        _piecewise_e_strength(
            T, lnT, mask_cold, mask_warm,
            (-7.69735, -1.30743, 0.697638, -0.111338, 0.705277e-2),
            (3.50609e2, -1.87474e2, 3.61803e1, -3.03283, 9.38138e-2),
            gamma20, tmp_a, tmp_b,
        )
        _piecewise_e_strength(
            T, lnT, mask_cold, mask_warm,
            (-7.4387, -0.57443, 0.358264, -4.18166e-2, 2.35272e-3),
            (3.86186e2, -2.02193e2, 3.85049e1, -3.19268, 9.78573e-2),
            gamma21, tmp_a, tmp_b,
        )

        # fac = 8.629e-8 * sqrt(1e4 / T)
        np.divide(1.0e4, T, out=fac)
        np.sqrt(fac, out=fac)
        np.multiply(fac, 8.629e-8, out=fac)

        # k_e_ij = fac * gamma_ij / g_upper
        np.multiply(gamma10, fac, out=k_e_10)
        np.multiply(k_e_10, 1.0 / _G1, out=k_e_10)
        np.multiply(gamma20, fac, out=k_e_20)
        np.multiply(k_e_20, 1.0 / _G2, out=k_e_20)
        np.multiply(gamma21, fac, out=k_e_21)
        np.multiply(k_e_21, 1.0 / _G2, out=k_e_21)

        # HI collision rates (Draine F.6)
        # k10HI = 1.26e-10 * T2 ** (0.115 + 0.057 * lnT2)
        np.multiply(lnT2, 0.057, out=tmp_a)
        np.add(tmp_a, 0.115, out=tmp_a)
        np.power(T2, tmp_a, out=k_HI_10)
        np.multiply(k_HI_10, 1.26e-10, out=k_HI_10)
        # k20HI = 0.89e-10 * T2 ** (0.228 + 0.046 * lnT2)
        np.multiply(lnT2, 0.046, out=tmp_a)
        np.add(tmp_a, 0.228, out=tmp_a)
        np.power(T2, tmp_a, out=k_HI_20)
        np.multiply(k_HI_20, 0.89e-10, out=k_HI_20)
        # k21HI = 2.64e-10 * T2 ** (0.231 + 0.046 * lnT2)
        np.multiply(lnT2, 0.046, out=tmp_a)
        np.add(tmp_a, 0.231, out=tmp_a)
        np.power(T2, tmp_a, out=k_HI_21)
        np.multiply(k_HI_21, 2.64e-10, out=k_HI_21)

        # H2 rates. Pattern: kij_p, kij_o each prefac * T2**(a + b*lnT2);
        # combine with f_para * para + f_ortho * ortho.
        def _h2(prefac_p, a_p, b_p, prefac_o, a_o, b_o, k_out):
            np.multiply(lnT2, b_p, out=tmp_a)
            np.add(tmp_a, a_p, out=tmp_a)
            np.power(T2, tmp_a, out=tmp_b)
            np.multiply(tmp_b, prefac_p * _F_PARA, out=tmp_b)
            np.multiply(lnT2, b_o, out=tmp_a)
            np.add(tmp_a, a_o, out=tmp_a)
            np.power(T2, tmp_a, out=k_out)
            np.multiply(k_out, prefac_o * _F_ORTHO, out=k_out)
            np.add(k_out, tmp_b, out=k_out)

        _h2(0.67e-10, -0.085, 0.102, 0.71e-10, -0.004, 0.049, k_H2_10)
        _h2(0.86e-10, -0.010, 0.048, 0.69e-10,  0.169, 0.038, k_H2_20)
        _h2(1.75e-10,  0.072, 0.064, 1.48e-10,  0.263, 0.031, k_H2_21)

        # q_ij = nH * (k_HI*xHI + k_H2*xH2 + k_e*xe)
        def _qij(k_HI, k_H2, k_e, q_out):
            np.multiply(k_HI, xHI, out=q_out)
            np.multiply(k_H2, xH2, out=tmp_a)
            np.add(q_out, tmp_a, out=q_out)
            np.multiply(k_e, xe, out=tmp_a)
            np.add(q_out, tmp_a, out=q_out)
            np.multiply(q_out, nH, out=q_out)

        _qij(k_HI_10, k_H2_10, k_e_10, q10)
        _qij(k_HI_20, k_H2_20, k_e_20, q20)
        _qij(k_HI_21, k_H2_21, k_e_21, q21)

        # Reverse Boltzmann factors.
        np.multiply(T, _K_B_CGS, out=tmp_a)
        np.divide(_E10, tmp_a, out=tmp_b)
        np.negative(tmp_b, out=tmp_b)
        np.exp(tmp_b, out=tmp_b)
        np.multiply(q10, tmp_b, out=q01)
        np.multiply(q01, _G1 / _G0, out=q01)
        np.divide(_E20, tmp_a, out=tmp_b)
        np.negative(tmp_b, out=tmp_b)
        np.exp(tmp_b, out=tmp_b)
        np.multiply(q20, tmp_b, out=q02)
        np.multiply(q02, _G2 / _G0, out=q02)
        np.divide(_E21, tmp_a, out=tmp_b)
        np.negative(tmp_b, out=tmp_b)
        np.exp(tmp_b, out=tmp_b)
        np.multiply(q21, tmp_b, out=q12)
        np.multiply(q12, _G2 / _G1, out=q12)

        cool_3level(
            q01, q10, q02, q20, q12, q21,
            _A10, _A20, _A21, _E10, _E20, _E21,
            xCI, out, tmp0, tmp1, tmp2,
        )

        if d_out is not None:
            d_out[:] = 0.0
