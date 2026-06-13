"""O I 3-level fine-structure cooling: [O I] 63 + 146 um.

Port of `pyathena.microphysics.cool.coolOI`. The O I 3-P term system:

    Level 0 = 3P_2 (ground, g = 5)
    Level 1 = 3P_1 (g = 3, E = 228 K above ground; [O I] 63 um line)
    Level 2 = 3P_0 (g = 1, E = 326 K above ground; [O I] 146 um line)

Collisional rates assembled from electron (Bell+1998), HI and H2
(Draine 2011 Table F.6) partners. The HI / e rates will be CHIANTI
table swaps in Phase 7; the H2 partner stays hand-coded.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from .base import CoolingChannel
from ._level_helpers import cool_3level

_K_B_CGS: float = 1.380649e-16

_G0: float = 5.0
_G1: float = 3.0
_G2: float = 1.0
_A10: float = 8.910e-5
_A20: float = 1.340e-10
_A21: float = 1.750e-5
_E10: float = 3.144e-14
_E20: float = 4.509e-14
_E21: float = 1.365e-14

_F_PARA: float = 0.25
_F_ORTHO: float = 0.75


def _assemble_kHI_kH2(T, T2, lnT2, scratch_T, scratch_o, scratch_p, k_out,
                     prefac_HI: float, alpha_HI: float, beta_HI: float,
                     prefac_p: float, alpha_p: float, beta_p: float,
                     prefac_o: float, alpha_o: float, beta_o: float,
                     k_HI: np.ndarray, k_H2: np.ndarray):
    """Helper: assemble k_HI and k_H2 for one transition.

    k_HI = prefac_HI * T2 ** (alpha_HI + beta_HI * lnT2)
    k_H2 = prefac_p * T2 ** (alpha_p + beta_p * lnT2) * f_para
         + prefac_o * T2 ** (alpha_o + beta_o * lnT2) * f_ortho

    Allocates nothing -- writes into caller-owned scratches.
    """
    # k_HI
    np.multiply(lnT2, beta_HI, out=scratch_T)
    np.add(scratch_T, alpha_HI, out=scratch_T)
    np.power(T2, scratch_T, out=k_HI)
    np.multiply(k_HI, prefac_HI, out=k_HI)
    # k_H2_p
    np.multiply(lnT2, beta_p, out=scratch_T)
    np.add(scratch_T, alpha_p, out=scratch_T)
    np.power(T2, scratch_T, out=scratch_p)
    np.multiply(scratch_p, prefac_p, out=scratch_p)
    # k_H2_o
    np.multiply(lnT2, beta_o, out=scratch_T)
    np.add(scratch_T, alpha_o, out=scratch_T)
    np.power(T2, scratch_T, out=scratch_o)
    np.multiply(scratch_o, prefac_o, out=scratch_o)
    # mix
    np.multiply(scratch_p, _F_PARA, out=scratch_p)
    np.multiply(scratch_o, _F_ORTHO, out=scratch_o)
    np.add(scratch_p, scratch_o, out=k_H2)


class OIFineStructureCooling(CoolingChannel):
    """O I 63 + 146 um fine-structure cooling. Phase 4b literal port."""

    name: ClassVar[str] = 'OIFineStructure'
    __version__: ClassVar[str] = '0.1@phase4b'

    def __init__(
        self,
        *,
        i_HI: int,
        i_H2: int,
        i_OI: int,
        i_electron: int,
    ) -> None:
        self._i_HI = int(i_HI)
        self._i_H2 = int(i_H2)
        self._i_OI = int(i_OI)
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
        xOI = state.x[self._i_OI]
        xe = state.x[self._i_electron]

        # Aliases for the per-transition rate buffers
        T2 = state.get_scratch('cooling:oi:T2')
        lnT2 = state.get_scratch('cooling:oi:lnT2')
        scratch_T = state.get_scratch('cooling:oi:tmp')
        scratch_o = state.get_scratch('cooling:oi:tmp_o')
        scratch_p = state.get_scratch('cooling:oi:tmp_p')
        k_HI = state.get_scratch('cooling:oi:k_HI')
        k_H2 = state.get_scratch('cooling:oi:k_H2')
        k_e = state.get_scratch('cooling:oi:k_e')
        q10 = state.get_scratch('cooling:oi:q10')
        q20 = state.get_scratch('cooling:oi:q20')
        q21 = state.get_scratch('cooling:oi:q21')
        q01 = state.get_scratch('cooling:oi:q01')
        q02 = state.get_scratch('cooling:oi:q02')
        q12 = state.get_scratch('cooling:oi:q12')
        tmp0 = state.get_scratch('cooling:oi:tmp0')
        tmp1 = state.get_scratch('cooling:oi:tmp1')
        tmp2 = state.get_scratch('cooling:oi:tmp2')

        np.multiply(T, 1.0e-2, out=T2)
        np.log(T2, out=lnT2)

        # ----- 1 <- 0 -----
        _assemble_kHI_kH2(
            T, T2, lnT2, scratch_T, scratch_o, scratch_p, q10,
            3.57e-10, 0.419, -0.003,
            1.49e-10, 0.264, 0.025,
            1.37e-10, 0.296, 0.043,
            k_HI, k_H2,
        )
        # k_e = 5.12e-10 * T**(-0.075)
        np.power(T, -0.075, out=k_e)
        np.multiply(k_e, 5.12e-10, out=k_e)
        # q10 = nH * (k_HI*xHI + k_H2*xH2 + k_e*xe)
        np.multiply(k_HI, xHI, out=q10)
        np.multiply(k_H2, xH2, out=scratch_T)
        np.add(q10, scratch_T, out=q10)
        np.multiply(k_e, xe, out=scratch_T)
        np.add(q10, scratch_T, out=q10)
        np.multiply(q10, nH, out=q10)

        # ----- 2 <- 0 -----
        _assemble_kHI_kH2(
            T, T2, lnT2, scratch_T, scratch_o, scratch_p, q20,
            3.19e-10, 0.369, -0.006,
            1.90e-10, 0.203, 0.041,
            2.23e-10, 0.237, 0.058,
            k_HI, k_H2,
        )
        np.power(T, -0.026, out=k_e)
        np.multiply(k_e, 4.86e-10, out=k_e)
        np.multiply(k_HI, xHI, out=q20)
        np.multiply(k_H2, xH2, out=scratch_T)
        np.add(q20, scratch_T, out=q20)
        np.multiply(k_e, xe, out=scratch_T)
        np.add(q20, scratch_T, out=q20)
        np.multiply(q20, nH, out=q20)

        # ----- 2 <- 1 -----
        _assemble_kHI_kH2(
            T, T2, lnT2, scratch_T, scratch_o, scratch_p, q21,
            4.34e-10, 0.755, -0.160,
            2.10e-12, 0.889, 0.043,
            3.00e-12, 1.198, 0.525,
            k_HI, k_H2,
        )
        np.power(T, 0.926, out=k_e)
        np.multiply(k_e, 1.08e-14, out=k_e)
        np.multiply(k_HI, xHI, out=q21)
        np.multiply(k_H2, xH2, out=scratch_T)
        np.add(q21, scratch_T, out=q21)
        np.multiply(k_e, xe, out=scratch_T)
        np.add(q21, scratch_T, out=q21)
        np.multiply(q21, nH, out=q21)

        # Reverse Boltzmann factors.
        # q01 = (g1 / g0) * q10 * exp(-E10 / (kB T))
        np.multiply(T, _K_B_CGS, out=scratch_T)
        np.divide(_E10, scratch_T, out=tmp0)
        np.negative(tmp0, out=tmp0)
        np.exp(tmp0, out=tmp0)
        np.multiply(q10, tmp0, out=q01)
        np.multiply(q01, _G1 / _G0, out=q01)

        np.divide(_E20, scratch_T, out=tmp0)
        np.negative(tmp0, out=tmp0)
        np.exp(tmp0, out=tmp0)
        np.multiply(q20, tmp0, out=q02)
        np.multiply(q02, _G2 / _G0, out=q02)

        np.divide(_E21, scratch_T, out=tmp0)
        np.negative(tmp0, out=tmp0)
        np.exp(tmp0, out=tmp0)
        np.multiply(q21, tmp0, out=q12)
        np.multiply(q12, _G2 / _G1, out=q12)

        cool_3level(
            q01, q10, q02, q20, q12, q21,
            _A10, _A20, _A21, _E10, _E20, _E21,
            xOI, out, tmp0, tmp1, tmp2,
        )

        if d_out is not None:
            d_out[:] = 0.0
