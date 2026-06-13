"""H I excitation cooling, Smith et al. 2021 fit.

Port of `pyathena.microphysics.cool.coolHISmith21`. The Smith+21 fit
covers H I excitation from n = 1 to n = 2, 3, 4, 5 via electron
collisions and improves on the DESPOTIC `coolHI` (Lya + Lyb +
2-photon only) at high temperature where high-n excitations matter.

The cooling rate is

    Lambda = x_HI * n_H * x_e * prefactor / (g1 * sqrt(T)) *
             sum_n  E_1n * exp(-T_1n / T) * Upsilon_1n_cool(T)

with prefactor = 8.62913e-6 (Maxwell-averaged collision strength
prefactor), `g1 = 2` (statistical weight of n = 1), and
`Upsilon_1n` piecewise-polynomial in T6 = T / 1e6 (cubic for
T6 <= 0.3, constant for T6 > 0.3).

Available as an alternative to `LymanAlphaCooling` (DESPOTIC) and
`LyaCooling` (NCR default). Production wiring keeps `LyaCooling`
unless a run specifically targets the Smith+21 high-T improvement.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from .base import CoolingChannel

_PREFACTOR: float = 8.62913e-6
_G1: float = 2.0

# Upsilon constants at T6 > 0.3 (Smith+21 Table)
_U12_HOT: float = 3.7354906
_U13_HOT: float = 0.8098996999999998
_U14_HOT: float = 0.3261425
_U15_HOT: float = 0.16427759999999997

# Excitation energies (erg) for n = 1 -> 2, 3, 4, 5
_E12: float = 1.63490e-11
_E13: float = 1.93766e-11
_E14: float = 2.04363e-11
_E15: float = 2.09267e-11

# Excitation temperatures (K)
_T12: float = 118415.6
_T13: float = 140344.4
_T14: float = 148019.5
_T15: float = 151572.0


def _u_cold(c0, c1, c2, c3, T6, T6_SQR, T6_CUB, out):
    """Upsilon at T6 <= 0.3: c0 + c1 T6 + c2 T6^2 + c3 T6^3."""
    np.multiply(T6_CUB, c3, out=out)
    np.multiply(T6_SQR, c2, out=out)  # would clobber: rewrite below
    # Re-do with Horner via temp slot: c3*T6^3 + c2*T6^2 + c1*T6 + c0
    out[:] = c3
    np.multiply(out, T6, out=out)
    np.add(out, c2, out=out)
    np.multiply(out, T6, out=out)
    np.add(out, c1, out=out)
    np.multiply(out, T6, out=out)
    np.add(out, c0, out=out)


class HISmith21Cooling(CoolingChannel):
    """H I excitation cooling (Smith+2021); alternative to Lya."""

    name: ClassVar[str] = 'HISmith21'
    __version__: ClassVar[str] = '0.1@phase4b'

    def __init__(self, *, i_HI: int, i_electron: int) -> None:
        self._i_HI = int(i_HI)
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
        xe = state.x[self._i_electron]

        Tinv = state.get_scratch('cooling:hi_smith21:Tinv')
        T6 = state.get_scratch('cooling:hi_smith21:T6')
        T6_SQR = state.get_scratch('cooling:hi_smith21:T6_SQR')
        T6_CUB = state.get_scratch('cooling:hi_smith21:T6_CUB')
        Upsilon = state.get_scratch('cooling:hi_smith21:Upsilon')
        u_cold = state.get_scratch('cooling:hi_smith21:u_cold')
        total = state.get_scratch('cooling:hi_smith21:total')
        tmp_a = state.get_scratch('cooling:hi_smith21:tmp_a')
        mask_cold = state.get_scratch('cooling:hi_smith21:mask_cold')
        mask_hot = state.get_scratch('cooling:hi_smith21:mask_hot')

        np.divide(1.0, T, out=Tinv)
        np.multiply(T, 1.0e-6, out=T6)
        np.multiply(T6, T6, out=T6_SQR)
        np.multiply(T6_SQR, T6, out=T6_CUB)

        np.less_equal(T6, 0.3, out=mask_cold)
        np.greater(T6, 0.3, out=mask_hot)

        total[:] = 0.0

        # 1 -> 2 channel
        _u_cold(0.616414, 16.8152, -32.0571, 35.5428,
                T6, T6_SQR, T6_CUB, u_cold)
        np.multiply(u_cold, mask_cold, out=u_cold)
        np.multiply(mask_hot, _U12_HOT, out=tmp_a)
        np.add(u_cold, tmp_a, out=Upsilon)
        np.multiply(Tinv, -_T12, out=tmp_a)
        np.exp(tmp_a, out=tmp_a)
        np.multiply(tmp_a, Upsilon, out=tmp_a)
        np.multiply(tmp_a, _E12, out=tmp_a)
        np.add(total, tmp_a, out=total)

        # 1 -> 3 channel
        _u_cold(0.217382, 3.92604, -10.6349, 13.7721,
                T6, T6_SQR, T6_CUB, u_cold)
        np.multiply(u_cold, mask_cold, out=u_cold)
        np.multiply(mask_hot, _U13_HOT, out=tmp_a)
        np.add(u_cold, tmp_a, out=Upsilon)
        np.multiply(Tinv, -_T13, out=tmp_a)
        np.exp(tmp_a, out=tmp_a)
        np.multiply(tmp_a, Upsilon, out=tmp_a)
        np.multiply(tmp_a, _E13, out=tmp_a)
        np.add(total, tmp_a, out=total)

        # 1 -> 4 channel
        _u_cold(0.0959324, 1.89951, -6.96467, 10.6362,
                T6, T6_SQR, T6_CUB, u_cold)
        np.multiply(u_cold, mask_cold, out=u_cold)
        np.multiply(mask_hot, _U14_HOT, out=tmp_a)
        np.add(u_cold, tmp_a, out=Upsilon)
        np.multiply(Tinv, -_T14, out=tmp_a)
        np.exp(tmp_a, out=tmp_a)
        np.multiply(tmp_a, Upsilon, out=tmp_a)
        np.multiply(tmp_a, _E14, out=tmp_a)
        np.add(total, tmp_a, out=total)

        # 1 -> 5 channel
        _u_cold(0.0747075, 0.670939, -2.28512, 3.4796,
                T6, T6_SQR, T6_CUB, u_cold)
        np.multiply(u_cold, mask_cold, out=u_cold)
        np.multiply(mask_hot, _U15_HOT, out=tmp_a)
        np.add(u_cold, tmp_a, out=Upsilon)
        np.multiply(Tinv, -_T15, out=tmp_a)
        np.exp(tmp_a, out=tmp_a)
        np.multiply(tmp_a, Upsilon, out=tmp_a)
        np.multiply(tmp_a, _E15, out=tmp_a)
        np.add(total, tmp_a, out=total)

        # Lambda = xHI * nH * xe * prefactor / (g1 sqrt(T)) * total
        np.sqrt(T, out=tmp_a)
        np.multiply(tmp_a, _G1, out=tmp_a)
        np.divide(_PREFACTOR, tmp_a, out=out)
        np.multiply(out, total, out=out)
        np.multiply(out, xHI, out=out)
        np.multiply(out, nH, out=out)
        np.multiply(out, xe, out=out)

        if d_out is not None:
            d_out[:] = 0.0
