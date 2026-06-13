"""H I Lyman-alpha 2-level cooling (NCR production default).

Port of `pyathena.microphysics.cool.coolLya`. This is the H I
cooling form used by tigris-ncr PhotochemistryNCR. It treats H I
as a 2-level system 1s -> 2p (Lyman-alpha), uses the Draine 2011
17.18 effective collision strength fit for the electron partner,
and applies the steady-state population balance:

    fac = 5.30856e-08 * T4**0.14897 / (1 + (0.2 * T4)**0.64897)
    k01e = fac * exp(-11.84 / T4)        (Boltzmann factor)
    q01 = k01e * n_e
    q10 = (g0 / g1) * fac * n_e
    Lambda = q01 / (q01 + q10 + A10) * A10 * E10 * x_HI

with `A10 = 6.265e8 s^-1`, `E10 = 1.634e-11 erg`, statistical weights
`g0 = 1` (1s), `g1 = 3` (2p), and `T4 = T / 1e4 K`.

This channel is the NCR default. The `LymanAlphaCooling` channel
ports DESPOTIC's `coolHI` which sums Lyman-alpha + Lyman-beta +
2-photon contributions; it is available for runs that match the
DESPOTIC convention (Krumholz 2014).
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from .base import CoolingChannel

_A10: float = 6.265e8
_E10: float = 1.634e-11
_G0: float = 1.0
_G1: float = 3.0


class LyaCooling(CoolingChannel):
    """H I Lyman-alpha 2-level cooling. NCR production default."""

    name: ClassVar[str] = 'Lya'
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

        T4 = state.get_scratch('cooling:lya:T4')
        fac = state.get_scratch('cooling:lya:fac')
        ne = state.get_scratch('cooling:lya:ne')
        tmp_a = state.get_scratch('cooling:lya:tmp_a')
        tmp_b = state.get_scratch('cooling:lya:tmp_b')

        # T4 = T / 1e4
        np.multiply(T, 1.0e-4, out=T4)

        # fac = 5.30856e-08 * T4**0.14897 / (1 + (0.2 * T4)**0.64897)
        np.power(T4, 0.14897, out=fac)
        np.multiply(fac, 5.30856e-08, out=fac)
        np.multiply(T4, 0.2, out=tmp_a)
        np.power(tmp_a, 0.64897, out=tmp_a)
        np.add(tmp_a, 1.0, out=tmp_a)
        np.divide(fac, tmp_a, out=fac)

        # ne = xe * nH
        np.multiply(xe, nH, out=ne)

        # q01 = fac * exp(-11.84 / T4) * ne; q10 = (g0/g1) * fac * ne
        np.divide(11.84, T4, out=tmp_a)
        np.negative(tmp_a, out=tmp_a)
        np.exp(tmp_a, out=tmp_a)
        np.multiply(fac, tmp_a, out=tmp_a)
        np.multiply(tmp_a, ne, out=tmp_a)    # tmp_a = q01

        np.multiply(fac, _G0 / _G1, out=tmp_b)
        np.multiply(tmp_b, ne, out=tmp_b)    # tmp_b = q10

        # Lambda = q01 / (q01 + q10 + A10) * A10 * E10 * xHI
        np.add(tmp_a, tmp_b, out=fac)
        np.add(fac, _A10, out=fac)
        np.divide(tmp_a, fac, out=out)
        np.multiply(out, _A10 * _E10, out=out)
        np.multiply(out, xHI, out=out)

        if d_out is not None:
            d_out[:] = 0.0
