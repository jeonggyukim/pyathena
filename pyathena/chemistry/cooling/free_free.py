"""H II free-free (bremsstrahlung) cooling.

Port of `pyathena.microphysics.cool.coolffH`. The H I free-free
emissivity from Draine 2011 10.11:

    Lambda = 1.422e-25 * g_ff(T) * (T / 1e4 K)**0.5 * n_H * x_e * x_HII

with frequency-averaged Gaunt factor

    g_ff(T) = 1 + 0.44 / (1 + 0.058 * ln(T / 10**5.4)**2)
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from .base import CoolingChannel

_PREFACTOR: float = 1.422e-25
_T_REF: float = 1.0e4
_T_GFF_REF: float = 10.0 ** 5.4  # 251188.6 K


class FreeFreeHCooling(CoolingChannel):
    """H II free-free (bremsstrahlung) cooling."""

    name: ClassVar[str] = 'FreeFreeH'
    SCRATCH_NAMES: ClassVar[tuple] = (
        'cooling:free_free:tmp',
        'cooling:free_free:gff',
        'cooling:free_free:L',
        'cooling:free_free:denom',
    )
    __version__: ClassVar[str] = '0.1@phase4b'

    def __init__(self, *, i_HII: int, i_electron: int) -> None:
        self._i_HII = int(i_HII)
        self._i_electron = int(i_electron)

    def evaluate(
        self,
        state: Any,
        out: np.ndarray,
        d_out: Optional[np.ndarray] = None,
    ) -> None:
        T = state.T
        nH = state.nH
        xHII = state.x[self._i_HII]
        xe = state.x[self._i_electron]

        scratch = state.get_scratch('cooling:free_free:tmp')
        gff = state.get_scratch('cooling:free_free:gff')
        L = state.get_scratch('cooling:free_free:L')
        denom = state.get_scratch('cooling:free_free:denom')

        # L = ln(T / T_gff_ref); denom = 1 + 0.058 L^2
        np.divide(T, _T_GFF_REF, out=L)
        np.log(L, out=L)
        np.multiply(L, L, out=denom)
        np.multiply(denom, 0.058, out=denom)
        np.add(denom, 1.0, out=denom)
        # gff = 1 + 0.44 / denom
        np.divide(0.44, denom, out=gff)
        np.add(gff, 1.0, out=gff)

        # Lambda = 1.422e-25 * g_ff * sqrt(T / 1e4) * nH * xe * xHII
        np.divide(T, _T_REF, out=scratch)
        np.sqrt(scratch, out=scratch)
        np.multiply(scratch, gff, out=out)
        np.multiply(out, _PREFACTOR, out=out)
        np.multiply(out, nH, out=out)
        np.multiply(out, xe, out=out)
        np.multiply(out, xHII, out=out)

        if d_out is not None:
            # Lambda = K * gff(T) * sqrt(T/1e4) with K = 1.422e-25 *
            # nH * xe * xHII. Define s = sqrt(T/1e4); then
            #   d(sqrt(T/1e4))/dT = 1 / (2 * 1e4 * s)
            # and using gff = 1 + 0.44/denom with denom = 1 + 0.058 L^2:
            #   dgff/dT = -0.44 / denom^2 * 0.116 * L * (1/T)
            #           = -0.05104 * L / (T * denom^2)
            # dLambda/dT = K * (dgff/dT * s + gff * 1/(2e4 s))
            #            = Lambda * (dgff/dT / gff + 1/(2 T))
            # because s = sqrt(T/1e4) -> 1/(2e4 s^2) = 1/(2T)
            # so the last term collapses to Lambda / (2T).

            # tmp := dgff/dT / gff = -0.05104 * L / (T * denom^2 * gff)
            np.multiply(denom, denom, out=scratch)        # denom^2
            np.multiply(scratch, gff, out=scratch)        # denom^2 * gff
            np.multiply(scratch, T, out=scratch)          # T * denom^2 * gff
            np.divide(L, scratch, out=d_out)              # L / above
            np.multiply(d_out, -0.05104, out=d_out)       # = dgff/dT / gff

            # add 1/(2T)
            np.divide(0.5, T, out=scratch)
            np.add(d_out, scratch, out=d_out)

            # multiply by Lambda to get dLambda/dT
            np.multiply(d_out, out, out=d_out)

            # convert to d/d(T/mu) by multiplying by mu
            mu = state.get_scratch('solver:mu_at_entry')
            np.multiply(d_out, mu, out=d_out)
