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

        # g_ff = 1 + 0.44 / (1 + 0.058 * ln(T / T_gff_ref)**2)
        np.divide(T, _T_GFF_REF, out=scratch)
        np.log(scratch, out=scratch)
        np.square(scratch, out=scratch)
        np.multiply(scratch, 0.058, out=scratch)
        np.add(scratch, 1.0, out=scratch)
        np.divide(0.44, scratch, out=gff)
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
            d_out[:] = 0.0
