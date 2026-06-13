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
    SCRATCH_NAMES: ClassVar[tuple] = (
        'cooling:lya:T4',
        'cooling:lya:fac',
        'cooling:lya:ne',
        'cooling:lya:tmp_a',
        'cooling:lya:tmp_b',
        'cooling:lya:v',
        'cooling:lya:d_ln_fac',
    )
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
        v = state.get_scratch('cooling:lya:v')

        # T4 = T / 1e4
        np.multiply(T, 1.0e-4, out=T4)

        # v = 1 + (0.2 * T4)**0.64897   (denominator of `fac` below)
        np.multiply(T4, 0.2, out=v)
        np.power(v, 0.64897, out=v)
        np.add(v, 1.0, out=v)

        # fac = 5.30856e-08 * T4**0.14897 / v
        np.power(T4, 0.14897, out=fac)
        np.multiply(fac, 5.30856e-08, out=fac)
        np.divide(fac, v, out=fac)

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
        np.add(fac, _A10, out=fac)            # fac now holds D = q01+q10+A10
        np.divide(tmp_a, fac, out=out)        # f1 = q01 / D
        np.multiply(out, _A10 * _E10, out=out)
        np.multiply(out, xHI, out=out)

        if d_out is not None:
            # Derivation with mu held fixed during the cooling
            # sub-step:
            #   d(ln fac)/dT = (1/T) * (0.64897 / v - 0.5)
            #     (from d/dT [ln(T4^0.14897) - ln v] with v as above
            #      and dv/dT4 = 0.64897 (v - 1) / T4.)
            # so
            #   d(q01)/dT = q01 * (d(ln fac)/dT + 11.84 / (T * T4))
            #   d(q10)/dT = q10 *  d(ln fac)/dT
            #   d(D)/dT  = d(q01)/dT + d(q10)/dT
            #   d(f1)/dT = (d(q01)/dT - f1 * d(D)/dT) / D
            # Lambda = f1 * A10 * E10 * xHI, so
            #   d(Lambda)/dT = d(f1)/dT * A10 * E10 * xHI
            # Multiply by mu_at_entry to convert to d/d(T/mu).
            d_ln_fac = state.get_scratch('cooling:lya:d_ln_fac')
            # d_ln_fac = (1/T) * (0.64897 / v - 0.5)
            np.divide(0.64897, v, out=d_ln_fac)
            np.subtract(d_ln_fac, 0.5, out=d_ln_fac)
            np.divide(d_ln_fac, T, out=d_ln_fac)

            # d_out := dq01/dT = q01 * (d_ln_fac + 11.84 / (T * T4))
            np.multiply(T, T4, out=d_out)
            np.divide(11.84, d_out, out=d_out)
            np.add(d_out, d_ln_fac, out=d_out)
            np.multiply(d_out, tmp_a, out=d_out)    # tmp_a still = q01

            # d_ln_fac := dq10/dT = q10 * d_ln_fac
            np.multiply(d_ln_fac, tmp_b, out=d_ln_fac)  # tmp_b = q10

            # v := dD/dT = dq01/dT + dq10/dT  (reuse `v` -- no longer
            # needed for the value computation).
            np.add(d_out, d_ln_fac, out=v)

            # d_ln_fac := f1 * dD/dT = (q01 / D) * dD/dT
            np.divide(tmp_a, fac, out=d_ln_fac)   # f1
            np.multiply(d_ln_fac, v, out=d_ln_fac)

            # d_out = (dq01/dT - f1 * dD/dT) / D = df1/dT
            np.subtract(d_out, d_ln_fac, out=d_out)
            np.divide(d_out, fac, out=d_out)

            # dLambda/dT = df1/dT * A10 * E10 * xHI
            np.multiply(d_out, _A10 * _E10, out=d_out)
            np.multiply(d_out, xHI, out=d_out)

            # Convert d/dT to d/d(T/mu) via the chain rule (mu fixed
            # during the cooling sub-step): d/d(T/mu) = mu * d/dT.
            mu = state.get_scratch('solver:mu_at_entry')
            np.multiply(d_out, mu, out=d_out)
