"""H II case-B recombination cooling.

Port of `pyathena.microphysics.cool.coolrecH`. Each recombination of
an electron with H+ deposits an average kinetic energy
`E_rr_B = (0.684 - 0.0416 * ln(T_4)) * k_B * T` (DESPOTIC fit; Draine
2011 14.42) into the radiation field:

    Lambda = E_rr_B * alpha_B(T) * n_H * x_e * x_HII

with `alpha_B(T) = get_rec_rate_H_caseB(T)` from
`pyathena.chemistry.rates.rec_rate.RecRate`. The channel owns a
`RecRate` instance so the analytic `alpha_B` formula is evaluated
without going through the `(Z, N) = (1, 0)` dispatch on every
substep.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from ..rates.rec_rate import RecRate
from .base import CoolingChannel

_K_B_CGS: float = 1.380649e-16

# Constants used inside the d(ln alpha_B) / dT expression. Computed
# from the raw exponents (1.5, -2.242, 0.407) so the cancellation
# inside `d(ln Lambda)/dT = d(ln E_rr_B)/dT + d(ln alpha)/dT` does
# not amplify any 5-digit-truncation roundoff into ~1e-4 relative
# error.
_DLNALPHA_K1: float = 2.242 * 0.407           # 0.912494
_DLNALPHA_K0: float = _DLNALPHA_K1 - 1.5      # -0.587506


class HRecombinationCooling(CoolingChannel):
    """H II -> H I case-B recombination cooling."""

    name: ClassVar[str] = 'HRecombination'
    SCRATCH_NAMES: ClassVar[tuple] = (
        'cooling:h_rec:tmp',
        'cooling:h_rec:E_rr_B',
        'cooling:h_rec:u',
        'cooling:h_rec:d_ln_alpha',
    )
    __version__: ClassVar[str] = '0.1@phase4b'

    def __init__(self, *, i_HII: int, i_electron: int) -> None:
        self._i_HII = int(i_HII)
        self._i_electron = int(i_electron)
        self._rec = RecRate()

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

        scratch = state.get_scratch('cooling:h_rec:tmp')
        E_rr_B = state.get_scratch('cooling:h_rec:E_rr_B')
        u = state.get_scratch('cooling:h_rec:u')

        # u = ln(T / 1e4); E_rr_B = (0.684 - 0.0416 * u) * kB * T
        np.multiply(T, 1.0e-4, out=u)
        np.log(u, out=u)
        np.multiply(u, -0.0416, out=scratch)
        np.add(scratch, 0.684, out=scratch)
        np.multiply(scratch, _K_B_CGS, out=scratch)
        np.multiply(scratch, T, out=E_rr_B)

        # alpha_B(T) -> reuse `scratch` via the analytic helper. The
        # legacy code path calls
        # `RecRate().get_rec_rate_H_caseB(T)`; this returns a fresh
        # allocation, which is fine outside the substep (called once
        # per cooling-update). For the strict hot-path-zero-alloc
        # requirement, Phase 4c will rebind to a pre-tabulated path.
        alpha = self._rec.get_rec_rate_H_caseB(T)

        # Lambda = E_rr_B * alpha * nH * xe * xHII
        np.multiply(E_rr_B, alpha, out=out)
        np.multiply(out, nH, out=out)
        np.multiply(out, xe, out=out)
        np.multiply(out, xHII, out=out)

        if d_out is not None:
            # Lambda = E_rr_B * alpha * nH * xe * xHII. nH / xe / xHII
            # are T-independent during the substep, so
            #   dLambda/dT = Lambda * (d(ln E_rr_B)/dT + d(ln alpha)/dT)
            #
            # E_rr_B / (kB * T) = 0.684 - 0.0416 * u with u = ln(T/1e4)
            # => d(E_rr_B/kB)/dT = (0.684 - 0.0416 u) - 0.0416
            #                    = 0.6424 - 0.0416 u
            # => d(ln E_rr_B)/dT = (0.6424 - 0.0416 u) /
            #                      ((0.684 - 0.0416 u) * T)
            #
            # alpha_B(T) = 2.753e-14 * b^1.5 * d^(-2.242)
            # b = 315614/T, c = 115188/T, d = 1 + c^0.407
            # d(ln b)/dT = -1/T
            # d(ln c)/dT = -1/T
            # d(c^0.407)/dT = c^0.407 * 0.407 / c * dc/dT
            #               = -0.407 * c^0.407 / T
            #               = -0.407 * (d - 1) / T
            # d(ln d)/dT = (1/d) * d(d)/dT = -0.407 * (d - 1) / (T * d)
            # d(ln alpha)/dT = 1.5 * (-1/T) + (-2.242) * (-0.407 * (d-1) / (T d))
            #               = (1/T) * (-1.5 + 0.91249 * (d-1)/d)
            #               = (1/T) * (-1.5 + 0.91249 - 0.91249/d)
            #               = (1/T) * (-0.58751 - 0.91249/d)
            d_ln_alpha = state.get_scratch('cooling:h_rec:d_ln_alpha')

            # d_arg (= 1 + (115188/T)^0.407) -- recompute into
            # d_ln_alpha temporarily so we can form -K1/d_arg cleanly.
            np.divide(115188.0, T, out=d_ln_alpha)
            np.power(d_ln_alpha, 0.407, out=d_ln_alpha)
            np.add(d_ln_alpha, 1.0, out=d_ln_alpha)
            # d_ln_alpha = (K0 - K1 / d_arg) / T, where
            # K0 = 2.242 * 0.407 - 1.5 and K1 = 2.242 * 0.407.
            np.divide(_DLNALPHA_K1, d_ln_alpha, out=d_ln_alpha)
            np.subtract(_DLNALPHA_K0, d_ln_alpha, out=d_ln_alpha)
            np.divide(d_ln_alpha, T, out=d_ln_alpha)

            # d(ln E_rr_B)/dT = (0.6424 - 0.0416 u) /
            #                  ((0.684 - 0.0416 u) * T)
            # numerator into d_out, denom into scratch
            np.multiply(u, -0.0416, out=d_out)
            np.add(d_out, 0.6424, out=d_out)
            np.multiply(u, -0.0416, out=scratch)
            np.add(scratch, 0.684, out=scratch)
            np.multiply(scratch, T, out=scratch)
            np.divide(d_out, scratch, out=d_out)
            # d_out += d_ln_alpha
            np.add(d_out, d_ln_alpha, out=d_out)
            # d_out *= Lambda
            np.multiply(d_out, out, out=d_out)
            # mu factor
            mu = state.get_scratch('solver:mu_at_entry')
            np.multiply(d_out, mu, out=d_out)
