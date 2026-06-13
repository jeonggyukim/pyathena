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


class HRecombinationCooling(CoolingChannel):
    """H II -> H I case-B recombination cooling."""

    name: ClassVar[str] = 'HRecombination'
    SCRATCH_NAMES: ClassVar[tuple] = (
        'cooling:h_rec:tmp',
        'cooling:h_rec:E_rr_B',
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

        # E_rr_B = (0.684 - 0.0416 * ln(T * 1e-4)) * kB * T
        np.multiply(T, 1.0e-4, out=scratch)
        np.log(scratch, out=scratch)
        np.multiply(scratch, -0.0416, out=scratch)
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
            d_out[:] = 0.0
