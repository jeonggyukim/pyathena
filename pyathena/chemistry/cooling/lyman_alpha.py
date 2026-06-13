"""H I Lyman-alpha + Lyman-beta + 2-photon cooling.

Compatibility port of `pyathena.microphysics.cool.coolHI` (lines
490-513). Three contributions sum into the channel total:

    Lambda_2p   = fac * exp(-TLyA/T) * upsilon_{2s} * xHI * xe * nH
                  * k_B * TLyA
    Lambda_LyA  = fac * exp(-TLyA/T) * upsilon_{2p} * xHI * xe * nH
                  * k_B * TLyA
    Lambda_LyB  = fac * exp(-TLyB/T) * (upsilon_{3s} + upsilon_{3p}
                  + upsilon_{3d}) * xHI * xe * nH * k_B * TLyB

with `fac = 8.629e-6 / (2 * sqrt(T))`. References go through the
DESPOTIC implementation Krumholz 2014 ApJS 211, 19.

Phase 4a: the channel evaluates Lambda only; the temperature
derivative is reported as zero. Phase 4b will fill in the analytic
`d(Lambda)/d(T/mu)` so the semi-implicit T/mu damping pulls in the
correct stiffness signal.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from .base import CoolingChannel


# Effective excitation temperatures of the n = 2 and n = 3 levels.
_T_LYA: float = 118415.63430152694
_T_LYB: float = 140344.45546847637

# Collision-strength prefactor from the Maxwell-averaged effective
# collision strength (the 8.629e-6 cm^3/s/K^0.5 constant out front of
# the Boltzmann factor in Draine 17.10).
_BETA_COLL: float = 8.629e-6

# Boltzmann constant in CGS (k_B in erg / K). Hardcoded here to avoid
# pulling astropy into the hot-path import chain; matches
# `astropy.constants.k_B.cgs.value` to all 17 digits.
_K_B_CGS: float = 1.380649e-16

# Effective collision strengths for the 2s / 2p / 3s / 3p / 3d levels
# (Draine 11.32 / 11.34 / 11.36).
_UPSILON_2S: float = 0.35
_UPSILON_2P: float = 0.69
_UPSILON_3S: float = 0.077
_UPSILON_3P: float = 0.14
_UPSILON_3D: float = 0.073
_UPSILON_N3_SUM: float = _UPSILON_3S + _UPSILON_3P + _UPSILON_3D


class LymanAlphaCooling(CoolingChannel):
    """H I Lyman-series + 2-photon cooling."""

    name: ClassVar[str] = 'LymanAlpha'
    __version__: ClassVar[str] = '0.1@phase4a'

    def __init__(self, *, i_HI: int, i_electron: int) -> None:
        """Cache the species-row indices once at construction.

        Parameters
        ----------
        i_HI : int
            Row index of `H I` in the SpeciesSet the driver uses.
        i_electron : int
            Row index of the electron ghost (the charge-sum ghost).
        """
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

        # fac = 8.629e-6 / (2 * sqrt(T))   [cm^3 / s, scaled]
        # exfacLyA = exp(-T_LYA / T)
        # exfacLyB = exp(-T_LYB / T)
        # Lambda_2p   = fac * exfacLyA * upsilon_2s * xHI * xe * nH * kB * T_LYA
        # Lambda_LyA  = fac * exfacLyA * upsilon_2p * xHI * xe * nH * kB * T_LYA
        # Lambda_LyB  = fac * exfacLyB * (sum upsilons n=3) * xHI * xe * nH * kB * T_LYB

        # Implementation strategy: assemble `xHI * xe * nH` once; then
        # for each line `fac * upsilon * exp(-T_eff/T) * kB * T_eff *
        # (above)`. Keep allocations off the hot path -- use `out` for
        # the running accumulator and a single temp slice off the
        # state scratch namespace.
        scratch = state.get_scratch('cooling:lyman_alpha:tmp')
        prefac = state.get_scratch('cooling:lyman_alpha:prefac')

        # prefac = xHI * xe * nH * (kB / (2 * sqrt(T))) * 8.629e-6
        # so each line's Lambda = prefac * upsilon * exp(-T_eff/T) * T_eff
        np.multiply(xHI, xe, out=prefac)
        np.multiply(prefac, nH, out=prefac)
        np.sqrt(T, out=scratch)
        np.multiply(scratch, 2.0, out=scratch)
        np.divide(prefac, scratch, out=prefac)
        np.multiply(prefac, _BETA_COLL * _K_B_CGS, out=prefac)

        # Lambda for the n=2 lines: 2-photon (2s) + Lyman-alpha (2p)
        #   Lambda_n2 = prefac * (upsilon_2s + upsilon_2p) * exfacLyA
        #             * T_LYA
        np.divide(_T_LYA, T, out=scratch)
        np.negative(scratch, out=scratch)
        np.exp(scratch, out=scratch)
        coeff_n2 = (_UPSILON_2S + _UPSILON_2P) * _T_LYA
        np.multiply(scratch, coeff_n2, out=scratch)
        np.multiply(scratch, prefac, out=out)

        # Lambda for the n=3 line: Lyman-beta
        np.divide(_T_LYB, T, out=scratch)
        np.negative(scratch, out=scratch)
        np.exp(scratch, out=scratch)
        coeff_n3 = _UPSILON_N3_SUM * _T_LYB
        np.multiply(scratch, coeff_n3, out=scratch)
        # accumulate: out += scratch * prefac
        np.multiply(scratch, prefac, out=scratch)
        np.add(out, scratch, out=out)

        if d_out is not None:
            # Phase 4b: analytic d(Lambda)/d(T/mu). Report zero for
            # Phase 4a so the dispatcher contract is honoured even
            # though the damping term in the semi-implicit T/mu update
            # underestimates stiffness around the steep Boltzmann
            # exponential.
            d_out[:] = 0.0

    @classmethod
    def required_scratch(cls) -> tuple:
        """Names + shapes of scratch slots the channel needs.

        The driver calls this once at setup time to allocate the
        channel's owned scratch (in addition to the per-channel
        `Lambda` / `dLambda` slots the aggregator allocates).
        """
        return (
            ('cooling:lyman_alpha:tmp', None),
            ('cooling:lyman_alpha:prefac', None),
        )
