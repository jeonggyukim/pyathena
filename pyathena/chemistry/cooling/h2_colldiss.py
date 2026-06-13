"""H2 collisional dissociation cooling (Glover + Mac Low 2007).

Port of `pyathena.microphysics.cool.coolH2colldiss` + the
`coeff_coll_H2` helper. The mechanism: at T > 700 K, collisions
with H I and H2 dissociate H2 molecules; each event removes 4.48 eV
from the gas thermal pool.

    Lambda = 4.48 eV * x_H2 * xi_coll_H2(n_H, T, x_HI, x_H2)

The collisional rate `xi_coll_H2` follows the GMacL07 density-
dependent fit:

    xi_coll_H2 = k_{H2,HI} * n_H * x_HI + k_{H2,H2} * n_H * x_H2

with `k_{H2,X}` interpolated between low- and high-density limits
in log space, weighted by `n_2ncr = n_H * (x_HI / n_crH_HI +
2*x_H2 / n_crH_H2)`. The rate is gated to zero below 700 K.

The C++ NCR port at `tigris-ncr/src/photchem/ncr_rates.hpp:919-...`
clips the low-T rate coefficients with a floor to avoid `log10(0)`
warnings. This channel mirrors the C++ floor; in cold cells the
final result is still exactly zero by the T > 700 K gate, matching
the legacy Python output byte-for-byte.

CHIANTI / Cloudy swap: this channel stays hand-coded; H2
collisional dissociation has no atomic-line equivalent.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from .base import CoolingChannel

_EV_CGS: float = 1.602176634e-12
_E_DISSOC_ERG: float = 4.48 * _EV_CGS
_TEMP_GATE: float = 7.0e2
_RATE_FLOOR: float = 1.0e-300


class H2CollDissCooling(CoolingChannel):
    """Cooling by H2 collisional dissociation (GMacL07)."""

    name: ClassVar[str] = 'H2CollDiss'
    SCRATCH_NAMES: ClassVar[tuple] = (
        'cooling:h2_colldiss:Tinv',
        'cooling:h2_colldiss:logT4',
        'cooling:h2_colldiss:sqrtT',
        'cooling:h2_colldiss:k9l',
        'cooling:h2_colldiss:k9h',
        'cooling:h2_colldiss:k10l',
        'cooling:h2_colldiss:k10h',
        'cooling:h2_colldiss:ncrH2',
        'cooling:h2_colldiss:ncrHI',
        'cooling:h2_colldiss:n2ncr',
        'cooling:h2_colldiss:k_H2_HI',
        'cooling:h2_colldiss:k_H2_H2',
        'cooling:h2_colldiss:tmp_a',
        'cooling:h2_colldiss:tmp_b',
        'cooling:h2_colldiss:gate',
    )
    __version__: ClassVar[str] = '0.1@phase4b'

    def __init__(self, *, i_HI: int, i_H2: int) -> None:
        self._i_HI = int(i_HI)
        self._i_H2 = int(i_H2)

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

        Tinv = state.get_scratch('cooling:h2_colldiss:Tinv')
        logT4 = state.get_scratch('cooling:h2_colldiss:logT4')
        sqrtT = state.get_scratch('cooling:h2_colldiss:sqrtT')
        k9l = state.get_scratch('cooling:h2_colldiss:k9l')
        k9h = state.get_scratch('cooling:h2_colldiss:k9h')
        k10l = state.get_scratch('cooling:h2_colldiss:k10l')
        k10h = state.get_scratch('cooling:h2_colldiss:k10h')
        ncrH2 = state.get_scratch('cooling:h2_colldiss:ncrH2')
        ncrHI = state.get_scratch('cooling:h2_colldiss:ncrHI')
        n2ncr = state.get_scratch('cooling:h2_colldiss:n2ncr')
        k_H2_HI = state.get_scratch('cooling:h2_colldiss:k_H2_HI')
        k_H2_H2 = state.get_scratch('cooling:h2_colldiss:k_H2_H2')
        tmp_a = state.get_scratch('cooling:h2_colldiss:tmp_a')
        tmp_b = state.get_scratch('cooling:h2_colldiss:tmp_b')
        gate = state.get_scratch('cooling:h2_colldiss:gate')

        np.divide(1.0, T, out=Tinv)
        np.multiply(T, 1.0e-4, out=tmp_a)
        np.log10(tmp_a, out=logT4)
        np.sqrt(T, out=sqrtT)

        # k9l = 6.67e-12 * sqrt(T) * exp(-(1 + 63590/T))
        np.multiply(Tinv, 63590.0, out=tmp_a)
        np.add(tmp_a, 1.0, out=tmp_a)
        np.negative(tmp_a, out=tmp_a)
        np.exp(tmp_a, out=tmp_a)
        np.multiply(tmp_a, sqrtT, out=k9l)
        np.multiply(k9l, 6.67e-12, out=k9l)
        # k9h = 3.52e-9 * exp(-43900/T)
        np.multiply(Tinv, -43900.0, out=tmp_a)
        np.exp(tmp_a, out=tmp_a)
        np.multiply(tmp_a, 3.52e-9, out=k9h)
        # k10l = 5.996e-30 * T^4.1881 / (1 + 6.761e-6 T)^5.6881 * exp(-54657.4/T)
        np.power(T, 4.1881, out=tmp_a)
        np.multiply(T, 6.761e-6, out=tmp_b)
        np.add(tmp_b, 1.0, out=tmp_b)
        np.power(tmp_b, 5.6881, out=tmp_b)
        np.divide(tmp_a, tmp_b, out=k10l)
        np.multiply(k10l, 5.996e-30, out=k10l)
        np.multiply(Tinv, -54657.4, out=tmp_a)
        np.exp(tmp_a, out=tmp_a)
        np.multiply(k10l, tmp_a, out=k10l)
        # k10h = 1.3e-9 * exp(-53300/T)
        np.multiply(Tinv, -53300.0, out=tmp_a)
        np.exp(tmp_a, out=tmp_a)
        np.multiply(tmp_a, 1.3e-9, out=k10h)

        # Floor the rate coefficients before log10 to avoid -Inf
        # warnings in cold cells (the T > 700 K gate at the end zeros
        # the contribution from those cells either way).
        np.maximum(k9l, _RATE_FLOOR, out=k9l)
        np.maximum(k9h, _RATE_FLOOR, out=k9h)
        np.maximum(k10l, _RATE_FLOOR, out=k10l)
        np.maximum(k10h, _RATE_FLOOR, out=k10h)

        # ncrH2 = 10^(4.845 - 1.3 logT4 + 1.62 logT4^2)
        np.multiply(logT4, logT4, out=tmp_a)
        np.multiply(tmp_a, 1.62, out=tmp_a)
        np.multiply(logT4, -1.3, out=tmp_b)
        np.add(tmp_a, tmp_b, out=tmp_a)
        np.add(tmp_a, 4.845, out=tmp_a)
        np.power(10.0, tmp_a, out=ncrH2)
        # ncrHI = 10^(3.0 - 0.416 logT4 - 0.327 logT4^2)
        np.multiply(logT4, logT4, out=tmp_a)
        np.multiply(tmp_a, -0.327, out=tmp_a)
        np.multiply(logT4, -0.416, out=tmp_b)
        np.add(tmp_a, tmp_b, out=tmp_a)
        np.add(tmp_a, 3.0, out=tmp_a)
        np.power(10.0, tmp_a, out=ncrHI)

        # ncrinv = max(xHI/ncrHI + 2 xH2/ncrH2, 0)
        np.divide(xHI, ncrHI, out=tmp_a)
        np.divide(xH2, ncrH2, out=tmp_b)
        np.multiply(tmp_b, 2.0, out=tmp_b)
        np.add(tmp_a, tmp_b, out=tmp_a)
        np.maximum(tmp_a, 0.0, out=tmp_a)
        np.multiply(tmp_a, nH, out=n2ncr)

        # weights f = n2ncr/(1+n2ncr), g = 1/(1+n2ncr)
        np.add(n2ncr, 1.0, out=tmp_a)
        np.divide(n2ncr, tmp_a, out=tmp_b)    # f
        np.divide(1.0, tmp_a, out=tmp_a)      # g  (re-uses tmp_a)
        # k_H2_HI = 10**(log10(k9h)*f + log10(k9l)*g)
        np.log10(k9h, out=k_H2_HI)
        np.multiply(k_H2_HI, tmp_b, out=k_H2_HI)
        np.log10(k9l, out=ncrHI)              # reuse ncrHI buffer for log10(k9l)
        np.multiply(ncrHI, tmp_a, out=ncrHI)
        np.add(k_H2_HI, ncrHI, out=k_H2_HI)
        np.power(10.0, k_H2_HI, out=k_H2_HI)
        # k_H2_H2 = 10**(log10(k10h)*f + log10(k10l)*g)
        np.log10(k10h, out=k_H2_H2)
        np.multiply(k_H2_H2, tmp_b, out=k_H2_H2)
        np.log10(k10l, out=ncrH2)             # reuse ncrH2 buffer for log10(k10l)
        np.multiply(ncrH2, tmp_a, out=ncrH2)
        np.add(k_H2_H2, ncrH2, out=k_H2_H2)
        np.power(10.0, k_H2_H2, out=k_H2_H2)

        # xi_coll_H2 = k_H2_H2 * nH * xH2 + k_H2_HI * nH * xHI
        np.multiply(k_H2_HI, xHI, out=tmp_a)
        np.multiply(k_H2_H2, xH2, out=tmp_b)
        np.add(tmp_a, tmp_b, out=tmp_a)
        np.multiply(tmp_a, nH, out=tmp_a)

        # Gate: out = (T > 700) * tmp_a * xH2 * E_dissoc
        np.greater(T, _TEMP_GATE, out=gate)
        np.multiply(tmp_a, gate, out=tmp_a)
        np.multiply(tmp_a, xH2, out=out)
        np.multiply(out, _E_DISSOC_ERG, out=out)

        if d_out is not None:
            d_out[:] = 0.0
