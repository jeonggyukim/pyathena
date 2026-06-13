"""H2 formation heating on grains.

Port of the NCR production form (Hollenbach + McKee 1979) at
`tigris-ncr/src/photchem/ncr_rates.hpp:1550`:

    Gamma_form = kgr * n_H * x_HI * (0.2 + 4.2 / (1 + n_crit / n_H))

with the temperature-dependent grain rate
`kgr = kgr_H2 * Z_d * sqrt(T2) * 2 / (1 + 0.4*sqrt(T2) + 0.2*T2 + 0.08*T2^2)`
when the temperature-dependent flag is on (the NCR default), where
`T2 = T / 100`. The critical density n_crit follows the HM79 form
`n_crit = 1e6 / sqrt(T) / (1.6 * x_HI * exp(-(400/T)^2) + 1.4 * x_H2
* exp(-12000/(T+1200)))`. Each H2 formed on a grain returns the
binding-energy fraction `0.2 + 4.2 * f` to the gas (units of eV);
the `f` factor accounts for whether the molecule thermalises with
the gas (f -> 0 at low density) or radiates (f -> 1 at high
density).

The kgr_H2 default is `3e-17 cm^3/s` (Z_d = 1 anchor); production
runs override via constructor argument. The eV -> erg conversion is
already applied so `out` carries erg / s / cm^3 like every other
channel.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from ..cooling.base import HeatingChannel

_EV_CGS: float = 1.602176634e-12


class H2FormationHeating(HeatingChannel):
    """H2 grain-surface formation heating (HM79 / NCR default)."""

    name: ClassVar[str] = 'H2Formation'
    SCRATCH_NAMES: ClassVar[tuple] = (
        'heating:h2_form:T2',
        'heating:h2_form:sqrtT2',
        'heating:h2_form:kgr',
        'heating:h2_form:ncrit',
        'heating:h2_form:tmp_a',
        'heating:h2_form:tmp_b',
        'heating:h2_form:f',
    )
    __version__: ClassVar[str] = '0.1@phase4b'

    def __init__(
        self,
        *,
        i_HI: int,
        i_H2: int,
        kgr_H2: float = 3.0e-17,
        xi_diss_H2: float = 0.0,
        temperature_dependent_kgr: bool = True,
    ) -> None:
        self._i_HI = int(i_HI)
        self._i_H2 = int(i_H2)
        self._kgr_H2 = float(kgr_H2)
        self._xi_diss_H2 = float(xi_diss_H2)
        self._temperature_dependent_kgr = bool(temperature_dependent_kgr)

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
        Z_d = state.Z_d

        T2 = state.get_scratch('heating:h2_form:T2')
        sqrtT2 = state.get_scratch('heating:h2_form:sqrtT2')
        kgr = state.get_scratch('heating:h2_form:kgr')
        ncrit = state.get_scratch('heating:h2_form:ncrit')
        tmp_a = state.get_scratch('heating:h2_form:tmp_a')
        tmp_b = state.get_scratch('heating:h2_form:tmp_b')
        f = state.get_scratch('heating:h2_form:f')

        np.multiply(T, 1.0e-2, out=T2)
        if self._temperature_dependent_kgr:
            # kgr = kgr_H2 * Z_d * sqrt(T2) * 2 / (1 + 0.4*sqrt(T2)
            #       + 0.2*T2 + 0.08*T2^2)
            np.sqrt(T2, out=sqrtT2)
            # denominator
            np.multiply(T2, T2, out=tmp_a)
            np.multiply(tmp_a, 0.08, out=tmp_a)
            np.multiply(T2, 0.2, out=tmp_b)
            np.add(tmp_a, tmp_b, out=tmp_a)
            np.multiply(sqrtT2, 0.4, out=tmp_b)
            np.add(tmp_a, tmp_b, out=tmp_a)
            np.add(tmp_a, 1.0, out=tmp_a)
            # kgr = 2 * sqrt(T2) / denom * (kgr_H2 * Z_d)
            np.multiply(sqrtT2, 2.0 * self._kgr_H2 * Z_d, out=kgr)
            np.divide(kgr, tmp_a, out=kgr)
        else:
            kgr[:] = self._kgr_H2 * Z_d

        # ncrit (HM79): 1e6 / sqrt(T) / (1.6 * xHI * exp(-(400/T)^2)
        #               + 1.4 * xH2 * exp(-12000/(T+1200)))
        np.divide(400.0, T, out=tmp_a)
        np.multiply(tmp_a, tmp_a, out=tmp_a)
        np.negative(tmp_a, out=tmp_a)
        np.exp(tmp_a, out=tmp_a)
        np.multiply(tmp_a, xHI, out=tmp_a)
        np.multiply(tmp_a, 1.6, out=tmp_a)
        # second term
        np.add(T, 1200.0, out=tmp_b)
        np.divide(12000.0, tmp_b, out=tmp_b)
        np.negative(tmp_b, out=tmp_b)
        np.exp(tmp_b, out=tmp_b)
        np.multiply(tmp_b, xH2, out=tmp_b)
        np.multiply(tmp_b, 1.4, out=tmp_b)
        np.add(tmp_a, tmp_b, out=tmp_a)
        # ncrit = 1e6 / sqrt(T) / tmp_a
        np.sqrt(T, out=ncrit)
        np.multiply(ncrit, tmp_a, out=ncrit)
        np.divide(1.0e6, ncrit, out=ncrit)

        # f = 1 / (1 + ncrit/nH) = nH / (nH + ncrit)
        np.add(nH, ncrit, out=tmp_a)
        np.divide(nH, tmp_a, out=f)

        # Gamma_form = kgr * nH * xHI * (0.2 + 4.2 * f) * eV
        np.multiply(f, 4.2, out=tmp_a)
        np.add(tmp_a, 0.2, out=tmp_a)
        np.multiply(kgr, nH, out=out)
        np.multiply(out, xHI, out=out)
        np.multiply(out, tmp_a, out=out)
        np.multiply(out, _EV_CGS, out=out)

        if d_out is not None:
            d_out[:] = 0.0
