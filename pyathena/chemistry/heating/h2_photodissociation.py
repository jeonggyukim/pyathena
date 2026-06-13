"""H2 photodissociation heating + pump heating (Hollenbach-McKee 1979 form).

Two channels live in this module:

- `H2DissociationHeating`: `Gamma_diss = xi_diss_H2 * x_H2 * 0.4 * eV`,
  the 0.4 eV per photodissociation event that goes into translational
  energy of the H atoms. Port of
  `tigris-ncr/src/photchem/ncr_rates.hpp:1554`.

- `H2PumpHeating`: `Gamma_pump = xi_diss_H2 * x_H2 * f_pump * mean_e
  * f * eV` with `f = 1 / (1 + ncrit/nH)`. The `ncrit` follows the
  Hollenbach-McKee 1979 form (`tigris-ncr/src/photchem/ncr_rates.hpp:
  965-967`). NCR default coefficients: `f_pump = 9.0`, `mean_e = 2.2`
  (matching iH2heating = 2 in the C++ enum). The Sternberg+2014 form
  (f_pump = 8.0, mean_e = 2.0) is the alternative, switched via the
  `form` keyword.

Both channels read `xi_diss_H2` from the channel state's optional
`xi_diss_H2` attribute (set by the radiation policy in Phase 4c)
with a default fallback of zero so the channel produces no heating
unless wired through a radiation source.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from ..cooling.base import HeatingChannel

_EV_CGS: float = 1.602176634e-12


class H2DissociationHeating(HeatingChannel):
    """0.4 eV per H2 photodissociation event (translational)."""

    name: ClassVar[str] = 'H2Dissociation'
    SCRATCH_NAMES: ClassVar[tuple] = (
        'heating:h2_pump:tmp_a',
        'heating:h2_pump:tmp_b',
        'heating:h2_pump:ncrit',
        'heating:h2_pump:f',
    )
    __version__: ClassVar[str] = '0.1@phase4b'

    def __init__(self, *, i_H2: int, xi_diss_H2: float = 0.0) -> None:
        self._i_H2 = int(i_H2)
        self._xi_diss_H2 = float(xi_diss_H2)

    def evaluate(
        self,
        state: Any,
        out: np.ndarray,
        d_out: Optional[np.ndarray] = None,
    ) -> None:
        xH2 = state.x[self._i_H2]
        np.multiply(xH2, self._xi_diss_H2 * 0.4 * _EV_CGS, out=out)
        if d_out is not None:
            d_out[:] = 0.0


class H2PumpHeating(HeatingChannel):
    """H2 UV pump heating (HM79 default; Sternberg+2014 optional)."""

    name: ClassVar[str] = 'H2Pump'
    __version__: ClassVar[str] = '0.1@phase4b'

    def __init__(
        self,
        *,
        i_HI: int,
        i_H2: int,
        xi_diss_H2: float = 0.0,
        form: str = 'HM79',
    ) -> None:
        self._i_HI = int(i_HI)
        self._i_H2 = int(i_H2)
        self._xi_diss_H2 = float(xi_diss_H2)
        if form == 'HM79':
            self._f_pump = 9.0
            self._mean_e = 2.2
        elif form == 'V18':
            # Sternberg+2014 / V18, i.e. NCR iH2heating = 1.
            self._f_pump = 8.0
            self._mean_e = 2.0
        else:
            raise ValueError(
                f'H2PumpHeating form={form!r} unknown; expected '
                "'HM79' or 'V18'.")
        self._form = form

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

        tmp_a = state.get_scratch('heating:h2_pump:tmp_a')
        tmp_b = state.get_scratch('heating:h2_pump:tmp_b')
        ncrit = state.get_scratch('heating:h2_pump:ncrit')
        f = state.get_scratch('heating:h2_pump:f')

        # HM79 ncrit (same formula as H2FormationHeating).
        np.divide(400.0, T, out=tmp_a)
        np.multiply(tmp_a, tmp_a, out=tmp_a)
        np.negative(tmp_a, out=tmp_a)
        np.exp(tmp_a, out=tmp_a)
        np.multiply(tmp_a, xHI, out=tmp_a)
        np.multiply(tmp_a, 1.6, out=tmp_a)
        np.add(T, 1200.0, out=tmp_b)
        np.divide(12000.0, tmp_b, out=tmp_b)
        np.negative(tmp_b, out=tmp_b)
        np.exp(tmp_b, out=tmp_b)
        np.multiply(tmp_b, xH2, out=tmp_b)
        np.multiply(tmp_b, 1.4, out=tmp_b)
        np.add(tmp_a, tmp_b, out=tmp_a)
        np.sqrt(T, out=ncrit)
        np.multiply(ncrit, tmp_a, out=ncrit)
        np.divide(1.0e6, ncrit, out=ncrit)

        # f = nH / (nH + ncrit)
        np.add(nH, ncrit, out=tmp_a)
        np.divide(nH, tmp_a, out=f)

        # Gamma = xi_diss_H2 * xH2 * f_pump * mean_e * f * eV
        amplitude = (self._xi_diss_H2 * self._f_pump * self._mean_e
                     * _EV_CGS)
        np.multiply(f, xH2, out=out)
        np.multiply(out, amplitude, out=out)

        if d_out is not None:
            d_out[:] = 0.0
