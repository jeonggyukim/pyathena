"""Cosmic-ray ionization heating.

Port of `pyathena.microphysics.cool.heatCR`. Heating per CR
ionization of H I is

    q_HI = (6.5 + 26.4 * sqrt(x_e / (x_e + 0.07))) * eV

(Glassgold et al. 2012; same shape as DESPOTIC Krumholz 2014
Appendix B). Heating per CR ionization of H2 is the Krumholz 2014
density-dependent fit:

    q_H2 = 10 eV                                 (log_nH < 2)
         = (10 + 1.5 * (log_nH - 2)) eV          (2 <= log_nH < 4)
         = (13 + (log_nH - 4) * 4/3) eV          (4 <= log_nH < 7)
         = (17 + (log_nH - 7) / 3) eV            (7 <= log_nH < 10)
         = 18 eV                                 (log_nH >= 10)

Total:

    Gamma = xi_CR * (x_HI * q_HI + 2 * x_H2 * q_H2)

The factor 2 on x_H2 accounts for both protons in the molecule being
available for ionization. The legacy `heatCR` omits the `+ 4.6e-10
xe` Bialy 2019 contribution; this port matches the legacy convention
byte-for-byte.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from ..cooling.base import HeatingChannel

_EV_CGS: float = 1.602176634e-12


class CosmicRayHeating(HeatingChannel):
    """Cosmic-ray ionization heating of H I + H2."""

    name: ClassVar[str] = 'CosmicRay'
    SCRATCH_NAMES: ClassVar[tuple] = (
        'heating:cosmic_ray:tmp',
        'heating:cosmic_ray:qHI',
        'heating:cosmic_ray:qH2',
        'heating:cosmic_ray:log_nH',
        'heating:cosmic_ray:mask_b1',
    )
    __version__: ClassVar[str] = '0.1@phase4b'

    def __init__(
        self,
        *,
        i_HI: int,
        i_H2: int,
        i_electron: int,
        xi_CR: float = 2.0e-16,
    ) -> None:
        self._i_HI = int(i_HI)
        self._i_H2 = int(i_H2)
        self._i_electron = int(i_electron)
        self._xi_CR = float(xi_CR)

    def evaluate(
        self,
        state: Any,
        out: np.ndarray,
        d_out: Optional[np.ndarray] = None,
    ) -> None:
        nH = state.nH
        xHI = state.x[self._i_HI]
        xH2 = state.x[self._i_H2]
        xe = state.x[self._i_electron]

        scratch = state.get_scratch('heating:cosmic_ray:tmp')
        qHI = state.get_scratch('heating:cosmic_ray:qHI')
        qH2 = state.get_scratch('heating:cosmic_ray:qH2')
        log_nH = state.get_scratch('heating:cosmic_ray:log_nH')

        # q_HI = (6.5 + 26.4 * sqrt(xe / (xe + 0.07))) * eV
        np.add(xe, 0.07, out=scratch)
        np.divide(xe, scratch, out=scratch)
        np.sqrt(scratch, out=scratch)
        np.multiply(scratch, 26.4, out=scratch)
        np.add(scratch, 6.5, out=qHI)
        np.multiply(qHI, _EV_CGS, out=qHI)

        # q_H2: piecewise in log10(n_H). Branch-free over the strip:
        # compute each piece as `mask * value` and sum.
        np.log10(nH, out=log_nH)

        # Piece 1: log_nH < 2.0 -> 10.0 eV
        np.less(log_nH, 2.0, out=scratch)
        np.multiply(scratch, 10.0, out=qH2)
        # Piece 2: 2 <= log_nH < 4 -> 10 + 1.5*(log_nH - 2)
        mask = state.get_scratch('heating:cosmic_ray:mask_b1')
        np.greater_equal(log_nH, 2.0, out=mask)
        np.less(log_nH, 4.0, out=scratch)
        np.logical_and(mask, scratch, out=scratch)
        np.subtract(log_nH, 2.0, out=mask)
        np.multiply(mask, 1.5, out=mask)
        np.add(mask, 10.0, out=mask)
        np.multiply(mask, scratch, out=mask)
        np.add(qH2, mask, out=qH2)
        # Piece 3: 4 <= log_nH < 7 -> 13 + (log_nH - 4)*4/3
        np.greater_equal(log_nH, 4.0, out=mask)
        np.less(log_nH, 7.0, out=scratch)
        np.logical_and(mask, scratch, out=scratch)
        np.subtract(log_nH, 4.0, out=mask)
        np.multiply(mask, 4.0 / 3.0, out=mask)
        np.add(mask, 13.0, out=mask)
        np.multiply(mask, scratch, out=mask)
        np.add(qH2, mask, out=qH2)
        # Piece 4: 7 <= log_nH < 10 -> 17 + (log_nH - 7)/3
        np.greater_equal(log_nH, 7.0, out=mask)
        np.less(log_nH, 10.0, out=scratch)
        np.logical_and(mask, scratch, out=scratch)
        np.subtract(log_nH, 7.0, out=mask)
        np.multiply(mask, 1.0 / 3.0, out=mask)
        np.add(mask, 17.0, out=mask)
        np.multiply(mask, scratch, out=mask)
        np.add(qH2, mask, out=qH2)
        # Piece 5: log_nH >= 10 -> 18.0
        np.greater_equal(log_nH, 10.0, out=scratch)
        np.multiply(scratch, 18.0, out=scratch)
        np.add(qH2, scratch, out=qH2)
        # Multiply by eV to bring from "in units of eV" to erg.
        np.multiply(qH2, _EV_CGS, out=qH2)

        # Gamma = xi_CR * (xHI * qHI + 2 * xH2 * qH2)
        np.multiply(xHI, qHI, out=out)
        np.multiply(xH2, qH2, out=scratch)
        np.multiply(scratch, 2.0, out=scratch)
        np.add(out, scratch, out=out)
        np.multiply(out, self._xi_CR, out=out)

        if d_out is not None:
            d_out[:] = 0.0
