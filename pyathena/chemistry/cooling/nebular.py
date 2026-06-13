"""Nebular metal-line cooling proxy (Gnat & Sternberg 2007 fit).

Port of `pyathena.microphysics.cool.coolneb`. This is a simple
Z_g-scaled proxy for the summed metal-line nebular cooling
(N II, O II, O III, S II, ...) when multi-ion abundances are not
tracked. It evaluates a 6th-order polynomial in `ln(T / 1e4 K)` for
the temperature dependence, plus a Hummer 1994 density-reduction
factor `f_red = 1 / (1 + 0.12 * (x_e n_H / 1e-2)^(0.38 - 0.12 ln T4))`,
and scales as `Z_g * x_HII * x_e * n_H / sqrt(T) * exp(-38585.52 / T)`.

This channel is a STOPGAP for runs that track H / H+ / H2 only
(NCRNetwork3). Phase 6 / Phase 7 will replace it with explicit
per-ion CHIANTI-table metal-line cooling once the multi-ion
network (NCRNetwork3PlusIons16) ships -- at that point each
followed ion (N II, O II, O III, S II, ...) contributes its own
analytic / CHIANTI-tabulated cooling channel and this proxy is
turned off in production.

CHIANTI / Phase 7 swap: replace this proxy entirely; do not retain
as a fallback. The explicit per-ion channels are strictly more
accurate and avoid the implicit-Z scaling assumption.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from .base import CoolingChannel

_PREFACTOR: float = 3.677602203699553e-21
_T_EXP: float = 38585.52

_ANEB = (
    -0.0050817, 0.00765822, 0.11832144, -0.50515842,
    0.81569592, -0.58648172, 0.69170381,
)


class NebularMetalLineCooling(CoolingChannel):
    """Stopgap nebular metal-line cooling (G&S 2007 fit). Phase 6/7
    replaces this with per-ion CHIANTI channels.
    """

    name: ClassVar[str] = 'NebularMetalLine'
    SCRATCH_NAMES: ClassVar[tuple] = (
        'cooling:neb:T4',
        'cooling:neb:lnT4',
        'cooling:neb:poly_fit',
        'cooling:neb:f_red',
        'cooling:neb:tmp_a',
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
        Z_g = state.Z_g

        T4 = state.get_scratch('cooling:neb:T4')
        lnT4 = state.get_scratch('cooling:neb:lnT4')
        poly_fit = state.get_scratch('cooling:neb:poly_fit')
        f_red = state.get_scratch('cooling:neb:f_red')
        tmp_a = state.get_scratch('cooling:neb:tmp_a')

        np.multiply(T, 1.0e-4, out=T4)
        np.log(T4, out=lnT4)

        # Horner: a[0] z^6 + a[1] z^5 + ... + a[6]; z = lnT4
        np.multiply(lnT4, _ANEB[0], out=poly_fit)
        np.add(poly_fit, _ANEB[1], out=poly_fit)
        np.multiply(poly_fit, lnT4, out=poly_fit)
        np.add(poly_fit, _ANEB[2], out=poly_fit)
        np.multiply(poly_fit, lnT4, out=poly_fit)
        np.add(poly_fit, _ANEB[3], out=poly_fit)
        np.multiply(poly_fit, lnT4, out=poly_fit)
        np.add(poly_fit, _ANEB[4], out=poly_fit)
        np.multiply(poly_fit, lnT4, out=poly_fit)
        np.add(poly_fit, _ANEB[5], out=poly_fit)
        np.multiply(poly_fit, lnT4, out=poly_fit)
        np.add(poly_fit, _ANEB[6], out=poly_fit)
        np.power(10.0, poly_fit, out=poly_fit)

        # f_red = 1 / (1 + 0.12 * (xe * nH * 1e-2)^(0.38 - 0.12 lnT4))
        np.multiply(xe, nH, out=tmp_a)
        np.multiply(tmp_a, 1.0e-2, out=tmp_a)
        np.multiply(lnT4, -0.12, out=f_red)
        np.add(f_red, 0.38, out=f_red)
        np.power(tmp_a, f_red, out=f_red)
        np.multiply(f_red, 0.12, out=f_red)
        np.add(f_red, 1.0, out=f_red)
        np.divide(1.0, f_red, out=f_red)

        # out = prefactor * Z_g * xHII * xe * nH / sqrt(T) *
        #       exp(-38585.52 / T) * poly_fit * f_red
        np.divide(_T_EXP, T, out=out)
        np.negative(out, out=out)
        np.exp(out, out=out)
        np.sqrt(T, out=tmp_a)
        np.divide(out, tmp_a, out=out)
        np.multiply(out, poly_fit, out=out)
        np.multiply(out, f_red, out=out)
        np.multiply(out, nH, out=out)
        np.multiply(out, xe, out=out)
        np.multiply(out, xHII, out=out)
        np.multiply(out, Z_g, out=out)
        np.multiply(out, _PREFACTOR, out=out)

        if d_out is not None:
            d_out[:] = 0.0
