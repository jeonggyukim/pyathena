"""H I collisional-ionization cooling.

Port of `pyathena.microphysics.cool.coolHIion`. The mechanism: a
thermal electron knocks an electron off neutral H I, transferring
13.6 eV of thermal energy into the ionization potential.

    Lambda = 13.6 eV * k_coll(T) * n_H * x_e * x_HI

with `k_coll(T)` an 8-term polynomial in `log(T_e)` (T in eV;
T_e = T * 8.6173e-5) taken from Janev 1987 via DESPOTIC (the same
formula appears in `pyathena.microphysics.cool.coeff_kcoll_H`).
The fit is gated to T > 3000 K; below that the rate is set to zero.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from .base import CoolingChannel

_EV_CGS: float = 1.602176634e-12  # 1 eV in erg
_E_IONIZATION_HI_ERG: float = 13.6 * _EV_CGS
# Boltzmann constant times 1 K in eV: T_in_K * 8.6173e-5 -> T in eV.
_KB_EV: float = 8.6173e-5

# Janev 1987 polynomial coefficients (DESPOTIC ordering: c0 + c1*y +
# c2*y^2 + ... + c8*y^8 where y = ln(T_e). Listed as in
# `coeff_kcoll_H(T)`.)
_C0: float = -3.271396786e1
_C1: float =  1.35365560e1
_C2: float = -5.73932875
_C3: float =  1.56315498
_C4: float = -2.877056e-1
_C5: float =  3.48255977e-2
_C6: float = -2.63197617e-3
_C7: float =  1.11954395e-4
_C8: float = -2.03914985e-6


class HICollisionalIonizationCooling(CoolingChannel):
    """Cooling due to thermal-electron collisional ionization of H I."""

    name: ClassVar[str] = 'HICollisionalIonization'
    SCRATCH_NAMES: ClassVar[tuple] = (
        'cooling:hi_coll_ion:tmp',
        'cooling:hi_coll_ion:y',
        'cooling:hi_coll_ion:kcoll',
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

        scratch = state.get_scratch('cooling:hi_coll_ion:tmp')
        y = state.get_scratch('cooling:hi_coll_ion:y')
        kcoll = state.get_scratch('cooling:hi_coll_ion:kcoll')

        # y = ln(T * kB_in_eV)
        np.multiply(T, _KB_EV, out=y)
        np.log(y, out=y)

        # Horner's evaluation: ((((((((c8 y + c7) y + c6) y + c5)
        # y + c4) y + c3) y + c2) y + c1) y + c0
        np.multiply(y, _C8, out=scratch)
        np.add(scratch, _C7, out=scratch)
        np.multiply(scratch, y, out=scratch)
        np.add(scratch, _C6, out=scratch)
        np.multiply(scratch, y, out=scratch)
        np.add(scratch, _C5, out=scratch)
        np.multiply(scratch, y, out=scratch)
        np.add(scratch, _C4, out=scratch)
        np.multiply(scratch, y, out=scratch)
        np.add(scratch, _C3, out=scratch)
        np.multiply(scratch, y, out=scratch)
        np.add(scratch, _C2, out=scratch)
        np.multiply(scratch, y, out=scratch)
        np.add(scratch, _C1, out=scratch)
        np.multiply(scratch, y, out=scratch)
        np.add(scratch, _C0, out=scratch)
        np.exp(scratch, out=kcoll)

        # Lambda = 13.6 eV * kcoll * nH * xe * xHI (without gate yet;
        # we gate Lambda and d_out together at the end so the d_out
        # path can still use `y` for the Horner-derivative polynomial)
        np.multiply(kcoll, nH, out=out)
        np.multiply(out, xe, out=out)
        np.multiply(out, xHI, out=out)
        np.multiply(out, _E_IONIZATION_HI_ERG, out=out)

        if d_out is not None:
            # k_coll(T) = exp(P(y)) with y = ln(T * kB_eV); y has
            # dy/dT = 1/T. Hence
            #   d(ln k_coll)/dT = P'(y) / T
            #   dLambda/dT = Lambda * P'(y) / T
            # P'(y) = c1 + 2 c2 y + 3 c3 y^2 + ... + 8 c8 y^7
            # Horner form:
            #   ((((((8 c8 y + 7 c7) y + 6 c6) y + 5 c5) y + 4 c4)
            #    y + 3 c3) y + 2 c2) y + c1
            np.multiply(y, 8.0 * _C8, out=scratch)
            np.add(scratch, 7.0 * _C7, out=scratch)
            np.multiply(scratch, y, out=scratch)
            np.add(scratch, 6.0 * _C6, out=scratch)
            np.multiply(scratch, y, out=scratch)
            np.add(scratch, 5.0 * _C5, out=scratch)
            np.multiply(scratch, y, out=scratch)
            np.add(scratch, 4.0 * _C4, out=scratch)
            np.multiply(scratch, y, out=scratch)
            np.add(scratch, 3.0 * _C3, out=scratch)
            np.multiply(scratch, y, out=scratch)
            np.add(scratch, 2.0 * _C2, out=scratch)
            np.multiply(scratch, y, out=scratch)
            np.add(scratch, _C1, out=scratch)
            # scratch = P'(y); d_out = Lambda * P'(y) / T
            np.divide(scratch, T, out=d_out)
            np.multiply(d_out, out, out=d_out)
            mu = state.get_scratch('solver:mu_at_entry')
            np.multiply(d_out, mu, out=d_out)

        # Gate: T > 3000 K applies to both Lambda and d_out. Reuse
        # `y` as the mask (no longer needed for the polynomial).
        np.greater(T, 3.0e3, out=y)
        np.multiply(out, y, out=out)
        if d_out is not None:
            np.multiply(d_out, y, out=d_out)
