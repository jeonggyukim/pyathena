"""Photoelectric heating from grains (Weingartner & Draine 2001).

Compatibility port of `pyathena.microphysics.cool.heatPE` (lines
78-86) and its `get_charge_param` helper (lines 73-77). The
WD01 fit (Table 2; Rv = 3.1, bC = 4.0, distribution A, ISRF):

    eps = (C_PE[0] + C_PE[1] * T**C_PE[4]) /
          (1 + C_PE[2] * x**C_PE[5] *
               (1 + C_PE[3] * x**C_PE[6]))
    Gamma_PE = 1.7e-26 * chi_PE * Z_d * eps

with the charge parameter

    x = 1.7 * chi_PE * sqrt(T) / (xe * nH * phi) + 50.0

(`phi = 1.0` for the WD01 fit; the BT94 variant uses `phi = 0.5`).
The `+ 50.0` offset clamps the WD01 fit away from the small-x regime
the fit is not valid in (WD01 note: do not use the formula for
x < 100). The legacy `pyathena.microphysics.cool.get_charge_param`
applies the same offset; the channel matches it byte-for-byte.

Phase 4a: this channel ports only the WD01 form (`heatPE`). The
BT94 and W03 variants land in Phase 4b alongside the config-flag
dispatch logic.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from ..cooling.base import HeatingChannel


# Weingartner & Draine 2001 Table 2 fit coefficients (Rv = 3.1,
# bC = 4.0, distribution A, ISRF).
_C_PE_0: float = 5.22
_C_PE_1: float = 2.25
_C_PE_2: float = 0.04996
_C_PE_3: float = 0.00430
_C_PE_4: float = 0.147
_C_PE_5: float = 0.431
_C_PE_6: float = 0.692

# Overall PE heating amplitude per unit chi_PE per unit Z_d
# (erg / s / cm^3). The WD01 rate is `1.7e-26 * chi_PE * Z_d * eps`.
# `chi_PE` is the local Habing-band intensity in Habing units; for
# the unattenuated ISRF chi_PE = 1.0.
_GAMMA0: float = 1.7e-26

# Numerical floor on the electron density to avoid division-by-zero
# in fully neutral cells.
_NE_FLOOR: float = 1.0e-10


class PhotoelectricHeating(HeatingChannel):
    """Weingartner & Draine 2001 photoelectric heating on dust."""

    name: ClassVar[str] = 'PhotoelectricWD01'
    SCRATCH_NAMES: ClassVar[tuple] = (
        'heating:photoelectric:tmp',
        'heating:photoelectric:ne_floor',
        'heating:photoelectric:eps_num',
        # Phase 4d FD-bootstrap derivative slots: T_orig snapshots
        # state.T for restoration, out_tp holds Gamma at the perturbed
        # T = T_orig * (1 + 1e-3). The full WD01 chain rule is too
        # tedious to write analytically (~40 ops through a quotient
        # rule on a 4-term denominator); FD bootstrap at the project
        # convention dT_rel = 1e-3 (see feedback_fd_bootstrap_convention)
        # matches the accuracy needed for stiffness damping with one
        # extra _compute_gamma call.
        'heating:photoelectric:T_orig',
        'heating:photoelectric:out_tp',
    )
    __version__: ClassVar[str] = '0.1@phase4a'

    def __init__(
        self,
        *,
        i_electron: int,
        chi_band: str = 'FUV',
        phi: float = 1.0,
    ) -> None:
        self._i_electron = int(i_electron)
        self._chi_band = chi_band
        self._phi = float(phi)

    def _compute_gamma(self, state: Any, out: np.ndarray) -> None:
        """Write Gamma into `out`. Reads state.T directly so the FD
        bootstrap can perturb T transiently."""
        T = state.T
        nH = state.nH
        xe = state.x[self._i_electron]
        Z_d = state.Z_d
        chi_PE = state.chi_for(self._chi_band)

        scratch = state.get_scratch('heating:photoelectric:tmp')
        ne_floor = state.get_scratch('heating:photoelectric:ne_floor')

        # charge param
        #     x = 1.7 * chi_PE * sqrt(T) / (xe * nH * phi) + 50.0
        # The denominator carries `phi`, NOT the numerator. The +50.0
        # offset clamps the fit away from its small-x invalid regime.
        np.multiply(xe, nH, out=ne_floor)
        np.multiply(ne_floor, self._phi, out=ne_floor)
        np.maximum(ne_floor, _NE_FLOOR, out=ne_floor)
        np.sqrt(T, out=scratch)
        np.multiply(scratch, chi_PE, out=scratch)
        np.multiply(scratch, 1.7, out=scratch)
        np.divide(scratch, ne_floor, out=scratch)
        np.add(scratch, 50.0, out=scratch)
        # scratch now holds the charge parameter x; reuse for the
        # numerator / denominator assembly below.

        # eps_num = C0 + C1 * T**C4
        # eps_den = 1 + C2 * x**C5 * (1 + C3 * x**C6)
        # Allocations would happen here if I assembled both pieces in
        # tmps. Reuse the strategy from the C++ port: compute the
        # denominator first (overwrites x), then the numerator into
        # `out`, then divide.
        eps_num = state.get_scratch('heating:photoelectric:eps_num')
        np.power(scratch, _C_PE_5, out=ne_floor)            # x**C5
        np.multiply(ne_floor, _C_PE_2, out=ne_floor)         # C2 * x**C5
        # second factor in denominator: 1 + C3 * x**C6
        np.power(scratch, _C_PE_6, out=eps_num)              # x**C6 (in eps_num scratch)
        np.multiply(eps_num, _C_PE_3, out=eps_num)           # C3 * x**C6
        np.add(eps_num, 1.0, out=eps_num)                    # 1 + C3 * x**C6
        np.multiply(ne_floor, eps_num, out=ne_floor)         # C2 * x**C5 * (1 + C3 * x**C6)
        np.add(ne_floor, 1.0, out=ne_floor)                  # denominator

        # numerator: C0 + C1 * T**C4
        np.power(T, _C_PE_4, out=eps_num)
        np.multiply(eps_num, _C_PE_1, out=eps_num)
        np.add(eps_num, _C_PE_0, out=eps_num)

        # eps = num / den; Gamma = 1.7e-26 * chi_PE * Z_d * eps
        np.divide(eps_num, ne_floor, out=out)
        np.multiply(out, chi_PE, out=out)
        np.multiply(out, Z_d, out=out)
        np.multiply(out, _GAMMA0, out=out)

    def evaluate(
        self,
        state: Any,
        out: np.ndarray,
        d_out: Optional[np.ndarray] = None,
    ) -> None:
        # Lambda at the current state.T.
        self._compute_gamma(state, out)

        if d_out is not None:
            # FD bootstrap at dT_rel = 1e-3 forward (the project
            # convention; see CoolingChannel.evaluate docstring and
            # the calibration sweep at
            # tests/chemistry/test_fd_calibration.py). One extra
            # _compute_gamma call at the perturbed temperature; the
            # WD01 analytic chain rule is mechanically tractable but
            # ~40 ops through a quotient rule on a 4-term denominator
            # and gives no meaningful accuracy gain for the substep
            # damping role.
            _DT_REL = 1.0e-3
            T_orig = state.get_scratch('heating:photoelectric:T_orig')
            out_tp = state.get_scratch('heating:photoelectric:out_tp')
            np.copyto(T_orig, state.T)
            np.multiply(state.T, 1.0 + _DT_REL, out=state.T)
            self._compute_gamma(state, out_tp)
            np.copyto(state.T, T_orig)
            # d_out = mu * (Gamma(T*(1+e)) - Gamma(T)) / (T * e)
            np.subtract(out_tp, out, out=d_out)
            np.divide(d_out, T_orig, out=d_out)
            np.multiply(d_out, 1.0 / _DT_REL, out=d_out)
            mu = state.get_scratch('solver:mu_at_entry')
            np.multiply(d_out, mu, out=d_out)
