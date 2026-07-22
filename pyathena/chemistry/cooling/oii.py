"""O II fine-structure cooling: 3727 + 3729 + 7320 + 7330 + 497 um.

Port of `pyathena.microphysics.cool.coolOII`. The O II 4-S / 2-D
term system, 3-level approximation:

    Level 0 = 4S_3/2  (g = 4, ground)
    Level 1 = 2D_5/2  (g = 6, [O II] 3728.8 A)
    Level 2 = 2D_3/2  (g = 4, [O II] 3726.0 A; 2D_5/2 -> 2D_3/2 497.1 um)

Electron-only collider (no significant HI or H2 contribution at the
densities/temperatures where O II abundance is non-negligible).
Effective collision strengths from Draine 2011 power-law fits.

CHIANTI swap (Phase 7): the e collision partner switches to a
CHIANTI lookup; the channel structure is unchanged.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from .base import CoolingChannel
from ._level_helpers import cool_3level

_K_B_CGS: float = 1.380649e-16
_H_CGS: float = 6.62607015e-27
_C_CGS: float = 2.99792458e10

_G0: float = 4.0
_G1: float = 6.0
_G2: float = 4.0
_A10: float = 3.6e-5
_A20: float = 1.6e-4
_A21: float = 1.3e-7

# E_ij = h c / lambda (Draine wavelengths)
_E10: float = _H_CGS * _C_CGS / (3728.8e-8)      # 3728.8 A
_E20: float = _H_CGS * _C_CGS / (3726.0e-8)      # 3726.0 A
_E21: float = _H_CGS * _C_CGS / (497.1e-4)       # 497.1 um


class OIIFineStructureCooling(CoolingChannel):
    """O II fine-structure cooling."""

    name: ClassVar[str] = 'OIIFineStructure'
    SCRATCH_NAMES: ClassVar[tuple] = (
        'cooling:oii:T4',
        'cooling:oii:lnT4',
        'cooling:oii:prefac',
        'cooling:oii:tmp_a',
        'cooling:oii:Omega',
        'cooling:oii:q10',
        'cooling:oii:q20',
        'cooling:oii:q21',
        'cooling:oii:q01',
        'cooling:oii:q02',
        'cooling:oii:q12',
        'cooling:oii:tmp0',
        'cooling:oii:tmp1',
        'cooling:oii:tmp2',
        'cooling:oii:T_orig',
        'cooling:oii:out_tp',
    )
    __version__: ClassVar[str] = '0.1@phase4b'

    def __init__(self, *, i_OII: int, i_electron: int) -> None:
        self._i_OII = int(i_OII)
        self._i_electron = int(i_electron)

    def _compute_lambda(self, state: Any, out: np.ndarray) -> None:
        T = state.T
        nH = state.nH
        xOII = state.x[self._i_OII]
        xe = state.x[self._i_electron]

        T4 = state.get_scratch('cooling:oii:T4')
        lnT4 = state.get_scratch('cooling:oii:lnT4')
        prefac = state.get_scratch('cooling:oii:prefac')
        tmp_a = state.get_scratch('cooling:oii:tmp_a')
        Omega = state.get_scratch('cooling:oii:Omega')
        q10 = state.get_scratch('cooling:oii:q10')
        q20 = state.get_scratch('cooling:oii:q20')
        q21 = state.get_scratch('cooling:oii:q21')
        q01 = state.get_scratch('cooling:oii:q01')
        q02 = state.get_scratch('cooling:oii:q02')
        q12 = state.get_scratch('cooling:oii:q12')
        tmp0 = state.get_scratch('cooling:oii:tmp0')
        tmp1 = state.get_scratch('cooling:oii:tmp1')
        tmp2 = state.get_scratch('cooling:oii:tmp2')

        np.multiply(T, 1.0e-4, out=T4)
        np.log(T4, out=lnT4)
        # prefac = 8.629e-8 / sqrt(T4)
        np.sqrt(T4, out=prefac)
        np.divide(8.629e-8, prefac, out=prefac)

        # Omega_10 = 0.803 * T4^(0.023 - 0.008 lnT4)
        np.multiply(lnT4, -0.008, out=tmp_a)
        np.add(tmp_a, 0.023, out=tmp_a)
        np.power(T4, tmp_a, out=Omega)
        np.multiply(Omega, 0.803, out=Omega)
        np.multiply(prefac, Omega, out=q10)
        np.multiply(q10, 1.0 / _G1, out=q10)

        # Omega_20 = 0.550 * T4^(0.054 - 0.004 lnT4)
        np.multiply(lnT4, -0.004, out=tmp_a)
        np.add(tmp_a, 0.054, out=tmp_a)
        np.power(T4, tmp_a, out=Omega)
        np.multiply(Omega, 0.550, out=Omega)
        np.multiply(prefac, Omega, out=q20)
        np.multiply(q20, 1.0 / _G2, out=q20)

        # Omega_21 = 1.434 * T4^(-0.176 + 0.004 lnT4)
        np.multiply(lnT4, 0.004, out=tmp_a)
        np.add(tmp_a, -0.176, out=tmp_a)
        np.power(T4, tmp_a, out=Omega)
        np.multiply(Omega, 1.434, out=Omega)
        np.multiply(prefac, Omega, out=q21)
        np.multiply(q21, 1.0 / _G2, out=q21)

        # q_ij = nH * k_ij_e * xe
        np.multiply(q10, nH, out=q10)
        np.multiply(q10, xe, out=q10)
        np.multiply(q20, nH, out=q20)
        np.multiply(q20, xe, out=q20)
        np.multiply(q21, nH, out=q21)
        np.multiply(q21, xe, out=q21)

        # Boltzmann reverses
        np.multiply(T, _K_B_CGS, out=tmp_a)
        np.divide(_E10, tmp_a, out=tmp0)
        np.negative(tmp0, out=tmp0)
        np.exp(tmp0, out=tmp0)
        np.multiply(q10, tmp0, out=q01)
        np.multiply(q01, _G1 / _G0, out=q01)
        np.divide(_E20, tmp_a, out=tmp0)
        np.negative(tmp0, out=tmp0)
        np.exp(tmp0, out=tmp0)
        np.multiply(q20, tmp0, out=q02)
        np.multiply(q02, _G2 / _G0, out=q02)
        np.divide(_E21, tmp_a, out=tmp0)
        np.negative(tmp0, out=tmp0)
        np.exp(tmp0, out=tmp0)
        np.multiply(q21, tmp0, out=q12)
        np.multiply(q12, _G2 / _G1, out=q12)

        cool_3level(
            q01, q10, q02, q20, q12, q21,
            _A10, _A20, _A21, _E10, _E20, _E21,
            xOII, out, tmp0, tmp1, tmp2,
        )

    def evaluate(
        self,
        state: Any,
        out: np.ndarray,
        d_out: Optional[np.ndarray] = None,
    ) -> None:
        self._compute_lambda(state, out)
        if d_out is not None:
            # FD bootstrap at dT_rel = 1e-3 forward (project
            # convention; see CoolingChannel.evaluate docstring and
            # tests/chemistry/test_fd_calibration.py). The analytic
            # chain rule runs through the Omega power-law fits and
            # the steady-state 3-level solve; FD bootstrap matches
            # the substep-damping accuracy in one extra
            # _compute_lambda call.
            _DT_REL = 1.0e-3
            T_orig = state.get_scratch('cooling:oii:T_orig')
            out_tp = state.get_scratch('cooling:oii:out_tp')
            np.copyto(T_orig, state.T)
            np.multiply(state.T, 1.0 + _DT_REL, out=state.T)
            self._compute_lambda(state, out_tp)
            np.copyto(state.T, T_orig)
            np.subtract(out_tp, out, out=d_out)
            np.divide(d_out, T_orig, out=d_out)
            np.multiply(d_out, 1.0 / _DT_REL, out=d_out)
            mu = state.get_scratch('solver:mu_at_entry')
            np.multiply(d_out, mu, out=d_out)
