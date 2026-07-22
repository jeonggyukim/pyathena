"""C II fine-structure cooling: the [C II] 158 um line.

Port of `pyathena.microphysics.cool.coolCII`. The 2-level system

    g0 = 2 (2P_{1/2} ground), g1 = 4 (2P_{3/2})
    A10 = 2.3e-6 s^-1
    E10 = 1.26e-14 erg  (158 um)

Collisional de-excitation rates assembled from electron, H I, and H2
partners (the ortho/para H2 mix is 0.75 / 0.25). The HI / e fits will
be CHIANTI-table swaps in Phase 7; the H2 fits stay hand-coded
because CHIANTI has no H2 collisional data. See
`pyathena.chemistry.cooling._level_helpers` for the steady-state
2-level solver.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from .base import CoolingChannel
from ._level_helpers import cool_2level

_K_B_CGS: float = 1.380649e-16

# Statistical weights and atomic constants.
_G0: float = 2.0
_G1: float = 4.0
_A10: float = 2.3e-6        # s^-1
_E10: float = 1.26e-14      # erg (158 um line)

# Ortho/para H2 mix used throughout pyathena.microphysics.cool.
_F_PARA: float = 0.25
_F_ORTHO: float = 0.75


class CIIFineStructureCooling(CoolingChannel):
    """C II 158 um fine-structure cooling. Phase 4b literal port."""

    name: ClassVar[str] = 'CIIFineStructure'
    SCRATCH_NAMES: ClassVar[tuple] = (
        'cooling:cii:T2',
        'cooling:cii:tmp',
        'cooling:cii:tmp_b',
        'cooling:cii:k10e',
        'cooling:cii:k10HI',
        'cooling:cii:k10H2',
        'cooling:cii:q10',
        'cooling:cii:q01',
        'cooling:cii:warm_mask',
        'cooling:cii:cold_mask',
        'cooling:cii:T_orig',
        'cooling:cii:out_tp',
    )
    __version__: ClassVar[str] = '0.1@phase4b'

    def __init__(
        self,
        *,
        i_HI: int,
        i_H2: int,
        i_CII: int,
        i_electron: int,
    ) -> None:
        self._i_HI = int(i_HI)
        self._i_H2 = int(i_H2)
        self._i_CII = int(i_CII)
        self._i_electron = int(i_electron)

    def _compute_lambda(self, state: Any, out: np.ndarray) -> None:
        T = state.T
        nH = state.nH
        xHI = state.x[self._i_HI]
        xH2 = state.x[self._i_H2]
        xCII = state.x[self._i_CII]
        xe = state.x[self._i_electron]

        T2 = state.get_scratch('cooling:cii:T2')
        tmp = state.get_scratch('cooling:cii:tmp')
        tmp_b = state.get_scratch('cooling:cii:tmp_b')
        k10e = state.get_scratch('cooling:cii:k10e')
        k10HI = state.get_scratch('cooling:cii:k10HI')
        k10H2 = state.get_scratch('cooling:cii:k10H2')
        q10 = state.get_scratch('cooling:cii:q10')
        q01 = state.get_scratch('cooling:cii:q01')

        np.multiply(T, 1.0e-2, out=T2)            # T2 = T / 100 K

        # k10e = 4.53e-8 * sqrt(1e4 / T)
        np.divide(1.0e4, T, out=k10e)
        np.sqrt(k10e, out=k10e)
        np.multiply(k10e, 4.53e-8, out=k10e)

        # k10HI = 7.58e-10 * T2 ** (0.1281 + 0.0087 * ln(T2))
        np.log(T2, out=tmp)
        np.multiply(tmp, 0.0087, out=tmp)
        np.add(tmp, 0.1281, out=tmp)              # tmp = exponent
        np.power(T2, tmp, out=k10HI)
        np.multiply(k10HI, 7.58e-10, out=k10HI)

        # H2 collision: piecewise in T.
        # cold (T < 500): k10oH2 = (5.33 + 0.11 * T2) * 1e-10
        #                 k10pH2 = (4.43 + 0.33 * T2) * 1e-10
        # warm (T >= 500): k10oH2 = 3.74757785025e-10 * T**0.07
        #                  k10pH2 = 3.88997286356e-10 * T**0.07
        # Branch-free over the strip: piecewise blend via
        # `result = warm_mask * warm + (1 - warm_mask) * cold`.
        warm_mask = state.get_scratch('cooling:cii:warm_mask')
        cold_mask = state.get_scratch('cooling:cii:cold_mask')
        np.greater_equal(T, 500.0, out=warm_mask)
        np.less(T, 500.0, out=cold_mask)

        np.power(T, 0.07, out=tmp)   # T**0.07 shared by ortho and para

        # k10oH2
        # cold piece into tmp_b
        np.multiply(T2, 0.11, out=tmp_b)
        np.add(tmp_b, 5.33, out=tmp_b)
        np.multiply(tmp_b, 1.0e-10, out=tmp_b)
        np.multiply(tmp_b, cold_mask, out=tmp_b)
        # warm piece into k10H2 (reused), then accumulate into tmp_b
        np.multiply(tmp, 3.74757785025e-10, out=k10H2)
        np.multiply(k10H2, warm_mask, out=k10H2)
        np.add(tmp_b, k10H2, out=tmp_b)
        # tmp_b now holds k10oH2

        # k10pH2
        np.multiply(T2, 0.33, out=k10H2)
        np.add(k10H2, 4.43, out=k10H2)
        np.multiply(k10H2, 1.0e-10, out=k10H2)
        np.multiply(k10H2, cold_mask, out=k10H2)
        # warm piece into tmp (reused beyond its T**0.07 lifetime)
        np.multiply(tmp, 3.88997286356e-10, out=tmp)
        np.multiply(tmp, warm_mask, out=tmp)
        np.add(k10H2, tmp, out=k10H2)
        # k10H2 now holds k10pH2

        # mix: k10H2_final = k10oH2 * f_ortho + k10pH2 * f_para
        np.multiply(tmp_b, _F_ORTHO, out=tmp_b)
        np.multiply(k10H2, _F_PARA, out=k10H2)
        np.add(k10H2, tmp_b, out=k10H2)

        # q10 = nH * (k10e * xe + k10HI * xHI + k10H2 * xH2)
        np.multiply(k10e, xe, out=q10)
        np.multiply(k10HI, xHI, out=tmp)
        np.add(q10, tmp, out=q10)
        np.multiply(k10H2, xH2, out=tmp)
        np.add(q10, tmp, out=q10)
        np.multiply(q10, nH, out=q10)

        # q01 = (g1 / g0) * q10 * exp(-E10 / (kB * T))
        np.multiply(T, _K_B_CGS, out=tmp)
        np.divide(_E10, tmp, out=tmp)
        np.negative(tmp, out=tmp)
        np.exp(tmp, out=tmp)
        np.multiply(q10, tmp, out=q01)
        np.multiply(q01, _G1 / _G0, out=q01)

        # Steady-state 2-level cooling
        cool_2level(q01, q10, _A10, _E10, xCII, out, tmp)

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
            # chain rule runs through the piecewise collision-partner
            # fits and the steady-state 2-level solve; FD bootstrap
            # matches the substep-damping accuracy in one extra
            # _compute_lambda call.
            _DT_REL = 1.0e-3
            T_orig = state.get_scratch('cooling:cii:T_orig')
            out_tp = state.get_scratch('cooling:cii:out_tp')
            np.copyto(T_orig, state.T)
            np.multiply(state.T, 1.0 + _DT_REL, out=state.T)
            self._compute_lambda(state, out_tp)
            np.copyto(state.T, T_orig)
            np.subtract(out_tp, out, out=d_out)
            np.divide(d_out, T_orig, out=d_out)
            np.multiply(d_out, 1.0 / _DT_REL, out=d_out)
            mu = state.get_scratch('solver:mu_at_entry')
            np.multiply(d_out, mu, out=d_out)
