"""H2 rotation-vibration cooling, Moseley et al. (2021).

Port of `pyathena.microphysics.cool.coolH2rovib`. This is the
NCR production default H2 cooling form (see
`pyathena.microphysics.get_cooling.py:127`). It evaluates four
rovibrational line series f1..f4 each excited by an effective
collider density x_i = n_HI + alpha_i * n_H2:

    f_k = A_k * sqrt(T3)**p_k * T3**q_k * exp(-b_k / T3) *
          (w1 * x_k / (1 + x_k / n_k) + w2 * x_k / (1 + x_k / (10 n_k)))

with the four (A, p, q, b, n, w1, w2, alpha) constants taken from
Moseley+21 fits as wired in the legacy helper. Lambda = xH2 * sum f_k.

Coolant family: gas-phase H2 line cooling. CHIANTI carries no H2
molecular data; Cloudy carries some. Stays hand-coded for now.

The H2Gong17Cooling channel implements the alternative Gong + Ostriker
+ Wolfire 2017 form; this Moseley+21 channel is the NCR default.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from .base import CoolingChannel


class H2Moseley21Cooling(CoolingChannel):
    """H2 rovibrational cooling (NCR default; Moseley+2021)."""

    name: ClassVar[str] = 'H2Moseley21'
    SCRATCH_NAMES: ClassVar[tuple] = (
        'cooling:h2_moseley:T3',
        'cooling:h2_moseley:T3inv',
        'cooling:h2_moseley:sqrtT3',
        'cooling:h2_moseley:nHI',
        'cooling:h2_moseley:nH2',
        'cooling:h2_moseley:tmp',
        'cooling:h2_moseley:x_eff',
        'cooling:h2_moseley:accum',
        'cooling:h2_moseley:term',
        'cooling:h2_moseley:ratio_lo',
        'cooling:h2_moseley:ratio_hi',
        # Phase 4d FD-bootstrap derivative slots. The analytic chain
        # rule on a sum of four sqrt(T3)/T3 * exp(b/T3) * saturation-
        # density terms is mechanically tractable but tedious; FD
        # bootstrap at dT_rel = 1e-3 forward (project convention; see
        # feedback_fd_bootstrap_convention) matches the substep-damping
        # accuracy in one extra _compute_lambda call.
        'cooling:h2_moseley:T_orig',
        'cooling:h2_moseley:out_tp',
    )
    __version__: ClassVar[str] = '0.1@phase4b'

    def __init__(
        self,
        *,
        i_HI: int,
        i_H2: int,
    ) -> None:
        self._i_HI = int(i_HI)
        self._i_H2 = int(i_H2)

    def _compute_lambda(self, state: Any, out: np.ndarray) -> None:
        """Write Lambda into `out`. Reads `state.T` directly so the FD
        bootstrap can perturb T transiently."""
        T = state.T
        nH = state.nH
        xHI = state.x[self._i_HI]
        xH2 = state.x[self._i_H2]

        T3 = state.get_scratch('cooling:h2_moseley:T3')
        T3inv = state.get_scratch('cooling:h2_moseley:T3inv')
        sqrtT3 = state.get_scratch('cooling:h2_moseley:sqrtT3')
        nHI = state.get_scratch('cooling:h2_moseley:nHI')
        nH2 = state.get_scratch('cooling:h2_moseley:nH2')
        tmp = state.get_scratch('cooling:h2_moseley:tmp')
        x_eff = state.get_scratch('cooling:h2_moseley:x_eff')
        accum = state.get_scratch('cooling:h2_moseley:accum')
        term = state.get_scratch('cooling:h2_moseley:term')
        ratio_lo = state.get_scratch('cooling:h2_moseley:ratio_lo')
        ratio_hi = state.get_scratch('cooling:h2_moseley:ratio_hi')

        np.multiply(T, 1.0e-3, out=T3)
        np.divide(1.0, T3, out=T3inv)
        np.sqrt(T3, out=sqrtT3)
        np.multiply(nH, xHI, out=nHI)
        np.multiply(nH, xH2, out=nH2)

        accum[:] = 0.0

        # f1: 1.1e-25 * sqrt(T3) * exp(-0.51/T3) * (0.7 * x1/(1 + x1/50)
        #     + 0.3 * x1/(1 + x1/500)); x1 = nHI + 5 * nH2
        # f2: 2.0e-25 * T3 * exp(-1.0/T3) * (0.35 * x2/(1 + x2/450)
        #     + 0.65 * x2/(1 + x2/4500)); x2 = nHI + 4.5 * nH2
        # f3: 2.4e-24 * sqrt(T3) * T3 * exp(-2.0/T3) * (x3/(1 + x3/25));
        #     x3 = nHI + 0.75 * nH2
        # f4: 1.7e-23 * sqrt(T3) * T3 * exp(-4.0/T3) * (0.45 * x4/(1 + x4/900)
        #     + 0.55 * x4/(1 + x4/9000)); x4 = nHI + 0.05 * nH2

        def _excited_density(alpha, out_arr):
            np.multiply(nH2, alpha, out=out_arr)
            np.add(out_arr, nHI, out=out_arr)

        def _saturated_pair(x, n_lo, w_lo, w_hi, n_hi, out_arr):
            """Compute w_lo * x / (1 + x / n_lo) + w_hi * x / (1 + x / n_hi)
            into `out_arr`. Two scratches: ratio_lo, ratio_hi.
            """
            np.divide(x, n_lo, out=ratio_lo)
            np.add(ratio_lo, 1.0, out=ratio_lo)
            np.divide(x, ratio_lo, out=ratio_lo)
            np.multiply(ratio_lo, w_lo, out=ratio_lo)
            np.divide(x, n_hi, out=ratio_hi)
            np.add(ratio_hi, 1.0, out=ratio_hi)
            np.divide(x, ratio_hi, out=ratio_hi)
            np.multiply(ratio_hi, w_hi, out=ratio_hi)
            np.add(ratio_lo, ratio_hi, out=out_arr)

        # f1
        _excited_density(5.0, x_eff)
        _saturated_pair(x_eff, 50.0, 0.7, 0.3, 500.0, term)
        np.multiply(sqrtT3, 1.1e-25, out=tmp)
        np.multiply(T3inv, -0.51, out=ratio_lo)
        np.exp(ratio_lo, out=ratio_lo)
        np.multiply(tmp, ratio_lo, out=tmp)
        np.multiply(term, tmp, out=term)
        np.add(accum, term, out=accum)

        # f2
        _excited_density(4.5, x_eff)
        _saturated_pair(x_eff, 450.0, 0.35, 0.65, 4500.0, term)
        np.multiply(T3, 2.0e-25, out=tmp)
        np.multiply(T3inv, -1.0, out=ratio_lo)
        np.exp(ratio_lo, out=ratio_lo)
        np.multiply(tmp, ratio_lo, out=tmp)
        np.multiply(term, tmp, out=term)
        np.add(accum, term, out=accum)

        # f3: single saturation only -> (x3 / (1 + x3/25))
        _excited_density(0.75, x_eff)
        np.divide(x_eff, 25.0, out=ratio_lo)
        np.add(ratio_lo, 1.0, out=ratio_lo)
        np.divide(x_eff, ratio_lo, out=term)
        np.multiply(sqrtT3, T3, out=tmp)
        np.multiply(tmp, 2.4e-24, out=tmp)
        np.multiply(T3inv, -2.0, out=ratio_lo)
        np.exp(ratio_lo, out=ratio_lo)
        np.multiply(tmp, ratio_lo, out=tmp)
        np.multiply(term, tmp, out=term)
        np.add(accum, term, out=accum)

        # f4
        _excited_density(0.05, x_eff)
        _saturated_pair(x_eff, 900.0, 0.45, 0.55, 9000.0, term)
        np.multiply(sqrtT3, T3, out=tmp)
        np.multiply(tmp, 1.7e-23, out=tmp)
        np.multiply(T3inv, -4.0, out=ratio_lo)
        np.exp(ratio_lo, out=ratio_lo)
        np.multiply(tmp, ratio_lo, out=tmp)
        np.multiply(term, tmp, out=term)
        np.add(accum, term, out=accum)

        # Lambda = xH2 * accum
        np.multiply(accum, xH2, out=out)

    def evaluate(
        self,
        state: Any,
        out: np.ndarray,
        d_out: Optional[np.ndarray] = None,
    ) -> None:
        # Lambda at current state.T.
        self._compute_lambda(state, out)
        if d_out is not None:
            # FD bootstrap at dT_rel = 1e-3 forward (project
            # convention; see CoolingChannel.evaluate docstring and
            # tests/chemistry/test_fd_calibration.py).
            _DT_REL = 1.0e-3
            T_orig = state.get_scratch('cooling:h2_moseley:T_orig')
            out_tp = state.get_scratch('cooling:h2_moseley:out_tp')
            np.copyto(T_orig, state.T)
            np.multiply(state.T, 1.0 + _DT_REL, out=state.T)
            self._compute_lambda(state, out_tp)
            np.copyto(state.T, T_orig)
            np.subtract(out_tp, out, out=d_out)
            np.divide(d_out, T_orig, out=d_out)
            np.multiply(d_out, 1.0 / _DT_REL, out=d_out)
            mu = state.get_scratch('solver:mu_at_entry')
            np.multiply(d_out, mu, out=d_out)
