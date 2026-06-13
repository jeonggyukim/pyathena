"""Gas-grain thermal coupling cooling.

Port of `pyathena.microphysics.cool.cooldust`. The gas couples to
dust grains through elastic collisions, transferring energy
proportional to `T - T_dust`. The DESPOTIC / Goldsmith 2001 form is

    Lambda = alpha_gd * Z_d * n_H * sqrt(T) * (T - T_dust)

with `alpha_gd = 3.2e-34`. Cooling when T > T_dust, heating when
T < T_dust (the formula is signed by convention).

This channel stays hand-coded indefinitely; gas-grain coupling is a
classical-physics rate, not an atomic-line process, and CHIANTI
carries no relevant data for it.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np

from .base import CoolingChannel

_ALPHA_GD: float = 3.2e-34


class DustGasCoupling(CoolingChannel):
    """Gas-grain thermal coupling. Signed: Lambda > 0 when gas is
    hotter than dust (cooling); Lambda < 0 when dust is hotter
    (heating).
    """

    name: ClassVar[str] = 'DustGasCoupling'
    SCRATCH_NAMES: ClassVar[tuple] = (
        'cooling:dust:tmp',
    )
    __version__: ClassVar[str] = '0.1@phase4b'

    def evaluate(
        self,
        state: Any,
        out: np.ndarray,
        d_out: Optional[np.ndarray] = None,
    ) -> None:
        T = state.T
        T_dust = state.T_dust
        nH = state.nH
        Z_d = state.Z_d

        scratch = state.get_scratch('cooling:dust:tmp')

        # Lambda = Z_d * alpha_gd * nH * sqrt(T) * (T - T_dust)
        np.subtract(T, T_dust, out=out)
        np.sqrt(T, out=scratch)
        np.multiply(out, scratch, out=out)
        np.multiply(out, nH, out=out)
        np.multiply(out, Z_d, out=out)
        np.multiply(out, _ALPHA_GD, out=out)

        if d_out is not None:
            d_out[:] = 0.0
