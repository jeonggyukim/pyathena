"""Placeholder policies for the radiation, opacity, and cooling roles.

The Phase 3 driver expects callable `update(state)` hooks on its
radiation / opacity / cooling collaborators so the substep loop can
refresh radiation fields and rebuild absorption coefficients before
each accepted substep. Phase 3 lands the explicit subcycling solver
without those policies being implemented yet, so the driver accepts
no-op stand-ins here and the real policies replace them in Phase 4
(cooling/heating ABCs) and Phase 5+ (radiation transport).

`RadiationStub` is mildly more than a no-op: it copies fixed FUV /
H I / H2 photo-rate scalars out of a configuration mapping into the
corresponding optional attributes on `state` so `NCRNetwork3` reads
non-zero radiation when the caller supplies a non-dark configuration
through `ChemistryConfig.radiation_params`. The fixed-rate fallback is
how the explicit-subcycling unit tests exercise a non-trivial H II
chemistry without standing up the radiation policy yet.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np


class CoolingStub:
    """No-op cooling policy. Replaced in Phase 4 by the cooling ABCs."""

    __version__: str = 'stub@phase3'

    def update(self, state: Any) -> None:
        """Refresh cached cooling lookup tables. No-op in the stub."""
        return None


class OpacityStub:
    """No-op opacity policy. Replaced in Phase 5 by the absorption-
    coefficient table layer.
    """

    __version__: str = 'stub@phase3'

    def update(self, state: Any) -> None:
        """Refresh per-cell absorption coefficients. No-op in the stub."""
        return None


class RadiationStub:
    """Constant-radiation policy.

    Copies the supplied scalar values (FUV intensity in Draine units,
    H I and H2 photoionisation / dissociation rates) into the optional
    state attributes that `NCRNetwork3.evaluate_CD` reads. When the
    config does not specify a key, the corresponding state attribute is
    left untouched, so callers that already set `state.chi_FUV` etc.
    keep their values.

    Parameters
    ----------
    radiation_params : mapping, optional
        Keys consumed: `chi_FUV`, `xi_ph_HI`, `xi_ph_H2`, `xi_diss_H2`.
        Anything else is ignored.
    """

    __version__: str = 'stub@phase3'

    def __init__(self,
                 radiation_params: Optional[Mapping[str, float]] = None) -> None:
        params = dict(radiation_params or {})
        self._chi_FUV: Optional[float] = params.get('chi_FUV')
        self._xi_ph_HI: Optional[float] = params.get('xi_ph_HI')
        self._xi_ph_H2: Optional[float] = params.get('xi_ph_H2')
        self._xi_diss_H2: Optional[float] = params.get('xi_diss_H2')

    def update(self, state: Any) -> None:
        """Broadcast the configured scalars into ncell-shaped buffers.

        Allocates on the first call (when the attribute is absent) and
        writes in place on subsequent calls so the solver hot path
        sees a stable buffer reference.
        """
        ncell = state.nH.shape[0]
        for attr_name, value in (
            ('chi_FUV', self._chi_FUV),
            ('xi_ph_HI', self._xi_ph_HI),
            ('xi_ph_H2', self._xi_ph_H2),
            ('xi_diss_H2', self._xi_diss_H2),
        ):
            if value is None:
                continue
            cur = getattr(state, attr_name, None)
            if cur is None or not isinstance(cur, np.ndarray) \
                    or cur.shape != (ncell,):
                setattr(state, attr_name,
                        np.full(ncell, float(value), dtype=np.float64))
            else:
                cur[:] = float(value)
