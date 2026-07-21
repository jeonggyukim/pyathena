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
    H I and H2 photoionisation / dissociation rates) into
    `state.chi_FUV` plus the per-species photo-rate dicts
    `state.zeta_pi` and `state.zeta_diss` that `NCRNetwork3.evaluate_CD`
    reads. Missing keys leave the existing state values alone, so
    callers that pre-populated the slots keep their values.

    Parameters
    ----------
    radiation_params : mapping, optional
        Keys consumed: `chi_FUV`, `zeta_pi_HI`, `zeta_pi_H2`,
        `zeta_diss_H2`. Anything else is ignored.
    """

    __version__: str = 'stub@phase3'

    def __init__(self,
                 radiation_params: Optional[Mapping[str, float]] = None) -> None:
        params = dict(radiation_params or {})
        self._chi_FUV: Optional[float] = params.get('chi_FUV')
        self._zeta_pi_HI: Optional[float] = params.get('zeta_pi_HI')
        self._zeta_pi_H2: Optional[float] = params.get('zeta_pi_H2')
        self._zeta_diss_H2: Optional[float] = params.get('zeta_diss_H2')

    def update(self, state: Any) -> None:
        """Broadcast the configured scalars into ncell-shaped buffers.

        Allocates on the first call (when the slot is absent) and
        writes in place on subsequent calls so the solver hot path
        sees a stable buffer reference.
        """
        ncell = state.nH.shape[0]
        # chi_FUV stays as a flat attribute on state (single FUV band).
        if self._chi_FUV is not None:
            cur = getattr(state, 'chi_FUV', None)
            if cur is None or not isinstance(cur, np.ndarray) \
                    or cur.shape != (ncell,):
                state.chi_FUV = np.full(
                    ncell, float(self._chi_FUV), dtype=np.float64)
            else:
                cur[:] = float(self._chi_FUV)
        # Per-species photo-rates go through the dict slots on state.
        zeta_pi = getattr(state, 'zeta_pi', None)
        if zeta_pi is None:
            state.zeta_pi = zeta_pi = {}
        for species, value in (
            ('HI', self._zeta_pi_HI),
            ('H2', self._zeta_pi_H2),
        ):
            if value is None:
                continue
            cur = zeta_pi.get(species)
            if cur is None or not isinstance(cur, np.ndarray) \
                    or cur.shape != (ncell,):
                zeta_pi[species] = np.full(
                    ncell, float(value), dtype=np.float64)
            else:
                cur[:] = float(value)
        zeta_diss = getattr(state, 'zeta_diss', None)
        if zeta_diss is None:
            state.zeta_diss = zeta_diss = {}
        if self._zeta_diss_H2 is not None:
            cur = zeta_diss.get('H2')
            if cur is None or not isinstance(cur, np.ndarray) \
                    or cur.shape != (ncell,):
                zeta_diss['H2'] = np.full(
                    ncell, float(self._zeta_diss_H2), dtype=np.float64)
            else:
                cur[:] = float(self._zeta_diss_H2)
