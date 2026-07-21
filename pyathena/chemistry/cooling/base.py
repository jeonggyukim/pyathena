"""CoolingChannel ABC and the CoolingChannels aggregator.

A `CoolingChannel` is one cooling mechanism (e.g. Lyman-alpha
de-excitation of H I; recombination cooling; CII fine-structure
emission). Each channel owns the data it needs at construction
(coefficient tables, atomic-data files, configuration flags) and
exposes a single `evaluate(state, out)` method that writes
`Lambda_chan` per cell into a caller-owned buffer.

`CoolingChannels` composes a tuple of channels and sums them into the
`solver:net_cool` scratch slot the explicit-subcycling solver
consumes, also writing the optional `solver:d_net_cool_d_temp_mu`
derivative when each channel reports one. The cooling policy slot on
`ChemistryDriver` is set to a `CoolingChannels` instance once Phase 4
wires it up; until then the solver runs against `CoolingStub` and
sees `net_cool == 0`.
"""
from __future__ import annotations

import abc
from typing import Any, ClassVar, Optional, Sequence, Tuple  # noqa: Tuple used in ClassVar annotations

import numpy as np


class CoolingChannel(abc.ABC):
    """One cooling mechanism.

    Subclasses set a short channel `name` (used in diagnostics, scratch
    namespacing, and debug printouts) and override
    `evaluate(state, out, d_out=None)`. The expectation is that the
    inner loop is branch-free over the strip and allocation-free given
    pre-allocated `out` / `d_out` buffers.
    """

    name: ClassVar[str] = '<unset>'

    # Internal scratch slots the channel allocates inside its
    # `evaluate` body. Concrete channels list every name they pass
    # to `state.get_scratch(...)`. Empty default for channels that
    # need no internal scratch (e.g. `H2DissociationHeating`).
    SCRATCH_NAMES: ClassVar[Tuple[str, ...]] = ()

    def allocate_scratch(self, state: Any) -> None:
        """Register channel-internal scratch buffers on `state`.

        Walks `SCRATCH_NAMES` and registers each as a `(ncell,)`
        float64 slot via `state.alloc_scratch`. Channels with
        non-trivial scratch shapes can override this method instead.
        """
        ncell = state.nH.shape[0]
        for name in self.SCRATCH_NAMES:
            state.alloc_scratch(name, (ncell,))

    @abc.abstractmethod
    def evaluate(
        self,
        state: Any,
        out: np.ndarray,
        d_out: Optional[np.ndarray] = None,
    ) -> None:
        """Write the channel's cooling rate to `out` (in place).

        Parameters
        ----------
        state : ChemState
            Current strip state. Concrete channels read whichever
            fields they need (`state.T`, `state.nH`, `state.x[...]`,
            `state.chi_for('FUV')`, etc.). The channel never mutates
            `state.x` or `state.T`.
        out : np.ndarray
            Caller-owned `(ncell,)` buffer to receive `Lambda_chan`
            in `erg / s / cm^3` units. Written in place. Existing
            contents are overwritten, not accumulated -- the
            aggregator's sum across channels is what accumulates.
        d_out : np.ndarray, optional
            Caller-owned `(ncell,)` buffer to receive the channel's
            DERIVATIVE OUTPUT (`d` is short for "derivative"):
            `d(Lambda_chan) / d(T/mu)` for cooling channels (or
            `d(Gamma_chan) / d(T/mu)` for heating channels), in
            `erg / s / cm^3 / K` units. Differentiation is with respect
            to `T/mu = temp_mu` because that is the substep-invariant
            variable the semi-implicit kernel updates; mu is held fixed
            during the cooling sub-step under standard operator
            splitting, so analytic implementations compute
            `d(Lambda) / dT` and multiply by mu (read from
            `state.get_scratch('solver:mu_at_entry')`) to convert.

            Optional because:

            - some channels lack a meaningful T derivative (e.g.
              cosmic-ray heating, where `q_HI(x_e)` and `q_H2(n_H)`
              are T-independent under operator splitting);
            - early-phase ports may not have written the analytic form
              yet (Phase 4d fills these in incrementally).

            When supplied, the channel writes the derivative; when
            None, the channel skips the derivative calculation. The
            semi-implicit kernel reads the sum across channels
            (`solver:d_net_cool_d_temp_mu`); a missing or zero entry
            just degrades the substep loop to forward Euler in that
            channel's contribution -- correct, but slower in stiff
            regimes.

            FD bootstrap convention (when the analytic chain rule is
            impractical, e.g. CIE-table-driven cooling that lacks a
            precomputed slope column): use 1-point forward FD at
            `dT_rel = 1e-3`. Pattern:

                T_orig = state.T.copy()
                state.T[:] = T_orig * (1.0 + 1.0e-3)
                channel.evaluate(state, out_Tp)
                state.T[:] = T_orig
                channel.evaluate(state, out_T)
                # d_out = mu * (out_Tp - out_T) / (T_orig * 1e-3)

            See `tests/chemistry/test_fd_calibration.py` for the
            sweep that validates this choice. Avoid `dT_rel = 2e-2`
            (the historical NCR convention): the same one-extra-eval
            cost at `1e-3` improves accuracy by an order of magnitude.
            Avoid per-channel-optimum `dT_rel` (sits on the roundoff
            cliff; platform-dependent; NaN risk).
        """


class CoolingChannels:
    """Aggregator that sums every channel's Lambda into `state.scratch`.

    The driver wires `cooling = CoolingChannels(channels=...)` into
    `ExplicitSubcyclingSolver` so that `_evaluate_cooling` finds the
    populated `solver:net_cool` / `solver:d_net_cool_d_temp_mu` slots
    on every substep.

    Heating channels can be folded in via the `heating` keyword: each
    heating channel's `Gamma` is subtracted from `Lambda_total` so the
    net result is the signed `net_cool = Lambda - Gamma` quantity the
    solver convention expects. Splitting the construction this way
    (cooling separate from heating) avoids the need for the channel
    base class to carry a "sign of contribution" flag.
    """

    __version__: ClassVar[str] = '0.1@phase4a'

    def __init__(
        self,
        channels: Sequence[CoolingChannel],
        heating: Optional[Sequence['HeatingChannel']] = None,
        *,
        scratch_prefix: str = 'cooling',
        provide_derivative: bool = True,
    ) -> None:
        if not channels and not heating:
            raise ValueError(
                'CoolingChannels: must supply at least one cooling or '
                'heating channel; the empty case is what CoolingStub '
                'is for.'
            )
        self._channels: Tuple[CoolingChannel, ...] = tuple(channels)
        self._heating: Tuple['HeatingChannel', ...] = (
            tuple(heating) if heating is not None else ()
        )
        self._prefix = scratch_prefix
        self._provide_derivative = provide_derivative

    def allocate_scratch(self, state: Any) -> None:
        """Register per-channel scratch on `state` so update() runs
        allocation-free.

        Each channel gets its own `(ncell,)` Lambda slot under the
        `<prefix>:Lambda:<name>` namespace; derivatives (when
        requested) live under `<prefix>:dLambda:<name>`. Heating
        channels mirror this with `Gamma` / `dGamma`. Each channel's
        own internal scratch (`cooling:cii:T2`,
        `heating:photoelectric:tmp` etc.) is allocated by calling
        the channel's `allocate_scratch(state)` hook.

        The aggregator does not own a `Lambda_total` buffer -- it
        writes directly into `solver:net_cool`, which the solver
        allocates.
        """
        ncell = state.nH.shape[0]
        for ch in self._channels:
            state.alloc_scratch(
                f'{self._prefix}:Lambda:{ch.name}', (ncell,))
            if self._provide_derivative:
                state.alloc_scratch(
                    f'{self._prefix}:dLambda:{ch.name}', (ncell,))
            ch.allocate_scratch(state)
        for hch in self._heating:
            state.alloc_scratch(
                f'{self._prefix}:Gamma:{hch.name}', (ncell,))
            if self._provide_derivative:
                state.alloc_scratch(
                    f'{self._prefix}:dGamma:{hch.name}', (ncell,))
            hch.allocate_scratch(state)

    def update(self, state: Any) -> None:
        """Recompute every channel and accumulate into the solver
        slots.

        The solver expects:
          - `solver:net_cool` = Lambda - Gamma (signed, cool-positive)
          - `solver:d_net_cool_d_temp_mu` = d/d(T/mu) of the above
        """
        net_cool = state.get_scratch('solver:net_cool')
        d_net = state.get_scratch('solver:d_net_cool_d_temp_mu')
        net_cool[:] = 0.0
        d_net[:] = 0.0
        for ch in self._channels:
            lam = state.get_scratch(f'{self._prefix}:Lambda:{ch.name}')
            if self._provide_derivative:
                dlam = state.get_scratch(
                    f'{self._prefix}:dLambda:{ch.name}')
                ch.evaluate(state, lam, dlam)
                np.add(d_net, dlam, out=d_net)
            else:
                ch.evaluate(state, lam, None)
            np.add(net_cool, lam, out=net_cool)
        for hch in self._heating:
            gam = state.get_scratch(f'{self._prefix}:Gamma:{hch.name}')
            if self._provide_derivative:
                dgam = state.get_scratch(
                    f'{self._prefix}:dGamma:{hch.name}')
                hch.evaluate(state, gam, dgam)
                np.subtract(d_net, dgam, out=d_net)
            else:
                hch.evaluate(state, gam, None)
            np.subtract(net_cool, gam, out=net_cool)


class HeatingChannel(abc.ABC):
    """One heating mechanism. Same shape as `CoolingChannel`; lives
    here to avoid a forward import cycle in `CoolingChannels`.
    """

    name: ClassVar[str] = '<unset>'
    SCRATCH_NAMES: ClassVar[Tuple[str, ...]] = ()

    def allocate_scratch(self, state: Any) -> None:
        """Register channel-internal scratch buffers on `state`.

        See `CoolingChannel.allocate_scratch` for the contract.
        """
        ncell = state.nH.shape[0]
        for name in self.SCRATCH_NAMES:
            state.alloc_scratch(name, (ncell,))

    @abc.abstractmethod
    def evaluate(
        self,
        state: Any,
        out: np.ndarray,
        d_out: Optional[np.ndarray] = None,
    ) -> None:
        """Write the channel's heating rate to `out` in
        `erg / s / cm^3`. See `CoolingChannel.evaluate` for the
        signature contract.
        """
