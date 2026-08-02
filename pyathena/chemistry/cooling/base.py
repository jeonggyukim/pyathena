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

    # Temperature fade, as `(T_full, T_off)`. The channel contributes
    # at full strength at `T_full`, ramps to zero at `T_off`, and is
    # exactly zero beyond. `T_full < T_off` fades the channel out with
    # rising temperature; `T_full > T_off` fades it out with falling
    # temperature. `None` means the channel contributes everywhere.
    #
    # Two uses. Fits calibrated for the cold and warm ISM must not be
    # evaluated in hot gas -- the WD01 grain-charge parameter carries
    # an additive floor of 50, so `GrainRecombinationCooling` keeps
    # scaling as T^1.02 with no radiation field and supplies about
    # seven times the entire metal-free CIE cooling at 1e6 K. And a
    # tracked ion whose abundance has gone gets its cooling switched
    # off rather than evaluated (the per-ion caps in
    # `ai-notes/docs-claude/pyathena/cie-pool-cooling-design.md`).
    #
    # The aggregator ramps rather than switching, because the solver
    # differentiates the cooling function and a step in Lambda would
    # put a delta in `d_net_cool_d_temp_mu`.
    T_FADE: ClassVar[Optional[Tuple[float, float]]] = None

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
        # Working space for the T_FADE ramp, shared across channels
        # because it is consumed within one channel's accumulation.
        if any(ch.T_FADE is not None
               for ch in tuple(self._channels) + tuple(self._heating)):
            state.alloc_scratch(f'{self._prefix}:fade_w', (ncell,))
            state.alloc_scratch(f'{self._prefix}:fade_dw', (ncell,))
            state.alloc_scratch(f'{self._prefix}:fade_tmp', (ncell,))

    @staticmethod
    def _fade_weight(T: np.ndarray, fade, w: np.ndarray,
                     dw: np.ndarray, tmp: np.ndarray) -> None:
        """Write the C2 smootherstep fade weight into `w` and `dw/dT`
        into `dw`.

        `fade` is `(T_full, T_off)`. The weight is 1 at `T_full`, 0 at
        `T_off`, and monotone between, with continuous first and second
        derivatives. C2 matters because the solver differentiates the
        cooling function: with the cheaper C1 smoothstep `3s^2 - 2s^3`
        the resulting `dLambda/dT` would be continuous but kinked.

            s     = clamp((T - T_off) / (T_full - T_off), 0, 1)
            w     = s^3 (s (6 s - 15) + 10)
            dw/dT = 30 s^2 (s - 1)^2 / (T_full - T_off)

        Written with `T_off` as the origin so one expression serves
        either direction: `T_full > T_off` fades out downwards,
        `T_full < T_off` fades out upwards. Allocation-free; `tmp` is
        working space and holds `s` on return.
        """
        t_full, t_off = fade
        inv_span = 1.0 / (t_full - t_off)
        np.subtract(T, t_off, out=tmp)
        np.multiply(tmp, inv_span, out=tmp)
        np.clip(tmp, 0.0, 1.0, out=tmp)      # tmp = s

        np.multiply(tmp, 6.0, out=dw)
        np.subtract(dw, 15.0, out=dw)
        np.multiply(dw, tmp, out=dw)
        np.add(dw, 10.0, out=dw)             # dw = s (6 s - 15) + 10
        np.multiply(tmp, tmp, out=w)
        np.multiply(w, tmp, out=w)           # w = s^3
        np.multiply(w, dw, out=w)            # w = s^3 (s (6 s - 15) + 10)

        np.subtract(tmp, 1.0, out=dw)
        np.multiply(dw, dw, out=dw)          # (s - 1)^2
        np.multiply(dw, tmp, out=dw)
        np.multiply(dw, tmp, out=dw)         # s^2 (s - 1)^2
        np.multiply(dw, 30.0 * inv_span, out=dw)

    def _fade_is_off_everywhere(self, state: Any, channel: Any) -> bool:
        """True when no cell in the strip is inside `channel`'s window.

        Checked before `evaluate` so a channel outside its range is not
        computed at all. That is the point of the window: evaluating a
        three-level solve for C I at 1e7 K is not merely wasted work,
        it overflows and returns NaN.

        A strip spanning the window still evaluates everywhere, and the
        out-of-range cells are zeroed afterwards in `_apply_fade`.
        Removing that case as well would need the channel to clamp its
        own temperature internally.
        """
        fade = channel.T_FADE
        if fade is None:
            return False
        t_full, t_off = fade
        if t_full < t_off:                 # fades out with rising T
            return bool(np.all(state.T >= t_off))
        return bool(np.all(state.T <= t_off))

    def _apply_fade(self, state: Any, channel: Any, val: np.ndarray,
                    dval: Optional[np.ndarray]) -> None:
        """Scale one channel's contribution by its `T_FADE` weight.

        With `L_eff = w(T) L(T)` the derivative needs the product rule,

            dL_eff/dT = w dL/dT + L dw/dT

        so scaling `dval` alone would be wrong inside the ramp -- it
        would miss the term that carries the switching itself. Channels
        report the derivative in `temp_mu` space (already multiplied by
        `mu`), so the new term picks up the same factor.

        The `L dw/dT` term is computed before `val` is scaled, since it
        needs the unfaded value.
        """
        fade = channel.T_FADE
        if fade is None:
            return
        w = state.get_scratch(f'{self._prefix}:fade_w')
        dw = state.get_scratch(f'{self._prefix}:fade_dw')
        tmp = state.get_scratch(f'{self._prefix}:fade_tmp')
        self._fade_weight(state.T, fade, w, dw, tmp)

        off = w == 0.0
        if dval is not None:
            mu = state.get_scratch('solver:mu_at_entry')
            np.multiply(dw, val, out=tmp)     # L dw/dT
            np.multiply(tmp, mu, out=tmp)     # into temp_mu space
            np.multiply(dval, w, out=dval)    # w dL/dT
            np.add(dval, tmp, out=dval)
            np.copyto(dval, 0.0, where=off)
        np.multiply(val, w, out=val)
        # Assign rather than rely on the multiply where the channel is
        # off. A fit evaluated far outside its range can return inf or
        # NaN -- the C I three-level solve overflows above about 1e7 K
        # and returns NaN -- and `NaN * 0` is still NaN, which would
        # then poison the whole of `net_cool`. Overwriting makes the
        # fade a numerical guard as well as a physical gate.
        np.copyto(val, 0.0, where=off)

    def update(self, state: Any) -> None:
        """Recompute every channel and accumulate into the solver
        slots.

        The solver expects:
          - `solver:net_cool` = Lambda - Gamma (signed, cool-positive),
            per unit volume, erg / s / cm^3
          - `solver:d_net_cool_d_temp_mu` = d/d(T/mu) of the above

        Individual channels write `n_H * Lambda_NCR`, that is the rate
        per hydrogen nucleus (erg / s / H), matching the per-H^2
        Tigris-NCR convention documented in
        `tables/chianti_v11/build_cool.py` where the volumetric rate is
        `n_H^2 * Lambda_NCR`. The accumulated sum is therefore short one
        factor of `n_H` against what the solver consumes, and this
        method applies it. The C++ side does the same multiplication in
        the solver rather than the channel
        (`tigris-ncr/src/photchem/ncr_solver.hpp:263`,
        `var.cool_rate = var.nH * cool * ...`).
        """
        net_cool = state.get_scratch('solver:net_cool')
        d_net = state.get_scratch('solver:d_net_cool_d_temp_mu')
        net_cool[:] = 0.0
        d_net[:] = 0.0
        for ch in self._channels:
            lam = state.get_scratch(f'{self._prefix}:Lambda:{ch.name}')
            if self._fade_is_off_everywhere(state, ch):
                lam[...] = 0.0
                if self._provide_derivative:
                    state.get_scratch(
                        f'{self._prefix}:dLambda:{ch.name}')[...] = 0.0
                continue
            if self._provide_derivative:
                dlam = state.get_scratch(
                    f'{self._prefix}:dLambda:{ch.name}')
                ch.evaluate(state, lam, dlam)
                self._apply_fade(state, ch, lam, dlam)
                np.add(d_net, dlam, out=d_net)
            else:
                ch.evaluate(state, lam, None)
                self._apply_fade(state, ch, lam, None)
            np.add(net_cool, lam, out=net_cool)
        for hch in self._heating:
            gam = state.get_scratch(f'{self._prefix}:Gamma:{hch.name}')
            if self._fade_is_off_everywhere(state, hch):
                gam[...] = 0.0
                if self._provide_derivative:
                    state.get_scratch(
                        f'{self._prefix}:dGamma:{hch.name}')[...] = 0.0
                continue
            if self._provide_derivative:
                dgam = state.get_scratch(
                    f'{self._prefix}:dGamma:{hch.name}')
                hch.evaluate(state, gam, dgam)
                self._apply_fade(state, hch, gam, dgam)
                np.subtract(d_net, dgam, out=d_net)
            else:
                hch.evaluate(state, gam, None)
                self._apply_fade(state, hch, gam, None)
            np.subtract(net_cool, gam, out=net_cool)

        # Per-H -> per-volume. See the convention note in the docstring.
        np.multiply(net_cool, state.nH, out=net_cool)
        np.multiply(d_net, state.nH, out=d_net)


class HeatingChannel(abc.ABC):
    """One heating mechanism. Same shape as `CoolingChannel`; lives
    here to avoid a forward import cycle in `CoolingChannels`.
    """

    name: ClassVar[str] = '<unset>'
    SCRATCH_NAMES: ClassVar[Tuple[str, ...]] = ()

    # Temperature fade, `(T_full, T_off)`. Same contract as
    # `CoolingChannel.T_FADE`. Heating channels need it for the same
    # reason: `PhotoelectricHeating` shares the WD01 grain-charge
    # chain with `GrainRecombinationCooling`, so it too is a
    # cold-and-warm-ISM fit that must not run in hot gas.
    T_FADE: ClassVar[Optional[Tuple[float, float]]] = None

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
