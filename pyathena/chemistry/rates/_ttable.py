"""Shared temperature-table helper for rate and cooling modules.

`ci_rate`, `rec_rate` and `ct_rate` each grew their own copy of the
same build-and-look-up code. This holds one implementation, on the
grid all three already share, so the cooling channels can tabulate
their temperature-only factors without a fourth copy.

The grid is 2000 points from 1 K to 1e9 K, log10-spaced for
`InterpMode.LogLog` and `nqt1_log`-spaced for the NQT modes.

Sharing the grid is what makes tabulation pay, and the API is shaped
around it. Finding a cell's position on the axis costs a transcendental
(`log10`, or bit manipulation for the NQT modes); interpolating a value
at a known position costs a few arithmetic operations plus one decode.
So a cell that reads N quantities should call `locate` once and
`interp` N times.

Calling `lookup` per table instead recomputes the position every time,
and then tabulation can lose outright. Lyman alpha is the worked
example: its two temperature-only factors replace three analytic
transcendentals, but two independent LogLog lookups cost two
transcendentals each, so four to replace three. One `locate` plus two
`interp` calls costs two, and wins.

Values are stored log-encoded, so a tabulated quantity must be positive
over the range where it is read. Non-positive entries are floored at
`RATE_FLOOR` and, where a column vanishes identically, a flag returns
exact zeros instead.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from ..enums import InterpMode
from . import _nqt


RATE_FLOOR: float = 1.0e-100
TAB_N: int = 2000
TAB_T_MIN: float = 1.0
TAB_T_MAX: float = 1.0e9


class TemperatureTable:
    """One tabulated function of temperature.

    Parameters
    ----------
    fn : callable
        Evaluated on an ndarray of temperatures. Must be positive
        wherever the table will be read.
    mode : InterpMode
        `LogLog`, `Nqt2` or `Nqt1`. `Exact` is not a table and should
        be handled by the caller before constructing one.
    clamp : (float, float), optional
        When the tabulated quantity is only defined on a window and is
        held constant outside it, pass the window here and build `fn`
        *without* its own clamping. Lookup then clips `T` into the
        window, which keeps the resulting slope discontinuity out of
        the interpolant. Tabulating an already-clamped function
        instead costs about two orders of magnitude in accuracy near
        the window edges, and refining the grid does not fix it --
        see the charge-transfer entry dated 2026-07-30 in
        `chemistry/CHANGES.md`.

        A clamped table cannot share a position with its neighbours,
        because the clip changes `T`. Such a table has to go through
        `lookup`.
    """

    __slots__ = ('mode', '_x_min', '_inv_dx', '_tab', '_tab_hi',
                 '_all_zero', '_clamp', 'T_grid')

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray],
                 mode: InterpMode,
                 clamp: Optional[Tuple[float, float]] = None) -> None:
        if mode == InterpMode.Exact:
            raise ValueError(
                'TemperatureTable is for the table modes; handle '
                'InterpMode.Exact analytically in the caller.')
        self.mode = mode
        self._clamp = clamp

        if mode == InterpMode.LogLog:
            log_T = np.linspace(np.log10(TAB_T_MIN),
                                np.log10(TAB_T_MAX), TAB_N)
            self.T_grid = 10.0 ** log_T
            self._x_min = float(log_T[0])
            self._inv_dx = float((TAB_N - 1) / (log_T[-1] - log_T[0]))
        else:
            y_min = float(_nqt.nqt1_log(np.array([TAB_T_MIN]))[0])
            y_max = float(_nqt.nqt1_log(np.array([TAB_T_MAX]))[0])
            y_grid = np.linspace(y_min, y_max, TAB_N)
            self.T_grid = _nqt.nqt1_exp(y_grid)
            self._x_min = y_min
            self._inv_dx = float((TAB_N - 1) / (y_max - y_min))

        if clamp is None:
            T_eval = self.T_grid
        else:
            # Widen by two bins so the bins straddling the window edges
            # hold true unclamped values on both sides. Beyond that the
            # column is never read, because lookup clips first.
            t_lo, t_hi = clamp
            pad = 10.0 ** (2.0 * (np.log10(TAB_T_MAX)
                                  - np.log10(TAB_T_MIN)) / (TAB_N - 1))
            T_eval = np.clip(self.T_grid,
                             max(t_lo / pad, TAB_T_MIN),
                             min(t_hi * pad, TAB_T_MAX))

        values = np.asarray(fn(T_eval), dtype=np.float64)
        self._all_zero = not np.any(values > 0.0)
        values = np.where(values > RATE_FLOOR, values, RATE_FLOOR)
        if mode == InterpMode.LogLog:
            self._tab = np.log10(values)
        elif mode == InterpMode.Nqt2:
            self._tab = _nqt.nqt2_log(values)
        else:
            self._tab = _nqt.nqt1_log(values)
        # View onto the upper node of each bracketing pair, so `interp`
        # never has to form `idx + 1` (which would allocate).
        self._tab_hi = self._tab[1:]

    # ---- Scratch -------------------------------------------------------
    @staticmethod
    def alloc_scratch(state, prefix: str) -> None:
        """Register the buffers the lookup path needs on `state`.

        `<prefix>:idx` and `<prefix>:frac` hold the axis position, which
        is shared across every unclamped table on this grid.
        `<prefix>:lo` and `<prefix>:hi` are per-interpolation working
        space and may be reused between tables.
        """
        ncell = state.nH.shape[0]
        state.alloc_scratch(f'{prefix}:idx', (ncell,), dtype=np.int64)
        state.alloc_scratch(f'{prefix}:frac', (ncell,))
        state.alloc_scratch(f'{prefix}:lo', (ncell,))
        state.alloc_scratch(f'{prefix}:hi', (ncell,))

    @staticmethod
    def get_scratch(state, prefix: str):
        """Fetch `(idx, frac, lo, hi)` in the order the calls expect."""
        return (state.get_scratch(f'{prefix}:idx'),
                state.get_scratch(f'{prefix}:frac'),
                state.get_scratch(f'{prefix}:lo'),
                state.get_scratch(f'{prefix}:hi'))

    # ---- Two-step path: locate once, interpolate many ------------------
    def locate(self, T: np.ndarray, idx: np.ndarray,
               frac: np.ndarray) -> None:
        """Write the grid position of `T` into `idx` and `frac`.

        This is the step that costs a transcendental, and the step to
        run once per cell rather than once per table. Every unclamped
        table on this grid accepts the same `(idx, frac)`.
        """
        np.copyto(frac, T)
        if self.mode == InterpMode.LogLog:
            np.log10(frac, out=frac)
        else:
            np.copyto(frac, _nqt.nqt1_log(frac))
        np.subtract(frac, self._x_min, out=frac)
        np.multiply(frac, self._inv_dx, out=frac)
        # Clip the position to the grid before truncating. Clipping to
        # TAB_N - 2 first would discard the top node: `frac` could never
        # reach 1, so the last grid point would return its neighbour's
        # value. Truncation equals floor here because the value is
        # non-negative after the clip.
        np.clip(frac, 0.0, float(TAB_N - 1), out=frac)
        np.copyto(idx, frac, casting='unsafe')
        np.clip(idx, 0, TAB_N - 2, out=idx)
        np.subtract(frac, idx, out=frac)
        np.clip(frac, 0.0, 1.0, out=frac)

    def interp(self, idx: np.ndarray, frac: np.ndarray, out: np.ndarray,
               lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Interpolate at a position produced by `locate`.

        No transcendental beyond the single decode, and no index
        arithmetic. `np.take` rather than fancy indexing because
        `self._tab[idx]` allocates.
        """
        if self._all_zero:
            out[...] = 0.0
            return out
        np.take(self._tab, idx, out=lo)
        np.take(self._tab_hi, idx, out=hi)
        np.subtract(hi, lo, out=hi)
        np.multiply(hi, frac, out=hi)
        np.add(lo, hi, out=out)
        if self.mode == InterpMode.LogLog:
            np.power(10.0, out, out=out)
        elif self.mode == InterpMode.Nqt2:
            np.copyto(out, _nqt.nqt2_exp(out))
        else:
            np.copyto(out, _nqt.nqt1_exp(out))
        return out

    # ---- One-step path -------------------------------------------------
    def lookup(self, T: np.ndarray, out: np.ndarray, scratch) -> np.ndarray:
        """`locate` then `interp` for a single table.

        Correct anywhere, and the only option for a clamped table, but
        it pays the position cost per table. Prefer one `locate` plus
        several `interp` when a cell reads more than one table.
        """
        idx, frac, lo, hi = scratch
        if self._all_zero:
            out[...] = 0.0
            return out
        if self._clamp is None:
            self.locate(T, idx, frac)
        else:
            np.clip(T, self._clamp[0], self._clamp[1], out=lo)
            self.locate(lo, idx, frac)
        return self.interp(idx, frac, out, lo, hi)

    def __call__(self, T: np.ndarray,
                 out: Optional[np.ndarray] = None) -> np.ndarray:
        """Convenience form that allocates its own scratch.

        For setup, tests and plotting. The substep loop should use
        `locate` once and `interp` per table.
        """
        T_arr = np.asarray(T, dtype=np.float64)
        result = out if out is not None else np.empty_like(T_arr)
        scratch = (np.empty(T_arr.shape, dtype=np.int64),
                   np.empty_like(T_arr),
                   np.empty_like(T_arr), np.empty_like(T_arr))
        return self.lookup(T_arr, result, scratch)
