"""Steady-state n-level population helpers for fine-structure cooling.

Two routines mirror `pyathena.microphysics.cool.cool2Level_` and
`cool3Level_`. They evaluate the analytic steady-state population of
a 2- or 3-level atomic system under collisional excitation /
de-excitation and spontaneous emission (no radiative pumping, no
stimulated emission), returning the cooling rate per cell.

The wrapping channel modules supply the per-mechanism collision
strength evaluations (q_ij rates) and the cooling rate is the
weighted sum of the line energies times their emission rates.

NOTE on the eventual CHIANTI swap (plan §5 Phase 7): CHIANTI is
purely atomic. The metal fine-structure channels (CII / CI / OI /
OII / OIII / NI / NII / SI / SII / SIII) will swap their e/HI
collision-partner rate fits to CHIANTI table lookups, with the H2
collision-partner fits left in place (Wiesenfeld+2014 CII-H2,
Lique+2018 OI-H2, ...). Everything outside the atomic metal-line
cooling stays hand-coded indefinitely: H2 cooling (rovib /
colldiss / G17), dust coupling, gas-grain charge-exchange, free-free,
and CR / PE heating. These steady-state level helpers are
collision-partner-agnostic and stay as-is across the swap; only the
q_ij assembly inside each metal channel changes.
"""
from __future__ import annotations

import numpy as np


def cool_2level(
    q01: np.ndarray,
    q10: np.ndarray,
    A10: float,
    E10: float,
    xs: np.ndarray,
    out: np.ndarray,
    tmp: np.ndarray,
) -> None:
    """Steady-state 2-level cooling rate.

    `Lambda = f1 * A10 * E10 * xs` with the steady-state upper-level
    fraction `f1 = q01 / (q01 + q10 + A10)`. All arrays must share
    shape with `q01`; `out` is overwritten in place.
    """
    np.add(q01, q10, out=tmp)
    np.add(tmp, A10, out=tmp)
    np.divide(q01, tmp, out=out)
    np.multiply(out, A10 * E10, out=out)
    np.multiply(out, xs, out=out)


def cool_3level(
    q01: np.ndarray, q10: np.ndarray,
    q02: np.ndarray, q20: np.ndarray,
    q12: np.ndarray, q21: np.ndarray,
    A10: float, A20: float, A21: float,
    E10: float, E20: float, E21: float,
    xs: np.ndarray,
    out: np.ndarray,
    tmp0: np.ndarray, tmp1: np.ndarray, tmp2: np.ndarray,
) -> None:
    """Steady-state 3-level cooling rate (Draine 2011 19.7-19.10).

    Equilibrium populations of levels 1 and 2 from the rate matrix:

        a0 = R10 R20 + R10 R21 + q12 R20
        a1 = q01 R20 + q01 R21 + R21 q02
        a2 = R10 q02 + q01 q12 + q02 q12       (with R_uv = q_uv + A_uv)

    Level-fraction `f_i = a_i / (a0 + a1 + a2)`. The cooling rate is

        Lambda = (f1 * A10 * E10 + f2 * (A20 * E20 + A21 * E21)) * xs

    Three temporary buffers required:
    `tmp0`, `tmp1`, `tmp2` (caller-owned, `(ncell,)`-shaped).
    """
    # R10 / R20 / R21 reuse the same tmp slots: build a0 first.
    # a0 = (q10 + A10) * (q20 + A20) + (q10 + A10) * (q21 + A21)
    #      + q12 * (q20 + A20)
    np.add(q10, A10, out=tmp0)   # tmp0 = R10
    np.add(q20, A20, out=tmp1)   # tmp1 = R20
    np.add(q21, A21, out=tmp2)   # tmp2 = R21
    np.multiply(tmp0, tmp1, out=out)             # R10*R20
    np.multiply(tmp0, tmp2, out=tmp0)            # R10*R21 (tmp0 reused)
    np.add(out, tmp0, out=out)
    np.multiply(q12, tmp1, out=tmp0)             # q12*R20
    np.add(out, tmp0, out=out)
    # `out` now holds a0; `tmp1` holds R20; `tmp2` holds R21.

    # a1 = q01 * R20 + q01 * R21 + R21 * q02
    np.multiply(q01, tmp1, out=tmp0)             # q01*R20
    # accumulate q01*R21 into tmp0
    np.multiply(q01, tmp2, out=tmp1)             # q01*R21 (tmp1 reused)
    np.add(tmp0, tmp1, out=tmp0)
    np.multiply(tmp2, q02, out=tmp1)             # R21*q02
    np.add(tmp0, tmp1, out=tmp0)                 # tmp0 = a1

    # a2 = R10 * q02 + q01 * q12 + q02 * q12
    # We lost R10 when we reused tmp0/tmp1; rebuild it.
    np.add(q10, A10, out=tmp1)                   # tmp1 = R10
    np.multiply(tmp1, q02, out=tmp1)             # R10*q02
    np.multiply(q01, q12, out=tmp2)              # q01*q12
    np.add(tmp1, tmp2, out=tmp1)
    np.multiply(q02, q12, out=tmp2)              # q02*q12
    np.add(tmp1, tmp2, out=tmp1)                 # tmp1 = a2

    # f0 = a0 / (a0 + a1 + a2)  (not needed)
    # f1 = a1 / sum;  f2 = a2 / sum.
    # Lambda = (f1 * A10 * E10 + f2 * (A20*E20 + A21*E21)) * xs
    np.add(out, tmp0, out=tmp2)                  # tmp2 = a0 + a1
    np.add(tmp2, tmp1, out=tmp2)                 # tmp2 = sum
    np.divide(tmp0, tmp2, out=tmp0)              # tmp0 = f1
    np.divide(tmp1, tmp2, out=tmp1)              # tmp1 = f2
    coeff_2 = A20 * E20 + A21 * E21
    np.multiply(tmp0, A10 * E10, out=tmp0)
    np.multiply(tmp1, coeff_2, out=tmp1)
    np.add(tmp0, tmp1, out=out)
    np.multiply(out, xs, out=out)
