"""Single source of truth for enum values shared between
`pyathena.chemistry` and the tigris-ncr C++ port.

The C++ side reads these values via a codegen pass that emits the
matching enum members in `src/photchem/photchem_enums.hpp` (planned).
Reordering or renumbering values in this file therefore has C++
ABI implications — keep them stable.

Two enum families today:

  Regime    — per-cell physical regime tag used by sorted-strip
              dispatch (Phase B.3b / C2). Set by `SolverBase.pre_scan`.

  InterpMode — rate-table interpolation mode. Mirrors the existing
              `NCRRates::InterpMode` enum class in
              `src/photchem/ncr_rates.hpp`.
"""
from __future__ import annotations

import enum


class Regime(enum.IntEnum):
    """Per-cell physical-regime tag for sorted-strip dispatch.

    Values are deliberately small int8-compatible integers; the C++
    port maps them to `enum class Regime : int8_t` and stores them
    in `NCRStrip` as `int8` to keep the strip footprint small.
    """
    HOT          = 0    # T > 1e5 K — fully ionized, collisional cooling dominant
    WARM_NEUTRAL = 1    # 1e3 K < T < 1e4 K, x_HII small — CII / OI cooling
    COLD_MOL     = 2    # T < 100 K, x_H2 > 0.5 — molecular / dust cooling dominant
    # FRONT (value 3) is reserved for sharp-gradient cells (ionization
    # fronts, WNM/CNM transition, post-shock cooling regions). It is
    # experimental: only consumed once Phase C0 perf experiments
    # validate that sorted-strip dispatch actually helps SIMD
    # throughput. Reserved here so the C++ ABI gets the value early.
    FRONT        = 3
    OTHER        = 99   # uncategorised; safe default before pre_scan runs


class InterpMode(enum.IntEnum):
    """Rate-table interpolation mode.

    Mirrors `NCRRates::InterpMode` in `src/photchem/ncr_rates.hpp`.
    Selectable per-run via the `interp_mode` parameter in the
    `[photchem_ncr]` input block.

    - kExact  : full analytical form (slow; one log/exp/pow per rate)
    - kLogLog : log-log linear interpolation (1 log10 + 1 pow10 per rate)
    - kNqt2   : NQT grid, nqt2_log encoding (1 sqrt + integer ops)
    - kNqt1   : NQT grid, nqt1_log encoding (integer ops only)
    """
    kExact  = 0
    kLogLog = 1
    kNqt2   = 2
    kNqt1   = 3
