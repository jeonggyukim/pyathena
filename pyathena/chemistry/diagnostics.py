"""Fixed-field diagnostic counters carried inside `ChemState`.

`SolverDiag` is intentionally a small POD with named scalar fields,
not a dict. Reasons:

1. The tigris-ncr C++ port carries the same diagnostics as a plain
   C struct attached to `PhotochemistryNCR`; a dict has no C++ analog.
2. Fixed-field access is grep-able and statically checkable; new
   diagnostics force an explicit field addition (and a CHANGES.md
   entry) rather than an opaque `state.diag['new_thing'] = ...`.
3. Field names match the planned C++ struct member names byte-for-byte
   so codegen / parity tools can map across languages without a
   translation table.

When a new counter is needed, add a field here AND propagate to the
C++ port. Avoid adding free-form dicts to the solver hot path.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SolverDiag:
    """Per-step diagnostic counters written by the solver.

    Resetting: `reset_step()` zeroes all counters at the head of every
    hydro step. The driver reads them after the step to expose
    aggregate statistics; the solver writes them during the step.
    """
    # ---- Strip / subcycling shape ----
    n_strips_total: int       = 0   # strips processed this step
    n_strips_capped: int      = 0   # strips that hit nsub_max and were extended
    n_extension_passes: int   = 0   # extension passes triggered by capped strips
    n_per_cell_fallback: int  = 0   # cells dispatched to per-cell solver (Phase G)

    # ---- Substep counts ----
    n_substeps_total: int     = 0   # total substeps summed over cells
    nsub_max_seen: int        = 0   # largest nsub seen in any cell this step

    # ---- Convergence / safety ----
    n_thermal_solves: int     = 0   # implicit thermal updates attempted
    n_thermal_iters_total: int = 0  # Newton iterations summed across cells
    n_floor_clamps: int       = 0   # x_i clamped to 1e-20 floor
    n_nan_traps: int          = 0   # NaN detections (should always be zero)

    # ---- Rate-table cache ----
    n_table_misses: int       = 0   # T-grid lookups that missed the per-strip cache

    def reset(self) -> None:
        """Zero all counters. Called by `ChemState.reset_step`."""
        for f in self.__dataclass_fields__:
            setattr(self, f, 0)

    def as_dict(self) -> dict:
        """Snapshot for logging / serialisation."""
        return {f: getattr(self, f) for f in self.__dataclass_fields__}
