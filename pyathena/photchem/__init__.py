"""Pyathena photochemistry subpackage.

Per-ion metal-line cooling functions + a generic N-level statistical
equilibrium solver, plus the offline CHIANTI table builder. Designed
to live alongside the existing `pyathena.microphysics.photchem`
PhotChem class without disturbing it; can be reorganized later.

Subpackages:
  - `coolants/` -- per-ion cooling functions (OIII, NII, SII, ...)
  - `data/` -- CHIANTI-derived atomic-data tables (numpy npz)

Top-level:
  - `n_level` -- generic statistical equilibrium solver
"""

from .n_level import solve_5level_steady_state, cooling_from_populations  # noqa: F401
