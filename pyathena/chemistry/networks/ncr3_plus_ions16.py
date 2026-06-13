"""NCRNetwork3PlusIons16 — Phase 6 planning stub.

Architectural placeholder for the Phase 6 multi-ion network. The
concrete rate equations land alongside the `SequentialIonSweepSolver`
implementation; this module exists today so the SpeciesSet layout,
ghost-kind composition, element-group declaration, and closure
algorithm can be reviewed against the surrounding scaffolding
(`SpeciesSet`, `NetworkBase.element_groups`, `CIEHighPool` from the
Phase 4 `build_cie_high_pool.py` tables) before any rate is wired.

See `chemistry-rewrite-plan.md` §4a (ghost-species KINDS and the
per-element closure formulation) and §5 Phase 6 (hot-regime fast
path) for the design context this stub captures.

Layout summary
==============

Evolved (13 rows, integrated by the solver as ODE state):

    HI, HII, H2,
    CI, CII,
    NI, NII,
    OI, OII, OIII,
    SI, SII, SIII.

Ghost (6 rows, rebuilt algebraically every substep by `fill_ghosts`):

    e          : charge-sum ghost (kind 1).  x_e = sum_q q * x_q.
    CO         : prescription ghost (kind 2).  Reuses
                 `NCRNetwork3.CalculateCOAbundance` until the GOW17
                 chain lands in Phase 9.
    x_high_C   : CIE-lumped pool ghost (kind 3).  Sum of x_CIII..x_CVI
                 in CIE.  Table from
                 `data/chemistry/cie_high_pool_C.nc`.
    x_high_N   : Sum of x_NIII..x_NVII in CIE.  Table for N.
    x_high_O   : Sum of x_OIV..x_OVIII in CIE.  Table for O.
    x_high_S   : Sum of x_SIV..x_SXVI in CIE.  Table for S.

He is left as a Phase 6 follow-up: when the runs in scope need
non-equilibrium He, `NCRNetwork3PlusHe` (a separate network class)
promotes He I / II / III out of the prescription-ghost layer; until
then He stays algebraic and contributes to `x_e` via the prescription.

Closure (per substep, after `evaluate_CD` integrates the evolved rows)
======================================================================

For each element in `element_groups`:

    x_low_avail(T) = (1 - x_high_frac(T)) * x_std * Z_g
    sum_evolved    = sum(x_q for q in evolved_ion_names)

If `sum_evolved > x_low_avail(T)` in any cell, the evolved rows of
that element are renormalised by the factor
`x_low_avail(T) / sum_evolved` (per cell, branch-free) before the
ghost rows are written.  The pool ghost stores
`x_high_pool = x_high_frac(T) * x_std * Z_g` directly.  See
chemistry-rewrite-plan.md §4a.3 for the worked example
(O at T = 1e5 K).

Hot-regime fast path (Phase 6)
==============================

Before the per-element implicit-Euler updates, the solver pre-scans
every strip cell:

    hot = all(x_low_avail(T) < eps for elem in element_groups)

Cells flagged `hot` skip the evolved-ion updates entirely; the
evolved rows are frozen at their incoming values (typically already
at `x_floor` after a few prior substeps in the same regime),
`fill_ghosts` writes the CIE-pool rows directly from the tables, and
cooling evaluates
`Lambda = sum_elem Lambda_high(T) * x_std * Z_g` instead of summing
per-ion `Lambda_q`.  Threshold `eps` defaults to 1e-3; the regression
suite covers cells on both sides so the fast path and the slow path
agree at the boundary.

Status
======

THIS IS A PLANNING STUB.  Instantiating the class raises
`NotImplementedError` — every method body is held over until Phase 6.
The class exists today to (a) anchor the architectural review against
the surrounding scaffolding, (b) document the `element_groups`
declaration the driver consumes, and (c) reserve the module path so
later imports do not need a churn rename.
"""
from __future__ import annotations

from typing import Any, ClassVar, Tuple

import numpy as np

from .base import NetworkBase


class NCRNetwork3PlusIons16(NetworkBase):
    """Phase 6 multi-ion NCR network — planning stub (see module docstring)."""

    # --------------------------------------------------------------
    # Structural metadata.  Concrete rate equations land in Phase 6.
    # --------------------------------------------------------------

    evolved: ClassVar[Tuple[str, ...]] = (
        'HI', 'HII', 'H2',
        'CI', 'CII',
        'NI', 'NII',
        'OI', 'OII', 'OIII',
        'SI', 'SII', 'SIII',
    )

    ghost: ClassVar[Tuple[str, ...]] = (
        'e', 'CO',
        'x_high_C', 'x_high_N', 'x_high_O', 'x_high_S',
    )

    element_groups: ClassVar[
        Tuple[Tuple[str, int, Tuple[str, ...]], ...]
    ] = (
        ('C', 2, ('CI', 'CII')),
        ('N', 2, ('NI', 'NII')),
        ('O', 3, ('OI', 'OII', 'OIII')),
        ('S', 3, ('SI', 'SII', 'SIII')),
    )

    # Walk order for `SequentialIonSweepSolver`.  Ground-state first
    # per element; `ne` is refreshed between elements (documented
    # behaviour, not a bug — see §6 of `multi_ion_chemistry_plan.md`).
    walk_order: ClassVar[Tuple[Tuple[str, ...], ...]] = (
        ('CI', 'CII'),
        ('NI', 'NII'),
        ('OI', 'OII', 'OIII'),
        ('SI', 'SII', 'SIII'),
    )

    kNeedsJacobian: ClassVar[bool] = False

    # --------------------------------------------------------------
    # Stub bodies.  Phase 6 implements these against `IonRates`,
    # `CIEHighPool`, and the per-element CT wiring from
    # `pyathena_ct_fixes_plan.md`.
    # --------------------------------------------------------------

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            'NCRNetwork3PlusIons16 is a Phase 6 planning stub. The rate '
            'equations, hot-regime fast path, and CIEHighPool integration '
            'land alongside SequentialIonSweepSolver. Use NCRNetwork3 '
            'until then.'
        )

    def evaluate_CD(
        self,
        state: Any,
        out_C: np.ndarray,
        out_D: np.ndarray,
    ) -> None:
        raise NotImplementedError('Phase 6')

    def closure(self, state: Any) -> None:
        raise NotImplementedError('Phase 6')

    def electron_fraction(self, state: Any) -> np.ndarray:
        raise NotImplementedError('Phase 6')

    def fill_ghosts(self, state: Any) -> None:
        raise NotImplementedError('Phase 6')

    def allocate_scratch(self, state: Any) -> None:
        raise NotImplementedError('Phase 6')
