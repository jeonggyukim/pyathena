"""NetworkBase — abstract base class for chemistry networks.

A network owns the species inventory and the rate equations that
populate the semi-implicit (C, D) decomposition. The solver layer
calls `evaluate_CD(state, out_C, out_D)` to fill caller-owned buffers
with creation rates C_i and destruction frequencies D_i such that

    dx_i / dt = C_i - D_i * x_i

then advances `x` in place using whatever integration scheme it
implements. See `chemistry-rewrite-plan.md` §4 NetworkBase for the
full policy contract.

Design notes:

- All methods take and mutate `ChemState` in place. None allocate.
  `evaluate_CD` writes into caller-owned `out_C`, `out_D` of shape
  `(nspec, ncell)`.
- `closure(state)` enforces algebraic conservation (e.g. H mass) and
  applies the species floor. It is called once per substep by the
  solver, never mid-sweep.
- `electron_fraction(state)` returns x_e implied by the current `x`
  and the species charges. The driver / solver may write `n_e` from
  this, but networks must not depend on stale `state.x[electron_idx]`.
- Networks must be branch-free over the strip axis: any temperature-
  or composition-based selection uses `np.where` over the full
  `ncell`-shaped array, never a Python `if`. Sorted-strip dispatch is
  the solver's responsibility.

The abstract class is deliberately small. Two class-level attributes
declare structural facts about the network — `species` (the
`SpeciesSet` instance describing names, indices, charges) and
`walk_order` (per-element ion sequence for the Phase D sequential
sweep) — and two ClassVar booleans advertise capability flags to the
solver layer: `kSupportsStrips` (can `evaluate_CD` consume an
`ncell > 1` state) and `kNeedsJacobian` (does any planned solver
require `jacobian` to be implemented). Concrete networks override the
abstract methods; the base class provides a no-op `fill_ghosts` and a
`jacobian` that raises `NotImplementedError`.
"""
from __future__ import annotations

import abc
from typing import Any, ClassVar, Tuple

import numpy as np


class NetworkBase(abc.ABC):
    """Abstract base for chemistry networks.

    Subclasses must set the class-level attributes `species` and
    `walk_order`, and override `evaluate_CD`, `closure`, and
    `electron_fraction`. Optional overrides: `fill_ghosts` (default
    no-op) and `jacobian` (default raises).
    """

    # Class-level structural metadata. Concrete networks override.
    # `species` is typed `Any` to avoid a forward-import cycle on
    # `SpeciesSet` (the integration agent will tighten this once a
    # shared SpeciesSet module lands).
    species: ClassVar[Any] = None
    walk_order: ClassVar[Tuple[Tuple[str, ...], ...]] = ()

    # Capability flags consumed by the solver / driver layer.
    kSupportsStrips: ClassVar[bool] = True
    kNeedsJacobian: ClassVar[bool] = False

    # ---- Abstract policy methods -----------------------------------
    @abc.abstractmethod
    def evaluate_CD(
        self,
        state: Any,
        out_C: np.ndarray,
        out_D: np.ndarray,
    ) -> None:
        """Fill `out_C` and `out_D` with the semi-implicit rate split.

        Writes:
            out_C[i, :] : creation rate for species i              [s^-1]
            out_D[i, :] : destruction frequency for species i      [s^-1]

        such that `dx_i/dt = C_i - D_i * x_i` (or, equivalently for
        the implicit-Euler substep, `x_i^new = (x_i + C_i*dt) /
        (1 + D_i*dt)`). Buffers have shape `(nspec, ncell)` and are
        caller-owned — this method never allocates.

        Species rows that the concrete network does not track may be
        left untouched (their cells are managed by other networks /
        closure). All written values must be non-negative.
        """

    @abc.abstractmethod
    def closure(self, state: Any) -> None:
        """Apply algebraic conservation + species floors in place.

        Called once per substep by the solver, after `evaluate_CD` has
        been integrated. Subclasses are responsible for clamping
        species fractions to `>= x_floor` and renormalising so that
        per-element conservation holds.
        """

    @abc.abstractmethod
    def electron_fraction(self, state: Any) -> np.ndarray:
        """Return x_e implied by the current `state.x` and charges.

        Returns an `(ncell,)` array. The driver may write
        `state.x[electron_row, :] = nH * x_e` (or however the strip
        encodes electrons); the network itself does not mutate
        electrons here — that is the closure / driver's job.
        """

    # ---- Concrete defaults -----------------------------------------
    def fill_ghosts(self, state: Any) -> None:
        """Fill algebraically-derived species (e.g., GOW17 CHx from
        H2, C). Default no-op for networks with no derived species.
        """
        return None

    def jacobian(self, state: Any, out_J: np.ndarray) -> None:
        """Write the sparse analytic Jacobian `dx_i/dt w.r.t. x_j`
        into `out_J`. Default raises `NotImplementedError` — solvers
        that need a Jacobian must use a network that overrides this.
        """
        raise NotImplementedError(
            f'{type(self).__name__} does not implement jacobian; '
            'choose a network with kNeedsJacobian=True or a solver '
            'that does not require one'
        )
