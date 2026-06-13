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

The abstract class is deliberately small. Four class-level attributes
declare structural facts about the network — `species` (the
`SpeciesSet` instance describing names, indices, charges),
`walk_order` (per-element ion sequence for the Phase D sequential
sweep), `evolved` (the tuple of species names the solver integrates
as ODE state), and `ghost` (the tuple of species names that
`fill_ghosts` reconstructs algebraically each substep). Two ClassVar
booleans advertise capability flags to the solver layer:
`kSupportsStrips` (can `evaluate_CD` consume an `ncell > 1` state)
and `kNeedsJacobian` (does any planned solver require `jacobian`).
Concrete networks override the abstract methods; the base class
provides a default `fill_ghosts` no-op (so networks with no ghost
species need not override) and a `jacobian` that raises
`NotImplementedError`.
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

    # Evolved / ghost species declarations. `evolved` lists the names
    # the solver treats as ODE state variables (operated on by
    # `evaluate_CD` and integrated by the solver); `ghost` lists the
    # names rebuilt algebraically by `fill_ghosts` each substep.
    # The two tuples must cover `species.names` exactly with no
    # overlap; `SpeciesSet` enforces that when constructed via the
    # `ncr3_with_ghosts` family of factories, so this declaration is
    # informational and used by the solver to size scratch buffers.
    evolved: ClassVar[Tuple[str, ...]] = ()
    ghost:   ClassVar[Tuple[str, ...]] = ()

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
        """Rebuild the ghost-species rows of `state.x` in place.

        Contract:

        - Pure algebra: reads only `state.x[evolved_idx]`, `state.T`,
          `state.nH`, `state.Z_g`, `state.Z_d`, and any radiation /
          column inputs the concrete network needs.
        - Idempotent: calling `fill_ghosts(state)` twice in a row
          leaves `state.x` unchanged on the second call.
        - Mutates only `state.x[ghost_idx]`; never touches evolved
          rows.
        - Allocates nothing in the inner loop; the ghost rows are
          written by indexed assignment into the pre-existing
          `state.x` buffer.

        Default no-op for networks that declare no ghost species
        (e.g., a pure GOW17 network with the full species tracked
        as ODE variables). Concrete networks override this to
        materialise the algebraic closure.
        """
        return None

    def allocate_scratch(self, state: Any) -> None:
        """Register network-owned scratch buffers on `state`.

        Called once at construction time by `ChemState.from_grid` when
        the network is supplied. Subclasses with multi-buffer scratch
        needs (e.g., a future NCR3-with-ions network reusing a
        `metal_CT` 4xN strip) override this and call
        `state.alloc_scratch(name, shape, dtype)` for each buffer.
        Default: no-op.
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
