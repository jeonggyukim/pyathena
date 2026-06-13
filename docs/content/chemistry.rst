=========
Chemistry
=========

The ``pyathena.chemistry`` package is a typed, policy-based rewrite of
``pyathena.microphysics``. The driver composes seven independent policy slots
(network, solver, cooling, heating, opacity, radiation, thermo) at
construction time; each slot is an abstract base class with multiple
concrete implementations swappable per run. The same shape is the target for
the tigris-ncr C++ implementation, so this package doubles as the reference
implementation while the C++ port is in progress. Replaces the old package
over a deprecation window. See the package ``README.md`` and
``tigris-notes/docs-claude/pyathena/chemistry-rewrite-plan.md`` for full
design and roadmap.

The current public API is small (Phase 0-1-2 in progress). This page renders
the docstrings of the already-landed modules; new sections appear as further
phases land.


Per-ion coolant solver
======================

The 5-level statistical-equilibrium solver and its per-ion wrapper. Docstrings
include the relevant atomic-physics equations rendered via MathJax.

.. autoclass:: pyathena.chemistry.coolants.base.IonCoolant
   :members:

.. autofunction:: pyathena.chemistry.coolants.n_level.solve_5level_steady_state


Rate coefficients
=================

Phase 1 ports of the leaf rate modules; parity-tested at rtol = 1e-12 against
the corresponding ``pyathena.microphysics`` modules.

.. autoclass:: pyathena.chemistry.rates.photx.PhotX
   :members:

.. autofunction:: pyathena.chemistry.rates.photx.get_sigma_pi_H2

.. autoclass:: pyathena.chemistry.rates.ci_rate.CollIonRate
   :members:

.. autoclass:: pyathena.chemistry.rates.rec_rate.RecRate
   :members:

.. autoclass:: pyathena.chemistry.rates.ct_rate.ChargeTransferRate
   :members:


State + enums + diagnostics
===========================

The Phase 0 skeleton: state object that every chemistry policy reads and
writes, plus shared enums and a fixed-field diagnostic counter block.

.. autoclass:: pyathena.chemistry.state.ChemState
   :members:

.. autoclass:: pyathena.chemistry.enums.Regime
   :members:

.. autoclass:: pyathena.chemistry.enums.InterpMode
   :members:

.. autoclass:: pyathena.chemistry.diagnostics.SolverDiag
   :members:

.. autofunction:: pyathena.chemistry.datapaths.chianti_v11_dir

.. autofunction:: pyathena.chemistry.datapaths.gf12_dir

.. autofunction:: pyathena.chemistry.datapaths.chemistry_dir
