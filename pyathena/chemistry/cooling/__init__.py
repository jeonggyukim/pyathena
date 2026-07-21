"""Per-channel cooling policies.

Each module in this package owns one cooling mechanism (Lyman-alpha,
recombination, fine-structure, dust, ...). The driver composes a
`CoolingChannels` policy that sums the per-channel `Lambda` columns
into the `solver:net_cool` scratch buffer the explicit-subcycling
solver consumes.

The abstract base is `pyathena.chemistry.cooling.base.CoolingChannel`.
Concrete channels mirror the legacy `pyathena.microphysics.cool`
functions one-for-one so the Phase 4 parity tests can swap the new
path in module by module.
"""
from __future__ import annotations
