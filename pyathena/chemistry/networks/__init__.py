"""Chemistry network policies (Phase 2+).

Each module defines one concrete network — species inventory plus the
semi-implicit (C, D) rate decomposition that the solver layer consumes.
The abstract base lives in `base.py`; see the chemistry-rewrite-plan §4
NetworkBase paragraph for the policy contract.
"""
