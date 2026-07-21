"""Data-file path resolvers.

All tables shipped with pyathena live under `<repo-root>/data/`. This
module is the single place that converts logical names like
'chianti_v11' or 'gf12' into absolute paths. Modules under
`pyathena.chemistry` should never hard-code paths to data files.

Reasoning: pyathena's repo root is reachable from
`pyathena/chemistry/<this file>` by going up three directories. The
data directory layout mirrors the producer:
    data/microphysics/    — files produced by pyathena.microphysics
                            (CHIANTI builders, GF12 reference tables, etc.)
    data/chemistry/       — files produced by pyathena.chemistry
                            (new table formats, 2-D Lambda(T, n_e), ...)
"""
from __future__ import annotations

import os

_THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# repo-root = pyathena/chemistry/<this file> -> ../../../
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_FILE_DIR, '..', '..'))
_DATA_ROOT = os.path.join(_REPO_ROOT, 'data')


def chianti_v11_dir() -> str:
    """Absolute path to the CHIANTI v11 derived tables (ioneq, cool,
    per-ion atomic data). Files live under
    `data/microphysics/chianti_v11/` — built by
    `pyathena.chemistry.tables.chianti_v11.*`.
    """
    return os.path.join(_DATA_ROOT, 'microphysics', 'chianti_v11')


def gf12_dir() -> str:
    """Absolute path to the Gnat & Ferland 2012 per-element CIE
    cooling tables. Files live under
    `data/microphysics/Gnat_Ferland12_tables/`.
    """
    return os.path.join(_DATA_ROOT, 'microphysics', 'Gnat_Ferland12_tables')


def chemistry_dir() -> str:
    """Absolute path to the `pyathena.chemistry`-produced tables
    (new formats not present in `microphysics`, e.g., 2-D
    Lambda(T, n_e) tables, per-band SED-averaged opacities).
    Created on demand; not guaranteed to exist on a fresh checkout.
    """
    return os.path.join(_DATA_ROOT, 'chemistry')
