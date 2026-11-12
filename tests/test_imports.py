"""Every pyathena module imports, apart from missing optional packages."""

import importlib
import pkgutil
import warnings

import pytest

import pyathena

# Modules that are broken today. Remove an entry once the module is fixed.
KNOWN_BROKEN = {
    'pyathena.classic.preprocessing',     # imports removed pyathena.parse_par
    'pyathena.classic.set_plt',           # imports removed pyathena.set_units
    'pyathena.sf_cloud.cloud_evol',       # Python 2 code
    'pyathena.sf_cloud.plt_snapshot',     # imports removed load_all_alphabeta
    'pyathena.sf_cloud_rad.plt_snapshot',  # imports nonexistent module
}

# Packages that are not pyathena dependencies; modules needing them skip.
OPTIONAL = {'mpi4py', 'pygc', 'aenum'}

MODULES = sorted(m.name for m in
                 pkgutil.walk_packages(pyathena.__path__, 'pyathena.'))


@pytest.mark.parametrize('name', MODULES)
def test_import(name):
    if name in KNOWN_BROKEN:
        pytest.xfail('known broken')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name and e.name.split('.')[0] in OPTIONAL:
            pytest.skip(f'needs {e.name}')
        raise
