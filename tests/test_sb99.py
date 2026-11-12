"""SB99 reader on the Starburst99 datasets bundled in data/sb99/."""

from pathlib import Path

import numpy as np
import pytest

from pyathena.util.sb99 import SB99

DATA = Path(__file__).resolve().parents[1] / 'data' / 'sb99'


@pytest.fixture(scope='module')
def rr():
    return SB99(basedir=str(DATA / 'Z014_M1E6_GenevaV00_dt02'),
                verbose=False).read_rad()


def test_time_grid(rr):
    t = np.asarray(rr['time_Myr'])
    assert len(t) == 251
    np.testing.assert_allclose(np.diff(t), 0.2, rtol=1e-6)


def test_band_luminosities(rr):
    L = rr['L']
    assert set(L) == {'tot', 'LyC', 'LW', 'PE', 'OPT', 'IR', 'FUV', 'UV'}
    for k in L:
        assert np.all(np.isfinite(L[k])) and np.all(np.asarray(L[k]) >= 0)
    for k in ('LyC', 'LW', 'PE'):
        assert np.all(np.asarray(L[k]) <= np.asarray(L['UV']))
    assert np.all(np.asarray(L['UV']) <= np.asarray(L['tot']))


def test_log_spaced_dataset():
    rr = SB99(basedir=str(DATA / 'Z014_M1E6_GenevaV00_logdt_10Gyr'),
              verbose=False).read_rad()
    assert len(rr['time_Myr']) == 77
