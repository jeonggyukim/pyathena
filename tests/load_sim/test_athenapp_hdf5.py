"""LoadSim on a fake Athena++ run with hdf5 output split into 4 meshblocks."""

import warnings

import numpy as np
import pytest

import pyathena as pa

VARS = ['rho', 'press', 'vel1', 'vel2', 'vel3']


@pytest.fixture(scope='module')
def sim(athenapp_hdf5_run):
    basedir, _ = athenapp_hdf5_run
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        return pa.LoadSim(str(basedir))


def test_metadata(sim, athenapp_hdf5_run):
    _, e = athenapp_hdf5_run
    assert sim.athena_variant == 'athena++'
    assert sim.problem_id == e['pid']
    assert sim.ff.out_fmt == ['hst', 'hdf5']
    assert sim.ff.hdf5_outid == [2]
    assert sim.ff._hdf5_outvar_def == 'prim'


def test_nums(sim, athenapp_hdf5_run):
    _, e = athenapp_hdf5_run
    assert sim.ff.nums_hdf5['prim'] == e['nums']
    assert sim.nums == e['nums']


def _check(ds, e, num):
    ref = e['fields'][num]
    assert sorted(ds.data_vars) == sorted(VARS)
    for v in VARS:
        np.testing.assert_array_equal(ds[v].values, ref[v])
    xc = e['le'][0] + e['dx'][0]*(np.arange(e['nx'][0]) + 0.5)
    yc = e['le'][1] + e['dx'][1]*(np.arange(e['nx'][1]) + 0.5)
    np.testing.assert_allclose(ds['x'].values, xc, rtol=1e-6)
    np.testing.assert_allclose(ds['y'].values, yc, rtol=1e-6)


@pytest.mark.parametrize('num', [0, 1])
def test_load_hdf5(sim, athenapp_hdf5_run, num):
    _, e = athenapp_hdf5_run
    ds = sim.load_hdf5(num=num)
    assert float(ds.attrs['Time']) == float(num)
    assert ds.attrs['dx1'] == pytest.approx(e['dx'][0])
    _check(ds, e, num)


def test_load_hdf5_dask(sim, athenapp_hdf5_run):
    _, e = athenapp_hdf5_run
    ds = sim.load_hdf5(num=1, chunks=dict(x=4, y=3, z=4))
    _check(ds.compute(), e, 1)
