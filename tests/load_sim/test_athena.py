"""LoadSim on a fake Athena (C) run with vtk files split over two MPI ranks."""

import warnings

import numpy as np
import pytest

import pyathena as pa


@pytest.fixture(scope='module')
def sim(athena_run):
    basedir, _ = athena_run
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        return pa.LoadSim(str(basedir))


def test_metadata(sim, athena_run):
    _, e = athena_run
    assert sim.athena_variant == 'athena'
    assert sim.problem_id == e['pid']
    assert sim.ff.out_fmt == ['hst', 'vtk']
    assert sim.par['domain1']['Nx1'] == e['nx'][0]
    np.testing.assert_array_equal(sim.domain['Nx'], e['nx'])
    np.testing.assert_allclose(sim.domain['le'], e['le'])
    np.testing.assert_allclose(sim.domain['re'], e['re'])


def test_nums(sim, athena_run):
    _, e = athena_run
    assert sim.nums == e['nums']
    assert sim.ff.nums_id0 == e['nums']


@pytest.mark.parametrize('num', [0, 1])
def test_load_vtk(sim, athena_run, num):
    _, e = athena_run
    ds = sim.load_vtk(num=num)
    assert ds.domain['time'] == float(num)
    assert ds.domain['ngrid'] == 2
    assert ds.field_list == ['density', 'velocity', 'pressure']
    np.testing.assert_array_equal(ds.domain['Nx'], e['nx'])
    np.testing.assert_allclose(ds.domain['le'], e['le'])
    np.testing.assert_allclose(ds.domain['dx'], e['dx'])

    d = ds.get_field(['density', 'velocity', 'pressure'])
    ref = e['fields'][num]
    np.testing.assert_array_equal(d['density'].values, ref['density'])
    np.testing.assert_array_equal(d['pressure'].values, ref['pressure'])
    for c in range(3):
        np.testing.assert_array_equal(d[f'velocity{c+1}'].values,
                                      ref['velocity'][..., c])

    xc = e['le'][0] + e['dx'][0]*(np.arange(e['nx'][0]) + 0.5)
    zc = e['le'][2] + e['dx'][2]*(np.arange(e['nx'][2]) + 0.5)
    np.testing.assert_allclose(d['x'].values, xc)
    np.testing.assert_allclose(d['z'].values, zc)


def test_get_slice(sim, athena_run):
    _, e = athena_run
    ds = sim.load_vtk(num=1)
    slc = ds.get_slice('z', 'density', pos=0.5)
    iz = int((0.5 - e['le'][2])/e['dx'][2])
    np.testing.assert_array_equal(slc['density'].values,
                                  e['fields'][1]['density'][iz])


def test_read_hst(sim):
    h = pa.read_hst(sim.files['hst'])
    assert list(h.columns) == ['time', 'dt', 'mass', 'totalE']
    assert len(h) == 21
    np.testing.assert_allclose(h['time'].iloc[-1], 2.0)


def test_findfiles_attribute_fallback_warns(sim, athena_run):
    _, e = athena_run
    with pytest.warns(FutureWarning, match=r's\.ff\.nums_id0'):
        nums = sim.nums_id0
    assert nums == e['nums']
    with pytest.raises(AttributeError):
        sim.no_such_attribute
