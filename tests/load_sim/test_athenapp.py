"""LoadSim on a fake Athena++ run with vtk output and no hdf5."""

import warnings

import numpy as np
import pytest

import pyathena as pa


@pytest.fixture(scope='module')
def sim(athenapp_run):
    basedir, _ = athenapp_run
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        return pa.LoadSim(str(basedir))


def test_metadata(sim, athenapp_run):
    _, e = athenapp_run
    assert sim.athena_variant == 'athena++'
    assert sim.problem_id == e['pid']
    assert sim.ff.out_fmt == ['hst', 'vtk']
    np.testing.assert_array_equal(sim.domain['Nx'], e['nx'])
    assert sim.u.length.to('pc').value == pytest.approx(1.0, rel=1e-12)


def test_nums_fall_back_to_vtk(sim, athenapp_run):
    _, e = athenapp_run
    assert sim.ff.nums_vtk == e['nums']
    assert sim.nums == e['nums']


@pytest.mark.parametrize('num', [0, 1])
def test_load_vtk(sim, athenapp_run, num):
    _, e = athenapp_run
    dat = sim.load_vtk(num=num)
    ref = e['fields'][num]
    nb = e['nx'][0]//e['nblock']
    assert dat['rho'].shape[0] == e['nblock']

    # Blocks may come back in any file order; order them by left x face.
    order = np.argsort(dat['x1f'][:, 0])
    for ib, b in enumerate(order):
        sl = np.s_[:, :, ib*nb:(ib + 1)*nb]
        np.testing.assert_array_equal(dat['rho'][b], ref['rho'][sl])
        np.testing.assert_array_equal(dat['press'][b], ref['press'][sl])
        np.testing.assert_array_equal(dat['vel'][b], ref['vel'][sl])
        x1f = e['le'][0] + e['dx'][0]*np.arange(ib*nb, (ib + 1)*nb + 1)
        np.testing.assert_allclose(dat['x1f'][b], x1f)
