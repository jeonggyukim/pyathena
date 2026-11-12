import pytest

from .fake_runs import make_athena_run, make_athenapp_run


@pytest.fixture(scope='module')
def athena_run(tmp_path_factory):
    basedir = tmp_path_factory.mktemp('athena') / 'run'
    basedir.mkdir()
    expected = make_athena_run(basedir)
    return basedir, expected


@pytest.fixture(scope='module')
def athenapp_run(tmp_path_factory):
    basedir = tmp_path_factory.mktemp('athenapp') / 'run'
    expected = make_athenapp_run(basedir)
    return basedir, expected
