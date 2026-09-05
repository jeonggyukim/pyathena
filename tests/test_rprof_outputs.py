import logging
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pyathena.find_files import FindFiles
from pyathena.load_sim import LoadSim
from pyathena.io.read_radial_profile import read_radial_profile
from test_read_radial_profile import _write_fixture


class TestRadialProfileOutputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def discover(self):
        finder = FindFiles.__new__(FindFiles)
        finder.basedir = str(self.root)
        finder.problem_id = 'test'
        finder.out_fmt = ['rprof']
        finder.files = {}
        finder.logger = logging.getLogger('test')
        finder.find_rprof()
        sim = LoadSim.__new__(LoadSim)
        sim._files = finder.files
        sim.nums_rprof = finder.nums_rprof
        sim.nums = [99]
        return sim

    def write(self, num, time, stream='rprof', directory=None):
        path = (directory or self.root)/f'test.{stream}.{num:05d}.rprof'
        _write_fixture(path)
        with path.open('r+b') as handle:
            handle.seek(32)
            handle.write(struct.pack('<Q', num))
            handle.seek(48)
            handle.write(struct.pack('<d', time))
        return path

    def test_discovery_does_not_read_headers(self):
        self.write(7, 1.25)
        self.write(8, 1.35)
        with patch('pyathena.load_sim.read_radial_profile', side_effect=AssertionError('header read')):
            sim = self.discover()
        self.assertEqual(sim.nums_rprof, [7, 8])
        self.assertEqual(sim.nums, [99])
        self.assertFalse(hasattr(sim, 'rprof_outputs'))

    def test_selected_load_and_explicit_rediscovery(self):
        self.write(7, 1.25)
        sim = self.discover()
        self.write(8, 1.35)
        with patch.object(FindFiles, 'find_match', side_effect=AssertionError('rescan')):
            ds = sim.load_rprof(7, center_ids=[29])
            self.assertEqual(ds.rho.shape, (1, 3))
            with self.assertRaises(FileNotFoundError):
                sim.load_rprof(8)
        self.assertEqual(self.discover().nums_rprof, [7, 8])

    def test_metadata_skips_payload(self):
        path = self.write(7, 1.25)
        with path.open('r+b') as handle:
            handle.seek(-8, 2)
            handle.write(struct.pack('<d', np.nan))
        ds = self.discover().load_rprof(7, metadata_only=True)
        self.assertEqual(len(ds.data_vars), 0)
        np.testing.assert_array_equal(ds.center_id, [11, 29])
        with self.assertRaisesRegex(ValueError, 'Nonfinite|non-finite'):
            read_radial_profile(path)

    def test_gap_is_for_analysis_to_validate(self):
        self.write(7, 1.25)
        self.write(9, 1.45)
        self.assertEqual(self.discover().nums_rprof, [7, 9])

    def test_ambiguous_streams(self):
        self.write(7, 1.25)
        self.write(8, 1.35, stream='other')
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            self.discover()

    def test_subdirectory_precedence(self):
        self.write(7, 1.25)
        directory = self.root/'rprof'
        directory.mkdir()
        self.write(8, 1.35, stream='custom', directory=directory)
        self.assertEqual(self.discover().nums_rprof, [8])

    def test_empty_and_invalid_filename(self):
        sim = self.discover()
        with self.assertRaises(FileNotFoundError):
            sim.load_rprof(0)
        (self.root/'test.invalid.rprof').touch()
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            self.discover()


if __name__ == '__main__':
    unittest.main()
