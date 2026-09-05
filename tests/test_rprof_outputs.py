import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pyathena.load_sim import LoadSim
from pyathena.io.read_radial_profile import read_radial_profile
from test_read_radial_profile import _write_fixture


class TestRadialProfileOutputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sim = LoadSim.__new__(LoadSim)
        self.sim._basedir = str(self.root)
        self.sim._problem_id = 'test'
        self.sim.nums = [99]

    def tearDown(self):
        self.temp.cleanup()

    def write(self, num, time, stream='rprof'):
        path = self.root / f'test.{stream}.{num:05d}.rprof'
        _write_fixture(path)
        with path.open('r+b') as handle:
            handle.seek(32)
            handle.write(struct.pack('<Q', num))
            handle.seek(48)
            handle.write(struct.pack('<d', time))
        return path

    def test_discovery_selection_and_refresh(self):
        self.write(7, 1.25)
        self.write(8, 1.35)
        self.assertEqual(list(self.sim.rprof_outputs.index), [7, 8])
        self.assertEqual(self.sim.nums, [99])
        ds = self.sim.load_rprof(8, center_ids=[29])
        self.assertEqual(ds.rho.shape, (1, 3))
        self.write(9, 1.45001)  # No interval-uniformity requirement.
        self.assertEqual(list(self.sim.rprof_outputs.index), [7, 8, 9])

    def test_metadata_skips_variable_payload(self):
        path = self.write(7, 1.25)
        # Nonfinite field payload is rejected by a full read but never read by
        # metadata-only discovery.
        with path.open('r+b') as handle:
            handle.seek(-8, 2)
            handle.write(struct.pack('<d', np.nan))
        ds = read_radial_profile(path, metadata_only=True)
        self.assertEqual(len(ds.data_vars), 0)
        np.testing.assert_array_equal(ds.center_id, [11, 29])
        with self.assertRaisesRegex(ValueError, 'Nonfinite|non-finite'):
            read_radial_profile(path)

    def test_gap(self):
        self.write(7, 1.25)
        self.write(9, 1.45)
        with self.assertRaisesRegex(ValueError, 'Gap'):
            _ = self.sim.rprof_outputs

    def test_duplicate_and_stream_ambiguity(self):
        self.write(7, 1.25)
        self.write(7, 1.25, stream='other')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            _ = self.sim.rprof_outputs

    def test_header_number_mismatch(self):
        path = self.write(7, 1.25)
        path.rename(self.root / 'test.rprof.00008.rprof')
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            _ = self.sim.rprof_outputs


if __name__ == '__main__':
    unittest.main()
