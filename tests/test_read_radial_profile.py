import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pyathena.io.read_radial_profile import read_radial_profile


def _write_fixture(path):
    ncenter, nr, nvariable = 2, 3, 2
    header_size = 148 + 56*nvariable
    id_offset = header_size
    position_offset = id_offset + 8*ncenter
    radius_offset = position_offset + 24*ncenter
    rho_offset = radius_offset + 8*nr
    velocity_offset = rho_offset + 8*ncenter*nr
    with path.open("wb") as stream:
        stream.write(struct.pack(
            "<8sIIQQQQdIII6dddIIII",
            b"ATHRPRF\0", 1, 0x01020304, 0, header_size, 7, 42, 1.25,
            8, 8, 8, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
            0.25, 0.5, 4, ncenter, nr, nvariable,
        ))
        stream.write(struct.pack(
            "<32sIIQQ", b"rho", 1, 1, ncenter*nr, rho_offset
        ))
        stream.write(struct.pack(
            "<32sIIQQ", b"velx_origin", 0, 1, ncenter, velocity_offset
        ))
        stream.write(struct.pack("<2Q", 11, 29))
        stream.write(struct.pack("<6d", -0.5, 0.0, 0.5, 0.5, 0.0, -0.5))
        stream.write(struct.pack("<3d", 0.0, 0.25, 0.5))
        stream.write(struct.pack("<6d", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        stream.write(struct.pack("<2d", -2.0, 3.0))


class TestReadRadialProfile(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name)/"sample.rprof"
        _write_fixture(self.path)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_read_and_select_centers(self):
        dataset = read_radial_profile(self.path, center_ids=[29, 11])
        np.testing.assert_array_equal(dataset.center_id, [29, 11])
        np.testing.assert_allclose(
            dataset.rho, [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]
        )
        np.testing.assert_allclose(dataset.velx_origin, [3.0, -2.0])
        np.testing.assert_allclose(dataset.r, [0.0, 0.25, 0.5])
        self.assertEqual(dataset.attrs["num"], 7)
        self.assertEqual(dataset.attrs["cycle"], 42)
        self.assertEqual(dataset.attrs["time"], 1.25)

    def test_missing_center_is_precise(self):
        with self.assertRaisesRegex(KeyError, "99"):
            read_radial_profile(self.path, center_ids=[99])

    def test_rejects_malformed_files(self):
        original = self.path.read_bytes()
        for mutation in ("magic", "version", "truncate"):
            with self.subTest(mutation=mutation):
                data = bytearray(original)
                if mutation == "magic":
                    data[0] = 0
                elif mutation == "version":
                    data[8:12] = struct.pack("<I", 2)
                else:
                    del data[-4:]
                self.path.write_bytes(data)
                with self.assertRaises(ValueError):
                    read_radial_profile(self.path)


if __name__ == "__main__":
    unittest.main()
