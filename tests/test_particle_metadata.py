import struct
import tempfile
import unittest
from pathlib import Path

from pyathena.io.read_particles import read_parbin, read_partab


class TestParticleMetadata(unittest.TestCase):
    def test_empty_binary_preserves_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'sample.parbin')
            with path.open('wb') as stream:
                stream.write(b'# Athena++ particle output\n# pid,local_id,mass,x1,x2,x3,\n')
                stream.write(struct.pack('<4i', 2, 4, 1, 0))
                stream.write(struct.pack('<12d', *([0.]*3+[1.]*9)))
                stream.write(struct.pack('<4i2di', 8, 8, 8, 0, .125, .01, 12))
                stream.write(struct.pack('<2i', 0, 0))
            header = read_parbin(path, header_only=True)
            frame = read_parbin(path)
            self.assertTrue(frame.empty)
            self.assertEqual(frame.attrs, header)
            self.assertEqual(header['time'], .125)
            self.assertEqual(header['cycle'], 12)

    def test_table_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'sample.tab')
            path.write_text('# Athena++ particle data at time = 1.250000000000000000e-01\n')
            self.assertEqual(read_partab(path, header_only=True), {'time': .125})


if __name__ == '__main__':
    unittest.main()
