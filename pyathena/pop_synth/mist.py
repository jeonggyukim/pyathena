import re
import os
from pathlib import Path
import pandas as pd
import numpy as np
import tarfile

from .read_mist_models import EEP
from .downloader import downloader

def is_valid_txz(fname):
    try:
        with tarfile.open(fname, "r:xz") as tar:
            tar.getmembers()
        return True
    except (tarfile.TarError, ValueError, OSError, EOFError) as e:
        print(f"Invalid .txz file: {e}")
        return False

class PopSynthMIST(object):
    urls = {
        'EEPS_v1.2_vvcrit0.4_feh_m4.00_afe_p0.0':
        'https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_feh_m4.00_afe_p0.0_vvcrit0.4_EEPS.txz',
        'EEPS_v1.2_vvcrit0.4_feh_m3.50_afe_p0.0':
        'https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_feh_m3.50_afe_p0.0_vvcrit0.4_EEPS.txz',
        'EEPS_v1.2_vvcrit0.4_feh_m3.00_afe_p0.0':
        'https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_feh_m3.00_afe_p0.0_vvcrit0.4_EEPS.txz'
    }

    def __init__(self, rootdir=None, model='feh_m3.00_afe_p0.0_vvcrit0.4', force_download=False):
        if rootdir is None:
            rootdir = Path(__file__).parent.absolute() / '../../data/pop_synth/mist'
            if not rootdir.is_dir():
                rootdir.mkdir(parents=True, exist_ok=True)

        self.rootdir = Path(rootdir)

        for k, url in self.urls.items():
            message = f'Downloading {k} from MESA Isochrones & Stellar Tracks.\n'\
                'https://waps.cfa.harvard.edu/MIST/model_grids.html'
            fname = self.rootdir / url.split('/')[-1]
#            if fname.with_suffix('').exists():

            if not force_download:
                if not fname.with_suffix('').exists():
                    force_download = True
                elif not is_valid_txz(fname):
                    force_download = True

            downloader(fname, url, message, force_download=force_download)
            if not fname.with_suffix('').exists() or force_download:
                print(f'Extracting {fname.name}')
                self.extract_one(fname, self.rootdir, delete_txz=False)

        self._find_models_and_dirs()
        print(self.models)
        self.set_model(model)

    @staticmethod
    def extract_one(fname, extractdir, delete_txz=False):
        """
        Unzips a single ZIP file.
        """
        # Ensure output directory exists
        os.makedirs(extractdir, exist_ok=True)
        if not tarfile.is_tarfile(fname):
            raise IOError(f'{fname} is not a valid txz file. '
                          'Try again with `force_download=True`')
        with tarfile.open(fname, 'r:xz') as tar:
            tar.extractall(path=extractdir)

        if delete_txz and fname.exists():
            print(fname)
            fname.unlink()

        return extractdir

    def set_model(self, model):
        if model not in self.models:
            raise ValueError('model {0:s} not found.'.format(model))
        self.model = model
        self.df = self.dfa[model]

    def read_track(self, M, model=None, as_pandas=True):
        if model is not None:
            self.set_model(model)

        d = self.df.loc[(self.df['M'] - M).abs().idxmin()]
        eep = EEP(d['fname'].as_posix(), verbose=True)
        if as_pandas:
            return pd.DataFrame(eep.eeps)
        else:
            return eep

    def _find_models_and_dirs(self):

        files = []
        self.models = []
        self.files = dict()
        self.M = dict()
        self.dfa = dict()
        for dd in self.rootdir.iterdir():
            if dd.is_dir():
                mdl = re.search(r'feh_.*?vvcrit\d+\.\d+', dd.name).group()
                self.models.append(mdl)
                files = []
                for d in dd.iterdir():
                    if d.name.endswith('track.eep'):
                        files.append(d)

                self.files[mdl] = sorted(files)
                self.M[mdl] = np.array([float(f.name.split('.')[0][:-1])/1e2 \
                                        for f in self.files[mdl]])

        for mdl in self.models:
            self.dfa[mdl] = pd.DataFrame({'M': self.M[mdl],
                                          'fname': self.files[mdl]})
