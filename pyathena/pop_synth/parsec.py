import os
import bs4
import requests
import zipfile
from pathlib import Path
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm

class PopSynthParsec(object):

    def __init__(self, rootdir=None, model_def='Z0.014', force_download=False):
        if rootdir is None:
            rootdir = Path(__file__).parent.absolute() / '../../data/pop_synth'
            if not rootdir.is_dir():
                rootdir.mkdir(parents=True, exist_ok=True)

        self.rootdir = Path(rootdir)
        self.dirs = dict()
        self.force_download = force_download
        self.url = {
            'photons': 'https://stev.oapd.inaf.it/PARSEC/Database/PARSECv2.0_VMS/all_photons.zip',
            'ejecta': 'https://stev.oapd.inaf.it/PARSEC/Database/PARSECv2.0_VMS/all_ejecta.zip',
            'tracks': 'https://stev.oapd.inaf.it/PARSEC/Database/PARSECv2.0_VMS/all_tracks.zip'
        }
        self.files_zip = {
            'photons': self.rootdir / 'all_photons.zip',
            'ejecta': self.rootdir / 'all_ejecta.zip',
            'tracks': self.rootdir / 'all_tracks.zip'
        }

        # Download files and unzip
        for k in self.files_zip.keys():
            self.download_file(k, force_download=self.force_download)
            self.dirs[k] = osp.splitext(self.files_zip[k])[0]
            if not osp.exists(self.dirs[k]) or force_download:
                self.unzip_one(self.files_zip[k], self.dirs[k])
                self.unzip_all(self.dirs[k])

        # self.files = self.get_all_files(self.dirs['photons'])
        self._find_models_and_dirs()

        # self.M = dict()
        # self.files = dict()
        # for mdl in self.models:
        #     self.get_M_and_files(mdl)

        # self.dfa = dict()
        # for mdl in self.models:
        #     self.dfa[mdl] = pd.DataFrame({'M': self.M[mdl], 'fname': self.files[mdl]})

        # self.set_model(model_def)

    def set_model(self, model):
        if model not in self.models:
            raise ValueError('model {0:s} not found.'.format(model))
        self.model = model
        self.df = dict()
        self.df['photons'] = self.dfa['photons'][model]
        self.df['tracks'] = self.dfa['tracks'][model]

    def download_file(self, kind, force_download=False):
        fname = self.files_zip[kind]
        url = self.url[kind]
        if not osp.exists(fname) or force_download:
            print('Downloading {0:s} tables from PAdova TRieste Stellar Evolutionary Code.'.\
                  format(kind))
            print('https://stev.oapd.inaf.it/PARSEC/tracks_v2_VMS.html')

            response = requests.get(url, stream=True)
            if not response.ok:
                raise Exception('Failed to download file. Check URL.')

            total_size = int(response.headers.get('content-length', 0))  # Get file size
            block_size = 1024

            with open(fname, 'wb') as f, tqdm(
                desc='Downloading',
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024  # Convert to KB/MB
            ) as bar:
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    bar.update(len(chunk))  # Update progress bar

    @staticmethod
    def unzip_one(fname, extractdir=None, delete_zip=False):
        """
        Unzips a single ZIP file.
        """
        if extractdir is None:
            # Extract to a folder with the ZIP file's name
            extractdir = osp.splitext(fname)[0]

        os.makedirs(extractdir, exist_ok=True)  # Ensure output directory exists
        with zipfile.ZipFile(fname, 'r') as zip_ref:
            zip_ref.extractall(extractdir)
            # if verbose:
            #print(f'Extracted: {fname} -> {extractdir}')

        if delete_zip:
            os.remove(fname)

        return extractdir

    def unzip_all(self, dirname, delete_zip=True):
        for root, _, files in os.walk(dirname):
            for f in files:
                if f.endswith('.zip'):
                    fname = osp.join(root, f)
                    extractdir = osp.join(root, f.replace('.zip', ''))
                    self.unzip_one(fname, extractdir, delete_zip)

    @staticmethod
    def get_all_files(dirname):
        """Returns a list of all file names
        """
        return [f for f in Path(dirname).rglob('*.QH') if f.is_file()]

    def _find_models_and_dirs(self):
        self.dfa = dict()
        self.M = dict()
        self.files = dict()
        self.models = dict()

        for k in ['photons', 'tracks']:
            self.dfa[k] = dict()
            self.M[k] = dict()
            self.files[k] = dict()
            self.models[k] = []
            self.dirs['models_' + k] = dict()
            for d in sorted(Path(self.dirs[k]).iterdir()):
                if d.is_dir():
                    mdl = d.name.split('_', 1)[0]
                    if 'D-' in mdl:
                        dnew = d.with_name(d.name.replace('D-', 'E-'))
                        d.rename(dnew)
                        mdl = mdl.replace('D-', 'E-')

                    self.models[k].append(mdl)
                    self.dirs['models_' + k][mdl] = d

                    files = self.get_all_files(self.dirs['models_' + k][mdl])
                    M = []
                    for f in files:
                        name = f.stem
                        M_ = float(name.split('_')[2].removesuffix('.TAB').removeprefix('M'))
                        M.append(M_)

                    sidx = np.argsort(M)
                    self.M[k][mdl] = np.array(M)[sidx]
                    self.files[k][mdl] = np.array(files)[sidx]

        if self.models['photons'] == self.models['tracks']:
            self.models = self.models['photons']

        for k in ['photons', 'tracks']:
            for mdl in self.models:
                self.dfa[k][mdl] = pd.DataFrame({'M': self.M[k][mdl],
                                                 'fname': self.files[k][mdl]})


    # def get_M_and_files(self, mdl):
    #     files = self.get_all_files(self.dirs['models'][mdl])
    #     M = []
    #     for f in files:
    #         name = f.stem
    #         M_ = float(name.split('_')[2].removesuffix('.TAB').removeprefix('M'))
    #         M.append(M_)

    #     sidx = np.argsort(M)
    #     self.M[mdl] = np.array(M)[sidx]
    #     self.files[mdl] = np.array(files)[sidx]

    def read_track(self, M, model=None):
        if model is not None:
            self.set_model(model)

        d = self.df['photons'].loc[(self.df['photons']['M'] - M).abs().idxmin()]
        return self.read_photon_file(d['fname'])

    def read_photon(self, M, model=None):
        if model is not None:
            self.set_model(model)

        d = self.df['photons'].loc[(self.df['photons']['M'] - M).abs().idxmin()]
        return self.read_photon_file(d['fname'])

    @staticmethod
    def read_photon_file(fname):
        with open(fname, 'r') as f:
            # Read lines before the table
            lines = []
            while True:
                l = f.readline().strip()
                if not l or l.startswith('#'):
                    lines.append(l)
                else:
                    header = l.strip()
                    break

            # Process metadata
            lines = lines[1:]
            metadata = {}
            for l in lines:
                if l.startswith('#'):
                    l = l.strip('# ').strip()  # Remove '#' and leading spaces
                    parts = l.split(',')
                    for part in parts:
                        k, v = part.split('=')
                        metadata[k.strip()] = v.strip()

            columns = header.split()
            df = pd.read_csv(f, sep=r'\s+', header=None, names=columns)

        df.attrs = metadata
        return df
