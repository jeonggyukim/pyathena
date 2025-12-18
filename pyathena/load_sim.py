import os
import sys
import glob
import re
import getpass
import warnings
import functools
import pickle
try:
    import yt
except ModuleNotFoundError:
    pass
import tarfile
import shutil
import dateutil

import numpy as np
import pandas as pd
import xarray as xr
import os.path as osp

from inherit_docstring import inherit_docstring
from abc import ABC
from collections.abc import Mapping

from .find_files import FindFiles
from .logger import create_logger, _verbose_to_level

from .classic.vtk_reader import AthenaDataSet as AthenaDataSetClassic
from .io.read_vtk import AthenaDataSet, read_vtk_athenapp
from .io.read_vtk_tar import AthenaDataSetTar
from .io.read_hdf5 import read_hdf5
from .io.read_particles import read_partab, read_parbin, read_parhst
from .io.athenak import read_particle_vtk
from .io.read_rst import read_rst
from .io.read_starpar_vtk import read_starpar_vtk
from .io.read_zprof import read_zprof_all
from .io.read_athinput import read_athinput
from .io.athena_read import athinput
from .util.units import Units
from .fields.fields import DerivedFields
from .plt_tools.make_movie import make_movie


class LoadSimBase(ABC):
    """Common properties to all LoadSim classes

    Parameters
    ----------

    Attributes
    ----------
    basedir : str
        Directory where simulation output files are stored.
    savdir : str
        Directory where pickles and figures are saved.
    basename : str
        basename (last component) of `basedir`.
    load_method : str
        Load vtk/hdf5 snapshots using 'xarray', 'pythena_classic' (vtk only),
        or 'yt'. Defaults to 'xarray'.
    athena_variant : str
        [athena, athena++, athenak]
    problem_id : str
        Prefix for output files.
    domain : dict
        Domain information such as box size and number of cells.
    files : dict
        Dictionary containing output file paths.
    par : dict
        Dictionary of dictionaries containing input parameters and configure
        options read from log fileoutput file names.
    config_time : pandas.Timestamp
        Date and time when the athena code is configured.
    verbose : bool or str or int
        If True/False, set logging level to 'INFO'/'WARNING'.
        Otherwise, one of valid logging levels
        ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        or their numerical values (0, 10, 20, 30, 40, 50).
        (see https://docs.python.org/3/library/logging.html#logging-levels)
    """

    # TODO: Can use pathlib.Path but there is a backward compatibility risk.
    @property
    def basedir(self):
        return self._basedir

    @property
    def basename(self):
        return self._basename

    @property
    def savdir(self):
        return self._savdir

    @savdir.setter
    def savdir(self, value):
        if value is None:
            self._savdir = self._basedir
        else:
            self._savdir = value

    @property
    def load_method(self):
        return self._load_method

    @load_method.setter
    def load_method(self, value):
        if value in ['xarray', 'pyathena_classic', 'yt']:
            self._load_method = value
        else:
            raise ValueError('Unrecognized load_method: ', value)

    @property
    def problem_id(self):
        return self._problem_id

    @property
    def athena_variant(self):
        return self._athena_variant

    @property
    def domain(self):
        return self._domain

    @property
    def config_time(self):
        return self._config_time

    @property
    def par(self):
        return self._par

    @property
    def files(self):
        return self._files

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        if hasattr(self, 'logger'):
            self.logger.setLevel(_verbose_to_level(value))
        if hasattr(self, 'ff'):
            if hasattr(self, 'logger'):
                self.ff.logger.setLevel(_verbose_to_level(value))

        self._verbose = value

@inherit_docstring
class LoadSim(LoadSimBase):
    """Class to prepare Athena simulation data analysis. Read input parameters
    and find simulation output files.

    Parameters
    ----------
    basedir : str
        Directory where simulation output files are stored.
    savdir : str, optional
        Directory where pickles and figures are saved. Defaults to `basedir`.
    load_method : {'xarray', 'pyathena_classic', 'yt'}, optional
        Load vtk/hdf5 snapshots using 'xarray', 'pythena_classic', or 'yt'.
        Defaults to 'xarray'.
    verbose : bool or str or int
        If True/False, set logging level to 'INFO'/'WARNING'.
        Otherwise, one of valid logging levels
        ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        or their numerical values (0, 10, 20, 30, 40, 50).
        (see https://docs.python.org/3/library/logging.html#logging-levels)

    Attributes
    ----------
    ds : AthenaDataSet or yt DataSet
        Class for reading vtk file
    nums : list of int
        vtk/hdf5 output numbers
    u : Units object
        Simulation unit
    dfi : dict
        Derived field information

    Methods
    -------
    load_vtk() :
        reads vtk file using pythena or yt and returns DataSet object
    load_starpar_vtk() :
        reads starpar vtk file and returns pandas DataFrame object
    print_all_properties() :
        prints all attributes and callable methods

    Examples
    --------
    >>> import pyathena as pa
    >>> s = pa.LoadSim('/path/to/basedir", verbose=True)
    """

    def __init__(self, basedir, savdir=None, load_method='xarray',
                 units=Units(kind='LV', muH=1.4271),
                 verbose=False):

        self.verbose = verbose
        self.logger = create_logger(self.__class__.__name__.split('.')[-1],
                                    verbose)
        self._basedir = basedir.rstrip('/')
        self._basename = osp.basename(self.basedir)
        self.savdir = savdir
        self.load_method = load_method

        self.logger.info('basedir: {0:s}'.format(self.basedir))
        self.logger.info('savdir: {:s}'.format(self.savdir))
        self.logger.info('load_method: {:s}'.format(self.load_method))

        self.find_files(verbose)

        # Set metadata
        self._get_domain_from_par(self.par)
        try:
            if self.athena_variant == 'athena++':
                k = 'Configure_date'
            elif self.athena_variant == 'athena':
                k = 'config_date'
            elif self.athena_variant == 'athenak':
                # TODO currently AthenaK does not print configure date
                pass
            tmp = self.par['configure'][k]
            for tz in ['PDT', 'EST', 'UTC']:
                tmp = tmp.replace(tz, '').strip()

            self._config_time = pd.to_datetime(dateutil.parser.parse(tmp))
        except:
            self._config_time = None

        # Set units and derived field infomation
        if self.athena_variant == 'athena':
            try:
                muH = self.par['problem']['muH']
                self.u = Units(kind='LV', muH=muH)
            except KeyError:
                try:
                    # Some old simulations run with new cooling may not have muH
                    # parameter printed out
                    if self.par['problem']['Z_gas'] != 1.0:
                        self.logger.warning(
                            'Z_gas={0:g} but muH is not found in par. '.\
                            format(self.par['problem']['Z_gas']) +
                            'Caution with muH={0:s}'.format(muH))
                    self.u = units
                except:
                    self.u = units
                    pass

            # TODO(SMOON) Make DerivedFields work with athena++
            self.dfi = DerivedFields(self.par).dfi
        elif self.athena_variant == 'athena++':
            self.u = Units(kind='custom', units_dict=self.par['units'])

    def find_files(self, verbose=None):
        """Find output files under base directory and update the `files`
        attribute.

        Parameters
        ----------
        verbose : bool or str or int
            If True/False, set logging level to 'INFO'/'WARNING'.
            Otherwise, one of valid logging levels
            ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            or their numerical values (0, 10, 20, 30, 40, 50).
            (see https://docs.python.org/3/library/logging.html#logging-levels)
        """
        if verbose is None:
            verbose = self.verbose
        try:
            self.ff = FindFiles(self.basedir, verbose)
        except OSError as e:
            raise OSError(e)

        # Transfer attributes of FindFiles to LoadSim
        # TODO: Some of these attributes don't need to be transferred.
        attrs_transfer = [
            'files', 'athena_variant', 'par', 'problem_id', 'out_fmt',
            'nums',
            # particle (Athena++, AthenaK)
            'nums_partab', 'nums_parbin', 'partags', 'pids',
            'partab_outid', 'parbin_outid', 'pvtk_outvar',
            # hdf5 (Athena++)
            'nums_hdf5', 'hdf5_outid', 'hdf5_outvar', '_hdf5_outid_def',
            '_hdf5_outvar_def',
            # zprof
            'nums_zprof', 'phase',
            # rst
            'nums_rst',
            # starpar (Athena)
            'nums_starpar',  'nums_sphst',
            # vtk
            'nums_vtk', 'nums_id0', 'nums_tar', 'nums_vtk',
            '_fmt_vtk2d_not_found']
        for attr in attrs_transfer:
            if hasattr(self.ff, attr):
                if attr in ['files', 'par', 'athena_variant', 'problem_id']:
                    setattr(self, '_' + attr, getattr(self.ff, attr))
                else:
                    setattr(self, attr, getattr(self.ff, attr))
            else:
                pass

    def load_vtk(self, num=None, ivtk=None, id0=True, load_method=None):
        """Function to read Athena vtk file using pythena or yt and
        return DataSet object.

        Parameters
        ----------
        num : int
           Snapshot number, e.g., /basedir/vtk/problem_id.xxxx.vtk
        ivtk : int
           Read i-th file in the vtk file list. Overrides num if both are given.
        id0 : bool
           Read vtk file in /basedir/id0. Default value is True.
        load_method : str
           'xarray', 'pyathena_classic' or 'yt'

        Returns
        -------
        ds : AthenaDataSet or yt datasets
        """

        if num is None and ivtk is None:
            raise ValueError('Specify either num or ivtk')

        # Override load_method
        if load_method is not None:
            self.load_method = load_method

        if self.athena_variant == 'athena++':
            def filter_vtk_files(kind='vtk', num=None):
                def func(num):
                    return lambda fname: '.{0:05d}.vtk'.format(num) in fname

                return list(filter(func(num), self.files[kind]))

            fnames = filter_vtk_files('vtk', num)
            return read_vtk_athenapp(fnames)

        if not 'vtk_id0' in self.files.keys():
            id0 = False

        if id0:
            kind = ['vtk_id0', 'vtk', 'vtk_tar']
        else:
            if 'vtk' in self.files.keys():
                kind = ['vtk', 'vtk_tar', 'vtk_id0']
            else:
                kind = ['vtk_tar', 'vtk', 'vtk_id0']

        self.fname = self._get_filename(kind[0], num, ivtk)
        if self.fname is None or not osp.exists(self.fname):
            if id0:
                self.logger.info('[load_vtk]: Vtk file does not exist. ' + \
                                 'Try joined/tarred vtk')
            else:
                self.logger.info('[load_vtk]: Vtk file does not exist. ' + \
                                 'Try vtk in id0')

            for kind_ in (kind[1], kind[2]):
                # Check if joined vtk (or vtk in id0) exists
                self.fname = self._get_filename(kind_, num, ivtk)
                if self.fname is None or not osp.exists(self.fname):
                    self.logger.info('[load_vtk]: Vtk file does not exist.')
                else:
                    self.logger.info('[load_vtk]: Found vtk file in "{}"\
                                     '.format(kind_))
                    break

        if self.fname.endswith('vtk'):
            if self.load_method == 'xarray':
                self.ds = AthenaDataSet(self.fname, units=self.u, dfi=self.dfi)
                self._domain = self.ds.domain
                self.logger.info('[load_vtk]: {0:s}. Time: {1:f}'.format(\
                    osp.basename(self.fname), self.ds.domain['time']))

            elif self.load_method == 'pyathena_classic':
                self.ds = AthenaDataSetClassic(self.fname)
                self._domain = self.ds.domain
                self.logger.info('[load_vtk]: {0:s}. Time: {1:f}'.format(\
                    osp.basename(self.fname), self.ds.domain['time']))

            elif self.load_method == 'yt':
                if hasattr(self, 'u'):
                    units_override = self.u.units_override
                else:
                    units_override = None
                self.ds = yt.load(self.fname, units_override=units_override)
            else:
                self.logger.error('load_method "{0:s}" not recognized.'.format(
                    self.load_method) + \
                    ' Use either "yt", "pyathena", "pyathena_classic".')
        elif self.fname.endswith('tar'):
            if self.load_method == 'xarray':
                self.ds = AthenaDataSetTar(self.fname, units=self.u,
                                           dfi=self.dfi)
                self._domain = self.ds.domain
                self.logger.info('[load_vtk_tar]: {0:s}. Time: {1:f}'.format(\
                    osp.basename(self.fname), self.ds.domain['time']))
            elif self.load_method == 'yt':
                if hasattr(self, 'u'):
                    units_override = self.u.units_override
                else:
                    units_override = None
                self.ds = yt.load(self.fname, units_override=units_override)
            else:
                self.logger.error('load_method "{0:s}" not recognized.'.format(
                    self.load_method) + ' Use either "yt" or "pyathena".')

        return self.ds

    def load_hdf5(self, num=None, **kwargs):
        """Wrapper function to read Athena/AthenaK hdf5 files

        See documentation of `_load_hdf5_athenapp` and `_load_hdf5_atheanak` for details.

        Parameters
        ----------
        num : int
           Snapshot number
        """

        if self.athena_variant == 'athenak':
            ds = self._load_hdf5_athenak(num=num, **kwargs)
        elif self.athena_variant == 'athena++':
            ds = self._load_hdf5_athenapp(num=num, **kwargs)
        else:
            self.logger.error('Athena hdf5 reading not implemented yet for '
                              f'{self.athena_variant}.')
            ds = None

        return ds

    def load_partab(self, num=None, ipartab=None,
                    partag='par0', **kwargs):
        """Read Athena++ partab file.

        Parameters
        ----------
        num : int
           Snapshot number.
           e.g., /basedir/partab/problem_id.out?.num.par?.tab.
        ipartab : int
           Read i-th file in the partab file list.
           Overrides num if both are given.
        partag : int
           Particle id in the input file. Default value is 'par0'

        Returns
        -------
        pds : pandas.DataFrame
            Particle data
        """
        if num is None and ipartab is None:
            raise ValueError('Specify either num or ipartab')

        self.fpartab = self._get_fpartab(self.partab_outid, partag, num, ipartab)
        if self.fpartab is None or not osp.exists(self.fpartab):
            self.logger.info('[load_partab]: partab file does not exist. ')

        self.pds = read_partab(self.fpartab, **kwargs)

        return self.pds

    def load_parbin(self, num=None, iparbin=None,
                    partag='par0', **kwargs):
        """Read Athena++ parbin file.

        Parameters
        ----------
        num : int
           Snapshot number.
           e.g., /basedir/parbin/problem_id.out?.num.par?.parbin.
        iparbin : int
           Read i-th file in the parbin file list.
           Overrides num if both are given.
        partag : int
           Particle id in the input file. Default value is 'par0'

        Returns
        -------
        pds : pandas.DataFrame
            Particle data
        """
        if num is None and iparbin is None:
            raise ValueError('Specify either num or iparbin')

        self.fparbin = self._get_fparbin(self.parbin_outid, partag, num, iparbin)
        if self.fparbin is None or not osp.exists(self.fparbin):
            self.logger.info('[load_parbin]: parbin file does not exist. ')

        self.pds = read_parbin(self.fparbin, **kwargs)

        return self.pds

    def load_pvtk(self, num=None, fidx=None,
                  partag='par0', **kwargs):
        """Read AthenaK particle vtk file.

        Parameters
        ----------
        num : int
           Snapshot number.
           e.g., /basedir/pvtk/problem_id.outvar.num.part.vtk.
        fidx : int
           Read i-th file in the pvtk file list.
           Overrides num if both are given.
        partag : int
           Particle id in the input file. Default value is 'par0'

        Returns
        -------
        pds : dict
            Particle data
        """
        if num is None and fidx is None:
            raise ValueError('Specify either num or fidx')

        self.fpvtk = self._get_fpvtk(self.pvtk_outvar, partag, num, fidx)
        if self.fpvtk is None or not osp.exists(self.fpvtk):
            self.logger.info('[load_pvtk]: pvtk file does not exist. ')

        self.pds = read_particle_vtk(self.fpvtk, **kwargs)

        return self.pds

    def load_parhst(self, pid, **kwargs):
        """Read Athena++ individual particle history

        Parameters
        ----------
        pid : int
           Particle id, e.g., /basedir/parhst/problem_id.pid.csv

        Returns
        -------
        phst : pandas.DataFrame
            Individual particle history
        """

        self.fparhst = self._get_fparhst(pid)
        if self.fparhst is None or not osp.exists(self.fparhst):
            self.logger.info('[load_parhst]: parhst file does not exist. ')

        self.phst = read_parhst(self.fparhst, **kwargs)

        return self.phst

    def load_starpar_vtk(self, num=None, ivtk=None, force_override=False,
                         verbose=False):
        """Function to read Athena starpar_vtk file using pythena and
        return DataFrame object.

        Parameters
        ----------
        num : int
           Snapshot number, e.g., /basedir/starpar/problem_id.xxxx.starpar.vtk
        ivtk : int
           Read i-th file in the vtk file list. Overrides num if both are given.
        force_override : bool
           Flag to force read of starpar_vtk file even when pickle exists

        Returns
        -------
        sp : Pandas DataFrame object
        """

        if num is None and ivtk is None:
            raise ValueError('Specify either num or ivtk')

        # get starpar_vtk file name and check if it exist
        self.fstarvtk = self._get_filename('starpar_vtk', num, ivtk)
        if self.fstarvtk is None or not osp.exists(self.fstarvtk):
            self.logger.error('[load_starpar_vtk]: Starpar vtk file does not exist.')

        self.sp = read_starpar_vtk(self.fstarvtk,
                force_override=force_override, verbose=verbose)
        self.logger.info('[load_starpar_vtk]: {0:s}. Time: {1:f}'.format(\
                 osp.basename(self.fstarvtk), self.sp.time))

        return self.sp

    def load_rst(self, num=None, irst=None, verbose=False):
        if num is None and irst is None:
            raise ValueError('Specify either num or irst')

        # get starpar_vtk file name and check if it exist
        self.frst = self._get_filename('rst', num, irst)
        if self.frst is None or not osp.exists(self.frst):
            self.logger.error('[load_rst]: rst file does not exist.')

        self.rh = read_rst(self.frst, verbose=verbose)
        self.logger.info('[load_rst]: {0:s}. Time: {1:f}'.format(\
                 osp.basename(self.frst), self.rh.time))

        return self.rh

    def create_tar_all(self,remove_original=False,kind='vtk'):
        for num in self.nums_id0:
            self.move_to_tardir(num=num, kind=kind)
        raw_tardirs = self._find_match([(kind,"????")])
        for num in [int(f[-4:]) for f in raw_tardirs]:
            self.create_tar(num=num, remove_original=remove_original, kind=kind)

    def move_to_tardir(self, num=None, kind='vtk'):
        """Move vtk files from id* to vtk/XXXX

        Parameters
        ----------
        num : int
           Snapshot number, e.g., /basedir/vtk/xxxx

        """
        # set tar file name
        dirname = osp.join(self.basedir,kind)
        fpattern = '{0:s}.{1:04d}.tar'
        tarname = osp.join(dirname, fpattern.format(self.problem_id, num))
        tardir = os.path.join(dirname,'{0:04d}'.format(num))

        # move files to vtk/num/*.num.tar
        if osp.isdir(tardir):
            self.logger.info('[move_to_tardir] {:s} exists'.format(tardir))
            return

        # move files under id* to vtk/num
        # create folder
        # self.logger.info('[create_vtk_tar] create a folder {:s}'.format(tardir))
        os.makedirs(tardir)
        # find files
        if kind == 'vtk':
            id_files = [self._get_filename('vtk_id0',num=num)]
        elif kind == 'rst':
            id_files = [self._get_filename('rst',num=num)]
        id_files += self._find_match([('id*','{0:s}-id*.{1:04d}.{2:s}'.\
                                     format(self.problem_id, num, kind))])
        # move each file
        self.logger.info('[move_to_tardir] moving {:d} files to {:s}'.\
                         format(len(id_files),tardir))
        for f in id_files: shutil.move(f,tardir)

    def create_tar(self, num=None, remove_original=False, kind='vtk'):
        """Creating tarred vtk/rst from rearranged output

        Parameters
        ----------
        num : int
           Snapshot number, e.g., /basedir/vtk/xxxx
        remove_original : bool
           Remove original after tar it if True
        kind : string
           vtk or rst
        """
        # set tar file name
        dirname = osp.join(self.basedir,kind)
        fpattern = '{0:s}.{1:04d}.tar'
        tarname = osp.join(dirname, fpattern.format(self.problem_id, num))
        tardir = os.path.join(dirname,'{0:04d}'.format(num))

        # remove originals
        def remove_tardir():
            if osp.isdir(tardir) and remove_original:
                self.logger.info('[create_tar] removing originals'
                                 ' at {}'.format(tardir))
                try:
                    shutil.rmtree(tardir)
                except OSError as e:
                    print ("Error: %s - %s." % (e.filename, e.strerror))

        # check file existence
        if osp.isfile(tarname):
            # if tar file exists, remove original and quit
            self.logger.info('[create_tar] tar file already exists')
            remove_tardir()
            return

        # tar to vtk/problem_id.num.tar
        self.logger.info('[create_tar] tarring {:s}'.format(tardir))

        tf = tarfile.open(tarname,'x')
        tf.add(tardir)
        tf.close()

        # remove_original
        remove_tardir()

    def print_all_properties(self):
        """Print all attributes and callable methods
        """

        attr_list = list(self.__dict__.keys())
        print('Attributes:\n', attr_list)
        print('\nMethods:')
        method_list = []
        for func in sorted(dir(self)):
            if not func.startswith("__"):
                if callable(getattr(self, func)):
                    method_list.append(func)
                    print(func, end=': ')
                    print(getattr(self, func).__doc__)
                    print('-------------------------')

    def make_movie(self, fname_glob=None, fname_out=None, fps_in=10, fps_out=10,
                   force_override=False, display=False):

        if fname_glob is None:
            fname_glob = osp.join(self.basedir, 'snapshots', '*.png')
        if fname_out is None:
            fname_out = osp.join('/tigress/{0:s}/movies/{1:s}.mp4'.format(
                getpass.getuser(), self.basename))

        if force_override or not osp.exists(fname_out):
            self.logger.info('Make a movie from files: {0:s}'.format(fname_glob))
            make_movie(fname_glob, fname_out, fps_in, fps_out)
            self.logger.info('Movie saved to {0:s}'.format(fname_out))
        else:
            self.logger.info('File already exists: {0:s}'.format(fname_out))

    def _load_hdf5_athenapp(self, num=None, ihdf5=None,
                            outvar=None, outid=None, load_method=None,
                            file_only=False, **kwargs):
        """Function to read Athena hdf5 file using pythena or yt and
        return DataSet object.

        Parameters
        ----------
        num : int
           Snapshot number, e.g., /basedir/problem_id.out?.?????.athdf
        ihdf5 : int
           Read i-th file in the hdf5 file list. Overrides num if both are given.
        outvar : str
           Variable name, e.g, 'prim', 'cons', 'uov'. Default value is 'prim'
           or 'cons'. Overrides outid.
        outid : int
           output block number (output[n] in the input file).
        load_method : str
           'xarray' or 'yt'

        Returns
        -------
        ds : xarray AthenaDataSet or yt datasets

        Examples
        --------
        >>> from pyathena.load_sim import LoadSim
        >>> s = LoadSim("/path/to/basedir")
        >>> # Load everything at snapshot number 30.
        >>> ds = s.load_hdf5(30)
        >>> # Read the domain information only, without loading the fields.
        >>> ds = s.load_hdf5(30, header_only=True)
        >>> # Load the selected fields.
        >>> ds = s.load_hdf5(30, quantities=['dens', 'mom1', 'mom2', 'mom3'])
        >>> # Load the selected region.
        >>> ds = s.load_hdf5(30, x1_min=-0.5, x1_max=0.5, x2_min=1, x2_max=1.2)
        >>> # Load everything at fifth snapshot with ghost cells
        >>> num_ghost = s.par['configure']['Number_of_ghost_cells'] if \
                 s.par[f'output{s.hdf5_outid[0]}']['ghost_zones'] == 'true' else 0
        >>> ds = s.load_hdf5(ihdf5=5, num_ghost=num_ghost)
        """

        if num is None and ihdf5 is None:
            raise ValueError('Specify either num or ihdf5')

        # Override load_method
        if load_method is not None:
            self.load_method = load_method

        if outid is None and outvar is None:
            outid = self._hdf5_outid_def
            outvar = self._hdf5_outvar_def
        elif outid is not None:
            if not outid in self.hdf5_outid:
                self.logger.error('Invalid hdf5 output id!')
            idx = [i for i,v in enumerate(self.hdf5_outid) if v == outid][0]
            outvar = self.hdf5_outvar[idx]
        elif outvar is not None:
            if not outvar in self.hdf5_outvar:
                self.logger.error('Invalid hdf5 variable!')
            idx = [i for i,v in enumerate(self.hdf5_outvar) if v == outvar][0]
            outid = self.hdf5_outid[idx]

        self.fhdf5 = self._get_fhdf5(outid, outvar, num, ihdf5)
        if self.fhdf5 is None or not osp.exists(self.fhdf5):
            self.logger.info('[load_hdf5]: hdf5 file does not exist. ')

        if file_only:
            return

        if self.load_method == 'xarray':
            try:
                refinement = self.par['mesh']['refinement']
            except KeyError:
                # Cannot determine if refinement is turned on/off without reading the
                # HDF5 file and without <refinement> block in athinput.
                # This part needs to be improved later.
                refinement = 'none'

            if refinement != 'none':
                self.logger.error('load_method "{0:s}" does not support mesh\
                        refinement data. Use "yt" instead'.format(self.load_method))
                ds = None
            else:
                ds = read_hdf5(self.fhdf5, **kwargs)

        elif self.load_method == 'yt':
            if hasattr(self, 'u'):
                units_override = self.u.units_override
            else:
                units_override = None
            ds = yt.load(self.fhdf5, units_override=units_override)
        else:
            self.logger.error('load_method "{0:s}" not recognized.'.format(
                self.load_method) + ' Use either "xarray" or "yt".')

        return ds

    def _load_hdf5_athenak(self, num, **kwargs):
        num_output_vars = len(self.files['hdf5'].keys())
        if num_output_vars == 1:
            outvar = list(self.files['hdf5'].keys())[0]
            fpattern = '{0:s}.{1:s}.{2:05d}.athdf'
            dirname = osp.dirname(self.files['hdf5'][outvar][0])
            fhdf5 = osp.join(dirname, fpattern.format(
                             self.problem_id, outvar, num))
        else:
            self.logger.error('AthenaK hdf5 reading not implemented yet for\
                              multiple output variables.')
            return None
        ds = read_hdf5(fhdf5, **kwargs)
        return ds

    def _get_domain_from_par(self, par):
        """Get domain info from par['domain1']. Time is set to None.
        """
        domain = dict()
        if self.athena_variant in ['athena++', 'athenak']:
            d = par['mesh']
            domain['Nx'] = np.array([d['nx1'], d['nx2'], d['nx3']])
        elif self.athena_variant == 'athena':
            d = par['domain1']
            domain['Nx'] = np.array([d['Nx1'], d['Nx2'], d['Nx3']])
        domain['ndim'] = np.sum(domain['Nx'] > 1)
        domain['le'] = np.array([d['x1min'], d['x2min'], d['x3min']])
        domain['re'] = np.array([d['x1max'], d['x2max'], d['x3max']])
        domain['Lx'] = domain['re'] - domain['le']
        domain['dx'] = domain['Lx']/domain['Nx']
        domain['center'] = 0.5*(domain['le'] + domain['re'])
        domain['time'] = None

        self._domain = domain

    def find_files_vtk2d(self):
        self.logger.info('Find 2d vtk: {0:s}'.format(' '.join(self._fmt_vtk2d_not_found)))
        for fmt in self._fmt_vtk2d_not_found:
            fmt = fmt.split('.')[0]
            patterns = [('id*', '*.????.{0:s}.vtk'.format(fmt)),
                ('{0:s}'.format(fmt), '*.????.{0:s}.vtk'.format(fmt))]
            files = self._find_match(patterns)
            if files:
                self.files[f'{fmt}'] = files
                setattr(self, f'nums_{fmt}', [int(osp.basename(f).split('.')[1]) \
                                              for f in self.files[f'{fmt}']])
            else:
                self.logger.info('{0:s} files not found '.format(fmt))

    def _get_filename(self, kind, num=None, ivtk=None):
        """Get file path
        """

        # TODO: can be moved to FindFiles?
        try:
            dirname = osp.dirname(self.files[kind][0])
        except IndexError:
            return None
        if ivtk is not None:
            fname = self.files[kind][ivtk]
        else:
            if kind == 'starpar_vtk':
                fpattern = '{0:s}.{1:04d}.starpar.vtk'
            elif kind == 'rst':
                fpattern = '{0:s}.{1:04d}.rst'
            elif kind == 'rst_tar':
                fpattern = '{0:s}.{1:04d}.rst'
            elif kind == 'vtk_tar':
                fpattern = '{0:s}.{1:04d}.tar'
            else:
                fpattern = '{0:s}.{1:04d}.vtk'
            fname = osp.join(dirname, fpattern.format(self.problem_id, num))

        return fname

    def _get_fhdf5(self, outid, outvar, num=None, ihdf5=None):
        """Get hdf5 file path
        """

        try:
            dirname = osp.dirname(self.files['hdf5'][outvar][0])
        except IndexError:
            return None
        if ihdf5 is not None:
            fhdf5 = self.files['hdf5'][outvar][ihdf5]
        else:
            fpattern = '{0:s}.out{1:d}.{2:05d}.athdf'
            fhdf5 = osp.join(dirname, fpattern.format(
                self.problem_id, outid, num))

        return fhdf5

    def _get_fpartab(self, outid, partag, num=None, ipartab=None):
        """Get partab file path
        """

        try:
            dirname = osp.dirname(self.files['partab'][partag][0])
        except IndexError:
            return None
        if ipartab is not None:
            fpartab = self.files['partab'][partag][ipartab]
        else:
            fpattern = '{0:s}.out{1:d}.{2:05d}.{3:s}.tab'
            fpartab = osp.join(dirname, fpattern.format(
                self.problem_id, outid, num, partag))

        return fpartab

    def _get_fparbin(self, outid, partag, num=None, iparbin=None):
        """Get parbin file path
        """

        try:
            dirname = osp.dirname(self.files['parbin'][partag][0])
        except IndexError:
            return None
        if iparbin is not None:
            fparbin = self.files['parbin'][partag][iparbin]
        else:
            fpattern = '{0:s}.out{1:d}.{2:05d}.{3:s}.parbin'
            fparbin = osp.join(dirname, fpattern.format(
                self.problem_id, outid, num, partag))

        return fparbin

    def _get_fpvtk(self, outvar, partag, num=None, idx=None):
        """Get pvtk file path
        """

        try:
            dirname = osp.dirname(self.files['pvtk'][partag][0])
        except IndexError:
            return None
        if idx is not None:
            fname = self.files['pvtk'][partag][idx]
        else:
            fpattern = f'{self.problem_id}.{outvar}.{num:05d}.part.vtk'
            fname = osp.join(dirname, fpattern)
        return fname

    def _get_fparhst(self, pid):
        """Get parhst file path
        """

        try:
            dirname = osp.dirname(self.files['parhst'][0])
        except IndexError:
            return None
        fpattern = '{0:s}.par{1:d}.csv'
        fparhst = osp.join(dirname, fpattern.format(self.problem_id, pid))

        return fparhst

    class Decorators(object):
        """Class containing a collection of decorators for prompt reading of analysis
        output, (reprocessed) hst, and zprof. Used in child classes.

        """

        # JKIM: I'm sure there is a better way to achieve this, but this works
        # anyway..
        def check_pickle(read_func):
            @functools.wraps(read_func)
            def wrapper(cls, *args, **kwargs):

                # Convert positional args to keyword args
                from inspect import getcallargs
                call_args = getcallargs(read_func, cls, *args, **kwargs)
                call_args.pop('self')
                kwargs = call_args

                try:
                    prefix = kwargs['prefix']
                except KeyError:
                    prefix = '_'.join(read_func.__name__.split('_')[1:])

                if kwargs['savdir'] is not None:
                    savdir = kwargs['savdir']
                else:
                    savdir = osp.join(cls.savdir, prefix)

                if 'filebase' in kwargs and kwargs['filebase'] is not None:
                    filebase = kwargs['filebase']
                else:
                    filebase = prefix

                force_override = kwargs['force_override']

                # Create savdir if it doesn't exist
                try:
                    if not osp.exists(savdir):
                        force_override = True
                        os.makedirs(savdir)
                except FileExistsError:
                    print('Directory exists: {0:s}'.format(savdir))
                except PermissionError as e:
                    print('Permission Error: ', e)

                if 'num' in kwargs:
                    fpkl = osp.join(savdir, f'{filebase}_{kwargs["num"]:04d}.p')
                else:
                    fpkl = osp.join(savdir, f'{filebase}.p')

                if not force_override and osp.exists(fpkl):
                    cls.logger.info('Read from existing pickle: {0:s}'.format(fpkl))
                    with open(fpkl, 'rb') as fb:
                        res = pickle.load(fb)
                    return res
                else:
                    cls.logger.info('[check_pickle]: Read original dump.')
                    # If we are here, force_override is True or history file is updated.
                    res = read_func(cls, **kwargs)
                    try:
                        with open(fpkl, 'wb') as fb:
                            pickle.dump(res, fb)
                    except (IOError, PermissionError) as e:
                        cls.logger.warning('Could not pickle to {0:s}.'.format(fpkl))
                    return res

            return wrapper

        def check_netcdf(read_func):
            @functools.wraps(read_func)
            def wrapper(cls, *args, **kwargs):

                # Convert positional args to keyword args
                from inspect import getcallargs
                call_args = getcallargs(read_func, cls, *args, **kwargs)
                call_args.pop('self')
                call_args.pop('dryrun')
                kwargs = call_args

                try:
                    prefix = kwargs['prefix']
                except KeyError:
                    print("prefix must be provided")

                if kwargs['savdir'] is not None:
                    savdir = kwargs['savdir']
                else:
                    savdir = osp.join(cls.savdir, prefix)

                if kwargs['filebase'] is not None:
                    filebase = kwargs['filebase']
                else:
                    filebase = prefix

                if "outid" in kwargs:
                    if kwargs["outid"] is not None:
                        filebase += f'.out{kwargs["outid"]:d}'

                force_override = kwargs['force_override']
                mtime = read_func(cls,dryrun=True,**kwargs)

                # Create savdir if it doesn't exist
                try:
                    if not osp.exists(savdir):
                        force_override = True
                        os.makedirs(savdir)
                except FileExistsError:
                    print('Directory exists: {0:s}'.format(savdir))
                except PermissionError as e:
                    print('Permission Error: ', e)

                if 'num' in kwargs:
                    fnetcdf = osp.join(savdir, f'{filebase}.{kwargs["num"]:05d}.nc')
                else:
                    fnetcdf = osp.join(savdir, f'{filebase}.nc')

                if not force_override and osp.exists(fnetcdf) and \
                   osp.getmtime(fnetcdf) > mtime:
                    cls.logger.info('Read from existing netcdf: {0:s}'.format(fnetcdf))
                    with xr.open_dataset(fnetcdf) as fb:
                        res = fb.load()
                    return res
                else:
                    cls.logger.info('[check_netcdf]: Read original dump.')
                    # If we are here, force_override is True or history file is updated.
                    res = read_func(cls, **kwargs)

                    # Delete file first
                    if osp.exists(fnetcdf):
                        os.remove(fnetcdf)
                    try:
                        res.to_netcdf(fnetcdf)
                    except (IOError, PermissionError) as e:
                        cls.logger.warning('Could not create {0:s}.'.format(fnetcdf))
                    return res

            return wrapper

        def check_pickle_hst(read_hst):

            @functools.wraps(read_hst)
            def wrapper(cls, *args, **kwargs):
                if 'savdir' in kwargs:
                    savdir = kwargs['savdir']
                else:
                    savdir = osp.join(cls.savdir, 'hst')

                if 'force_override' in kwargs:
                    force_override = kwargs['force_override']
                else:
                    force_override = False

                # Create savdir if it doesn't exist
                if not osp.exists(savdir):
                    try:
                        os.makedirs(savdir)
                        force_override = True
                    except (IOError, PermissionError) as e:
                        cls.logger.warning('Could not make directory')

                fpkl = osp.join(savdir, osp.basename(cls.files['hst']) +
                                 '.{0:s}.mod.p'.format(cls.basename))
                #fpkl = osp.join(savdir, osp.basename(cls.files['hst']) + '.mod.p')

                # Check if the original history file is updated
                if not force_override and osp.exists(fpkl) and \
                   osp.getmtime(fpkl) > osp.getmtime(cls.files['hst']):
                    cls.logger.info('[read_hst]: Reading pickle.')
                    #print('[read_hst]: Reading pickle.')
                    hst = pd.read_pickle(fpkl)
                    cls.hst = hst
                    return hst
                else:
                    cls.logger.info('[read_hst]: Reading original hst file.')
                    # If we are here, force_override is True or history file is updated.
                    # Call read_hst function
                    hst = read_hst(cls, *args, **kwargs)
                    try:
                        hst.to_pickle(fpkl)
                    except (IOError, PermissionError) as e:
                        cls.logger.warning('[read_hst]: Could not pickle hst to {0:s}.'.format(fpkl))
                    return hst

            return wrapper

        def check_netcdf_zprof(_read_zprof):

            @functools.wraps(_read_zprof)
            def wrapper(cls, *args, **kwargs):
                if 'savdir' in kwargs:
                    savdir = kwargs['savdir']
                    if savdir is None:
                        savdir = osp.join(cls.savdir, 'zprof')
                else:
                    savdir = osp.join(cls.savdir, 'zprof')

                if 'force_override' in kwargs:
                    force_override = kwargs['force_override']
                else:
                    force_override = False

                if 'phase' in kwargs:
                    phase = kwargs['phase']
                else:
                    phase = 'whole'

                # Create savdir if it doesn't exist
                if not osp.exists(savdir):
                    os.makedirs(savdir)
                    force_override = True

                fnetcdf = '{0:s}.{1:s}.zprof.{2:s}.mod.nc'.format(
                    cls.problem_id, phase, cls.basename)
                fnetcdf = osp.join(savdir, fnetcdf)

                # Check if the original history file is updated
                mtime = max([osp.getmtime(f) for f in cls.files['zprof']])

                if not force_override and osp.exists(fnetcdf) and \
                   osp.getmtime(fnetcdf) > mtime:
                    cls.logger.info('[read_zprof]: Read {0:s}'.format(phase) + \
                                    ' zprof from existing NetCDF dump.')
                    ds = xr.open_dataset(fnetcdf)
                    return ds
                else:
                    cls.logger.info('[read_zprof]: Read from original {0:s}'.\
                        format(phase) + ' zprof dump and renormalize.'.format(phase))
                    # If we are here, force_override is True or zprof files are updated.
                    # Read original zprof dumps.
                    ds = _read_zprof(cls, phase, savdir, force_override)

                    # Somehow overwriting with mode='w' in to_netcdf doesn't work..
                    # Delete file first
                    if osp.exists(fnetcdf):
                        os.remove(fnetcdf)

                    try:
                        ds.to_netcdf(fnetcdf, mode='w')
                    except (IOError, PermissionError) as e:
                        cls.logger.warning('[read_zprof]: Could not netcdf to {0:s}.'\
                                           .format(fnetcdf))
                    return ds

            return wrapper


# Would be useful to have something like this for each problem
class LoadSimAll(object):
    """Class to load multiple simulations

    """
    def __init__(self, models, load_sim_class=LoadSim):
        # Default models
        if models is None:
            models = dict()
        self.models = []
        self.basedirs = dict()
        self.simdict = dict()
        self.load_sim_class = load_sim_class
        for mdl, basedir in models.items():
            if not osp.exists(basedir):
                print(
                    "[LoadSimAll]: Model {0:s} doesn't exist: {1:s}".format(
                        mdl, basedir
                    )
                )
            else:
                self.models.append(mdl)
                self.basedirs[mdl] = basedir

    def set_model(self, model, savdir=None,
                  load_method='xarray',
                  load_sim_class=None,
                  units=Units(kind='LV', muH=1.4271),
                  verbose=False):
        if load_sim_class is None:
            load_sim_class = self.load_sim_class
        try:
            self.sim = self.simdict[model]
        except KeyError:
            self.sim = load_sim_class(
                self.basedirs[model],
                savdir=savdir,
                load_method=load_method,
                verbose=verbose,
                units=units
            )
            self.simdict[model] = self.sim
        self.model = model
        return self.sim