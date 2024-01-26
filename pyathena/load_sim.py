from __future__ import print_function

import os
import sys
import glob, re
import getpass
import warnings
import logging
import os.path as osp
import functools
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import yt
import tarfile
import shutil

from .classic.vtk_reader import AthenaDataSet as AthenaDataSetClassic
from .io.read_vtk import AthenaDataSet, read_vtk_athenapp
from .io.read_vtk_tar import AthenaDataSetTar
from .io.read_hdf5 import read_hdf5
from .io.read_particles import read_partab, read_parhst
from .io.read_rst import read_rst
from .io.read_starpar_vtk import read_starpar_vtk
from .io.read_zprof import read_zprof_all
from .io.read_athinput import read_athinput
from .io.athena_read import athinput
from .util.units import Units
from .fields.fields import DerivedFields
from .plt_tools.make_movie import make_movie

class LoadSim(object):
    """Class to prepare Athena simulation data analysis. Read input parameters,
    find simulation output (vtk, starpar_vtk, hst, sn, zprof) files.

    Properties
    ----------
        basedir : str
            base directory of simulation output
        basename : str
            basename (tail) of basedir
        files : dict
            output file paths for vtk, starpar, hst, sn, zprof
        problem_id : str
            prefix for (vtk, starpar, hst, zprof) output
        par : dict
            input parameters and configure options read from log file
        ds : AthenaDataSet or yt DataSet
            class for reading vtk file
        domain : dict
            info about dimension, cell size, time, etc.
        load_method : str
            'pyathena' or 'yt' or 'pyathenaclassic'
        num : list of int
            vtk output numbers
        u : Units object
            simulation unit
        dfi : dict
            derived field information

    Methods
    -------
        load_vtk() :
            reads vtk file using pythena or yt and returns DataSet object
        load_starpar_vtk() :
            reads starpar vtk file and returns pandas DataFrame object
        print_all_properties() :
            prints all attributes and callable methods

    Parameters
    ----------
        basedir : str
            Name of the directory where all data is stored
        savdir : str
            Name of the directory where pickled data and figures will be saved.
            Default value is basedir.
        load_method : str
            Load vtk using 'pyathena', 'pythena_classic', or 'yt'.
            Default value is 'pyathena'.
            If None, savdir=basedir. Default value is None.
        verbose : bool or str or int
            Print verbose messages using logger. If True/False, set logger
            level to 'DEBUG'/'WARNING'. If string, it should be one of the string
            representation of python logging package:
            ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            Numerical values from 0 ('NOTSET') to 50 ('CRITICAL') are also
            accepted.

    Examples
    --------
        >>> s = LoadSim('/Users/jgkim/Documents/R4_8pc.RT.nowind', verbose=True)
        LoadSim-INFO: basedir: /Users/jgkim/Documents/R4_8pc.RT.nowind
        LoadSim-INFO: athinput: /Users/jgkim/Documents/R4_8pc.RT.nowind/out.txt
        LoadSim-INFO: problem_id: R4
        LoadSim-INFO: hst: /Users/jgkim/Documents/R4_8pc.RT.nowind/hst/R4.hst
        LoadSim-INFO: sn: /Users/jgkim/Documents/R4_8pc.RT.nowind/hst/R4.sn
        LoadSim-WARNING: No vtk files are found in /Users/jgkim/Documents/R4_8pc.RT.nowind.
        LoadSim-INFO: starpar: /Users/jgkim/Documents/R4_8pc.RT.nowind/starpar nums: 0-600
        LoadSim-INFO: zprof: /Users/jgkim/Documents/R4_8pc.RT.nowind/zprof nums: 0-600
        LoadSim-INFO: timeit: /Users/jgkim/Documents/R4_8pc.RT.nowind/timeit.txt
    """

    def __init__(self, basedir, savdir=None, load_method='pyathena',
                 units=Units(kind='LV', muH=1.4271),
                 verbose=False):
        """Constructor for LoadSim class.

        """

        self.basedir = basedir.rstrip('/')
        self.basename = osp.basename(self.basedir)

        self.load_method = load_method
        self.logger = self._get_logger(verbose=verbose)

        if savdir is None:
            self.savdir = self.basedir
        else:
            self.savdir = savdir

        self.logger.info('savdir : {:s}'.format(self.savdir))

        self._find_files()

        # Get domain info
        try:
            self.domain = self._get_domain_from_par(self.par)
        except:
            pass

        # Get config time
        try:
            config_time = self.par['configure']['config_date']
            # Avoid un-recognized timezone FutureWarning
            config_time = config_time.replace('PDT ', '')
            config_time = config_time.replace('EDT ', '')
            self.config_time = pd.to_datetime(config_time).tz_localize('US/Pacific')
            #self.config_time = self.config_time
        except:
            try:
                # set it using hst file creation time
                self.config_time = pd.to_datetime(osp.getctime(self.files['hst']),
                                                  unit='s')
            except:
                self.config_time = None

        try:
            muH = self.par['problem']['muH']
            self.u = Units(kind='LV', muH=muH)
        except KeyError:
            try:
                # Some old simulations run with new cooling may not have muH
                # parameter printed out
                if self.par['problem']['Z_gas'] != 1.0:
                    self.logger.warning('Z_gas={0:g} but muH is not found in par. '.\
                                        format(self.par['problem']['Z_gas']) +
                                        'Caution with muH={0:s}'.format(muH))
                self.u = units
            except:
                self.u = units
                pass
        if not self.athena_pp:
            # TODO(SMOON) Make DerivedFields work with athena++
            self.dfi = DerivedFields(self.par).dfi

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
           'pyathena', 'pyathena_classic' or 'yt'

        Returns
        -------
        ds : AthenaDataSet or yt datasets
        """

        if num is None and ivtk is None:
            raise ValueError('Specify either num or ivtk')

        # Override load_method
        if load_method is not None:
            self.load_method = load_method

        if self.athena_pp:
            def filter_vtk_files(kind='vtk', num=None):
                def func(num):
                    return lambda fname: '.{0:05d}.vtk'.format(num) in fname

                return list(filter(func(num), self.files[kind]))

            fnames = filter_vtk_files('vtk', num)
            return read_vtk_athenapp(fnames)

        if not self.files['vtk_id0']:
            id0 = False

        if id0:
            kind = ['vtk_id0', 'vtk', 'vtk_tar']
        else:
            if self.files['vtk']:
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
            if self.load_method == 'pyathena':
                self.ds = AthenaDataSet(self.fname, units=self.u, dfi=self.dfi)
                self.domain = self.ds.domain
                self.logger.info('[load_vtk]: {0:s}. Time: {1:f}'.format(\
                    osp.basename(self.fname), self.ds.domain['time']))

            elif self.load_method == 'pyathena_classic':
                self.ds = AthenaDataSetClassic(self.fname)
                self.domain = self.ds.domain
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
                    self.load_method) + ' Use either "yt", "pyathena", "pyathena_classic".')
        elif self.fname.endswith('tar'):
            if self.load_method == 'pyathena':
                self.ds = AthenaDataSetTar(self.fname, units=self.u, dfi=self.dfi)
                self.domain = self.ds.domain
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

    def load_hdf5(self, num=None, ihdf5=None,
                  outvar=None, outid=None, load_method=None, **kwargs):
        """Function to read Athena hdf5 file using pythena or yt and
        return DataSet object.

        Parameters
        ----------
        num : int
           Snapshot number, e.g., /basedir/problem_id.out?.?????.athdf
        ihdf5 : int
           Read i-th file in the hdf5 file list. Overrides num if both are given.
        outvar : str
           variable name, e.g, 'prim', 'cons', 'uov'. Default value is 'prim' or 'cons'.
           Overrides outid.
        outid : int
           output block number (output[n] in the input file).
        load_method : str
           'pyathena' or 'yt'

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

        if self.load_method == 'pyathena':
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
                self.ds = None
            else:
                self.ds = read_hdf5(self.fhdf5, **kwargs)

        elif self.load_method == 'yt':
            if hasattr(self, 'u'):
                units_override = self.u.units_override
            else:
                units_override = None
            self.ds = yt.load(self.fhdf5, units_override=units_override)
        else:
            self.logger.error('load_method "{0:s}" not recognized.'.format(
                self.load_method) + ' Use either "yt" or "pyathena".')

        return self.ds

    def load_partab(self, num=None, ipartab=None,
                    partag=None, **kwargs):
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

        if partag is None:
            partag = self._partab_partag_def

        self.fpartab = self._get_fpartab(self.partab_outid, partag, num, ipartab)
        if self.fpartab is None or not osp.exists(self.fpartab):
            self.logger.info('[load_partab]: partab file does not exist. ')

        self.pds = read_partab(self.fpartab, **kwargs)

        return self.pds

    def load_parhst(self, pid, **kwargs):
        """Read Athena++ individual particle history

        Parameters
        ----------
        pid : int
           Particle id, e.g., /basedir/partab/problem_id.pid.csv

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
        if num is None and ivtk is None:
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

    def _get_domain_from_par(self, par):
        """Get domain info from par['domain1']. Time is set to None.
        """
        domain = dict()
        if self.athena_pp:
            d = par['mesh']
            domain['Nx'] = np.array([d['nx1'], d['nx2'], d['nx3']])
        else:
            d = par['domain1']
            domain['Nx'] = np.array([d['Nx1'], d['Nx2'], d['Nx3']])
        domain['ndim'] = np.sum(domain['Nx'] > 1)
        domain['le'] = np.array([d['x1min'], d['x2min'], d['x3min']])
        domain['re'] = np.array([d['x1max'], d['x2max'], d['x3max']])
        domain['Lx'] = domain['re'] - domain['le']
        domain['dx'] = domain['Lx']/domain['Nx']
        domain['center'] = 0.5*(domain['le'] + domain['re'])
        domain['time'] = None
        self.domain = domain

        return domain

    def _find_match(self, patterns):
        glob_match = lambda p: sorted(glob.glob(osp.join(self.basedir, *p)))
        for p in patterns:
            f = glob_match(p)
            if f:
                break

        return f

    def _find_files(self):
        """Function to find all output files under basedir and create "files" dictionary.

        hst: problem_id.hst

        (athena only)
        vtk: problem_id.num.vtk
        sn: problem_id.sn (file format identical to hst)
        vtk_tar: problem_id.num.tar
        starpar_vtk: problem_id.num.starpar.vtk
        zprof: problem_id.num.phase.zprof
        sphst: *.star
        timeit: timtit.txt

        (athena_pp only)
        hdf5: problem_id.out?.num.athdf
        partab: problem_id.out?.num.par?.tab
        parhst: problem_id.pid.csv
        loop_time: problem_id.loop_time.txt
        task_time: problem_id.task_time.txt
        """

        self._out_fmt_def = ['hst', 'vtk']

        if not osp.isdir(self.basedir):
            raise IOError('basedir {0:s} does not exist.'.format(self.basedir))

        self.files = dict()

        athinput_patterns = [
            ('athinput.runtime',),
            ('out.txt',),
            ('out*.txt',),
            ('stdout.txt',),
            ('log.txt',),
            ('*.par',),
            ('*.out',),
            ('athinput.*',),
            ('slurm-*',)]

        hst_patterns = [('id0', '*.hst'),
                        ('hst', '*.hst'),
                        ('*.hst',)]

        sphst_patterns = [('id0', '*.star'),
                          ('hst', '*.star'),
                          ('*.star',)]

        sn_patterns = [('id0', '*.sn'),
                       ('hst', '*.sn'),
                       ('*.sn',)]

        vtk_patterns = [('vtk', '*.????.vtk'),
                        ('*.????.vtk',)]

        vtk_id0_patterns = [('vtk', 'id0', '*.' + '[0-9]'*4 + '.vtk'),
                            ('id0', '*.' + '[0-9]'*4 + '.vtk')]

        vtk_tar_patterns = [('vtk', '*.????.tar')]

        vtk_athenapp_patterns = [('*.block*.out*.?????.vtk',),
                                 ('*.joined.out*.?????.vtk',)]

        hdf5_patterns = [('hdf5', '*.out?.?????.athdf'),
                         ('*.out?.?????.athdf',)]

        starpar_patterns = [('starpar', '*.????.starpar.vtk'),
                            ('id0', '*.????.starpar.vtk'),
                            ('*.????.starpar.vtk',)]

        partab_patterns = [('partab', '*.out?.?????.par?.tab'),
                         ('*.out?.?????.par?.tab',)]

        parhst_patterns = [('parhst', '*.par*.csv'),
                         ('*.par*.csv',)]

        zprof_patterns = [('zprof', '*.zprof'),
                          ('id0', '*.zprof')]

        timeit_patterns = [('timeit.txt',),
                           ('timeit', 'timeit.txt')]

        looptime_patterns = [('*.loop_time.txt',),]

        tasktime_patterns = [('*.task_time.txt',),]

        self.logger.info('basedir: {0:s}'.format(self.basedir))

        # Read athinput files
        # Throw warning if not found
        fathinput = self._find_match(athinput_patterns)
        if fathinput:
            self.files['athinput'] = fathinput[0]
            try:
                self.par = read_athinput(self.files['athinput'])
            except:
                self.par = athinput(self.files['athinput'])
            self.logger.info('athinput: {0:s}'.format(self.files['athinput']))
            # self.out_fmt = [self.par[k]['out_fmt'] for k in self.par.keys() \
            #                 if 'output' in k]
            # Determine if this is Athena++ or Athena data
            if 'mesh' in self.par:
                self.athena_pp = True
                self.logger.info('athena_pp simulation')
            else:
                self.athena_pp = False
                self.logger.info('athena simulation')

            self.out_fmt = []
            self.partags = []
            if self.athena_pp:
                # read output blocks
                for k in self.par.keys():
                    if k.startswith('output'):
                        self.out_fmt.append(self.par[k]['file_type'])

                    # Save particle output tags
                    if k.startswith('particle'):
                        par_id = int(k.strip('particle')) - 1
                        partag = 'par{}'.format(par_id)
                        self.partags.append(partag)
                        self._partab_partag_def = partag

                # if there are hdf5 outputs, save some info
                if self.out_fmt.count('hdf5') > 0:
                    self.hdf5_outid = []
                    self.hdf5_outvar = []
                    for k in self.par.keys():
                        if k.startswith('output') and self.par[k]['file_type'] == 'hdf5':
                            self.hdf5_outid.append(int(re.split(r'(\d+)',k)[1]))
                            self.hdf5_outvar.append(self.par[k]['variable'])
                    for i,v in zip(self.hdf5_outid,self.hdf5_outvar):
                        if 'prim' in v or 'cons' in v:
                            self._hdf5_outid_def = i
                            self._hdf5_outvar_def = v

                # if there are partab outputs, save some info
                if 'partab' in self.out_fmt:
                    for k in self.par.keys():
                        if k.startswith('output') and self.par[k]['file_type'] == 'partab':
                            self.partab_outid = int(re.split(r'(\d+)',k)[1])

            else:
                for k in self.par.keys():
                    if k.startswith('output'):
                        # Skip if the block number XX (<outputXX>) is greater than maxout
                        if int(k.replace('output','')) > self.par['job']['maxout']:
                            continue
                        if self.par[k]['out_fmt'] == 'vtk' and \
                           not (self.par[k]['out'] == 'prim' or self.par[k]['out'] == 'cons'):
                            self.out_fmt.append(self.par[k]['id'] + '.' + \
                                                self.par[k]['out_fmt'])
                        else:
                            self.out_fmt.append(self.par[k]['out_fmt'])

            self.problem_id = self.par['job']['problem_id']
            self.logger.info('problem_id: {0:s}'.format(self.problem_id))
        else:
            self.par = None
            self.logger.warning('Could not find athinput file in {0:s}'.\
                                format(self.basedir))
            self.out_fmt = self._out_fmt_def

        if not self.athena_pp:
            # Find timeit.txt
            ftimeit = self._find_match(timeit_patterns)
            if ftimeit:
                self.files['timeit'] = ftimeit[0]
                self.logger.info('timeit: {0:s}'.format(self.files['timeit']))
            else:
                self.logger.info('timeit.txt not found.')

        if self.athena_pp:
            # Find problem_id.loop_time.txt
            flooptime = self._find_match(looptime_patterns)
            if flooptime:
                self.files['loop_time'] = flooptime[0]
                self.logger.info('loop_time: {0:s}'.format(self.files['loop_time']))
            else:
                self.logger.info('{}.loop_time.txt not found.'.format(self.problem_id))

            # Find problem_id.task_time.txt
            ftasktime = self._find_match(tasktime_patterns)
            if ftasktime:
                self.files['task_time'] = ftasktime[0]
                self.logger.info('task_time: {0:s}'.format(self.files['task_time']))
            else:
                self.logger.info('{}.task_time.txt not found.'.format(self.problem_id))

        # Find history dump and
        # Extract problem_id (prefix for output file names)
        # Assumes that problem_id does not contain '.'
        if 'hst' in self.out_fmt:
            fhst = self._find_match(hst_patterns)
            if fhst:
                self.files['hst'] = fhst[0]
                if not hasattr(self, 'problem_id'):
                    self.problem_id = osp.basename(self.files['hst']).split('.')[:-1]
                self.logger.info('hst: {0:s}'.format(self.files['hst']))
            else:
                self.logger.warning('Could not find hst file in {0:s}'.\
                                    format(self.basedir))

        # Find sn dump
        if self.athena_pp:
            fsn = self._find_match(sn_patterns)
            if fsn:
                self.files['sn'] = fsn[0]
                self.logger.info('sn: {0:s}'.format(self.files['sn']))
            else:
                if self.par is not None:
                    # Issue warning only if iSN is nonzero
                    try:
                        if self.par['feedback']['iSN'] != 0:
                            self.logger.warning('Could not find sn file in {0:s},' +
                            ' but <feedback>/iSN={1:d}'.\
                            format(self.basedir, self.par['feedback']['iSN']))
                    except KeyError:
                        pass

            # Find sphst dump
            fsphst = self._find_match(sphst_patterns)
            if fsphst:
                self.files['sphst'] = fsphst
                self.nums_sphst = [int(f[-10:-5]) for f in self.files['sphst']]
                self.logger.info('sphst: {0:s} nums: {1:d}-{2:d}'.format(
                    osp.dirname(self.files['sphst'][0]),
                    self.nums_sphst[0], self.nums_sphst[-1]))

        # Find vtk files
        # vtk files in both basedir (joined) and in basedir/id0
        if 'vtk' in self.out_fmt and not self.athena_pp:
            self.files['vtk'] = self._find_match(vtk_patterns)
            self.files['vtk_id0'] = self._find_match(vtk_id0_patterns)
            self.files['vtk_tar'] = self._find_match(vtk_tar_patterns)
            if not self.files['vtk'] and not self.files['vtk_id0'] and \
               not self.files['vtk_tar']:
                self.logger.warning(
                    'vtk files not found in {0:s}'.format(self.basedir))
                self.nums = None
                self.nums_id0 = None
            else:
                self.nums = [int(f[-8:-4]) for f in self.files['vtk']]
                self.nums_id0 = [int(f[-8:-4]) for f in self.files['vtk_id0']]
                self.nums_tar = [int(f[-8:-4]) for f in self.files['vtk_tar']]
                if self.nums_id0:
                    self.logger.info('vtk in id0: {0:s} nums: {1:d}-{2:d}'.format(
                        osp.dirname(self.files['vtk_id0'][0]),
                        self.nums_id0[0], self.nums_id0[-1]))
                    if not hasattr(self, 'problem_id'):
                        self.problem_id = osp.basename(self.files['vtk_id0'][0]).split('.')[-2:]
                if self.nums:
                    self.logger.info('vtk (joined): {0:s} nums: {1:d}-{2:d}'.format(
                        osp.dirname(self.files['vtk'][0]),
                        self.nums[0], self.nums[-1]))
                    if not hasattr(self, 'problem_id'):
                        self.problem_id = osp.basename(self.files['vtk'][0]).split('.')[-2:]
                else:
                    self.nums = self.nums_id0
                if self.nums_tar:
                    self.logger.info('vtk in tar: {0:s} nums: {1:d}-{2:d}'.format(
                        osp.dirname(self.files['vtk_tar'][0]),
                        self.nums_tar[0], self.nums_tar[-1]))
                    if not hasattr(self, 'problem_id'):
                        self.problem_id = osp.basename(self.files['vtk_tar'][0]).split('.')[-2:]
                    self.nums = self.nums_tar
            self.nums_vtk_all = list(set(self.nums)|set(self.nums_id0)|set(self.nums_tar))
            self.nums_vtk_all.sort()

            # Check (joined) vtk file size
            sizes = [os.stat(f).st_size for f in self.files['vtk']]
            if len(set(sizes)) > 1:
                size = max(set(sizes), key=sizes.count)
                flist = [(i, s // 1024**2) for i, s in enumerate(sizes) if s != size]
                self.logger.warning('Vtk file size is not unique.')
                for f in flist:
                   self.logger.debug('vtk num:', f[0], 'size [MB]:', f[1])

            # Check (tarred) vtk file size
            sizes = [os.stat(f).st_size for f in self.files['vtk_tar']]
            if len(set(sizes)) > 1:
                size = max(set(sizes), key=sizes.count)
                flist = [(i, s // 1024**2) for i, s in enumerate(sizes) if s != size]
                self.logger.warning('Vtk file size is not unique.')
                for f in flist:
                   self.logger.debug('vtk num:', f[0], 'size [MB]:', f[1])
        elif 'vtk' in self.out_fmt and self.athena_pp:
            # Athena++ vtk files
            self.files['vtk'] = self._find_match(vtk_athenapp_patterns)
            self.nums_vtk = list(set([int(f[-9:-4]) for f in self.files['vtk']]))
            self.nums_vtk.sort()

        # Find hdf5 files
        # hdf5 files in basedir
        if 'hdf5' in self.out_fmt:
            self.files['hdf5'] = dict()
            self.nums_hdf5 = dict()
            for i,v in zip(self.hdf5_outid,self.hdf5_outvar):
                hdf5_patterns_ = []
                for p in hdf5_patterns:
                    p = list(p)
                    p[-1] = p[-1].replace('out?', 'out{0:d}'.format(i))
                    hdf5_patterns_.append(tuple(p))
                self.files['hdf5'][v] = self._find_match(hdf5_patterns_)
                if not self.files['hdf5'][v]:
                    self.logger.warning(
                        'hdf5 ({0:s}) files not found in {1:s}'.\
                        format(v,self.basedir))
                    self.nums_hdf5[v] = None
                else:
                    self.nums_hdf5[v] = [int(f[-11:-6]) \
                                              for f in self.files['hdf5'][v]]
                    self.logger.info('hdf5 ({0:s}): {1:s} nums: {2:d}-{3:d}'.format(
                        v, osp.dirname(self.files['hdf5'][v][0]),
                        self.nums_hdf5[v][0], self.nums_hdf5[v][-1]))
                    if not hasattr(self, 'problem_id'):
                        self.problem_id = osp.basename(
                            self.files['hdf5'][self._hdf5_outvar_def][0]).split('.')[-2:]
            # Set nums array
            self.nums = self.nums_hdf5[self._hdf5_outvar_def]

        # Find starpar files
        if 'starpar_vtk' in self.out_fmt:
            fstarpar = self._find_match(starpar_patterns)
            if fstarpar:
                self.files['starpar_vtk'] = fstarpar
                self.nums_starpar = [int(f[-16:-12]) for f in self.files['starpar_vtk']]
                self.logger.info('starpar_vtk: {0:s} nums: {1:d}-{2:d}'.format(
                    osp.dirname(self.files['starpar_vtk'][0]),
                    self.nums_starpar[0], self.nums_starpar[-1]))
            else:
                self.logger.warning(
                    'starpar files not found in {0:s}.'.format(self.basedir))

        # Find partab files
        if 'partab' in self.out_fmt:
            self.files['partab'] = dict()
            self.nums_partab = dict()
            for partag in self.partags:
                partab_patterns_ = []
                for p in partab_patterns:
                    p = list(p)
                    p[-1] = p[-1].replace('par?', partag)
                    partab_patterns_.append(tuple(p))
                self.files['partab'][partag] = self._find_match(partab_patterns_)
                if not self.files['partab'][partag]:
                    self.logger.warning(
                        'partab ({0:s}) files not found in {1:s}'.\
                        format(partag, self.basedir))
                    self.nums_partab[partag] = None
                else:
                    self.nums_partab[partag] = [int(f[-14:-9])
                                                 for f in self.files['partab'][partag]]
                    self.logger.info('partab ({0:s}): {1:s} nums: {2:d}-{3:d}'.format(
                        partag, osp.dirname(self.files['partab'][partag][0]),
                        self.nums_partab[partag][0], self.nums_partab[partag][-1]))

        # Find parhst files
        if self.athena_pp and any(['particle' in k for k in self.par.keys()]):
            fparhst = self._find_match(parhst_patterns)
            if fparhst:
                self.files['parhst'] = fparhst
                self.pids = [int(f.split('/')[-1].split('.')[1].strip('par'))
                            for f in self.files['parhst']]
                self.pids.sort()
                self.logger.info('parhst: {0:s} pids: {1:d}-{2:d}'.format(
                    osp.dirname(self.files['parhst'][0]),
                    self.pids[0], self.pids[-1]))
            else:
                self.pids = []
                self.logger.warning(
                    'parhst files not found in {0:s}.'.format(self.basedir))

        # Find zprof files
        # Multiple zprof files for each snapshot.
        if 'zprof' in self.out_fmt:
            fzprof = self._find_match(zprof_patterns)
            if fzprof:
                self.files['zprof'] = fzprof
                self.nums_zprof = dict()
                self.phase = []
                for f in self.files['zprof']:
                    _, num, ph, _ = osp.basename(f).split('.')[-4:]
                    try:
                        self.nums_zprof[ph].append(int(num))
                    except KeyError:
                        self.phase.append(ph)
                        self.nums_zprof[ph] = []
                        self.nums_zprof[ph].append(int(num))

                # Check if number of files for each phase matches
                num = [len(self.nums_zprof[ph]) for ph in self.nums_zprof.keys()]
                if not all(num):
                    self.logger.warning('Number of zprof files doesn\'t match.')
                    self.logger.warning(', '.join(['{0:s}: {1:d}'.format(ph, \
                        len(self.nums_zprof[ph])) for ph in self.phase][:-1]))
                else:
                    self.logger.info('zprof: {0:s} nums: {1:d}-{2:d}'.format(
                    osp.dirname(self.files['zprof'][0]),
                    self.nums_zprof[self.phase[0]][0],
                    self.nums_zprof[self.phase[0]][-1]))

            else:
                self.logger.warning(
                    'zprof files not found in {0:s}.'.format(self.basedir))

        # 2d vtk files
        self._fmt_vtk2d_not_found = []
        for fmt in self.out_fmt:
            if '.vtk' in fmt:
                fmt = fmt.split('.')[0]
                patterns = [('id0', '*.????.{0:s}.vtk'.format(fmt)),
                    ('{0:s}'.format(fmt), '*.????.{0:s}.vtk'.format(fmt))]
                files = self._find_match(patterns)
                if files:
                    self.files[f'{fmt}'] = files
                    setattr(self, f'nums_{fmt}', [int(osp.basename(f).split('.')[1]) \
                                                  for f in self.files[f'{fmt}']])
                else:
                    # Some 2d vtk files may not be found in id0 folder (e.g., slices)
                    self._fmt_vtk2d_not_found.append(fmt)

        if self._fmt_vtk2d_not_found:
            self.logger.info('These vtk files need to be found ' + \
                             'using find_files_vtk2d() method: ' + \
                             ', '.join(self._fmt_vtk2d_not_found))
        # Find rst files
        if 'rst' in self.out_fmt:
            if hasattr(self,'problem_id'):
                rst_patterns = [('rst','{}.*.rst'.format(self.problem_id)),
                                ('rst','{}.*.tar'.format(self.problem_id)),
                                ('id0','{}.*.rst'.format(self.problem_id)),
                                ('{}.*.rst'.format(self.problem_id),)]
                frst = self._find_match(rst_patterns)
                if frst:
                    self.files['rst'] = frst
                    if self.athena_pp:
                        numbered_rstfiles = [f for f in self.files['rst'] if 'final' not in f]
                        self.nums_rst = [int(f[-9:-4]) for f in numbered_rstfiles]
                    else:
                        self.nums_rst = [int(f[-8:-4]) for f in self.files['rst']]
                    self.logger.info('rst: {0:s} nums: {1:d}-{2:d}'.format(
                                     osp.dirname(self.files['rst'][0]),
                                     self.nums_rst[0], self.nums_rst[-1]))
                else:
                    self.logger.warning(
                        'rst files not found in {0:s}.'.format(self.basedir))

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

    def _get_logger(self, verbose=False):
        """Function to set logger and default verbosity.

        Parameters
        ----------
        verbose: bool or str or int
            Set logging level to "INFO"/"WARNING" if True/False.
        """

        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

        if verbose is True:
            self.loglevel = 'INFO'
        elif verbose is False:
            self.loglevel = 'WARNING'
        elif verbose in levels + [l.lower() for l in levels]:
            self.loglevel = verbose.upper()
        elif isinstance(verbose, int):
            self.loglevel = verbose
        else:
            raise ValueError('Cannot recognize option {0:s}.'.format(verbose))

        l = logging.getLogger(self.__class__.__name__.split('.')[-1])

        try:
            if not l.hasHandlers():
                h = logging.StreamHandler()
                f = logging.Formatter('[%(name)s-%(levelname)s] %(message)s')
                # f = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
                h.setFormatter(f)
                l.addHandler(h)
                l.setLevel(self.loglevel)
            else:
                l.setLevel(self.loglevel)
        except AttributeError: # for python 2 compatibility
            if not len(l.handlers):
                h = logging.StreamHandler()
                f = logging.Formatter('[%(name)s-%(levelname)s] %(message)s')
                # f = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
                h.setFormatter(f)
                l.addHandler(h)
                l.setLevel(self.loglevel)
            else:
                l.setLevel(self.loglevel)

        return l

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
                    fpkl = osp.join(savdir, '{0:s}_{1:04d}.p'.format(prefix, kwargs['num']))
                else:
                    fpkl = osp.join(savdir, '{0:s}.p'.format(prefix))

                if not force_override and osp.exists(fpkl):
                    cls.logger.info('Read from existing pickle: {0:s}'.format(fpkl))
                    res = pickle.load(open(fpkl, 'rb'))
                    return res
                else:
                    cls.logger.info('[check_pickle]: Read original dump.')
                    # If we are here, force_override is True or history file is updated.
                    res = read_func(cls, **kwargs)
                    try:
                        pickle.dump(res, open(fpkl, 'wb'))
                    except (IOError, PermissionError) as e:
                        cls.logger.warning('Could not pickle to {0:s}.'.format(fpkl))
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
    def __init__(self, models):

        self.models = list(models.keys())
        self.basedirs = dict()

        for mdl, basedir in models.items():
            self.basedirs[mdl] = basedir

    def set_model(self, model, savdir=None, load_method='pyathena',
                  units=Units(kind='LV', muH=1.4271),
                  verbose=False):
        self.model = model
        self.sim = LoadSim(self.basedirs[model], savdir=savdir,
                           load_method=load_method,
                           units=units, verbose=verbose)
        return self.sim
