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
from .io.read_vtk import AthenaDataSet
from .io.read_vtk_tar import AthenaDataSetTar
from .io.read_rst import read_rst
from .io.read_starpar_vtk import read_starpar_vtk
from .io.read_zprof import read_zprof_all
from .io.read_athinput import read_athinput
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
    """

    def __init__(self, basedir, savdir=None, load_method='pyathena',
                 units=Units(kind='LV', muH=1.4271),
                 verbose=False):
        """Constructor for LoadSim class.

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
            self.config_time = pd.to_datetime(self.par['configure']['config_date'])
            if 'PDT' in self.par['configure']['config_date']:
                self.config_time = self.config_time.tz_localize('US/Pacific')
        except:
            # set it using hst file creation time
            self.config_time = pd.to_datetime(osp.getctime(self.files['hst']),unit='s')

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

        if id0:
            kind = ['vtk_id0', 'vtk', 'vtk_tar']
        else:
            kind = ['vtk', 'vtk_tar', 'vtk_id0']

        self.fvtk = self._get_fvtk(kind[0], num, ivtk)
        if self.fvtk is None or not osp.exists(self.fvtk):
            if id0:
                self.logger.info('[load_vtk]: Vtk file does not exist. ' + \
                                 'Try joined/tarred vtk')
            else:
                self.logger.info('[load_vtk]: Vtk file does not exist. ' + \
                                 'Try vtk in id0')

            # Check if joined vtk (or vtk in id0) exists
            self.fvtk = self._get_fvtk(kind[1], num, ivtk)
            if self.fvtk is None or not osp.exists(self.fvtk):
                self.logger.info('[load_vtk]: Vtk file does not exist.')

            # Check if joined vtk (or vtk in id0) exists
            self.fvtk = self._get_fvtk(kind[2], num, ivtk)
            if self.fvtk is None or not osp.exists(self.fvtk):
                self.logger.error('[load_vtk]: Vtk file does not exist.')

        if self.fvtk.endswith('vtk'):
            if self.load_method == 'pyathena':
                self.ds = AthenaDataSet(self.fvtk, units=self.u, dfi=self.dfi)
                self.domain = self.ds.domain
                self.logger.info('[load_vtk]: {0:s}. Time: {1:f}'.format(\
                    osp.basename(self.fvtk), self.ds.domain['time']))

            elif self.load_method == 'pyathena_classic':
                self.ds = AthenaDataSetClassic(self.fvtk)
                self.domain = self.ds.domain
                self.logger.info('[load_vtk]: {0:s}. Time: {1:f}'.format(\
                    osp.basename(self.fvtk), self.ds.domain['time']))

            elif self.load_method == 'yt':
                if hasattr(self, 'u'):
                    units_override = self.u.units_override
                else:
                    units_override = None
                self.ds = yt.load(self.fvtk, units_override=units_override)
            else:
                self.logger.error('load_method "{0:s}" not recognized.'.format(
                    self.load_method) + ' Use either "yt", "pyathena", "pyathena_classic".')
        elif self.fvtk.endswith('tar'):
            self.ds = AthenaDataSetTar(self.fvtk, units=self.u, dfi=self.dfi)
            self.domain = self.ds.domain
            self.logger.info('[load_vtk_tar]: {0:s}. Time: {1:f}'.format(\
                osp.basename(self.fvtk), self.ds.domain['time']))

        return self.ds

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
        self.fstarvtk = self._get_fvtk('starpar_vtk', num, ivtk)
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
        self.frst = self._get_fvtk('rst', num, irst)
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
            id_files = [self._get_fvtk('vtk_id0',num=num)]
        elif kind == 'rst':
            id_files = [self._get_fvtk('rst',num=num)]
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
        d = par['domain1']
        domain = dict()

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
        sn: problem_id.sn (file format identical to hst)
        vtk: problem_id.num.vtk
        vtk_tar: problem_id.num.tar
        starpar_vtk: problem_id.num.starpar.vtk
        zprof: problem_id.num.phase.zprof
        timeit: timtit.txt
        """

        self._out_fmt_def = ['hst', 'vtk']

        if not osp.isdir(self.basedir):
            raise IOError('basedir {0:s} does not exist.'.format(self.basedir))

        self.files = dict()

        athinput_patterns = [('stdout.txt',),
                             ('out.txt',),
                             ('out*.txt',),
                             ('log.txt',),
                             ('*.par',),
                             ('*.out',),
                             ('slurm-*',),
                             ('athinput.*',),
                             ]

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
                            # ('vtk', '[0-9]'*4, '*.' + '[0-9]'*4 + '.vtk'),
                            ('id0', '*.' + '[0-9]'*4 + '.vtk')]

        vtk_tar_patterns = [('vtk', '*.????.tar')]

        starpar_patterns = [('starpar', '*.????.starpar.vtk'),
                            ('id0', '*.????.starpar.vtk'),
                            ('*.????.starpar.vtk',)]

        zprof_patterns = [('zprof', '*.zprof'),
                          ('id0', '*.zprof')]

        timeit_patterns = [('timeit.txt',),
                           ('timeit', 'timeit.txt')]

        self.logger.info('basedir: {0:s}'.format(self.basedir))

        # Read athinput files
        # Throw warning if not found
        fathinput = self._find_match(athinput_patterns)
        if fathinput:
            self.files['athinput'] = fathinput[0]
            self.par = read_athinput(self.files['athinput'])
            self.logger.info('athinput: {0:s}'.format(self.files['athinput']))
            # self.out_fmt = [self.par[k]['out_fmt'] for k in self.par.keys() \
            #                 if 'output' in k]
            self.out_fmt = []
            for k in self.par.keys():
                if 'output' in k:
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

        # Find timeit.txt
        ftimeit = self._find_match(timeit_patterns)
        if ftimeit:
            self.files['timeit'] = ftimeit[0]
            self.logger.info('timeit: {0:s}'.format(self.files['timeit']))
        else:
            self.logger.info('timeit.txt not found.')

        # Find history dump and
        # Extract problem_id (prefix for vtk and hitsory file names)
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
        if 'vtk' in self.out_fmt:
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
                                ('id0','{}.*.rst'.format(self.problem_id))]
                frst = self._find_match(rst_patterns)
                if frst:
                    self.files['rst'] = frst
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


    def _get_fvtk(self, kind, num=None, ivtk=None):
        """Get vtk file path
        """

        try:
            dirname = osp.dirname(self.files[kind][0])
        except IndexError:
            return None
        if ivtk is not None:
            fvtk = self.files[kind][ivtk]
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
            fvtk = osp.join(dirname, fpattern.format(self.problem_id, num))

        return fvtk

    def _get_logger(self, verbose=False):
        """Function to set logger and default verbosity.

        Parameters
        ----------
        verbose: bool or str or int
            Set logging level to "INFO"/"WARNING" if True/False.
        """

        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

        if verbose is True:
            self.loglevel_def = 'INFO'
        elif verbose is False:
            self.loglevel_def = 'WARNING'
        elif verbose in levels + [l.lower() for l in levels]:
            self.loglevel_def = verbose.upper()
        elif isinstance(verbose, int):
            self.loglevel_def = verbose
        else:
            raise ValueError('Cannot recognize option {0:s}.'.format(verbose))

        l = logging.getLogger(self.__class__.__name__.split('.')[-1])

        try:
            if not l.hasHandlers():
                h = logging.StreamHandler()
                f = logging.Formatter('%(name)s-%(levelname)s: %(message)s')
                # f = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
                h.setFormatter(f)
                l.addHandler(h)
                l.setLevel(self.loglevel_def)
            else:
                l.setLevel(self.loglevel_def)
        except AttributeError: # for python 2 compatibility
            if not len(l.handlers):
                h = logging.StreamHandler()
                f = logging.Formatter('%(name)s-%(levelname)s: %(message)s')
                # f = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
                h.setFormatter(f)
                l.addHandler(h)
                l.setLevel(self.loglevel_def)
            else:
                l.setLevel(self.loglevel_def)

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

