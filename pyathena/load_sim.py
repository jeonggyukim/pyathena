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

from .classic.vtk_reader import AthenaDataSet as AthenaDataSetClassic
from .io.read_vtk import AthenaDataSet
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
        
        self.u = units
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
           Read vtk file in /basedir/id0. Default value is False.
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
            kind = ['vtk_id0', 'vtk']
        else:
            kind = ['vtk', 'vtk_id0']

        self.fvtk = self._get_fvtk(kind[0], num, ivtk)
        if self.fvtk is None or not osp.exists(self.fvtk):
            if id0:
                self.logger.info('[load_vtk]: Vtk file does not exist. ' + \
                                 'Try joined vtk')
            else:
                self.logger.info('[load_vtk]: Vtk file does not exist. ' + \
                                 'Try vtk in id0')
                
            # Check if joined vtk (or vtk in id0) exists
            self.fvtk = self._get_fvtk(kind[1], num, ivtk)
            if self.fvtk is None or not osp.exists(self.fvtk):
                self.logger.error('[load_vtk]: Vtk file does not exist.')
                
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
        self.fstarvtk = self._get_fvtk('starpar', num, ivtk)
        if self.fstarvtk is None or not osp.exists(self.fstarvtk):
            self.logger.error('[load_starpar_vtk]: Starpar vtk file does not exist.')

        self.sp = read_starpar_vtk(self.fstarvtk,
                force_override=force_override, verbose=verbose)
        self.logger.info('[load_starpar_vtk]: {0:s}. Time: {1:f}'.format(\
                 osp.basename(self.fstarvtk), self.sp.time))

        return self.sp
    
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

    def _find_files(self):
        """Function to find all output files under basedir and create "files" dictionary.

        hst: problem_id.hst
        sn: problem_id.sn (file format identical to hst)
        vtk: problem_id.num.vtk
        starpar_vtk: problem_id.num.starpar.vtk
        zprof: problem_id.num.phase.zprof
        timeit: timtit.txt
        """

        self._out_fmt_def = ['hst', 'vtk']

        if not osp.isdir(self.basedir):
            raise IOError('basedir {0:s} does not exist.'.format(self.basedir))
        
        self.files = dict()
        def find_match(patterns):
            glob_match = lambda p: sorted(glob.glob(osp.join(self.basedir, *p)))
               
            for p in patterns:
                f = glob_match(p)
                if f:
                    break
                
            return f

        athinput_patterns = [('stdout.txt',),    # Jeong-Gyu
                             ('out.txt',),    # Jeong-Gyu
                             ('log.txt',),     # Jeong-Gyu
                             ('*.out',),      # Chang-Goo's stdout
                             ('slurm-*',),    # Erin
                             ('athinput.*',), # Chang-Goo's restart
                             ('*.par',)]
        
        hst_patterns = [('id0', '*.hst'),
                        ('hst', '*.hst'),
                        ('*.hst',)]
        
        sn_patterns = [('id0', '*.sn'),
                       ('hst', '*.sn'),
                       ('*.sn',)]
        
        vtk_patterns = [('vtk', '*.????.vtk'),
                        ('*.????.vtk',)]

        vtk_id0_patterns = [('vtk', 'id0', '*.????.vtk'),
                            ('id0', '*.????.vtk')]
        
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
        fathinput = find_match(athinput_patterns)
        if fathinput:
            self.files['athinput'] = fathinput[0]
            self.par = read_athinput(self.files['athinput'])
            self.logger.info('athinput: {0:s}'.format(self.files['athinput']))
            # self.out_fmt = [self.par[k]['out_fmt'] for k in self.par.keys() \
            #                 if 'output' in k]
            self.out_fmt = []
            for k in self.par.keys():
                if 'output' in k:
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

        # Find history dump and
        # Extract problem_id (prefix for vtk and hitsory file names)
        # Assumes that problem_id does not contain '.'
        if 'hst' in self.out_fmt:
            fhst = find_match(hst_patterns)
            if fhst:
                self.files['hst'] = fhst[0]
                self.problem_id = osp.basename(self.files['hst']).split('.')[0]
                self.logger.info('hst: {0:s}'.format(self.files['hst']))
            else:
                self.logger.warning('Could not find hst file in {0:s}'.\
                                    format(self.basedir))

        # Find sn dump
        fsn = find_match(sn_patterns)
        if fsn:
            self.files['sn'] = fsn[0]
            self.logger.info('sn: {0:s}'.format(self.files['sn']))
        else:
            if self.par is not None:
                # Issue warning only if iSN is nonzero
                try:
                    if self.par['feedback']['iSN'] != 0:
                        self.logger.warning('Could not find sn file in {0:s}'.\
                                            format(self.basedir))
                except KeyError:
                    pass
                
        # Find vtk files
        # vtk files in both basedir (joined) and in basedir/id0
        if 'vtk' in self.out_fmt:
            self.files['vtk'] = find_match(vtk_patterns)
            self.files['vtk_id0'] = find_match(vtk_id0_patterns)
            if not self.files['vtk'] and not self.files['vtk_id0']:
                self.logger.warning(
                    'No vtk files are found in {0:s}'.format(self.basedir))
                self.nums = None
                self.nums_id0 = None
            else:
                self.nums = [int(f[-8:-4]) for f in self.files['vtk']]
                self.nums_id0 = [int(f[-8:-4]) for f in self.files['vtk_id0']]
                if self.nums_id0:
                    self.logger.info('vtk in id0: {0:s} nums: {1:d}-{2:d}'.format(
                        osp.dirname(self.files['vtk_id0'][0]),
                        self.nums_id0[0], self.nums_id0[-1]))
                    if not hasattr(self, 'problem_id'):
                        self.problem_id = osp.basename(self.files['vtk_id0'][0]).split('.')[0]
                if self.nums:
                    self.logger.info('vtk (joined): {0:s} nums: {1:d}-{2:d}'.format(
                        osp.dirname(self.files['vtk'][0]),
                        self.nums[0], self.nums[-1]))
                    if not hasattr(self, 'problem_id'):
                        self.problem_id = osp.basename(self.files['vtk'][0]).split('.')[0]
                else:
                    self.nums = self.nums_id0
                

            # Check (joined) vtk file size
            sizes = [os.stat(f).st_size for f in self.files['vtk']]
            if len(set(sizes)) > 1:
                size = max(set(sizes), key=sizes.count)
                flist = [(i, s // 1024**2) for i, s in enumerate(sizes) if s != size]
                self.logger.warning('Vtk file size is not unique.')
                for f in flist:
                   self.logger.debug('vtk num:', f[0], 'size [MB]:', f[1])

        # Find starpar files
        if 'starpar_vtk' in self.out_fmt:
            fstarpar = find_match(starpar_patterns)
            if fstarpar:
                self.files['starpar'] = fstarpar
                self.nums_starpar = [int(f[-16:-12]) for f in self.files['starpar']]
                self.logger.info('starpar: {0:s} nums: {1:d}-{2:d}'.format(
                    osp.dirname(self.files['starpar'][0]),
                    self.nums_starpar[0], self.nums_starpar[-1]))
            else:
                self.logger.warning(
                    'No starpar files are found in {0:s}.'.format(self.basedir))

        # Find zprof files
        # Multiple zprof files for each snapshot.
        if 'zprof' in self.out_fmt:
            fzprof = find_match(zprof_patterns)
            if fzprof:
                self.files['zprof'] = fzprof
                self.nums_zprof = dict()
                self.phase = []
                for f in self.files['zprof']:
                    _, num, ph, _ = osp.basename(f).split('.')
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
                    'No zprof files are found in {0:s}.'.format(self.basedir))

        # 2d vtk files
        for fmt in self.out_fmt:
            if '.vtk' in fmt:
                fmt = fmt.split('.')[0]
                patterns = [('id0', '*.????.{0:s}.vtk'.format(fmt)),
                    ('{0:s}'.format(fmt), '*.????.{0:s}.vtk'.format(fmt))]
                files = find_match(patterns)
                self.files[f'{fmt}'] = files
                setattr(self, f'nums_{fmt}', [int(osp.basename(f).split('.')[1]) \
                                              for f in self.files[f'{fmt}']])


        # Find timeit.txt
        ftimeit = find_match(timeit_patterns)
        if ftimeit:
            self.files['timeit'] = ftimeit[0]
            self.logger.info('timeit: {0:s}'.format(self.files['timeit']))
        else:
            self.logger.info('No timeit.txt found.')

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
            if kind == 'starpar':
                fpattern = '{0:s}.{1:04d}.starpar.vtk'
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

