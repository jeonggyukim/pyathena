from __future__ import print_function

import os
import sys
import glob, re
import warnings
import logging
import os.path as osp
import functools
import pandas as pd

import yt
from .classic.vtk_reader import AthenaDataSet as AthenaDataSetClassic
from .io.read_athinput import read_athinput
from .io.read_vtk import AthenaDataSet

class LoadSim(object):
    """Class to prepare Athena simulation data analysis. Read input parameters, 
    find vtk, starpar_vtk, zprof, and history files.

    """

    def __init__(self, basedir, savdir=None, load_method='pyathena',
                 verbose=True):
        """
        The constructor for LoadSim class.

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
        >>> s = LoadSim('/projects/EOSTRIKE/TIGRESS_XCO_ART/R2_2pc_L256_B2.noHII.Z1.shldA')
        LoadSim-INFO: basedir: /projects/EOSTRIKE/TIGRESS_XCO_ART/R2_2pc_L256_B2.noHII.Z1.shldA
        LoadSim-WARNING: Could not find athinput file in /projects/EOSTRIKE/TIGRESS_XCO_ART/R2_2pc_L256_B2.noHII.Z1.shldA
        LoadSim-INFO: hst: /projects/EOSTRIKE/TIGRESS_XCO_ART/R2_2pc_L256_B2.noHII.Z1.shldA/id0/R2_2pc_L256_B2.hst
        LoadSim-INFO: vtk in id0:  /projects/EOSTRIKE/TIGRESS_XCO_ART/R2_2pc_L256_B2.noHII.Z1.shldA/id0 nums: 0-1

        """

        self.basedir = basedir.rstrip('/')
        self.basename = osp.basename(self.basedir)

        self.load_method = load_method
        self.logger = self._get_logger(verbose=verbose)
        self._find_files()

        if savdir is None:
            self.savdir = self.basedir
        else:
            self.savdir = savdir
            self.logger.info('savdir : {:s}'.format(savdir))

    def load_vtk(self, num=None, ivtk=None, id0=False, load_method=None):
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

        def get_fvtk(kind, num=None, ivtk=None):
            try:
                dirname = osp.dirname(self.files[kind][0])
            except IndexError:
                return None
            if ivtk is not None:
                fvtk = self.files[kind][ivtk]
            else:
                fvtk = osp.join(dirname, '{0:s}.{1:04d}.vtk'.\
                                format(self.problem_id, num))

            return fvtk

        self.fvtk = get_fvtk(kind[0], num, ivtk)
        if self.fvtk is None or not osp.exists(self.fvtk):
            if id0:
                self.logger.info('[load_vtk]: Vtk file does not exist. ' + \
                                 'Try joined vtk')
            else:
                self.logger.info('[load_vtk]: Vtk file does not exist. ' + \
                                 'Try vtk in id0')
            # Check if joined vtk (or vtk in id0) exists
            self.fvtk = get_fvtk(kind[1], num, ivtk)
            if not osp.exists(self.fvtk):
                self.logger.error('[load_vtk]: Vtk file does not exist {:s}'.\
                                  format(self.fvtk))
                                
        if self.load_method == 'pyathena':
            self.ds = AthenaDataSet(self.fvtk)
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
            self.logger.error('load_method "{0:s}" not recognized.'.format(\
                self.load_method) + ' Use either "yt" or "pyathena".')
            raise
        
        return self.ds

    
    def print_all_properties(self):
        """Print all attributes and callable methods
        """
        
        attr_list = list(self.__dict__.keys())
        print('Attributes:\n', attr_list)
        print('\nMethods:')
        method_list = []
        for func in sorted(dir(self)):
            if not func.startswith("_"):
                if callable(getattr(self, func)):
                    method_list.append(func)
                    print(func, end=': ')
                    print(getattr(self, func).__doc__)
                    print('-------------------------')


    def _find_files(self):
        """Function to find all output files under basedir and create "files" dictionary.

        hst: problem_id.hst
        vtk: problem_id.num.vtk
        starpar_vtk: problem_id.num.starpar.vtk
        zprof: problem_id.num.phase.zprof
        """

        self._out_fmt_def = ['hst', 'vtk']

        if not osp.isdir(self.basedir):
            raise IOError('basedir {0:s} does not exist.'.format(self.basedir))
        
        self.files = dict()
        def find_match(patterns):
            glob_match = lambda p: \
                         sorted(glob.glob(osp.join(self.basedir, *p)))
               
            for p in patterns:
                f = glob_match(p)
                if f:
                    break
                
            return f

        athinput_patterns = [('out.txt',), # Jeong-Gyu
                             ('*.out',), # Chang-Goo's stdout
                             ('slurm-*',), # Erin
                             ('athinput.*',), # Chang-Goo's restart
                             ('*.par',)]
        
        hst_patterns = [('id0', '*.hst'),
                        ('hst', '*.hst'),
                        ('*.hst',)]
        
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
            self.out_fmt = [self.par[k]['out_fmt'] for k in self.par.keys() \
                            if 'output' in k]
            self.problem_id = self.par['job']['problem_id']
            self.logger.info('problem_id: {0:s}'.format(self.problem_id))
        else:
            self.logger.warning('Could not find athinput file in {0:s}'.\
                                format(self.basedir))
            self.out_fmt = self._out_fmt_def

        # Find history dump and
        # Extract problem_id (prefix for vtk and hitsory file names)
        if 'hst' in self.out_fmt:
            fhst = find_match(hst_patterns)
            if fhst:
                self.files['hst'] = fhst[0]
                self.problem_id = osp.basename(self.files['hst']).split('.')[0]
                self.logger.info('hst: {0:s}'.format(self.files['hst']))
            else:
                raise IOError('Could not find history file in {0:s}'.\
                              format(self.basedir))

        # Find vtk files
        # vtk files in both basedir (joined) and in basedir/id0
        if 'vtk' in self.out_fmt:
            self.files['vtk'] = find_match(vtk_patterns)
            self.files['vtk_id0'] = find_match(vtk_id0_patterns)
            if not self.files['vtk'] and not self.files['vtk_id0']:
                self.logger.error(
                    'No vtk files are found in {0:s}.'.format(self.basedir))
            else:
                self.nums = [int(f[-8:-4]) for f in self.files['vtk']]
                self.nums_id0 = [int(f[-8:-4]) for f in self.files['vtk_id0']]
                if self.nums:
                    self.logger.info('vtk (joined): {0:s} nums: {1:d}-{2:d}'.format(
                        osp.dirname(self.files['vtk'][0]),
                        self.nums[0], self.nums[-1]))
                if self.nums_id0:
                    self.logger.info('vtk in id0: {0:s} nums: {1:d}-{2:d}'.format(
                        osp.dirname(self.files['vtk_id0'][0]),
                        self.nums_id0[0], self.nums_id0[-1]))

            # Check (joined) vtk file size
            sizes = [os.stat(f).st_size for f in self.files['vtk']]
            if len(set(sizes)) > 1:
                size = max(set(sizes), key=sizes.count)
                flist = [(i, s // 1024**2) for i, s in enumerate(sizes) if s != size]
                self.logger.warning('Vtk file size is not unique.')
                # for f in flist:
                #    self.logger.debug('vtk num:', f[0], 'size [MB]:', f[1])

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

        # Find timeit.txt
        ftimeit = find_match(timeit_patterns)
        if ftimeit:
            self.files['timeit'] = ftimeit[0]
            self.logger.info('timeit: {0:s}'.format(self.files['timeit']))
        else:
            self.logger.info('No timeit.txt found.')
                
                
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
        
        # JKIM: I am sure there is a better way to do this..but this works anyway..
        def check_pickle_hst(read_hst):
            
            @functools.wraps(read_hst)
            def wrapper(cls, *args, **kwargs):
                if 'savdir' in kwargs:
                    savdir = kwargs['savdir']
                else:
                    savdir = os.path.join(cls.savdir, 'hst')

                if 'force_override' in kwargs:
                    force_override = kwargs['force_override']
                else:
                    force_override = False

                # Create savdir if it doesn't exist
                if not os.path.exists(savdir):
                    os.makedirs(savdir)
                    force_override = True

                fpkl = os.path.join(savdir,
                                    os.path.basename(cls.files['hst']) + '.mod.p')

                # Check if the original history file is updated
                if not force_override and os.path.exists(fpkl) and \
                   os.path.getmtime(fpkl) > os.path.getmtime(cls.files['hst']):
                    cls.logger.info('[read_hst]: Reading from existing pickle.')
                    hst = pd.read_pickle(fpkl)
                    cls.hst = hst
                    return hst
                else:
                    cls.logger.info('[read_hst]: Reading from original hst dump.')
                    # If we are here, force_override is True or history file is updated.
                    # Call read_hst function
                    hst = read_hst(cls, *args, **kwargs)
                    try:
                        hst.to_pickle(fpkl)
                    except IOError:
                        self.logger.warning('[read_hst]: Could not pickle hst to {0:s}.'.format(fpkl))
                    return hst

            return wrapper

    
class LoadSimAll(object):
    """Class to load multiple simulations
    """
    def __init__(self, models):

        self.models = list(models.keys())
        self.basedirs = dict()
        
        for mdl, basedir in models.items():
            self.basedirs[mdl] = basedir

    def set_model(self, model, savdir=None, load_method='pyathena', verbose=False):
        self.model = model
        self.sim = LoadSim(self.basedir[model], savdir=savdir,
                           load_method=load_method, verbose=verbose)
