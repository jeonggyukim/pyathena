from __future__ import print_function

import os
import sys
import glob, re
import warnings
import logging
import os.path as osp

import yt
from ..vtk_reader import AthenaDataSet
from .io.read_athinput import read_athinput

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
            Load vtk using 'pyathena' or 'yt'. Default value is 'pyathena'.
            If None, savdir=basedir. Default value is None.
        verbose : bool or str or int
            Print verbose messages using logger. If True/False, set logger
            level to 'DEBUG'/'WARNING'. If string, it should be one of the string
            representation of python logging package:
            ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            Numerical values from 0 ('NOTSET') to 50 ('CRITICAL') are also
            accepted.
        """

        self.basedir = basedir.rstrip('/')
        self.name = osp.basename(self.basedir)
        
        self.load_method = load_method
        self.logger = self._get_logger(verbose=verbose)
        self._find_files()

        if savdir is None:
            self.savdir = self.basedir
        else:
            self.savdir = savdir
            self.logger.info('savdir : {:s}'.format(savdir))
            

    def load_vtk(self, num=None, ivtk= None, id0=False, load_method=None,
                 verbose=True):
        """Function to read Athena vtk file using pythena or yt and 
        return DataSet object.
        
        Parameters
        ----------
        num : int
           Snapshot number. For example, basedir/vtk/problem_id.{num}.vtk
        ivtk : int
           Read i-th file in the vtk file list. Overrides num.
        id0 : bool
           Read vtk file in basedir/id0. Default value is False.
        load_method : str
           'pyathena' or 'yt'
        
        Returns
        -------
        ds : AthenaDataSet
        """
        
        # Override load_method
        if load_method is not None:
            self.load_method = load_method

        if num is not None:
            if id0:
                dirname = osp.dirname(self.files['vtk_id0'][0])
            else:
                dirname = osp.dirname(self.files['vtk'][0])
               
            self.fvtk = osp.join(dirname, '{0:s}.{1:04d}.vtk'.\
                                     format(self.problem_id, num))

        if ivtk is not None:
            if id0:
                self.fvtk = self.files['vtk_id0'][ivtk]
            else:
                self.fvtk = self.files['vtk'][ivtk]

        if not osp.exists(self.fvtk):
            self.logger.error('[load_vtk]: Vtk file does not exist {:s}'.\
                              format(self.fvtk))
            raise
                
        if self.load_method == 'pyathena':
            self.ds = AthenaDataSet(self.fvtk)
            self.domain = self.ds.domain
            if verbose:
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
                          ('id0', '*.????.zprof')]

        self.logger.info('basedir: {0:s}'.format(self.basedir))

        # Read athinput files
        fathinput = find_match(athinput_patterns)
        if fathinput:
            self.files['athinput'] = fathinput[0]
            self.par = read_athinput(self.files['athinput'])
            self.logger.info('athinput: {0:s}'.format(self.files['athinput']))
        else:
            self.logger.warning('Could not find athinput file in {0:s}'.\
                                format(self.basedir))

        self.out_fmt = [self.par[k]['out_fmt'] for k in self.par.keys() \
                        if 'output' in k]
        self.problem_id = self.par['job']['problem_id']
        self.logger.info('problem_id: {0:s}'.format(self.problem_id))

        # find history dump and
        # extract problem_id (prefix for vtk and hitsory file names)
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
        # vtk files in basedir (joined) and in basedir/id0
        if 'vtk' in self.out_fmt:
            self.files['vtk'] = find_match(vtk_patterns)
            self.files['vtk_id0'] = find_match(vtk_id0_patterns)
            if not self.files['vtk'] and not self.files['vtk_id0']:
                self.logger.error(
                    'No vtk files are found in {0:s}.'.format(self.basedir))
            else:
                self.nums = [int(f[-7:-4]) for f in self.files['vtk']]
                self.nums_id0 = [int(f[-7:-4]) for f in self.files['vtk_id0']]
                if self.nums:
                    self.logger.info('vtk (joined): {0:s} nums: {1:d}-{2:d}'.format(
                        osp.dirname(self.files['vtk'][0]),
                        self.nums[0], self.nums[-1]))
                if self.nums_id0:
                    self.logger.info('vtk in id0:  {0:s} nums: {1:d}-{2:d}'.format(
                        osp.dirname(self.files['vtk_id0'][0]),
                        self.nums_id0[0], self.nums_id0[-1]))

            # Check (joined) vtk file size
            sizes = [os.stat(f).st_size for f in self.files['vtk']]
            if len(set(sizes)) > 1:
                size = max(set(sizes), key=sizes.count)
                flist = [(i, s // 1024**2) for i, s in enumerate(sizes) if s != size]
                self.logger.warning('Vtk file size is not unique.')
                #for f in flist:
                #    self.logger.warning('vtk num:', f[0], 'size:', f[1])

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
    
