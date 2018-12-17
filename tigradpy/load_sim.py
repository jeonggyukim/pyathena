import os, sys
import glob
import warnings
import logging

import yt
from ..vtk_reader import AthenaDataSet
from .io.read_athinput import read_athinput

class LoadSim(object):
    """Class to prepare Athena simulation data analysis. Read input parameters, find
    vtk and history files.

    """

    def __init__(self, basedir, savdir=None, load_method='pyathena', verbose=True):
        """
        The constructor for load_sim class.

        Parameters
        ----------
        basedir : str
            Name of the directory where all data is stored
        savdir: str
            Name of the directory where pickled data and figures will be saved.
            Default value is basedir.
        load_method: str
            Load vtk using 'pyathena' or 'yt'. Default value is 'pyathena'.
            If None, savdir=basedir. Default value is None.
        verbose: bool
            Print verbose messages using logger.
        """
        
        self.basedir = basedir
        self.load_method = load_method
        self.logger = self._set_logger(verbose=verbose)
        self._find_files()

        if savdir is None:
            self.savdir = self.basedir
        else:
            self.savdir = savdir
            self.logger.info('savdir : {:s}'.format(savdir))
            

    def load_vtk(self, num=None, ivtk= None, id0=False, load_method=None):
        """Function to read Athena vtk file using pythena or yt and return DataSet object.
        
        Parameters
        ----------
        num : int
           Snapshot number. For example, basedir/vtk/problem_id.{num}.vtk
        ivtk: int
           Read i-th file in the vtk file list. Overrides num.
        id0: bool
           Read vtk file in basedir/id0. Default value is False.
        load_method: str
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
                dirname = os.path.dirname(self.files['vtk_id0'][0])
            else:
                dirname = os.path.dirname(self.files['vtk'][0])
               
            self.fvtk = os.path.join(dirname, '{0:s}.{1:04d}.vtk'.\
                                     format(self.problem_id, num))

        if ivtk is not None:
            if id0:
                self.fvtk = self.files['vtk_id0'][ivtk]
            else:
                self.fvtk = self.files['vtk'][ivtk]

        if not os.path.exists(self.fvtk):
            self.logger.error('[load_vtk]: Vtk file does not exist {:s}'.format(self.fvtk))
            raise
                
        if self.load_method == 'pyathena':
            self.ds = AthenaDataSet(self.fvtk)
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
        print('Attributes:', attr_list)
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
a
        vtk: problem_id.num.vtk
        hst: problem_id.hst
        zprof: problem_id.zprof
        starpar: problem_id.num.starpar.vtk"""
        
        self.files = dict()
        def find_match(patterns):
            glob_match = lambda p: \
                     sorted(glob.glob(os.path.join(self.basedir, *p)))
    
            for p in patterns:
                f = glob_match(p)
                if f:
                    break
                
            return f
        
        hst_patterns = [('id0', '*.hst'), ('hst', '*.hst'), ('*.hst',)]
        athinput_patterns = [('out.txt',), # Jeong-Gyu
                             ('slurm-*',), # Erin
                             ('athinput.*',), # Chang-Goo's restart
                             ('*.par',)]
        vtk_patterns = [('*.????.vtk',), ('vtk', '*.????.vtk',)]
        vtk_id0_patterns = [('id0', '*.????.vtk',)]
        star_patterns = [('id0', '*.????.starpar.vtk'),
                         ('starpar', '*.????.starpar.vtk'),
                         ('*.????.starpar.vtk',)]
        
        # find history dump and
        # extract problem_id (prefix for vtk and hitsory file names)
        fhst = find_match(hst_patterns)
        if fhst:
            self.files['hst'] = fhst[0]
            self.problem_id = os.path.basename(self.files['hst']).split('.')[0]
            self.logger.info('basedir: {0:s}'.format(self.basedir))
            self.logger.info('problem_id: {0:s}'.format(self.problem_id))
            self.logger.info('hst: {0:s}'.format(self.files['hst']))
        else:
            raise IOError('Could not find history file in {0:s}'.format(self.basedir))

        fathinput = find_match(athinput_patterns)
        if fathinput:
            self.files['athinput'] = fathinput[0]
            self.par = read_athinput(self.files['athinput'])
            self.logger.info('athinput: {0:s}'.format(self.files['athinput']))
        else:
            self.logger.warning('Could not find athinput file in {0:s}'.\
                                format(self.basedir))

        # Find vtk files
        # vtk files in basedir (joined) and in basedir/id0
        self.files['vtk'] = find_match(vtk_patterns)
        self.files['vtk_id0'] = find_match(vtk_id0_patterns)
        
        if not self.files['vtk'] and not self.files['vtk_id0']:
            self.logger.error(
                'No vtk files are found in {0:s}.'.format(self.basedir))
        else:
            self.nums = [int(f[-7:-4]) for f in self.files['vtk']]
            self.nums_id0 = [int(f[-7:-4]) for f in self.files['vtk_id0']]
            if self.nums:
                self.logger.info('vtk (joined) : {0:d}-{1:d}'.format(\
                            self.nums[0], self.nums[-1]))
            if self.nums_id0:
                self.logger.info('vtk in id0 : {0:d}-{1:d}'.format(\
                            self.nums_id0[0], self.nums_id0[-1]))

        # Find starpar files
        fstar = find_match(star_patterns)
        if fstar:
            self.files['star'] = fstar
            self.nums_star = [int(f[-16:-12]) for f in self.files['star']]
            self.logger.info('starpar: {0:d}-{1:d}'.format(\
                        self.nums_star[0], self.nums_star[-1]))
        else:
            self.logger.warning(
                'No starpar files are found in {0:s}.'.format(self.basedir))

        # Check (joined) vtk file size
        sizes = [os.stat(f).st_size for f in self.files['vtk']]
        if len(set(sizes)) > 1:
            size = max(set(sizes), key=sizes.count)
            flist = [(i, s) for i, s in enumerate(sizes) if s != size]
            self.logger.warning('Check vtk file size. Should be {0:d} bytes.'.format(size))
            for f in flist:
                self.logger.warning('vtk num:', f[0], 'size:', f[1])
        
    def _set_logger(self, verbose=False):
        """Function to set logger

        Parameters
        ----------
        verbose: bool
            Set logging level to DEBUG/WARNING if True/False.
        """
        if verbose:
            loglevel = logging.DEBUG
        else:
            loglevel = logging.WARNING
        
        l = logging.getLogger(__class__.__name__.split('.')[-1])
        if not l.hasHandlers():
            h = logging.StreamHandler()
            f = logging.Formatter('%(name)s-%(levelname)s: %(message)s')
            # f = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            h.setFormatter(f)
            l.addHandler(h)
            l.setLevel(loglevel)
        else:
            l.setLevel(loglevel)

        return l
    
