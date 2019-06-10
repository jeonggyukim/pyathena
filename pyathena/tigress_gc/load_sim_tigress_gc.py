import os
import pandas as pd

from ..load_sim import LoadSim
from ..util.units import Units

from .read_hst import ReadHst
# from .read_zprof import ReadZprof
# from .plt_hst_zprof import PltHstZprof

class LoadSimTIGRESSGC(LoadSim, ReadHst): #, ReadZprof, PltHstZprof):
    """LoadSim class for analyzing TIGRESS-GC simulations.
    """

    def __init__(self, basedir, savdir=None, load_method='pyathena',
                 muH=1.4271, verbose=True):
        """The constructor for LoadSimTIGRESSGC class

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

        super(LoadSimTIGRESSGC,self).__init__(basedir, savdir=savdir,
                                               load_method=load_method, verbose=verbose)

        # Set unit
        self.u = Units(muH=muH)
        
        # Get domain info
        if not self.files['vtk']:
            self.logger.info('Loading {0:s}'.format(self.files['vtk_id0'][0]))
            self.ds = self.load_vtk(num=0, id0=True, load_method=load_method)
        else:
            self.ds = self.load_vtk(ivtk=0, load_method=load_method)


class LoadSimTIGRESSGCAll(object):
    """Class to load multiple simulations"""
    def __init__(self, models=None):

        # Default models
        if models is None:
            models = dict()
            # M1_4pc
            models['M1_4pc_pathfinder'] = '/home/smoon/data/gc/M1_4pc_pathfinder'
            models['TM1_4pc_fast_inflow'] = '/home/smoon/data/gc/TM1_4pc_fast_inflow'
            models['TM1_4pc_slow_inflow'] = '/home/smoon/data/gc/TM1_4pc_slow_inflow'

        self.models = list(models.keys())
        self.basedirs = dict()
        
        for mdl, basedir in models.items():
            self.basedirs[mdl] = basedir

    def set_model(self, model, savdir=None, load_method='pyathena', verbose=False):
        
        self.model = model
        self.sim = LoadSimTIGRESSGC(self.basedirs[model], savdir=savdir,
                                    load_method=load_method, verbose=verbose)
        return self.sim
