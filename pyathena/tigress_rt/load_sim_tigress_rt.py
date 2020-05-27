import os
import pandas as pd
import os.path as osp

from ..load_sim import LoadSim
from ..util.units import Units

from .hst import Hst
from .zprof import Zprof
from .slc_prj import SliceProj

class LoadSimTIGRESSRT(LoadSim, Hst, Zprof, SliceProj):
    """LoadSim class for analyzing TIGRESS-RT simulations.
    """
    
    def __init__(self, basedir, savdir=None, load_method='pyathena',
                 verbose=False):
        """The constructor for LoadSimTIGRESSRT class

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

        super(LoadSimTIGRESSRT,self).__init__(basedir, savdir=savdir,
                                              load_method=load_method, verbose=verbose)
        
        # Set unit and domain
        self.u = Units(muH=1.4271)
        self.domain = self._get_domain_from_par(self.par)

    # def load_vtk(self, num=None, ivtk=None, id0=False, load_method=None,
    #              AthenaDataSet=AthenaDataSet):

    #     self.ds = super(LoadSimTIGRESSRT,self).load_vtk(num, ivtk, id0, load_method,
    #                                                     AthenaDataSet)
    #     return self.ds

class LoadSimTIGRESSRTAll(object):
    """Class to load multiple simulations"""
    def __init__(self, models=None):

        # Default models
        if models is None:
            models = dict()
            models['R8_8pc_rad'] = '/perseus/scratch/gpfs/jk11/TIGRESS-RT/R8_8pc_rad.implicit.test'

        self.models = list(models.keys())
        self.basedirs = dict()
        
        for mdl, basedir in models.items():
            self.basedirs[mdl] = basedir

    def set_model(self, model, savdir=None, load_method='pyathena', verbose=False):
        
        self.model = model
        self.sim = LoadSimTIGRESSRT(self.basedirs[model], savdir=savdir,
                                    load_method=load_method, verbose=verbose)
        return self.sim
