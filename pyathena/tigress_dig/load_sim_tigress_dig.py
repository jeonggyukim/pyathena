import os
import pandas as pd

from ..load_sim import LoadSim
from ..util.units import Units

from .hst import Hst
from .zprof import Zprof
from .plt_hst_zprof import PltHstZprof
from .extract_data import ExtractData

class LoadSimTIGRESSDIG(LoadSim, Hst, Zprof, PltHstZprof, ExtractData):
    """LoadSim class for analyzing TIGRESS DIG simulations.
    """

    def __init__(self, basedir, savdir=None, load_method='pyathena', verbose=False):
        """The constructor for LoadSimTIGRESSDIG class

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

        super(LoadSimTIGRESSDIG,self).__init__(basedir, savdir=savdir,
                                        load_method=load_method, verbose=verbose)

        # Set unit
        self.u = Units(muH=1.4271)

        # Get domain info
        if not self.files['vtk']:
            self.logger.info('Loading {0:s}'.format(self.files['vtk_id0'][0]))
            self.ds = self.load_vtk(num=0, id0=True, load_method=load_method)
        else:
            self.ds = self.load_vtk(ivtk=0, load_method=load_method)



class LoadSimTIGRESSDIGAll(object):
    """Class to load multiple simulations"""
    def __init__(self, models=None):

        # Default models
        if models is None:
            models = dict()

        self.models = list(models.keys())
        self.basedirs = dict()

        for mdl, basedir in models.items():
            self.basedirs[mdl] = basedir

    def set_model(self, model, savdir=None, load_method='pyathena', verbose=False):

        self.model = model
        self.sim = LoadSimTIGRESSDIG(self.basedirs[model], savdir=savdir,
                                     load_method=load_method, verbose=verbose)
        return self.sim
