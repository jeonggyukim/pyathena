import os
import pandas as pd

from ..load_sim import LoadSim
from ..util.units import Units

from .hst import Hst

class LoadSimTIGRESSSingleSN(LoadSim, Hst):
    """LoadSim class for analyzing TIGRESS-SINGLE-SN simulations.
    """

    def __init__(self, basedir, savdir=None, load_method='pyathena',
                 verbose=False):
        """The constructor for LoadSimTIGRESSSingleSN class

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

        super(LoadSimTIGRESSSingleSN,self).__init__(basedir, savdir=savdir,
                                                    load_method=load_method,
                                                    verbose=verbose)

        # Set unit
        self.u = Units(muH=1.4271)

        # Get domain info
        if not self.files['vtk']:
            self.logger.info('Loading {0:s}'.format(self.files['vtk_id0'][0]))
            self.ds = self.load_vtk(num=0, id0=True, load_method=load_method)
        else:
            self.ds = self.load_vtk(ivtk=0, load_method=load_method)


class LoadSimTIGRESSSingleSNAll(object):
    """Class to load multiple simulations"""
    def __init__(self, models=None):

        # Default models
        if models is None:
            models = dict()
            # models['oldcool.implicit'] = '/perseus/scratch/gpfs/jk11/TIGRESS-SINGLE-SN/SNR_r1_N256.oldcool.cooldump.fcool05/'
            models['oldcool.fcool05.cool_before_io'] = '/perseus/scratch/gpfs/jk11/TIGRESS-SINGLE-SN/SNR_r1_N256.oldcool.cooldump.fcool05.cooling_called_before_output/'
            models['newcool.fcool05'] = '/perseus/scratch/gpfs/jk11/TIGRESS-SINGLE-SN/SNR_r1_N256.newcool.fcool05/'
            models['newcool.fcool10'] = '/perseus/scratch/gpfs/jk11/TIGRESS-SINGLE-SN/SNR_r1_N256.newcool.fcool10/'
            models['newcool.fcool20'] = '/perseus/scratch/gpfs/jk11/TIGRESS-SINGLE-SN/SNR_r1_N256.newcool.fcool20/'
            models['newcool.fcool10.fastexp'] = '/perseus/scratch/gpfs/jk11/TIGRESS-SINGLE-SN/SNR_r1_N256.newcool.fcool10-fastexp/'
            models['newcool.fcool10.fastexplog'] = '/perseus/scratch/gpfs/jk11/TIGRESS-SINGLE-SN/SNR_r1_N256.newcool.fcool10-fastexplog/'
            models['newcool.fcool10.fastexplogpow'] = '/perseus/scratch/gpfs/jk11/TIGRESS-SINGLE-SN/SNR_r1_N256.newcool.fcool10-fastexplogpow/'
            models['newcool.fcool10.fastexplogpow-further'] = '/perseus/scratch/gpfs/jk11/TIGRESS-SINGLE-SN/SNR_r1_N256.newcool.fcool10-fastexplogpow-further-optimization/'

        self.models = list(models.keys())
        self.basedirs = dict()

        for mdl, basedir in models.items():
            self.basedirs[mdl] = basedir

    def set_model(self, model, savdir=None, load_method='pyathena', verbose=False):

        self.model = model
        self.sim = LoadSimTIGRESSSingleSN(self.basedirs[model], savdir=savdir,
                                          load_method=load_method, verbose=verbose)
        return self.sim
