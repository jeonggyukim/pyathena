import os
import pandas as pd

from ..load_sim import LoadSim
from ..util.units import Units

from .hst import Hst

class LoadSimTIGRESSXCO(LoadSim, Hst):
    """LoadSim class for analyzing TIGRESS simulations.
    """

    def __init__(self, basedir, savdir=None, load_method='pyathena',
                 muH=1.4271, verbose=False):
        """The constructor for LoadSimRPS class

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

        super(LoadSimTIGRESSXCO,self).__init__(basedir, savdir=savdir,
                                               load_method=load_method, verbose=verbose)

        # Set unit
        self.u = Units(muH=muH)

        # Get domain info
        if not self.files['vtk']:
            self.logger.info('Loading {0:s}'.format(self.files['vtk_id0'][0]))
            self.ds = self.load_vtk(num=0, id0=True, load_method=load_method)
        else:
            self.ds = self.load_vtk(ivtk=0, load_method=load_method)


class LoadSimTIGRESSXCOAll(object):
    """Class to load multiple simulations"""
    def __init__(self, models=None):

        # Default models
        if models is None:
            models = dict()
            # R2_2pc_L256 Z1
            models['R2_2pc_L256.Z1.CR001.L001'] = '/projects/EOSTRIKE/TIGRESS_XCO_ART/R2_2pc_L256_B2.noHII.Z1.CR001.L001'
            models['R2_2pc_L256.Z1.CR010.L010'] = '/projects/EOSTRIKE/TIGRESS_XCO_ART/R2_2pc_L256_B2.noHII.Z1.CR010.L010'
            models['R2_2pc_L256.Z1.CR001.L010'] = '/projects/EOSTRIKE/TIGRESS_XCO_ART/R2_2pc_L256_B2.noHII.Z1.CR001.L010'

            # R2_2pc_L256 Z2
            models['R2_2pc_L256.Z2.CR001.L001'] = '/projects/EOSTRIKE/TIGRESS_XCO_ART/R2_2pc_L256_B2.noHII.Z2.CR001.L001'
            models['R2_2pc_L256.Z2.CR001.L010'] = '/projects/EOSTRIKE/TIGRESS_XCO_ART/R2_2pc_L256_B2.noHII.Z2.CR001.L010'
            models['R2_2pc_L256.Z2.CR010.L001'] = '/projects/EOSTRIKE/TIGRESS_XCO_ART/R2_2pc_L256_B2.noHII.Z2.CR010.L001'
            models['R2_2pc_L256.Z2.CR010.L010'] = '/projects/EOSTRIKE/TIGRESS_XCO_ART/R2_2pc_L256_B2.noHII.Z2.CR010.L010'

            # R4_2pc_L512
            models['R4_2pc_L512.Z1.CR001.L010'] = '/projects/EOSTRIKE/TIGRESS_XCO_ART/R4_2pc_L512_B10.noHII.Z1.CR001.L010'
            models['R4_2pc_L512.Z1.CR010.L010'] = '/projects/EOSTRIKE/TIGRESS_XCO_ART/R4_2pc_L512_B10.noHII.Z1.CR010.L010'

            models['R4_2pc_L512.Z2.CR010.L010'] = '/projects/EOSTRIKE/TIGRESS_XCO_ART/R4_2pc_L512_B10.noHII.Z2.CR010.L010'

            # R8_2pc_rst
            models['R8_2pc_rst.Z1.CR100.L100'] = '/projects/EOSTRIKE/TIGRESS_XCO_ART/R8_2pc_rst.noHII.Z1.CR100.L100'

            # M1_4pc - with starpar mask
            models['M1_4pc.Z2.CR010.L100'] = '/perseus/scratch/gpfs/jk11/TIGRESS-XCO/M1_4pc.noHII.Z2.CR010.L100'
            # M1_4pc - without starpar mask
            models['M1_4pc.Z2.CR010.L100.nomask'] = '/perseus/scratch/gpfs/jk11/TIGRESS-XCO/M1_4pc.noHII.Z2.CR010.L100.nomask'
            models['M1_4pc.Z2.CR010.L100.nomask.taumax20'] = '/perseus/scratch/gpfs/jk11/TIGRESS-XCO/M1_4pc.noHII.Z2.CR010.L100.nomask.taumax20'

            # M1_2pc_Tth50
            models['M1_2pc_Tth50.Z2.CR010.L100'] = '/projects/EOSTRIKE/TIGRESS_XCO_ART/M1_2pc_Tth50.noHII.Z2.CR010.L100'


        self.models = list(models.keys())
        self.basedirs = dict()

        for mdl, basedir in models.items():
            self.basedirs[mdl] = basedir

    def set_model(self, model, savdir=None, load_method='pyathena', verbose=False):

        self.model = model
        self.sim = LoadSimTIGRESSXCO(self.basedirs[model], savdir=savdir,
                                     load_method=load_method, verbose=verbose)
        return self.sim
