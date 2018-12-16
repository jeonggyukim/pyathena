import os
import pandas as pd

from ..load_sim import LoadSim
from ..util.units import Units

from .read_hst import ReadHst
from .plt_hst import PltHst

class LoadSimRPS(LoadSim, ReadHst, PltHst):
    """LoadSim class for radps_postproc simulations

    """
    
    def __init__(self, basedir, load_method='pyathena', verbose=True):
        """The constructor for LoadSimRPS class

        Parameters
        ----------
        basedir: str
            Name of the directory in which all data is stored
        """
        
        super().__init__(basedir, load_method='pyathena',
                         verbose=verbose)

        # Set unit
        self.u = Units(muH=1.4271)
        # Get domain info
        if not self.files['vtk']:
            self.ds = self.load_vtk(num=0, id0=True, load_method='pyathena')
        else:
            self.ds = self.load_vtk(ivtk=0, load_method='pyathena')    

class LoadSimRPSAll(object):
    def __init__(self, basedirs=None):

        models = ['/tigress/jk11/radps_postproc/R8_8pc_rst.xymax1024',
                  '/scratch/gpfs/jk11/radps_postproc/R8_8pc_rst.xymax2048'
        ]
