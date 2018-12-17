import os
import pandas as pd

from ..load_sim import LoadSim
from ..util.units import Units

from .read_hst import ReadHst
from .plt_hst import PltHst

class LoadSimRPS(LoadSim, ReadHst, PltHst):
    """LoadSim class for radps_postproc simulations

    """
    
    def __init__(self, basedir, savdir=None, load_method='pyathena', verbose=True):
        """The constructor for LoadSimRPS class

        Parameters
        ----------
        basedir: str
            Name of the directory in which all data is stored
        savdir: str
            Name of the directory where pickled data and figures will be saved.
            Default value is basedir.
        load_method: str
            Load vtk using 'pyathena' or 'yt'. Default value is 'pyathena'.
            If None, savdir=basedir. Default value is None.
        verbose: bool
            Print verbose messages using logger.
        """
        
        super().__init__(basedir, savdir=savdir,
                         load_method=load_method, verbose=verbose)

        # Set unit
        self.u = Units(muH=1.4271)
        
        # Get domain info
        if not self.files['vtk']:
            self.ds = self.load_vtk(num=0, id0=True, load_method='pyathena')
        else:
            self.ds = self.load_vtk(ivtk=0, load_method='pyathena')

    # def get_method_list(self):
    #     for func in dir(self):
    #         if not func.startswith("__"):
    #             if callable(getattr(self, func)):
    #                 print(func)
                
        #return [func for func in dir(self) if \
        #        callable(getattr(self.__weakref__, func)) and not func.startswith("__")]
            
class LoadSimRPSAll(object):
    def __init__(self, basedirs=None):

        models = ['/tigress/jk11/radps_postproc/R8_8pc_rst.xymax1024',
                  '/scratch/gpfs/jk11/radps_postproc/R8_8pc_rst.xymax2048'
        ]
