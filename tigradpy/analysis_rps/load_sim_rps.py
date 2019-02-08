import os
import pandas as pd

from ..load_sim import LoadSim
from ..util.units import Units

from .read_hst import ReadHst
from .read_zprof import ReadZprof
from .plt_hst import PltHst

class LoadSimRPS(LoadSim, ReadHst, ReadZprof, PltHst):
    """LoadSim class for analyzing radps_postproc simulations.
    """
    
    def __init__(self, basedir, savdir=None, load_method='pyathena', verbose=True):
        """The constructor for LoadSimRPS class

        Parameters
        ----------
        basedir : str
            Name of the directory in which all data is stored.
        savdir : str, optional
            Name of the directory where pickled data and figures will be saved.
            Default value is basedir.
        load_method : str, optional
            Load vtk using 'pyathena' or 'yt'. Default value is 'pyathena'.
        verbose : bool, optional
            If True, print verbose messages using logger.
        """
        
        super().__init__(basedir, savdir=savdir,
                         load_method=load_method, verbose=verbose)

        # Set unit
        self.u = Units(muH=1.4271)
        
        # Get domain info
        if not self.files['vtk']:
            self.logger.info('Loading {0:s}'.format(self.files['vtk_id0'][0]))
            self.ds = self.load_vtk(num=0, id0=True, load_method='pyathena', verbose=False)
        else:
            self.ds = self.load_vtk(ivtk=0, load_method='pyathena', verbose=False)
            
class LoadSimRPSAll(object):
    def __init__(self, basedirs=None):

        models = [
            '/tigress/jk11/radps_postproc/R8_4pc_newacc.xymax1024',
            '/tigress/jk11/radps_postproc/R8_4pc_newacc.xymax1024.runaway',
            '/tigress/jk11/radps_postproc/R8_8pc_rst.xymax1024',
            '/tigress/jk11/radps_postproc/R8_8pc_rst.xymax1024'
        ]
