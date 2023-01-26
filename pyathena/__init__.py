__version__ = '0.1'

# __all__ is taken to be the list of module names that should be imported when
# from `package import *` is encountered
# https://docs.python.org/3/tutorial/modules.html#importing-from-a-package

__all__ = [
           # Units class
           "Units",
           # LoadSim
           "LoadSim",
           "LoadSimAll",
           # SFCloud
           "LoadSimSFCloud",
           "LoadSimSFCloudAll",
           # SFCloudRad
           "LoadSimSFCloudRad",
           "LoadSimSFCloudRadAll",
           # FeedbackTest
           "LoadSimFeedbackTest",
           "LoadSimFeedbackTestAll",
           # DIG
           "LoadSimTIGRESSDIG",
           "LoadSimTIGRESSDIGAll",
           # NCR
           "LoadSimTIGRESSNCR",
           "LoadSimTIGRESSNCRAll",
           # Single SN
           "LoadSimTIGRESSSingleSN",
           "LoadSimTIGRESSSingleSNAll",
           # XCO
           "LoadSimTIGRESSXCO",
           "LoadSimTIGRESSXCOAll",
           # CoreCollapse
           "LoadSimCoreCollapse",
           "LoadSimCoreCollapseAll",
           ]

from .io.read_vtk import read_vtk, AthenaDataSet
from .io.read_athinput import read_athinput
from .io.read_hst import read_hst
from .io.read_sphst import read_sphst
from .io.read_timeit import read_timeit
from .io.read_starpar_vtk import read_starpar_vtk
from .io.read_zprof import read_zprof, read_zprof_all
from .io.read_rst import read_rst, RestartHandler

from .classic.vtk_reader import AthenaDataSet as AthenaDataSetClassic

# LoadSim classes
from .load_sim import LoadSim, LoadSimAll

# Problem specific subclasses
from .feedback_test.load_sim_feedback_test import LoadSimFeedbackTest, LoadSimFeedbackTestAll

from .sf_cloud.load_sim_sf_cloud import LoadSimSFCloud, LoadSimSFCloudAll
from .sf_cloud_rad.load_sim_sf_cloud_rad import LoadSimSFCloudRad, LoadSimSFCloudRadAll

from .tigress_dig.load_sim_tigress_dig import LoadSimTIGRESSDIG, LoadSimTIGRESSDIGAll
from .tigress_single_sn.load_sim_tigress_single_sn import LoadSimTIGRESSSingleSN, LoadSimTIGRESSSingleSNAll
from .tigress_xco.load_sim_tigress_xco import LoadSimTIGRESSXCO, LoadSimTIGRESSXCOAll
from .tigress_ncr.load_sim_tigress_ncr import LoadSimTIGRESSNCR, LoadSimTIGRESSNCRAll
from .core_collapse.load_sim_core_collapse import LoadSimCoreCollapse, LoadSimCoreCollapseAll

# ReadObs class
from .obs.read_obs import ReadObs

# Utils
from .util.units import Units, ac, au
from .util.rebin import rebin_xyz, rebin_xy
from .util.mass_to_lum import mass_to_lum

from .plt_tools.cmap import Colormaps, cmap_shift, get_cmap_planck, get_cmap_parula
from .plt_tools.plt_starpar import scatter_sp
from .plt_tools.make_movie import make_movie, display_movie
from .plt_tools.set_plt import set_plt_default, set_plt_fancy

# Microphysics
from .microphysics import cool, cool_gnat12, rec_rate, photx, dust_draine
