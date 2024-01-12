## Taken from yt/fields/xray_emission_fields.py
## https://yt-project.org/doc/analyzing/analysis_modules/xray_emission_fields.html#

from yt.utilities.on_demand_imports import _h5py as h5py
import numpy as np
import os

from yt.config import ytcfg
from yt.fields.derived_field import DerivedField
from yt.funcs import \
    mylog, \
    only_on_root, \
    parse_h5_attr
from yt.utilities.exceptions import YTFieldNotFound
from yt.utilities.exceptions import YTException
from yt.utilities.linear_interpolators import \
    UnilinearFieldInterpolator, BilinearFieldInterpolator
from yt.units.yt_array import YTArray, YTQuantity
from yt.utilities.cosmology import Cosmology

data_version = {"cloudy": 2,
                "apec": 2}

def _get_data_file(table_type, data_dir=None):
    data_file = "%s_emissivity_v%d.h5" % (table_type, data_version[table_type])
    if data_dir is None:
        supp_data_dir = ytcfg.get("yt", "supp_data_dir")
        data_dir = supp_data_dir if os.path.exists(supp_data_dir) else "."
    data_path = os.path.join(data_dir, data_file)
    if not os.path.exists(data_path):
        msg = "Failed to find emissivity data file %s in %s! " % (data_file, data_path) \
        + "Please download from http://yt-project.org/data!"
        mylog.error(msg)
        raise IOError(msg)
    return data_path

class EnergyBoundsException(YTException):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __str__(self):
        return "Energy bounds are %e to %e keV." % \
          (self.lower, self.upper)

class ObsoleteDataException(YTException):
    def __init__(self, table_type):
        data_file = "%s_emissivity_v%d.h5" % (table_type, data_version[table_type])
        self.msg = "X-ray emissivity data is out of date.\n"
        self.msg += "Download the latest data from %s/%s." % (data_url, data_file)

    def __str__(self):
        return self.msg

class XrayEmissivityIntegrator(object):
    r"""Class for making X-ray emissivity fields. Uses hdf5 data tables
    generated from Cloudy and AtomDB/APEC.

    Initialize an XrayEmissivityIntegrator object.

    Parameters
    ----------
    table_type : string
        The type of data to use when computing the emissivity values. If "cloudy",
        a file called "cloudy_emissivity.h5" is used, for photoionized
        plasmas. If, "apec", a file called "apec_emissivity.h5" is used for
        collisionally ionized plasmas. These files contain emissivity tables
        for primordial elements and for metals at solar metallicity for the
        energy range 0.1 to 100 keV.
    redshift : float, optional
        The cosmological redshift of the source of the field. Default: 0.0.
    data_dir : string, optional
        The location to look for the data table in. If not supplied, the file
        will be looked for in the location of the YT_DEST environment variable
        or in the current working directory.
    use_metals : boolean, optional
        If set to True, the emissivity will include contributions from metals.
        Default: True
    """
    def __init__(self, table_type, redshift=0.0, data_dir=None, use_metals=True):

        mylog.setLevel(50)
        filename = _get_data_file(table_type, data_dir=data_dir)
        only_on_root(mylog.info, "Loading emissivity data from %s." % filename)
        in_file = h5py.File(filename, "r")
        if "info" in in_file.attrs:
            only_on_root(mylog.info, parse_h5_attr(in_file, "info"))
        if parse_h5_attr(in_file, "version") != data_version[table_type]:
            raise ObsoleteDataException(table_type)
        else:
            only_on_root(mylog.info, "X-ray '%s' emissivity data version: %s." % \
                         (table_type, parse_h5_attr(in_file, "version")))

        self.log_T = in_file["log_T"][:]
        self.emissivity_primordial = in_file["emissivity_primordial"][:]
        if "log_nH" in in_file:
            self.log_nH = in_file["log_nH"][:]
        if use_metals:
            self.emissivity_metals = in_file["emissivity_metals"][:]
        self.ebin = YTArray(in_file["E"], "keV")
        in_file.close()
        self.dE = np.diff(self.ebin)
        self.emid = 0.5*(self.ebin[1:]+self.ebin[:-1]).to("erg")
        self.redshift = redshift

    def get_interpolator(self, data_type, e_min, e_max, energy=True):
        data = getattr(self, "emissivity_%s" % data_type)
        if not energy:
            data = data[..., :] / self.emid.v
        e_min = YTQuantity(e_min, "keV")*(1.0+self.redshift)
        e_max = YTQuantity(e_max, "keV")*(1.0+self.redshift)
        if (e_min - self.ebin[0]) / e_min < -1e-3 or \
          (e_max - self.ebin[-1]) / e_max > 1e-3:
            raise EnergyBoundsException(self.ebin[0], self.ebin[-1])
        e_is, e_ie = np.digitize([e_min, e_max], self.ebin)
        e_is = np.clip(e_is - 1, 0, self.ebin.size - 1)
        e_ie = np.clip(e_ie, 0, self.ebin.size - 1)

        my_dE = self.dE[e_is: e_ie].copy()
        # clip edge bins if the requested range is smaller
        my_dE[0] -= e_min - self.ebin[e_is]
        my_dE[-1] -= self.ebin[e_ie] - e_max

        interp_data = (data[..., e_is:e_ie]*my_dE).sum(axis=-1)
        if data.ndim == 2:
            emiss = UnilinearFieldInterpolator(np.log10(interp_data),
                                               [self.log_T[0],  self.log_T[-1]],
                                               "log_T", truncate=True)
        else:
            emiss = BilinearFieldInterpolator(np.log10(interp_data),
                                              [self.log_nH[0], self.log_nH[-1],
                                               self.log_T[0],  self.log_T[-1]],
                                              ["log_nH", "log_T"], truncate=True)

        return emiss

def get_xray_emissivity(T, Z=1.0, emin=0.5, emax=7.0, table_type='apec', energy=True):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, '../../data')
    x = XrayEmissivityIntegrator(table_type, data_dir=data_dir)
    log_em_0 = x.get_interpolator('primordial', emin, emax, energy=energy)
    log_em_z = x.get_interpolator('metals', emin, emax, energy=energy)

    dd = dict(log_nH=0.0, log_T=np.log10(T))
    em_tot = 10.0**log_em_0(dd) + Z*10.0**log_em_z(dd)
    return em_tot
