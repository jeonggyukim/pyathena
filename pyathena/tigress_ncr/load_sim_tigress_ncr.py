import os
import os.path as osp
import pandas as pd
import xarray as xr
import numpy as np
import astropy.constants as ac
# import astropy.units as au

from ..load_sim import LoadSim
from ..util.units import Units
from ..io.read_hst import read_hst
from ..classic.cooling import coolftn
from .pdf import PDF
from .h2 import H2
from .hst import Hst
from .zprof import Zprof
from .slc_prj import SliceProj
from .starpar import StarPar
from .snapshot_HIH2EM import Snapshot_HIH2EM
from .profile_1d import Profile1D
from .get_cooling import get_cooling_heating, get_pdfs


class LoadSimTIGRESSNCR(
    LoadSim, Hst, Zprof, SliceProj, StarPar, PDF, H2, Profile1D, Snapshot_HIH2EM
):
    """LoadSim class for analyzing TIGRESS-RT simulations."""

    def __init__(
        self, basedir, savdir=None, load_method="pyathena", muH=1.4271, verbose=False
    ):
        """The constructor for LoadSimTIGRESSNCR class

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

        super(LoadSimTIGRESSNCR, self).__init__(
            basedir, savdir=savdir, load_method=load_method, verbose=verbose
        )

        # Set unit and domain
        try:
            muH = self.par["problem"]["muH"]
        except KeyError:
            pass
        self.muH = muH
        self.u = Units(muH=muH)
        self.domain = self._get_domain_from_par(self.par)
        if self.test_newcool():
            self.test_newcool_params()

    def test_newcool(self):
        try:
            if self.par["configure"]["new_cooling"] == "ON":
                newcool = True
            else:
                newcool = False
        except KeyError:
            newcool = False
        return newcool

    def test_spiralarm(self):
        try:
            if self.par["configure"]["SpiralArm"] == "yes":
                arm = True
            else:
                arm = False
        except KeyError:
            arm = False
        return arm

    def test_newcool_params(self):
        s = self
        try:
            s.iCoolH2colldiss = s.par["cooling"]["iCoolH2colldiss"]
        except KeyError:
            s.iCoolH2colldiss = 0

        try:
            s.iCoolH2rovib = s.par["cooling"]["iCoolH2rovib"]
        except KeyError:
            s.iCoolH2rovib = 0

        try:
            s.ikgr_H2 = s.par["cooling"]["ikgr_H2"]
        except KeyError:
            s.ikgr_H2 = 0

        s.config_time = pd.to_datetime(s.par["configure"]["config_date"])
        if "PDT" in s.par["configure"]["config_date"]:
            s.config_time = s.config_time.tz_localize("US/Pacific")
        if s.config_time < pd.to_datetime("2021-06-30 20:29:36 -04:00"):
            s.iCoolHIcollion = 0
        else:
            s.iCoolHIcollion = 1

        # check this is run with corrected CR heating
        # 85a7857bb7c797686a4e9630cba71f326e1097cd
        if s.config_time < pd.to_datetime("2022-05-23 22:23:43 -04:00"):
            s.oldCRheating = 1
        else:
            s.oldCRheating = 0

        try:
            s.iH2heating = s.par["cooling"]["iH2heating"]
        except KeyError:
            s.iH2heating = -1

    def show_timeit(self):
        import matplotlib.pyplot as plt

        try:
            time = pd.read_csv(self.files["timeit"], delim_whitespace=True)

            tfields = [k.split("_")[0] for k in time.keys() if k.endswith("tot")]

            for tf in tfields:
                if tf == "rayt":
                    continue
                plt.plot(
                    time["time"], time[tf].cumsum() / time["all"].cumsum(), label=tf
                )
            plt.legend()
        except KeyError:
            print("No timeit plot is available")

    def get_timeit(self):
        try:
            time = pd.read_csv(self.files["timeit"], delim_whitespace=True)

            tfields = [k.split("_")[0] for k in time.keys() if k.endswith("tot")]

            return time[tfields]
        except KeyError:
            print("No timeit file is available")
            raise KeyError

    def get_timeit_mean(self):
        try:
            return self.get_timeit().mean()
        except KeyError:
            print("No timeit file is available")

    def get_classic_cooling_rate(self, ds):
        if not hasattr(self, "heat_ratio"):
            hst = read_hst(self.files["hst"])
            self.heat_ratio = hst["heat_ratio"]
            self.heat_ratio.index = hst["time"]
        dd = ds.get_field(["density", "pressure"])
        nH = dd["density"]
        heat_ratio = np.interp(
            ds.domain["time"], self.heat_ratio.index, self.heat_ratio
        )
        T1 = dd["pressure"] / dd["density"]
        T1 *= (self.u.velocity**2 * ac.m_p / ac.k_B).cgs.value
        T1data = np.clip(T1.data, 10, None)
        temp = nH / nH * coolftn().get_temp(T1data)
        Lambda_cool = nH / nH * coolftn().get_cool(T1data)
        cool = nH * nH * Lambda_cool
        heat = heat_ratio * nH * np.clip(coolftn().get_heat(T1data), 0.0, None)
        net_cool = cool - heat
        dd["T"] = temp
        dd["cool_rate"] = cool
        dd["heat_rate"] = heat
        dd["net_cool_rate"] = net_cool
        dd["Lambda_cool"] = Lambda_cool

        return dd

    def get_savdir_pdf(self, zrange=None):
        """return joint pdf savdir"""
        if not self.test_newcool():
            return "{}/jointpdf/cooling_heating/".format(self.savdir)
        if zrange is None:
            zmin, zmax = 0, self.domain["re"][2]
        else:
            zmin, zmax = zrange.start, zrange.stop
            if zmin < 0:
                zmin = 0
        savdir = "{}/jointpdf_z{:02d}-{:02d}/cooling_heating/".format(
            self.savdir, int(zmin / 100), int(zmax / 100)
        )
        return savdir

    def get_dhnu_PH(self):
        with open(self.files["athinput"].replace("out", "err"), "r") as fp:
            i = 0
            imax = 10000
            while i < imax:
                line = fp.readline()
                if "dhnu_HI" in line:
                    dhnu_HI_PH = float(line.split(":")[-1])
                if "dhnu_H2" in line:
                    dhnu_H2_PH = float(line.split(":")[-1])
                    break
                i += 1
        return dhnu_HI_PH, dhnu_H2_PH

    def get_coolheat_pdf(
        self,
        num,
        zrange=None,
        xHI=False,
        create=True,
        dhnu_HI_PH_default=3.45,
        dhnu_H2_PH_default=4.42,
    ):
        """return pdf from netcdf file

        ==========
        Parameters
        ==========

        xHI : bool
            return T-xHI pdfs if true else nH-T pdfs by default
        """

        savdir = self.get_savdir_pdf(zrange=zrange)
        if not os.path.isdir(savdir):
            os.makedirs(savdir)
        fcool = os.path.join(
            savdir, "{}.{:04d}.cool.pdf.nc".format(self.problem_id, num)
        )
        fheat = os.path.join(
            savdir, "{}.{:04d}.heat.pdf.nc".format(self.problem_id, num)
        )
        if xHI:
            fcool = os.path.join(
                savdir, "{}.{:04d}.xHI.cool.pdf.nc".format(self.problem_id, num)
            )
            fheat = os.path.join(
                savdir, "{}.{:04d}.xHI.heat.pdf.nc".format(self.problem_id, num)
            )
        if not self.test_newcool():
            # if no file exists, create
            if not (os.path.isfile(fcool)):
                if create:
                    ds = self.load_vtk(num=num)
                    self.create_coolheat_pdf(ds, zrange=zrange)
                    ds.close()
                else:
                    return

            with xr.open_dataset(fcool) as pdf_cool:
                pdf_cool.load()
            return pdf_cool

        # if no file exists, create
        if not (os.path.isfile(fcool) and os.path.isfile(fheat)):
            if create:
                ds = self.load_vtk(num=num)
                self.create_coolheat_pdf(ds, zrange=zrange)
                ds.close()
            else:
                return

        with xr.open_dataset(fcool) as pdf_cool:
            pdf_cool.load()
        with xr.open_dataset(fheat) as pdf_heat:
            pdf_heat.load()
        dhnu_HI_PH, dhnu_H2_PH = self.get_dhnu_PH()
        if "PH_HI" in pdf_heat.dims:
            pdf_heat["PH_HI"] = pdf_heat["PH_HI"] * dhnu_HI_PH / dhnu_HI_PH_default
        if "PH_H2" in pdf_heat.dims:
            pdf_heat["PH_H2"] = pdf_heat["PH_H2"] * dhnu_H2_PH / dhnu_H2_PH_default
        if "nH" in pdf_cool.dims:
            pdf_cool = pdf_cool.rename(nH="nH_bin")
        if "T" in pdf_cool.dims:
            pdf_cool = pdf_cool.rename(T="T_bin")
        if "xHI" in pdf_cool.dims:
            pdf_cool = pdf_cool.rename(xHI="xHI_bin")
        if "nH" in pdf_heat.dims:
            pdf_heat = pdf_heat.rename(nH="nH_bin")
        if "T" in pdf_heat.dims:
            pdf_heat = pdf_heat.rename(T="T_bin")
        if "xHI" in pdf_heat.dims:
            pdf_heat = pdf_heat.rename(xHI="xHI_bin")
        return pdf_cool, pdf_heat

    def create_coolheat_pdf(self, ds, zrange=None):
        savdir = self.get_savdir_pdf(zrange=zrange)
        coolfname = "{}.{:04d}.cool.pdf.nc".format(ds.problem_id, ds.num)
        heatfname = "{}.{:04d}.heat.pdf.nc".format(ds.problem_id, ds.num)

        data, coolrate, heatrate = get_cooling_heating(self, ds, zrange=zrange)

        # get total cooling from vtk output for normalization
        total_cooling = coolrate.attrs["total_cooling"]
        # get total heating from vtk output for normalization
        total_heating = heatrate.attrs["total_heating"]

        pdf_cool = (
            get_pdfs("nH", "T", data, coolrate).assign_coords(time=ds.domain["time"])
            / total_cooling
        )
        pdf_heat = (
            get_pdfs("nH", "T", data, heatrate).assign_coords(time=ds.domain["time"])
            / total_heating
        )

        pdf_cool_xHI = (
            get_pdfs("T", "xHI", data, coolrate).assign_coords(time=ds.domain["time"])
            / total_cooling
        )
        pdf_heat_xHI = (
            get_pdfs("T", "xHI", data, heatrate).assign_coords(time=ds.domain["time"])
            / total_heating
        )

        pdf_cool.attrs = coolrate.attrs
        pdf_heat.attrs = heatrate.attrs

        pdf_cool_xHI.attrs = coolrate.attrs
        pdf_heat_xHI.attrs = heatrate.attrs

        pdf_cool.to_netcdf(os.path.join(savdir, coolfname))
        pdf_heat.to_netcdf(os.path.join(savdir, heatfname))

        pdf_cool_xHI.to_netcdf(
            os.path.join(savdir, coolfname.replace(".cool.", ".xHI.cool."))
        )
        pdf_heat_xHI.to_netcdf(
            os.path.join(savdir, heatfname.replace(".heat.", ".xHI.heat."))
        )

    def get_merge_jointpdfs(
        self, nums=None, zrange=None, force_override=False, xHI=False
    ):
        savdir = self.get_savdir_pdf(zrange=zrange)
        merged_fname = os.path.join(savdir, "jointpdf_all.nc")
        if xHI:
            merged_fname = os.path.join(savdir, "jointpdf_all_xHI.nc")
        if os.path.isfile(merged_fname) and (not force_override):
            with xr.open_dataset(merged_fname) as pdf:
                pdf.load()
            return pdf
        if nums is None:
            nums = self.nums
        pdf = []
        for num in nums:
            pdfs = self.get_coolheat_pdf(num, zrange=zrange, xHI=xHI, create=False)
            if pdfs is not None:
                print(num, end=" ")
                if self.test_newcool():
                    pdf_cool, pdf_heat = pdfs
                    if "OIold" in pdf_cool:
                        pdf_cool = pdf_cool.drop_vars("OIold")
                    pdf_cool = (
                        pdf_cool.rename(total="total_cooling")
                        * pdf_cool.attrs["total_cooling"]
                    )
                    pdf_heat = (
                        pdf_heat.rename(total="total_heating")
                        * pdf_heat.attrs["total_heating"]
                    )
                    pdf_cool.update(pdf_heat)
                else:
                    pdf_cool = pdfs
                if "time" not in pdf_cool:
                    ds = self.load_vtk(num)
                    pdf_cool = pdf_cool.assign_coords(time=ds.domain["time"])
                    ds.close()
                pdf_cool = pdf_cool.assign_coords(
                    cool=pdf_cool.attrs["total_cooling"],
                    heat=pdf_cool.attrs["total_heating"],
                )
                pdf.append(pdf_cool)
        pdf = xr.concat(pdf, dim="time")
        pdf.to_netcdf(merged_fname)
        pdf.close()

        return pdf

    def load_chunk(self, num, scratch_dir="/scratch/gpfs/changgoo/TIGRESS-NCR/"):
        """Read in temporary outputs in scartch directory"""
        scratch_dir += osp.join(self.basename, "midplane_chunk")
        chunk_file = osp.join(
            scratch_dir, "{:s}.{:04d}.hLx.nc".format(self.problem_id, num)
        )
        if not osp.isfile(chunk_file):
            raise IOError("File does not exist: {}".format(chunk_file))
        with xr.open_dataset(chunk_file) as chunk:
            self.data_chunk = chunk

    def get_field_from_chunk(self, fields):
        """Get fields using temporary outputs in scartch directory"""
        dd = xr.Dataset()
        for f in fields:
            if f in self.data_chunk:
                dd[f] = self.data_chunk[f]
            elif f in self.dfi:
                dd[f] = self.dfi[f]["func"](self.data_chunk, self.u)
            else:
                raise IOError("{} is not available".format(f))
        return dd

    @staticmethod
    def get_phase_Tlist(kind="ncr"):
        if kind == "ncr":
            return [500, 6000, 15000, 35000, 5.0e5]
        elif kind == "classic":
            return [200, 5000, 15000, 20000, 5.0e5]

    @staticmethod
    def get_phase_T1list():
        return [500, 6000, 13000, 24000, 1.0e6]

    def ytload(self, num):
        import yt

        self.yt = yt
        self.u.units_override.update(dict(magnetic_unit=(self.u.muG * 1.0e-6, "gauss")))

        # define fields from TIGRESS-NCR output
        from yt.utilities.physical_constants import mh

        muH = self.muH
        Zsolar = 0.02

        def _ndensity(field, data):
            return data[("gas", "density")] / (muH * mh)

        def _nelectron(field, data):
            return data[("gas", "density")] * data[("athena", "xe")] / (muH * mh)

        def _temperature(field, data):
            return data[("athena", "temperature")] * yt.units.K

        def _EM(field, data):
            return (
                data[("gas", "H_nuclei_density")]
                * data[("gas", "El_number_density")]
                * data[("gas", "cell_volume")]
            )

        def _metallicity(field, data):
            return data[("athena", "specific_scalar[0]")]

        def _metallicity_solar(field, data):
            return data[("athena", "specific_scalar[0]")] / Zsolar

        fname = self._get_fvtk("vtk_tar", num=num)
        ds = yt.load(fname, units_override=self.u.units_override, unit_system="cgs")

        # add/override fields
        ds.add_field(
            ("gas", "H_nuclei_density"),
            function=_ndensity,
            force_override=True,
            units="cm**(-3)",
            display_name=r"$n_{\rm H}$",
            sampling_type="cell",
        )
        ds.add_field(
            ("gas", "El_number_density"),
            function=_nelectron,
            force_override=True,
            units="cm**(-3)",
            display_name=r"$n_{\rm e}$",
            sampling_type="cell",
        )
        ds.add_field(
            ("gas", "temperature"),
            function=_temperature,
            force_override=True,
            units="K",
            display_name=r"$T$",
            sampling_type="cell",
        )
        ds.add_field(
            ("gas", "emission_measure"),
            function=_EM,
            force_override=True,
            units="cm**(-3)",
            display_name=r"EM",
            sampling_type="cell",
        )
        ds.add_field(
            ("gas", "metallicity"),
            function=_metallicity,
            force_override=True,
            units="dimensionless",
            display_name=r"Z",
            sampling_type="cell",
        )
        ds.add_field(
            ("gas", "metallicity_solar"),
            function=_metallicity_solar,
            force_override=True,
            units="dimensionless",
            display_name=r"$Z/Z_\odot$",
            sampling_type="cell",
        )

        return ds


class LoadSimTIGRESSNCRAll(object):
    """Class to load multiple simulations"""

    def __init__(self, models=None, muH=None):
        # Default models
        if models is None:
            models = dict()
        if muH is None:
            muH = dict()
            for mdl in models:
                muH[mdl] = 1.4271
        self.models = []
        self.basedirs = dict()
        self.muH = dict()
        self.simdict = dict()

        for mdl, basedir in models.items():
            if not osp.exists(basedir):
                print(
                    "[LoadSimTIGRESSNCRAll]: Model {0:s} doesn't exist: {1:s}".format(
                        mdl, basedir
                    )
                )
            else:
                self.models.append(mdl)
                self.basedirs[mdl] = basedir
                if mdl in muH:
                    self.muH[mdl] = muH[mdl]
                else:
                    print(
                        "[LoadSimTIGRESSNCRAll]: muH for {0:s} has to be set".format(
                            mdl
                        )
                    )

    def set_model(self, model, savdir=None, load_method="pyathena", verbose=False):
        self.model = model
        try:
            self.sim = self.simdict[model]
        except KeyError:
            self.sim = LoadSimTIGRESSNCR(
                self.basedirs[model],
                savdir=savdir,
                muH=self.muH[model],
                load_method=load_method,
                verbose=verbose,
            )
            self.simdict[model] = self.sim

        return self.sim

    # adding two objects
    def __add__(self, o):
        for mdl in o.models:
            if mdl not in self.models:
                self.models += [mdl]
                self.basedirs[mdl] = o.basedirs[mdl]
                self.muH[mdl] = o.muH[mdl]
                if mdl in o.simdict:
                    self.simdict[mdl] = o.simdict[mdl]

        return self

    # get self class with only one key
    def __getitem__(self, key):
        return self.set_model(key)

    def __setitem__(self, key, value):
        if type(value) == LoadSimTIGRESSNCR:
            self.models.append(key)
            self.simdict[key] = value
            self.basedirs[key] = value.basedir
            self.muH[key] = value.muH
        else:
            print("Assigment only accepts LoadSimTIGRESSNCR")
