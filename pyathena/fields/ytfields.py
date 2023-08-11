import matplotlib.pyplot as plt
import numpy as np

import pyathena as pa
import astropy.constants as ac
import astropy.units as au
import yt
from yt.visualization.volume_rendering.api import Scene, create_volume_source

from pyathena.microphysics.cool import get_xCII, q10CII_

def add_fields(s, ds, xray=True, CII=True):
    # radiation fields

    Erad_PE0 = s.par['cooling']['Erad_PE0']
    Erad_LW0 = s.par['cooling']['Erad_LW0']

    def _Erad_FUV(field, data):
        return (data['athena','rad_energy_density_PE'] + data['athena','rad_energy_density_LW'])*s.u.energy_density.cgs.value*yt.units.erg/yt.units.cm**3
    def _chi_FUV(field, data):
        return (data['athena','rad_energy_density_PE'] + data['athena','rad_energy_density_LW'])*(s.u.energy_density.cgs.value/(Erad_PE0 + Erad_LW0))
    def _Erad_LyC(field, data):
        return data['athena','rad_energy_density_PH']*s.u.energy_density.cgs.value*yt.units.erg/yt.units.cm**3

    ds.add_field(("gas","radiation_energy_density_FUV"),
                function=_Erad_FUV,
                display_name=r"$\mathcal{E}_{\rm FUV}$",
                units='erg/cm**3',
                sampling_type="cell",force_override=True)
    ds.add_field(("gas","radiation_energy_density_LyC"),
                function=_Erad_LyC,
                display_name=r"$\mathcal{E}_{\rm LyC}$",
                units='erg/cm**3',
                sampling_type="cell",force_override=True)
    ds.add_field(("gas","chi_FUV"),
                function=_chi_FUV,
                display_name=r"$\chi_{\rm FUV}$",
                units='dimensionless',
                sampling_type="cell",force_override=True)

    # abundance fields in yt standard abundance field names
    def _nHI(field, data):
        xHI = data[("athena", "xHI")]
        return data[("gas", "H_nuclei_density")] * xHI * (xHI>0.01)

    def _nHII(field, data):
        xHII =  1 - data[("athena", "xHI")] - 2 * data[("athena", "xH2")]
        return data[("gas", "H_nuclei_density")] * xHII * (xHII>0.01)

    def _nH2(field, data):
        xH2 = data[("athena", "xH2")]
        return 2.0 * data[("gas", "H_nuclei_density")] * xH2 * (xH2>0.01)

    # emissivity/luminosity fields
    def _jHalpha(field, data):
        T4 = data[("athena", "temperature")].v / 1.0e4
        idx = (T4 > 0.1) * (T4 < 3)
        alpha_eff = 1.17e-13 * T4 ** (-0.942 - 0.031 * np.log(T4))
        hnu_halpha = 3.0269e-13
        return (
            alpha_eff
            * hnu_halpha
            * idx
            * data[("gas", "El_number_density")]
            * data[("gas", "H_p1_number_density")]
            * (yt.units.erg * yt.units.cm**3 / yt.units.s)
        )

    def _LHalpha(field, data):
        return data[("gas","H_alpha_emissivity")]*data[("gas", "cell_volume")]

    def _jHI21(field, data):
        C = 4*np.pi*1.6201623e-33*yt.units.erg/yt.units.s # (3/16pi)h*nu*A (KKO14)
        return C*data[("gas","H_p0_number_density")]

    def _LHI21(field, data):
        return data[("gas","HI_21cm_emissivity")]*data[("gas", "cell_volume")]

    # add/override fields
    ds.add_field(
        ("gas", "H_p0_number_density"),
        function=_nHI,
        force_override=True,
        units="cm**(-3)",
        display_name=r"$n_{\rm H^0}$",
        sampling_type="cell",
    )
    ds.add_field(
        ("gas", "H_p1_number_density"),
        function=_nHII,
        force_override=True,
        units="cm**(-3)",
        display_name=r"$n_{\rm H^+}$",
        sampling_type="cell",
    )
    ds.add_field(
        ("gas", "H2_number_density"),
        function=_nH2,
        force_override=True,
        units="cm**(-3)",
        display_name=r"$n_{\rm H_2}$",
        sampling_type="cell",
    )

    # Halpha
    ds.add_field(
        ("gas", "H_alpha_emissivity"),
        function=_jHalpha,
        force_override=True,
        units="erg/s/cm**3",
        display_name=r"$4\pi j_{\rm H\alpha}$",
        sampling_type="cell",
    )

    ds.add_field(
        ("gas", "H_alpha_luminosity"),
        function=_LHalpha,
        force_override=True,
        units="erg/s",
        display_name=r"$L_{\rm H\alpha}$",
        sampling_type="cell",
    )

    # HI_21cm
    ds.add_field(
        ("gas", "HI_21cm_emissivity"),
        function=_jHI21,
        force_override=True,
        units="erg/s/cm**3",
        display_name=r"$4\pi j_{\rm HI}$",
        sampling_type="cell",
    )

    ds.add_field(
        ("gas", "HI_21cm_luminosity"),
        function=_LHI21,
        force_override=True,
        units="erg/s",
        display_name=r"$L_{\rm HI}$",
        sampling_type="cell",
    )

    # xray
    if xray:
        import pyxsim

        emin = 0.1
        emax = 10
        nbins = 1000
        model = "spex"
        binscale = "log"
        Zmet = s.par["problem"]["Z_gas"]
        srcmdl = pyxsim.CIESourceModel(
            model, emin, emax, nbins, Zmet, binscale=binscale, abund_table="aspl"
        )
        xray_fields = srcmdl.make_source_fields(ds, 0.5, 7.0, force_override=True)

    if CII:
        # set total C, O abundance
        # xOstd = s.par["cooling"]["xOstd"]
        xCstd = s.par["cooling"]["xCstd"]
        # set metallicities
        Z_g = s.par["problem"]["Z_gas"]
        Z_d = s.par["problem"]["Z_dust"]
        # calculate normalized radiation fields
        Erad_PE0 = s.par["cooling"]["Erad_PE0"] / s.u.energy_density.cgs.value
        Erad_LW0 = s.par["cooling"]["Erad_LW0"] / s.u.energy_density.cgs.value
        # set flags
        try:
            CRphotC = True if s.par["cooling"]["iCRPhotC"] == 1 else False
        except KeyError:
            CRphotC = False
        try:
            iCII_rec_rate = True if s.par["cooling"]["iCRPhotC"] == 1 else False
        except KeyError:
            iCII_rec_rate = False

        def _xCII(field, data):
            nH = data[("athena", "density")].v
            xe = data[("athena", "xe")].v
            xH2 = data[("athena", "xH2")].v
            T = data[("athena", "temperature")].v
            G_PE = data[("athena", "rad_energy_density_PE")].v / Erad_PE0
            G_CI = data[("athena", "rad_energy_density_LW")].v / Erad_LW0
            CR_rate = data[("athena", "CR_ionization_rate")].v
            return get_xCII(
                nH,
                xe,
                xH2,
                T,
                Z_d,
                Z_g,
                CR_rate,
                G_PE,
                G_CI,
                xCstd=xCstd,
                gr_rec=True,
                CRPhotC=CRphotC,
                iCII_rec_rate=iCII_rec_rate,
            )

        def _Lambda_CII(field, data):
            nH = data[("athena", "density")].v
            xe = data[("athena", "xe")].v
            xHI = data[("athena", "xHI")].v
            xH2 = data[("athena", "xH2")].v
            xCII = data[("gas", "xCII")]
            T = data[("athena", "temperature")].v

            g0CII_ = 2.0
            g1CII_ = 4.0

            A10CII_ = 2.3e-6
            E10CII_ = 1.26e-14
            kB_cgs = ac.k_B.cgs.value

            q10 = q10CII_(nH, T, xe, xHI, xH2)
            q01 = (g1CII_ / g0CII_) * q10 * np.exp(-E10CII_ / (kB_cgs * T))
            T4 = data[("athena", "temperature")].v / 1.0e4
            idx = T4 < 3.5
            return (
                (q01 / (q01 + q10 + A10CII_) * A10CII_ * E10CII_ * xCII / nH)
                * idx
                * yt.units.erg
                / yt.units.s
                * yt.units.cm**3
            )

        def _jCII(field, data):
            return (
                data[("gas", "Lambda_CII")]
                * data[("gas", "H_nuclei_density")] ** 2
            )

        def _LCII(field, data):
            return data[("gas", "CII_emissivity")] * data[("gas", "cell_volume")]

        ds.add_field(
            ("gas", "xCII"),
            function=_xCII,
            force_override=True,
            display_name=r"$x_{\rm C^+}$",
            sampling_type="cell",
        )
        ds.add_field(
            ("gas", "Lambda_CII"),
            function=_Lambda_CII,
            force_override=True,
            units="erg*cm**3/s",
            display_name=r"$\Lambda_{\rm C^+}$",
            sampling_type="cell",
        )
        ds.add_field(
            ("gas", "CII_emissivity"),
            function=_jCII,
            force_override=True,
            units="erg/cm**3/s",
            display_name=r"$4\pi j_{\rm CII}$",
            sampling_type="cell",
        )
        ds.add_field(
            ("gas", "CII_luminosity"),
            function=_LCII,
            force_override=True,
            units="erg/s",
            display_name=r"$L_{\rm C^+}$",
            sampling_type="cell",
        )

    # more fields
    def _specific_enthalphy(field,data):
        return data["gas", "specific_thermal_energy"] + data["gas", "pressure"]/data["gas", "density"]
    def _specific_kinetic_energy(field,data):
        return 0.5*data["gas", "velocity_magnitude"]**2
    def _total_energy_density(field,data):
        return data["gas", "specific_total_energy"]*data["gas", "density"]
    def _total_energy_flux_z(field,data):
        return (data["gas", "specific_enthalphy"] + data["gas", "specific_kinetic_energy"])*data["gas", f"momentum_density_z"]
    def _vzout(field,data):
        return data["gas", "velocity_z"]*(data["gas","z"]/data["gas","z"])
    def _vzin(field,data):
        return data["gas", "velocity_z"]*(-data["gas","z"]/data["gas","z"])

    ds.add_field(("gas","specific_enthalphy"),
                function=_specific_enthalphy,
                units='(km/s)**2',
                sampling_type="cell",force_override=True)
    ds.add_field(("gas","specific_kinetic_energy"),
                function=_specific_kinetic_energy,
                units='(km/s)**2',
                sampling_type="cell",force_override=True)
    ds.add_field(("gas","total_energy_density"),
                function=_total_energy_density,
                units='erg/cm**3',
                sampling_type="cell",force_override=True)
    ds.add_field(("gas","total_energy_flux_z"),
                function=_total_energy_flux_z,
                units='erg/s/cm**2',
                sampling_type="cell",force_override=True)
    ds.add_field(("gas","vzout"),
                function=_vzout,
                units='km/s',
                sampling_type="cell",force_override=True)
    ds.add_field(("gas","vzin"),
                function=_vzin,
                units='km/s',
                sampling_type="cell",force_override=True)

    return ds