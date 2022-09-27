from ..microphysics.cool import *
import xarray as xr
import numpy as np
import pandas as pd


def f1(T, T0=2e4, T1=3.5e4):
    """transition function"""
    return np.where(
        T > T1,
        1.0,
        np.where(
            T <= T0,
            0.0,
            1.0 / (1.0 + np.exp(-10.0 * (T - 0.5 * (T0 + T1)) / (T1 - T0))),
        ),
    )


def get_cooling_heating(sim, ds, zrange=None, return_Lambda_e=True):
    """read necessary fields, calculate cooling from each coolnat"""

    if sim.config_time < pd.to_datetime("2022-02-10 13:21:32 -0500"):
        cooling_rate_unit = 1.0
    else:
        cooling_rate_unit = (sim.u.energy_density / sim.u.time).cgs.value

    # if sim.config_time < pd.to_datetime("2022-01-23 21:39:10 -0500"):
    #     return_Lambda_e = True
    # else:
    #     return_Lambda_e = False

    if 'iCoolHCIE' in sim.par['cooling']:
        iCoolHCIE = sim.par['cooling']['iCoolHCIE']
    else:
        iCoolHCIE = 0

    # unit definition
    unitT = sim.u.energy_density / ac.k_B / sim.muH * au.cm ** 3

    # read all necessary native fields
    field_to_read = [
        "nH",
        "pressure",
        "cool_rate",
        "heat_rate",  # velocity (for CO)
        "net_cool_rate",
        "xHI",
        "xH2",
        "xe",
    ]
    rad_fields = [
        "CR_ionization_rate",
        "rad_energy_density_PH",
        "rad_energy_density_LW",
        "rad_energy_density_PE",
        "rad_energy_density_LW_diss",
    ]
    for rf in rad_fields:
        if rf in ds.field_list:
            field_to_read += [rf]
    dd = ds.get_field(field_to_read)
    dd["cool_rate"] *= cooling_rate_unit
    dd["heat_rate"] *= cooling_rate_unit
    dd["net_cool_rate"] *= cooling_rate_unit
    total_cooling = dd["cool_rate"].sum().data
    total_heating = dd["heat_rate"].sum().data
    total_netcool = dd["net_cool_rate"].sum().data

    # apply zcut first
    if zrange is not None:
        dd = dd.sel(z=zrange)

    # set metallicities
    Z_g = sim.par["problem"]["Z_gas"]
    Z_d = sim.par["problem"]["Z_dust"]

    # get a few derived fields
    dd["xHII"] = 1 - dd["xHI"] - 2.0 * dd["xH2"]
    dd["T1"] = dd["pressure"] / dd["nH"] * unitT.cgs.value
    dd["mu"] = sim.muH / (1.1 + dd["xe"] - dd["xH2"])
    dd["T"] = dd["T1"] * dd["mu"]

    # get weight functions
    w2 = f1(dd["T"], T0=sim.par["cooling"]["Thot0"], T1=sim.par["cooling"]["Thot1"])
    w1 = 1 - w2

    # non-equilibrium hydrogen cooling
    cool_hyd = get_hydrogen_cooling(sim, dd) * dd["nH"]

    # other cooling at low T
    cool_other = get_other_cooling(sim, dd) * dd["nH"] * w1
    # CIE cooling by He and metal
    cool_CIE = get_Lambda_CIE(dd, return_Lambda_e=return_Lambda_e)
    cool_CIE["CIE_metal"] = cool_CIE["CIE_metal"] * Z_g * dd["nH"] ** 2 * w2
    cool_CIE["CIE_He"] = cool_CIE["CIE_He"] * dd["nH"] ** 2
    if (iCoolHCIE):
        # CIE hydrogen cooling
        cool_CIE["CIE_H"] = cool_CIE["CIE_H"] * dd["nH"] ** 2 * w2
        # non-equilibrium hydrogen cooling
        cool_hyd *= w1
    else:
        cool_CIE=cool_CIE[["CIE_metal","CIE_He"]]
        print("no HCIE")

    # heating
    heat = get_heating(sim, dd) * dd["nH"] * w1
    heat["total"] = dd["heat_rate"]

    # add ancillary fields
    cool = cool_hyd.update(cool_other).update(cool_CIE)
    cool["total"] = dd["cool_rate"]

    # add total cooling/heating
    cool.attrs["total_cooling"] = total_cooling
    cool.attrs["total_heating"] = total_heating
    cool.attrs["total_netcool"] = total_netcool
    heat.attrs["total_cooling"] = total_cooling
    heat.attrs["total_heating"] = total_heating
    heat.attrs["total_netcool"] = total_netcool

    return (
        dd,
        cool.assign_coords(time=ds.domain["time"]),
        heat.assign_coords(time=ds.domain["time"]),
    )


def get_heating(s, dd):
    """calculate heating"""
    if 'CR_ionization_rate' in dd: CR_rate = dd['CR_ionization_rate']
    else: CR_rate = s.par["problem"]["xi_CR_amp"]*2.e-16
    # set metallicities
    Z_g = s.par["problem"]["Z_gas"]
    Z_d = s.par["problem"]["Z_dust"]
    # normalization factors
    Erad_PE0 = s.par["cooling"]["Erad_PE0"] / s.u.energy_density.cgs.value
    Erad_LW0 = s.par["cooling"]["Erad_LW0"] / s.u.energy_density.cgs.value
    xi_diss_H2_conv = s.par["cooling"]["xi_diss_H2_ISRF"] / Erad_LW0
    eV_ = (1.0 * au.eV).cgs.value
    dhnu_H2_diss = 0.4 * eV_
    if "dhnu_HI_PH" in s.par["radps"]:
        dhnu_HI_PH = s.par["radps"]["dhnu_HI_PH"] * eV_
    else:
        dhnu_HI_PH = 3.45 * eV_
    if "dhnu_H2_PH" in s.par["radps"]:
        dhnu_H2_PH = s.par["radps"]["dhnu_H2_PH"] * eV_
    else:
        dhnu_H2_PH = 4.42 * eV_
    sigma_HI_PH = s.par["opacity"]["sigma_HI_PH"]
    sigma_H2_PH = s.par["opacity"]["sigma_H2_PH"]
    hnu_PH = s.par["radps"]["hnu_PH"] * eV_
    # calculate normalized radiation fields
    Erad_PE = 0.
    Erad_LW = 0.
    if 'rad_energy_density_PE' in dd:
        Erad_PE = dd["rad_energy_density_PE"]
    if 'rad_energy_density_LW' in dd:
        Erad_LW = dd["rad_energy_density_LW"]
    G_PE = (Erad_PE + Erad_LW) / (Erad_PE0 + Erad_LW0)
    if 'rad_energy_density_PH' in dd:
        Erad_PH = dd["rad_energy_density_PH"]
        xi_ph_HI = Erad_PH * s.u.energy_density.cgs.value
        xi_ph_HI *= ac.c.cgs.value * sigma_HI_PH / hnu_PH
        xi_ph_H2 = Erad_PH * s.u.energy_density.cgs.value
        xi_ph_H2 *= ac.c.cgs.value * sigma_H2_PH / hnu_PH
    if 'rad_energy_density_LW_diss' in dd:
        Erad_LW_diss = dd["rad_energy_density_LW_diss"]
        xi_diss_H2 = Erad_LW_diss * xi_diss_H2_conv
    else:
        xi_diss_H2 = 0.
        print('no Photo Dissociation')
    # set flags
    try:
        ikgr_H2 = s.par["cooling"]["ikgr_H2"]
    except KeyError:
        ikgr_H2 = 0
    try:
        kgr_H2 = s.par["cooling"]["kgr_H2"]
    except KeyError:
        kgr_H2 = 3.e-17
    try:
        iPEheating = s.par["cooling"]["iPEheating"]
    except KeyError:
        iPEheating = True

    heatrate = xr.Dataset()
    if iPEheating:
        heatrate["PE"] = heatPE(dd["nH"], dd["T"], dd["xe"], Z_d, G_PE)
    else:
        print("no PE heating")

    heatrate["CR"] = heatCR(
        dd["nH"], dd["xe"], dd["xHI"], dd["xH2"], CR_rate,
        old = s.oldCRheating
    )
    # if 'rad_energy_density_LW_diss' in dd:
    if s.iH2heating > 0:
        heatrate["H2"] = heatH2(
            dd["nH"],
            dd["T"],
            dd["xHI"],
            dd["xH2"],
            xi_diss_H2,
            Z_d,
            kind=s.iH2heating,
            xi_diss_H2_ISRF=s.par["cooling"]["xi_diss_H2_ISRF"],
            kgr_H2 = kgr_H2,
            ikgr_H2=ikgr_H2,
        )
    else:
        print("no H2 heating")
    # heatrate["H2_form"] = heatH2form(
    #     dd["nH"], dd["T"], dd["xHI"], dd["xH2"], Z_d, ikgr_H2=ikgr_H2
    # )
    # heatrate["H2_pump"] = heatH2pump(
    #     dd["nH"], dd["T"], dd["xHI"], dd["xH2"], xi_diss_H2
    # )
    # heatrate["H2_diss"] = heatH2diss(dd["xH2"], xi_diss_H2)
    if 'rad_energy_density_PH' in dd:
        heatrate["PH_HI"] = dd["xHI"] * xi_ph_HI * dhnu_HI_PH
        heatrate["PH_H2"] = dd["xH2"] * xi_ph_H2 * dhnu_H2_PH
    else:
        print("no PI heating")
    # no heating at high-T
    heatrate = heatrate.where(dd["T"] < s.par["cooling"]["Thot1"]).fillna(1.0e-35)

    return heatrate


def get_hydrogen_cooling(sim, dd):
    """a wrapper function to calculate H cooling"""
    if 'iCoolH2rovib' in sim.par['cooling']:
        iCoolH2rovib = sim.par['cooling']['iCoolH2rovib']
    else:
        iCoolH2rovib = 0
    if 'iCoolH2colldiss' in sim.par['cooling']:
        iCoolH2colldiss = sim.par['cooling']['iCoolH2colldiss']
    else:
        iCoolH2colldiss = 0

    if 'xHIcut' in sim.par['cooling']:
        xHIcut = sim.par['cooling']['xHIcut']
    else:
        xHIcut = 0.0
    if 'xHIIcut' in sim.par['cooling']:
        xHIIcut = sim.par['cooling']['xHIIcut']
    else:
        xHIIcut = 0.0
    if 'xH2cut' in sim.par['cooling']:
        xH2cut = sim.par['cooling']['xH2cut']
    else:
        xH2cut = 0.0

    cuts=dict(xHI=xHIcut, xHII=xHIIcut, xH2=xH2cut)
    for x in cuts:
        print('{} cut is {}'.format(x,cuts[x]))

    coolrate = xr.Dataset()
    # HI cooling
    x = "xHI"
    coolrate["HI_Lya"] = coolLya(dd["nH"], dd["T"], dd["xe"], dd["xHI"]).where(dd[x]>cuts[x],0.0)
    coolrate["HI_collion"] = coolHIion(dd["nH"], dd["T"], dd["xe"], dd["xHI"]).where(dd[x]>cuts[x],0.0)
    # HII cooling
    x = "xHII"
    coolrate["HII_ff"] = coolffH(dd["nH"], dd["T"], dd["xe"], dd["xHII"]).where(dd[x]>cuts[x],0.0)
    coolrate["HII_rec"] = coolrecH(dd["nH"], dd["T"], dd["xe"], dd["xHII"]).where(dd[x]>cuts[x],0.0)
    # H2 cooling
    x = "xH2"
    if iCoolH2rovib:
        coolrate["H2_rovib"] = coolH2rovib(dd["nH"], dd["T"], dd["xHI"], dd["xH2"]).where(dd[x]>cuts[x],0.0)  # rovib
    if iCoolH2colldiss:
        coolrate["H2_colldiss"] = coolH2colldiss(dd["nH"], dd["T"], dd["xHI"], dd["xH2"]).where(dd[x]>cuts[x],0.0)
    return coolrate


def get_other_cooling(s, dd):
    """function to other cooling at low T
    """
    if not ("xHII" in dd):
        raise KeyError("xHII must set before calling this function")

    if 'iChem' in s.par['cooling']:
        iChem = (s.par['cooling']['iChem'] == 1)
    else:
        iChem = True

    if (not iChem):
        print('no Chemistry')

    # set total C, O abundance
    xOstd = s.par["cooling"]["xOstd"]
    xCstd = s.par["cooling"]["xCstd"]
    # set metallicities
    Z_g = s.par["problem"]["Z_gas"]
    Z_d = s.par["problem"]["Z_dust"]
    # calculate normalized radiation fields
    Erad_PE0 = s.par["cooling"]["Erad_PE0"] / s.u.energy_density.cgs.value
    Erad_LW0 = s.par["cooling"]["Erad_LW0"] / s.u.energy_density.cgs.value

    Erad_PE = 0.
    Erad_LW = 0.
    if 'rad_energy_density_PE' in dd:
        Erad_PE = dd["rad_energy_density_PE"]
        if dd['rad_energy_density_PE'].max().data == 0:
            gr_rec=False
            print('no PE')
        else:
            gr_rec=True
    else:
        gr_rec=False
        print('no PE')
    if 'rad_energy_density_LW' in dd:
        Erad_LW = dd["rad_energy_density_LW"]
    G_PE = (Erad_PE + Erad_LW) / (Erad_PE0 + Erad_LW0)
    G_CI = Erad_LW / Erad_LW0
    G_CO = G_CI
    # set flags
    try:
        CRphotC = True if s.par["cooling"]["iCRPhotC"] == 1 else False
    except KeyError:
        CRphotC = False
    try:
        iCII_rec_rate = True if s.par["cooling"]["iCRPhotC"] == 1 else False
    except KeyError:
        iCII_rec_rate = False
    # calculate C, O species abundances
    dd["xOII"] = dd["xHII"] * s.par["cooling"]["xOstd"] * s.par["problem"]["Z_gas"]
    if 'CR_ionization_rate' in dd: CR_rate = dd['CR_ionization_rate']
    else: CR_rate = s.par["problem"]["xi_CR_amp"]*2.e-16
    if (iChem):
        dd["xCII"] = get_xCII(
            dd["nH"],
            dd["xe"],
            dd["xH2"],
            dd["T"],
            Z_d,
            Z_g,
            CR_rate,
            G_PE,
            G_CI,
            xCstd=xCstd,
            gr_rec=gr_rec,
            CRphotC=CRphotC,
            iCII_rec_rate=iCII_rec_rate
        )
        dd["xCO"], ncrit = get_xCO(
            dd["nH"],
            dd["xH2"],
            dd["xCII"],
            dd["xOII"],
            Z_d,
            Z_g,
            CR_rate,
            G_CO,
            xCstd=xCstd,
            xOstd=xOstd,
        )
        dd["xCI"] = np.clip(xCstd * Z_g - dd["xCII"] - dd["xCO"], 1.0e-20, None)
        dd["xOI"] = np.clip(xOstd * Z_g - dd["xOII"] - dd["xCO"], 1.0e-20, None)
    else:
        dd["xOI"] = np.clip(xOstd * Z_g - dd["xOII"], 1.0e-20, None)


    # cooling others
    coolrate = xr.Dataset()
    if (iChem):
        # CI
        coolrate["CI"] = coolCI(
            dd["nH"], dd["T"], dd["xe"], dd["xHI"], dd["xH2"], dd["xCI"]
        )
        # CII
        coolrate["CII"] = coolCII(
            dd["nH"], dd["T"], dd["xe"], dd["xHI"], dd["xH2"], dd["xCII"]
        )
        # OI
        coolrate["OI"] = coolOI(
            dd["nH"], dd["T"], dd["xe"], dd["xHI"], dd["xH2"], dd["xOI"]
        )
    else:
        # CII
        coolrate["CII"] = coolCII(
            dd["nH"], dd["T"], dd["xe"], dd["xHI"], dd["xH2"],
            s.par["cooling"]["xCstd"] * s.par["problem"]["Z_gas"]
        )
        # OI
        coolrate["OI"] = coolOI(
            dd["nH"], dd["T"], dd["xe"], dd["xHI"], dd["xH2"], dd["xOI"]
        )
        print("no CI cooling")
    # for now, this is too slow
    # set_dvdr(dd)
    # coolrate['CO'] = coolCO(dd['nH'],dd['T'],
    #        dd['xe'],dd['xHI'],dd['xH2'],dd['xCO'],dd['dvdr'])

    # coolrate["OIold"] = coolOI(
    #     dd["nH"], dd["T"], dd["xe"], dd["xHI"], dd["xHII"], dd["xOI"]
    # )
    # if "fac_coolingOII" in s.par["cooling"]:
    #     coolrate["OII"] = s.par["cooling"]["fac_coolingOII"] * coolOII(
    #         dd["nH"], dd["T"], dd["xe"], dd["xOII"]
    #     )
    # else:
    # coolrate["OII"] = 4 * coolOII(
    #     dd["nH"], dd["T"], dd["xe"], dd["xOII"]
    # )
    try:
        iPhotIon = s.par["radps"]["iPhotIon"]
    except KeyError:
        iPhotIon = True
    if (iPhotIon):
        coolrate["neb"] = coolneb(dd["nH"], dd["T"], dd["xe"], dd["xHII"], Z_g)
    else:
        print("no Nebular cooling")

    try:
        iPEheating = s.par["cooling"]["iPEheating"]
    except KeyError:
        iPEheating = True
    # if (iPEheating):
    coolrate["grRec"] = coolRec(dd["nH"], dd["T"], dd["xe"], Z_d, G_PE)
    # else:
        # print("no grain recom. cooling")

    return coolrate


# def set_dvdr(s,dd):
# velocity gradient for CO
# dx = dd.x[1]-dd.x[0]
# dy = dd.y[1]-dd.y[0]
# dz = dd.z[1]-dd.z[0]
# dvdx=0.5*(dd['velocity1'].shift(x=1)-dd['velocity1'].shift(x=-1))/dx
# dvdy=0.5*(dd['velocity2'].shift(y=1)-dd['velocity2'].shift(y=-1))/dy
# dvdz=0.5*(dd['velocity3'].shift(z=1)-dd['velocity3'].shift(z=-1))/dz

# dvdx.data[:,:,-1] = ((dd['velocity1'].isel(x=-1)-dd['velocity1'].isel(x=-2))/dx).data # evaluate at ie-1/2
# dvdx.data[:,:,0] = ((dd['velocity1'].isel(x=1)-dd['velocity1'].isel(x=0))/dx).data # evaluate at is+1/2
# dvdy.data[:,-1,:] = (0.5*(dd['velocity2'].isel(y=-2)-dd['velocity2'].isel(y=0))/dy).data # periodic in y
# dvdy.data[:,0,:] = (0.5*(dd['velocity2'].isel(y=1)-dd['velocity2'].isel(y=-1))/dy).data # periodic in y
# dvdz.data[-1,:,:] = ((dd['velocity3'].isel(z=-1)-dd['velocity3'].isel(z=-2))/dz).data # evaluate at ke-1/2
# dvdz.data[0,:,:] = ((dd['velocity3'].isel(z=1)-dd['velocity3'].isel(z=0))/dz).data # evaluate at ks-1/2

# dd['dvdr'] = 1/3.*(np.abs(dvdx)+np.abs(dvdy)+np.abs(dvdz))/s.u.time.cgs.value


def set_CIE_interpolator(return_xe=False, return_Lambda_e=False):
    """CIE cooling from Gnat12
    based on /tigress/jk11/notebook/NEWCOOL/paper-fig-transition.ipynb
    """
    # CIE cooling
    from ..microphysics.cool_gnat12 import CoolGnat12

    cg = CoolGnat12(abundance="Asplund09")
    elem_no_ion_frac = []
    xe = dict()
    xe_tot = np.zeros_like(cg.temp)
    cool = dict()
    cool_tot = np.zeros_like(cg.temp)
    # Elements for which CIE ion_frac is available
    elements = ["H", "He", "C", "N", "O", "Ne", "Mg", "Si", "S", "Fe"]

    for e in elements:
        xe[e] = np.zeros_like(cg.temp)
        cool[e] = np.zeros_like(cg.temp)

    # Note that Gnat & Ferland provided Lambda_GF = cool_rate/(n_elem*ne)
    # Need to get the total electron abundance first to obtain
    #   cool_rate/nH^2 = Lambda_GF*Abundance*x_e

    if return_Lambda_e:
        for e in elements:
            nstate = cg.info.loc[e]["number"] + 1
            A = cg.info.loc[e]["abd"]

            for i in range(nstate):
                xe[e] += A * i * cg.ion_frac[e + str(i)].values
                cool[e] += (
                    A * cg.ion_frac[e + str(i)].values * cg.cool_cie_per_ion[e][:, i]
                )

        for e in elements:
            xe_tot += xe[e]
            cool_tot += cool[e]
    else:
        for e in elements:
            nstate = cg.info.loc[e]["number"] + 1
            A = cg.info.loc[e]["abd"]

            for i in range(nstate):
                xe[e] += A * i * cg.ion_frac[e + str(i)].values

        for e in elements:
            xe_tot += xe[e]

        for e in elements:
            nstate = cg.info.loc[e]["number"] + 1
            A = cg.info.loc[e]["abd"]
            for i in range(nstate):
                cool[e] += (
                    xe_tot
                    * A
                    * cg.ion_frac[e + str(i)].values
                    * cg.cool_cie_per_ion[e][:, i]
                )

        for e in elements:
            cool_tot += cool[e]

    # Interpolation
    from scipy.interpolate import interp1d

    cgi_metal = interp1d(
        cg.temp, cool_tot - cool["He"] - cool["H"], bounds_error=False, fill_value=0.0
    )
    cgi_He = interp1d(cg.temp, cool["He"], bounds_error=False, fill_value=0.0)

    cgi_H = interp1d(cg.temp, cool["H"], bounds_error=False, fill_value=0.0)
    if return_xe:
        cgi_xe = interp1d(
            cg.temp, xe_tot, bounds_error=False, fill_value=0.0
        )
        cgi_xe_mHHe = interp1d(
            cg.temp, xe_tot - xe["H"] - xe["He"], bounds_error=False, fill_value=0.0
        )
        cgi_xe_He = interp1d(cg.temp, xe["He"], bounds_error=False, fill_value=0.0)
        return cgi_metal, cgi_He, cgi_H, cgi_xe, cgi_xe_mHHe, cgi_xe_He
    else:
        return cgi_metal, cgi_He, cgi_H


def get_Lambda_CIE(dd, return_Lambda_e=False):
    """return Lambda_CIE"""
    Lambda_cool = xr.Dataset()
    cgi_metal, cgi_He, cgi_H = set_CIE_interpolator(
        return_xe=False, return_Lambda_e=return_Lambda_e
    )
    Lambda_cool["CIE_metal"] = xr.DataArray(
        cgi_metal(dd["T"]), coords=[dd.z, dd.y, dd.x]
    )
    Lambda_cool["CIE_He"] = xr.DataArray(cgi_He(dd["T"]), coords=[dd.z, dd.y, dd.x])
    Lambda_cool["CIE_H"] = xr.DataArray(cgi_H(dd["T"]), coords=[dd.z, dd.y, dd.x])
    return Lambda_cool


def set_bins_default():
    binranges = dict(
        nH=(-6, 6),
        T=(0, 9),
        pok=(0, 10),
        nHI=(-6, 6),
        nH2=(-6, 6),
        nHII=(-6, 6),
        ne=(-6, 6),
        xH2=(0, 0.5),
        xHII=(0, 1.0),
        xHI=(0, 1.0),
        xe=(0, 1.2),
        chi_PE=(-3, 4),
        chi_FUV=(-3, 4),
        chi_H2=(-3, 4),
        xi_CR=(-18, -12),
        Erad_LyC=(-30, -10),
        net_cool_rate=(-30, -18),
        Lambda_cool=(-30, -18),
        cool_rate=(-30, -18),
        heat_rate=(-30, -18),
    )
    dbins = dict(vz=10, xH2=0.005, xHII=0.01, xHI=0.01, xe=0.01, nH=0.05, T=0.05)
    nologs = ["vz", "xH2", "xHII", "xHI", "xe"]
    bins = dict()
    for k, v in binranges.items():
        bmin, bmax = v
        try:
            dbin = dbins[k]
        except:
            dbin = 0.1
        nbin = int((bmax - bmin) / dbin) + 1
        if k in nologs:
            bins[k] = np.linspace(bmin, bmax, nbin)
        else:
            bins[k] = np.logspace(bmin, bmax, nbin)
    return bins, nologs


def get_pdf_xarray(x, y, w, xbin, ybin, xf, yf):
    h = np.histogram2d(x, y, weights=w, bins=[xbin, ybin])
    xe = h[1]
    ye = h[2]
    dx = xe[1] - xe[0]
    dy = ye[1] - ye[0]
    dx2 = xe[2] - xe[1]
    dy2 = ye[2] - ye[1]
    if dx != dx2:
        xe = np.log10(xe)
    if dy != dy2:
        ye = np.log10(ye)
    xc = 0.5 * (xe[1:] + xe[:-1])
    yc = 0.5 * (ye[1:] + ye[:-1])
    dx = xe[1] - xe[0]
    dy = ye[1] - ye[0]
    pdf = xr.DataArray(h[0].T / dx / dy, coords=[yc, xc], dims=[yf, xf])
    return pdf


def get_pdfs(xf, yf, data, rate, set_bins=set_bins_default):
    bins, nologs = set_bins()
    pdfs = xr.Dataset()
    sources = list(rate.keys())
    x = data[xf].data.flatten()
    y = data[yf].data.flatten()
    xbin = bins[xf]
    ybin = bins[yf]
    for s in sources:
        w = rate[s].data.flatten()
        pdf = get_pdf_xarray(x, y, w, xbin, ybin, xf, yf)
        pdfs[s] = pdf
    return pdfs
