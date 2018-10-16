import numpy as np
import yt.units as yu
from yt import physical_constants as phyc

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def add_fields(ds, IXN=3,
               kappa_dust=500.0*yu.cm**2/yu.g, 
               sigma_ph=3.0e-18*yu.cm**2,
               muH=1.4272,
               y=0.0,
               units='LT'):
    """
    Function to add derived fields to yt dataset.
    
    Parameters
    ----------
       ds: yt dataset

       IXN: integer
           index for neutral hydrogen in specific_scalar array
           in athena vtk dump
       kappa_dust: float
           (constant) dust opacity in FUV and EUV
       sigma_ph: float
           (constant) photoionization cross section
       y: float
           Contribution to electron density from all heavy elements
           ne = (1 - xn + y)*nH
    """
    
    # Set code units
    if units == 'LT':
        lunit = yu.pc
        tunit = yu.Myr
        munit = (muH*phyc.mass_hydrogen_cgs*(yu.pc/yu.cm)**3).in_units('Msun')
    elif units == 'LV':
        lunit = yu.pc
        tunit = yu.pc/(yu.km/yu.s)
        munit = (muH*phyc.mass_hydrogen_cgs*(yu.pc/yu.cm)**3).in_units('Msun')
    elif units == 'cgs':
        lunit = yu.cm
        tunit = yu.s
        munit = yu.g
        
    vunit = lunit/tunit
    Eunit = munit/(lunit*tunit**2)       # energy density
    Funit = (Eunit*lunit/tunit).in_cgs() # flux

    field_xn = "specific_scalar[{0:d}]".format(IXN)
    
    # Number density of H, HI and H neutral fraction
    def _nH(field, data):
        return data["density"]/(muH*phyc.mass_hydrogen_cgs)
    ds.add_field(("athena","nH"), sampling_type="cell", function=_nH,
                 units='cm**(-3)', display_name=r'$n_{\rm H}$')
    def _xn(field, data):
        return data[("athena",field_xn)]
    ds.add_field(("athena","xn"), sampling_type="cell",
                 function=_xn,units="dimensionless",
                 take_log=False, display_name=r'$x_{\rm n}$')
    def _nHI(field, data):
        return data[("athena","xn")]*data[("athena","nH")]
    ds.add_field(("athena","nHI"), sampling_type="cell", 
                 function=_nHI,units="cm**-3", take_log=True,
                 display_name=r'$n_{\rm HI}$')

    # electron number density: assume ne = (1 - xn + y)*nH
    @static_vars(y=y)
    def _nelec(field, data):
        return (1.0 - data[("athena",field_xn)] + y)*\
                data[("athena","nH")]
    ds.add_field(("athena","nelec"), sampling_type="cell",
                 function=_nelec, units="cm**-3", take_log=True,
                 display_name=r'$n_{e}$')
    def _nesq(field, data):
        return data["nelec"]**2
    ds.add_field("nesq", sampling_type="cell",
                 function=_nesq, units="cm**-6", take_log=True,
                 display_name=r'$n_{e}^2$')

    # Temperature
    def _Temperature(field, data):
        return data["pressure"]/((1.1*data[("athena","nH")] + data["nelec"])*phyc.kb)
    ds.add_field("Temperature", sampling_type="cell",
                 function=_Temperature, units="K", take_log=True,
                 display_name=r'$T$')
    
    def _density_ion(field, data):
        return (1.0 - data[("athena",field_xn)])*data[("athena","density")]
    ds.add_field(("athena","density_ion"), sampling_type="cell",
                 function=_density_ion, units="g*cm**-3", take_log=True, 
                 display_name=r'$\rho_{\rm i}$')
    def _density_neu(field, data):
        return data[("athena", field_xn)]*data[("athena", "density")]
    ds.add_field(("athena","density_neu"), sampling_type="cell",
                 function=_density_neu, units="g*cm**-3", take_log=True,
                 display_name=r'$\rho_{\rm n}$')

    # Mean radiation intensity relative to the solar nbhd value
    # (for FUV) 2.1e-4 [cgs units]
    def _Erad0(field, data):
        return data[("athena","rad_energy_density0")]*Eunit
    ds.add_field(("gas","Erad0"), sampling_type="cell", function=_Erad0,
                 units="erg/cm**3", take_log=True,
                 display_name=r'$\mathcal{E}_{\rm EUV}$')
    def _Jrad0(field, data):
        return data["Erad0"]/(4.0*np.pi*yu.sr)*phyc.clight
    ds.add_field("Jrad0", sampling_type="cell", function=_Jrad0,
                 units="erg/cm**2/sr/s", take_log=True,
                 display_name=r'$J_{\rm EUV}$')
    def _G0prime0(field, data):
        return data["Jrad0"]/(2.1e-4*yu.erg/yu.cm**2/yu.s/yu.sr)
    ds.add_field("G0prime0", sampling_type="cell",
                 function=_G0prime0, units="dimensionless", take_log=True,
                 display_name=r'$G_{0,{\rm EUV}}^{\prime}$')
    
    def _Erad1(field, data):
        return data[("athena","rad_energy_density1")]*Eunit
    ds.add_field(("gas","Erad1"), sampling_type="cell",
                 function=_Erad1, units="erg/cm**3", take_log=True,
                 display_name=r'$\mathcal{E}_{\rm FUV}$')
    def _Jrad1(field, data):
        return data["Erad1"]/(4.0*np.pi*yu.sr)*phyc.clight
    ds.add_field("Jrad1", sampling_type="cell",
                 function=_Jrad1, units="erg/cm**2/sr/s", take_log=True,
                 display_name=r'$J_{\rm FUV}$')
    def _G0prime1(field, data):
        return data["Jrad1"]/(2.1e-4*yu.erg/yu.cm**2/yu.s/yu.sr)
    ds.add_field("G0prime1", sampling_type="cell", 
                 function=_G0prime1, units="dimensionless", take_log=True,
                 display_name=r'$G_{0,{\rm FUV}}^{\prime}$')

    # absorption coefficient per unit length
    @static_vars(kappa_dust=kappa_dust)
    def _chi_nion(field, data):
        return _chi_nion.kappa_dust*data["density"]
    ds.add_field("chi_nion", sampling_type="cell", 
                 function=_chi_nion, units="1/cm", take_log=True,
                 display_name=r'$\chi_{\rm FUV}$')

    @static_vars(kappa_dust=kappa_dust)
    @static_vars(sigma_ph=sigma_ph)
    def _chi_ion(field, data):
        return _chi_ion.kappa_dust*data["density"] + \
               _chi_ion.sigma_ph*data[("athena","nHI")]
    ds.add_field("chi_ion", sampling_type="cell",
                 function=_chi_ion, units="1/cm", take_log=True,
                 display_name=r'$\chi_{\rm EUV}$')

    # for y slice or projection, make x-z as XY axes in the image
    ds.coordinates.x_axis[1] = 0
    ds.coordinates.x_axis['y'] = 0
    ds.coordinates.y_axis[1] = 2
    ds.coordinates.y_axis['y'] = 2

