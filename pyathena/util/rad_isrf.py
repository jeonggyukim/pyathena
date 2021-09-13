import numpy as np
import astropy.units as au
import astropy.constants as ac
from scipy import integrate
from scipy.interpolate import interp1d

def F_E_Dr78(E):
    """Photon flux for Draine's ISRF
    E : Photon energy (in eV)
    """
    if E is not None and not isinstance(E, au.quantity.Quantity):
        E = (E*au.eV).to(au.eV)

    Funit = 1/au.cm**2/au.s/au.sr/au.eV
    return (1.658e6*(E/au.eV) - 2.152e5*(E/au.eV)**2 + 6.919e3*(E/au.eV)**3)*Funit

def nuJnu_Dr78(E):
    return (E**2*F_E_Dr78(E)).to('erg s-1 cm-2 sr-1')

def Jnu_vD82(wav):
    if wav is not None and not isinstance(wav, au.quantity.Quantity):
        wav = (wav*au.angstrom).to(au.angstrom)
    else:
        wav = wav.to(au.angstrom)

    w = wav.value
    return 2.44e-16*w**2.7/au.cm**2/au.s/au.Hz

def Jnu_MMP83(wav):
    w = np.array([0.0912,0.1,0.11,0.13,0.143,0.18,0.2,0.21,0.216,0.23,0.25,0.346,0.435,0.55,0.7,0.9,
                  1.2,1.8,2.2,2.4,3.4,4.0,5.0,6.0,8.0])*1e4*au.angstrom
    unit = au.erg/au.cm**2/au.s/au.micron
    four_pi_Jlambda = np.array([1.07,1.47,2.04,2.05,1.82,1.24,1.04,0.961,0.917,0.825,0.727,1.3,1.5,1.57,1.53,1.32,
                                0.926,0.406,0.241,0.189,0.0649,0.0379,0.0176,0.00921,0.00322])*1e-2*unit
    Jnu = (four_pi_Jlambda/(4.0*np.pi*au.sr)*(w**2/ac.c)).to('erg cm-2 s-1 Hz-1 sr-1')
    nu = (ac.c/w).to('Hz')
    f_Jnu = interp1d(w, Jnu, bounds_error=False, fill_value=np.nan)
    
    return f_Jnu(wav)

# def Jnu_MMP83(wav):
    
#     if wav is not None and not isinstance(wav, au.quantity.Quantity):
#         wav = (wav*au.angstrom).to(au.angstrom)
#     else:
#         wav = wav.to(au.angstrom)

#     w = wav.value
#     u_nu = np.where(np.logical_and(w < 2460.0, w >= 1340.0),
#                       2.373e-14*(w*1e-4)**(-0.6678),0.0) + \
#              np.where(np.logical_and(wav.value < 1340.0, wav.value >= 1100.0),
#                       6.825e-13*(w*1e-4), 0.0) + \
#              np.where(np.logical_and(wav.value < 1100.0, wav.value >= 912.0),
#                       1.287e-9*(w*1e-4)**(4.4172), 0.0)
#     return u_nu*((ac.c/(4.0*np.pi))).cgs.value

