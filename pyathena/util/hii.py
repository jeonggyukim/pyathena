import numpy as np
from astropy import constants as ac
from astropy import units as au
from scipy.interpolate import interp1d
from scipy.optimize import fsolve


def rst(Qi, alphaB, nrms, fion=1.0):
    """
    Find Stromgren radius in cgs units
    rst=((3.0*fion*Qi)/(4.0*np.pi*alphaB*nrms**2))**(1.0/3.0)

    Parameters
    ----------
    Qi: ionizing power
    alphaB: case B recombination coefficient
    nrms: rms H number density inside the HII region
    fion: fraction of ionizing photons absorbed by HI
    """
    return ((3.0*fion*Qi)/(4.0*np.pi*alphaB*nrms**2))**(1.0/3.0)


def nion_rms(Qi, alphaB, rst, fion=1.0):
    """sin(Qi,alphaB,nrms,fion=1.0)
    Find rms number density
    rst=((3.0*fion*Qi)/(4.0*np.pi*alphaB*nrms**2))**(1.0/3.0)

    Parameters
    ----------
    Qi: ionizing power
    alphaB: case B recombination coefficient
    rst: Stromgren radius
    """
    return ((3.0*fion*Qi)/(4.0*np.pi*rst**3*alphaB))**0.5


def find_IF_1d(r, xn, log=False):
    """
    Find rIF where xn = 0.5
    r, xn are 1d arrays

    Parameters
    ----------
    log : bool, optional
        If True, compute rIF using r vs log_xn profile
    """

    n = 5
    idx0 = np.where(xn > 1e-1)[0][0]

    # Take the vicinity of IF
    r_ = r[idx0-n:idx0+n]
    xn_ = xn[idx0-n:idx0+n]

    if not log:
        f = interp1d(r_, xn_ - 0.5)
    else:
        f = interp1d(r_, np.log10(xn_/0.5))

    x0guess = r_.mean()
    x0 = fsolve(f, x0guess)

    return x0[0]


def alphaB(T):
    """
    Returns Case B recombination coefficient (Osterbrock 1989; Krumholz+07)

    2.59e-13*(T/1e4)**(-0.7)
    """
    return 2.59e-13*(T/1e4)**(-0.7)


def betaHI(T):
    """
    Returns collisional ionization rate coefficient
    (Cen 1992, Rosdahl & Teyssier 2013)

    5.85e-11*np.sqrt(T)/(1.0 + np.sqrt(T/1e5))* \
    np.exp(-157809.1/T)
    """
    return 5.85e-11*np.sqrt(T)/(1.0 + np.sqrt(T/1e5)) * np.exp(-157809.1/T)


def cool_metalion(ne, nH, xn, T, zoxygen=8.59e-3):
    """
    Returns cooling by collisionally excited optical lines of ionized metals
    (Appendix in Henney et al. 2009)

    2.905e-19*zoxygen*ne*nH*(1.0 - xn)*np.exp(-33610./T- (2180./T)**2)
    """
    return 2.905e-19*zoxygen*ne*nH*(1.0 - xn)*np.exp(-33610./T - (2180./T)**2)


def cool_metalneutral(ne, nH, xn, T, zoxygen=8.59e-3):
    """
    Returns cooling by collisionally excited lines of neutral metals
    (Appendix in Henney et al. 2009)

    4.477e-20*zoxygen*ne*nH*xn*np.exp(-28390./T + (1780/T)**2)
    """
    return 4.477e-20*zoxygen*ne*nH*xn*np.exp(-28390./T + (1780/T)**2)


def cool_PDR(n, T):
    """
    Returns a fit to the PDR cooling determined from CLOUDY models
    (Appendix in Henney et al. 2009)

    """
    return 3.981e-27*n**1.6*T**0.5*np.exp(-(70.0 + 220.0*(n/1e6)**0.2)/T)


def cool_KI(n, T):
    """
    Returns Koyama & Inutsuka (2002) cooling function

    """
    return 2e-19*n*n*(np.exp(-1.184e5/(T + 1e3)) +
                      1.4e-9*T**0.5*np.exp(-92.0/T))


def cool_CIE(T, zoxygen=8.59e-3):
    """
    Returns fit to the collisional ionization equilibrium-cooling curve
    (e.g., Dalgarno & McCray 1972)
    3.485e-15*zoxygen*T**(-0.63)*(1.0 - np.exp(-(1e-5*T)**1.63)
    """
    return 3.485e-15*zoxygen*T**(-0.63)*(1.0 - np.exp(-(1e-5*T)**1.63))


def cool_recomb(ne, nH, xn, T):
    """
    Returns cooling due to recombination of H
    see Krumholz et al. (2007)
    6.1e-10*ne*nH*(1.0 - xn)*kB*T*T**(-0.89) (for T>100K)
    """
    return 8.422e-26*ne*nH*(1.0 - xn)*T**0.11


def gff(nu, T, Z):
    """Draine 10.5
    Assuming that h*nu << kT
    """
    k_B = ac.k_B.cgs.value
    e = ac.e.gauss.value
    m_e = ac.m_e.cgs.value
    gam = 0.57721566490153286
    return np.sqrt(3.0)/np.pi*(np.log((2.0*k_B*T)**1.5/(np.pi*Z*e**2*m_e**0.5*nu))) - 2.5*gam


def jff(nu, T, ni=1.0, ne=1.0, Z=1.0):
    """Draine (10.1)"""
    m_e = ac.m_e.cgs.value
    e = ac.e.gauss.value
    c = ac.c.cgs.value
    k_B = ac.k_B.cgs.value
    h = ac.h.cgs.value
    # print h*nu/(k_B*T)
    # gaunt=gff(nu,T,Z)
    return gff(nu, T, Z)*8.0/3.0*(2.0*np.pi/3.0)**0.5*e**6/(m_e**2*c**3) *\
        (m_e/(k_B*T))**0.5*np.exp(-h*nu/(k_B*T))*ne*ni*Z**2
