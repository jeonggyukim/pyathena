import os
import pandas as pd
import numpy as np
from scipy.special import expn
import xarray as xr
import astropy.units as au
import astropy.constants as ac

def J_over_JUV_inside_slab(tau, tau_SF):
    """
    Compute the mean intensity at tau(z) assuming that source
    distribtuion in a layer of optical depth tau_SF

    Compute the mean intensity at tau(z) (optical depth from the midplane)
    0 < zz := tau(z)/(tau_SF/2) < 1.0
    """
    # if not np.all(np.abs(tau) <= 0.5*tau_SF):
    #     raise ValueError("tau must be smaller than or equal to tau_SF/2")

    return 0.5/tau_SF*(2.0 - expn(2,0.5*tau_SF - tau) - expn(2,0.5*tau_SF + tau))

def J_over_JUV_outside_slab(tau, tau_SF):
    """
    Compute the mean intensity at height |z| > Lz/2
    with tau(z) = tau > tau_SF/2
    """
    # if not np.all(np.abs(tau) >= 0.5*tau_SF):
    #     raise ValueError("optical depth must be larger than or equal to tau_SF/2")

    return 0.5/tau_SF*(expn(2,tau - 0.5*tau_SF) - expn(2,tau + 0.5*tau_SF))

def J_over_JUV_avg_slab(tau_SF):
    """
    Compute the mean intensity averaged over the entrie volume of the slab
    from -Lz/2 < z < Lz/2
    or
    from -tau_SF/2 < tau < tau_SF/2
    """

    return 1.0/tau_SF*(1.0 - (0.5 - expn(3,tau_SF))/tau_SF)


def read_rad_lost(filename, force_override=False, verbose=False):
    """
    Function to read rad_lost.txt and pickle

    Parameters:
       filename : string
           Name of the file to open, including extension
       force_override: bool
           Flag to force read of rad_lost file even when pickle exists

    Returns:
       df, da : tuple
          (pandas dataframe, xarray dataarray)
    """

    fpkl = filename + '.p'
    if not force_override and os.path.exists(fpkl) and \
       os.path.getmtime(fpkl) > os.path.getmtime(filename):
        df = pd.read_pickle(fpkl)
        if verbose:
            print('[read_radiators]: reading from existing pickle.')

    if verbose:
        print('[read_radiators]: pickle does not exist or file updated.' + \
                  ' Reading {0:s}'.format(filename))

    df = pd.read_csv(filename, sep=' ', header=None, skiprows=0)

    # drop nan column (due to space at the end of line in output file)
    df = df.drop(labels=[df.columns[-1]], axis=1)
    col = {0:'time',1:'nfreq',2:'nsrc',3:'N_mu'}
    nfreq = df[1][0]
    N_mu = df[3][0]
    for i in range(4, 4 + nfreq):
        col[i] = 'L_tot{0:d}'.format(i-4)

    df = df.rename(columns=col)

    return df

def calc_Jrad_pp(s, num):
    """
    Function to calculate z-profile of mean intensity Jrad in the plane-parallel approximation
    """

    from scipy import interpolate

    rsp = s.read_starpar(num, force_override=False)
    zpa = s.read_zprof('whole')

    domain = s.domain
    u = s.u
    par = s.par
    dz = s.domain['dx'][2]
    sigmad = dict()
    for f in ('LW','PE'):
        sigmad[f] = par['opacity']['sigma_dust_{0:s}0'.format(f)]*par['problem']['Z_dust']

    dz_cgs = dz*u.length.cgs.value
    LxLy = (s.domain['Lx'][0]*s.domain['Lx'][1])*u.pc**2

    # Cell centers (z plus center)
    zpc = zpa.z.data
    zmc = np.flipud(zpc)
    # Cell edges
    zpe = zpc + 0.5*domain['dx'][2]
    zme = zmc + 0.5*domain['dx'][2]

    # zprofile
    zp = zpa.sel(time=rsp['time'], method='nearest')

    zstar = rsp['sp_src']['x3']
    S4pi = dict()
    Jrad = dict()
    for f in ('LW','PE'):
        taup = sigmad[f]*np.cumsum(zp['d'].data)*dz_cgs
        taum = sigmad[f]*np.cumsum(np.flipud(zp['d'].data))*dz_cgs

        # interpolation function
        fp = interpolate.interp1d(zpe, taup)
        fm = interpolate.interp1d(zme, taum)

        # Surface density of luminosity over 4pi in cgs units
        S4pi[f] = rsp['sp_src'][f'L_{f}']/LxLy/(4.0*np.pi)*(1.0*au.Lsun/au.pc**2).cgs.value

        # plane-parallel approximation
        Jrad[f] = []
        for z_ in zpe:
            Jrad[f].append(calc_Jrad(z_, S4pi[f], zstar, fp, fm, dz))

        Jrad[f] = np.array(Jrad[f])

    return Jrad, zpc

def calc_Jrad(z, S4pi, zstar, fp, fm, dz_pc):
    J = 0.0
    for S4pi_, zstar_ in zip(S4pi, zstar):
        Dz = 0.2*dz_pc
        tau_SF = fp(zstar_ + Dz) - fp(zstar_ - Dz)
        #print(tau_SF)
        if z >= zstar_ + Dz:
            tau = fp(z) - fp(zstar_)
            J += S4pi_*0.5*expn(1, tau)
            #J += SFUV4pi_*J_over_JUV_outside_slab(tau, tau_SF)
        elif z <= zstar_ - Dz:
            tau = fm(z) - fm(zstar_)
            J += S4pi_*0.5*expn(1, tau)
            #J += SFUV4pi_*J_over_JUV_outside_slab(tau, tau_SF)
        else:
            J += S4pi_*J_over_JUV_avg_slab(tau_SF)
            #J += SFUV4pi_*J_over_JUV_inside_slab(0.0, tau_SF)
            pass

    return J
