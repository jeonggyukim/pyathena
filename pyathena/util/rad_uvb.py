import pathlib
import os.path as osp
import numpy as np
import astropy.units as au
import astropy.constants as ac
import xarray as xr
from scipy import integrate

import matplotlib.pyplot as plt

local = pathlib.Path(__file__).parent.absolute()

def read_FG20():
    """Function to read Faucher-Giguère (2020) UV background as functions of nu
    and z

    See : https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.1614F/abstract
    https://galaxies.northwestern.edu/uvb-fg20/

    Returns
    -------
    r : dict

    """

    fname = osp.join(local, '../../data/fg20_spec_lambda.dat')
    with open(fname) as fp:
        ll = fp.readlines()

    # Get redshift
    z = np.array(list(map(lambda zz: np.round(float(zz),3), ll[0].split())))
    zstr = np.array(list(map(lambda zz: str(np.round(float(zz),3)), ll[0].split())))
    nz = len(zstr)

    # Get wavelengths
    wav = []
    for l in ll[1:]:
        wav.append(float(l.split()[0]))

    wav = np.array(wav)*au.angstrom
    nwav = len(wav)

    # Read Jnu
    Jnu = np.zeros((nz,nwav))
    for i,l in enumerate(ll[1:]):
        Jnu[:,i] = np.array(list(map(float, l.split())))[1:]

    r = dict()
    r['nwav'] = nwav
    r['wav'] = wav
    r['nz'] = nz
    r['z'] = z
    r['Jnu'] = Jnu
    r['nu'] = (ac.c/wav).to('Hz').value

    da = xr.DataArray(data=r['Jnu'], dims=['z', 'wav'], coords=[r['z'],r['wav']],
                      attrs=dict(description='FG20 UVB', units='ergs/s/cm^2/Hz/sr'))
    r['ds'] = xr.Dataset(dict(Jnu=da))
    conv = (1.0/au.angstrom**2*au.erg/au.s/au.cm**2/au.Hz/au.sr*ac.c).\
        to('erg s-1 cm-2 sr-1 angstrom-1').value

    # Jlambda in unit of erg/s/cm^2/sr/ang
    r['ds']['Jlambda'] = (r['ds']['Jnu']/r['wav']**2)*conv

    # Note that LyC includes X-ray
    idx_FUV = np.logical_and(r['wav'].value < 2000.0, r['wav'].value > 912.0)
    idx_LyC = np.logical_and(r['wav'].value < 912.0, r['wav'].value > 0.0)
    idx_Xray = np.logical_and(r['wav'].value < 100.0, r['wav'].value > 0.0)
    r['J_FUV'] = -integrate.trapz(r['nu'][idx_FUV]*r['ds']['Jnu'][:,idx_FUV],
                                       x=np.log(r['nu'][idx_FUV]))
    r['J_LyC'] = -integrate.trapz(r['nu'][idx_LyC]*r['ds']['Jnu'][:,idx_LyC],
                                  x=np.log(r['nu'][idx_LyC]))
    r['J_Xray'] = -integrate.trapz(r['nu'][idx_Xray]*r['ds']['Jnu'][:,idx_Xray],
                                   x=np.log(r['nu'][idx_Xray]))
    r['idx_FUV'] = idx_FUV
    r['idx_LyC'] = idx_LyC
    r['idx_Xray'] = idx_Xray

    from ..microphysics.photx import PhotX
    E_LyC = (ac.h*r['nu'][idx_LyC]*au.Hz).to('eV').value
    E_Xray = (ac.h*r['nu'][idx_Xray]*au.Hz).to('eV').value

    px = PhotX()
    Eth_H = px.get_Eth(1,1)
    sigma_pi_H_LyC = px.get_sigma(1,1,E_LyC)
    sigma_pi_H_Xray = px.get_sigma(1,1,E_Xray)

    # Photoionization rate
    # $\zeta_{\rm pi} = \int \frac{4\pi J_{\nu}}{h\nu} \sigma_{\rm pi} d\nu$

    # $\mathcal{I}_{\rm phot} = n_{\rm HI}\int_{\nu_0}^{\infty}
    # \frac{4\pi J_{\nu}}{h\nu} \sigma_{\rm pi,\nu} d\nu = n_{\rm HI} \zeta_{\rm pi,H}$

    # Photoheating rate
    # $n_{\rm HI} q_{\rm pi,H}\zeta_{\rm pi,H} = n_{\rm H}\Gamma_{\rm pi} =
    # n_{\rm HI}\int_{\nu_0}^{\infty} \frac{4\pi J_{\nu}}{h\nu} \sigma_{\rm pi,\nu}(h\nu - h\nu_0) d\nu$

    zeta_pi_H = -integrate.trapz(4.0*np.pi*r['ds']['Jnu'][:,idx_LyC]*\
                                 sigma_pi_H_LyC/(E_LyC*au.eV).cgs.value,
                                 x=r['nu'][idx_LyC])
    q_zeta_pi_H = -integrate.trapz(4.0*np.pi*r['ds']['Jnu'][:,idx_LyC]*\
                                   sigma_pi_H_LyC/(E_LyC*au.eV).cgs.value*\
                                   (E_LyC*au.eV - Eth_H*au.eV).cgs.value,
                                   x=r['nu'][idx_LyC])
    q_pi_H = (q_zeta_pi_H/zeta_pi_H*au.erg).to('eV')
    sigma_mean_pi_H = -zeta_pi_H/(integrate.trapz(4.0*np.pi*r['ds']['Jnu'][:,idx_LyC]/\
                                                  (E_LyC*au.eV).cgs.value,
                                                  x=r['nu'][idx_LyC]))

    zeta_pi_H_Xray = -integrate.trapz(4.0*np.pi*r['ds']['Jnu'][:,idx_Xray]*\
                                      sigma_pi_H_Xray/(E_Xray*au.eV).cgs.value,
                                      x=r['nu'][idx_Xray])

    # Save J as dataset
    r['ds_int'] = xr.Dataset(data_vars=dict(J_LyC=('z',r['J_LyC']),
                                            J_FUV=('z',r['J_FUV']),
                                            zeta_pi_H=('z',zeta_pi_H),
                                            zeta_pi_H_Xray=('z',zeta_pi_H_Xray),
                                            q_zeta_pi_H=('z',q_zeta_pi_H),
                                            q_pi_H=('z',q_pi_H),
                                            sigma_mean_pi_H=('z',sigma_mean_pi_H),
                                            ),
                             coords=dict(z=r['z']))

    ## Save ISRF info for reference
    from .rad_isrf import nuJnu_Dr78

    E = np.linspace(6,13.6, 1000)*au.eV
    nuJnu_ISRF = nuJnu_Dr78(E)
    wav = (ac.h*ac.c/E).to('angstrom')
    J_FUV_ISRF = integrate.trapz(nuJnu_ISRF, x=np.log((ac.c/wav).to('Hz').value))

    r['ISRF_Dr'] = dict()
    r['ISRF_Dr']['E'] = E
    r['ISRF_Dr']['nuJnu'] = nuJnu_ISRF
    r['ISRF_Dr']['wav'] = wav
    r['ISRF_Dr']['J_FUV'] = J_FUV_ISRF

    return r


def plt_UVB_Jnu_FG20(redshifts=[0,1,2,3,8]):
    """Function to plot nuJnu in cgs units"""

    plt.figure()
    r = read_FG20()
    Jnu = r['ds']['Jnu']

    for z in redshifts:
        plt.loglog(r['wav'], r['nu']*Jnu.sel(z=z),
                   label=r'$z=${0:d}'.format(z))

    plt.loglog(r['ISRF_Dr']['wav'], r['ISRF_Dr']['nuJnu'],
               label='Draine ISRF',c='k',lw=3)

    plt.axvline(100, color='grey', lw=1) # Xray
    plt.axvline(912, color='grey', lw=1) # LyC
    plt.xlabel(r'${\rm wavlength}\;[\mathrm{\AA}]$')
    plt.ylabel(r'$\nu J_{\nu}\;[{\rm erg}\,{\rm s}^{-1}\,{\rm cm}^{-2}\,{\rm sr}^{-1}]$')
    plt.suptitle('FG20 UV background')
    plt.ylim(1e-9,1e-3)
    plt.legend(loc=2)

    return plt.gcf()

def plt_UVB_JFUV_JLyC_FG20():
    """plot frequency-integrated mean intensity
    J_FUV and J_LyC as a function of redshift
    """

    plt.figure()

    r = read_FG20()

    l, = plt.semilogy(r['z'],r['J_FUV'],label='FUV (912-2000)')
    plt.semilogy(r['z'], r['J_LyC'],label='LyC (<912)',c=l.get_color(),ls='--')
    plt.semilogy(r['z'], r['J_Xray'],label='X-ray only (<100)',c=l.get_color(),ls=':')
    plt.axhline(r['ISRF_Dr']['J_FUV'].value, c='C2',label='Draine ISRF')

    plt.xlabel('redshift')
    plt.ylabel(r'$J_{\rm FG20}\;[{\rm erg}\,{\rm cm}^{-2}\,{\rm s}^{-1}\,{\rm sr}^{-1}]$')
    plt.ylim(1e-9,5e-4)
    plt.legend()
    plt.suptitle('Frequency integrated mean intensity')

    return plt.gcf()

def plt_UVB_pi_heat_rates():
    """plot photoionization rate, heating rate, mean photon energy, etc"""
    r = read_FG20()
    rr = r['ds_int']
    fig,axes = plt.subplots(1,4,figsize=(18,4))
    plt.sca(axes[0])
    plt.semilogy(r['z'], rr['zeta_pi_H'])
    plt.xlabel('redshift')
    plt.ylabel(r'$\zeta_{\rm pi,H}\,[{\rm s}^{-1}]$')

    plt.sca(axes[1])
    plt.semilogy(r['z'], rr['q_pi_H']*(1.0*au.eV).cgs.value*rr['zeta_pi_H'])
    plt.xlabel('redshift')
    plt.ylabel(r'$q_{\rm pi,H}\zeta_{\rm pi,H}\;[{\rm erg}\,{\rm s}^{-1}]$')

    plt.sca(axes[2])
    plt.plot(r['z'], rr['q_pi_H'])
    plt.xlabel('redshift')
    plt.ylabel(r'$q_{\rm pi,H}=\langle h\nu - h\nu_{0}\rangle\;[{\rm eV}]$')

    plt.sca(axes[3])
    plt.plot(r['z'], rr['sigma_mean_pi_H'])
    plt.xlabel('redshift')
    plt.ylabel(r'$\langle \sigma_{\rm pi,H}\rangle\;[{\rm cm}^{2}]$')
    plt.tight_layout()
    plt.suptitle('FG20 UVB')

    return fig


def f_sshld_R13(nH, sigma_pi_mean, T, zeta_pi, fg=0.17, z=0.0):
    """Self-shielding factor by Rahmati+13
    """
    if z < 0.5:
        n0 = 10.0**-2.94
        alpha1 = -3.98
        alpha2 = -1.09
        beta = 1.29
        f = 0.01
    else:
        n0 = 6.73e-3*(sigma_pi_mean/2.49e-18)**(-2.0/3.0)*(T/1e4)**0.17*\
        (zeta_pi.value/1e-12)**(2.0/3.0)*(fg/0.17)**(-1.0/3.0)
        alpha1 = -2.28
        alpha2 = -0.84
        beta = 1.64
        f = 0.02

    x = nH/n0
    return (1.0 - f)*(1.0 + x**beta)**alpha1 + f*(1.0 + x)**alpha2
