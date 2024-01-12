# hst.py

import os
import os.path as osp
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

import astropy.constants as ac
import astropy.units as au
from scipy import integrate
from scipy.integrate import cumtrapz

from ..io.read_hst import read_hst
from ..load_sim import LoadSim
from ..util.derivative import deriv_convolve
from ..util.scp_to_pc import scp_to_pc

class Hst:

    @LoadSim.Decorators.check_pickle_hst
    def read_hst(self, savdir=None, force_override=False):
        """Function to read hst and convert quantities to convenient units
        """

        hst = read_hst(self.files['hst'], force_override=force_override)

        par = self.par
        u = self.u
        domain = self.domain
        # volume of resolution element (code unit)
        dvol = domain['dx'].prod()
        # total volume of domain (code unit)
        vol = domain['Lx'].prod()
        nscalars = par['configure']['nscalars']

        # Rename column names
        hst = hst.rename(columns={"mass": "Mgas",     # total gas mass
                                  "mass_sp": "Mstar", # star particle mass in the box
                                  "mass_sp_esc": "Mstar_esc"})
        try:
            hst = hst.rename(columns={"mass_sp_s0": "Mstar_s0"})
            hst = hst.rename(columns={"mass_sp_s1": "Mstar_s1"})
            hst = hst.rename(columns={"mass_sp_s2": "Mstar_s2"})
        except KeyError:
            pass

        # Time in code unit
        hst['time_code'] = hst['time']
        # Time in Myr
        hst['time'] *= u.Myr
        # Time step
        hst['dt_code'] = hst['dt']
        hst['dt'] *= u.Myr
        if par['configure']['new_cooling'] == 'ON' and \
           (par['configure']['radps'] == 'ON' or par['configure']['sixray'] == 'ON'):
            for c in ('dt_cool_min','dt_xH2_min','dt_xHII_min'):
                hst[c] *= u.Myr*vol

        # Mass of (gas, gas, starpar, cold/intermediate/warm/hot temperature gas,
        #          molecular,atomic,ionized) in Msun
        for c in ('Mgas','Mcold','Minter','Mwarm','Mhot',
                  'MH2','MHI','MH2_cl','MHI_cl','M_cl',
                  'Mstar','Mstar_esc','Mstar_s0','Mstar_s1','Mstar_s2'):
            try:
                hst[c] *= vol*u.Msun
            except KeyError:
                self.logger.warning('[read_hst]: Column {0:s} not found'.format(c))
                continue

        for i in range(nscalars):
            try:
                hst[f'scalar{i}'] *= vol*u.Msun
            except KeyError:
                self.logger.warning('Nscalar {0:i}, but column {0:s} not found'.\
                                    format(nscalars,c))
                continue

        # Convert energy unit [Msun*(km/s)**2]
        for c in hst.columns:
            if 'Emag' in c or 'Ekin' in c or 'Egrav' in c:
                hst[c] *=  vol*u.Msun*(u.kms)**2

        # Velocity dispersion
        # (mass-weighted rms velocity magnitude)
        try:
            hst['vdisp_cl'] = np.sqrt(2.0*(hst['Ekin_H2_cl'] + hst['Ekin_HI_cl'])
                                      /(hst['MHI_cl'] + hst['MH2_cl']))
        except KeyError:
            self.logger.warning('Could not compute vdisp_cl due to KeyError')

        # Mstar: total
        # Mstar_in: mass of sp currently in the domain
        # Mstar_esc: mass of sp escaped the domain
        if 'Mstar' in hst.columns and 'Mstar_esc' in hst.columns:
            hst['Mstar_in'] = hst['Mstar']
            hst['Mstar'] += hst['Mstar_esc']

        try:
            hst['MHII'] = hst['Mgas'] - hst['MHI'] - hst['MH2']
            hst['MHII_cl'] = hst['M_cl'] - hst['MHI_cl'] - hst['MH2_cl']

            # Volume
            hst['VHII'] = 1.0 - hst['VHI'] - hst['VH2']
            hst['VHII_cl'] = hst['V_cl'] - hst['VHI_cl'] - hst['VH2_cl']
        except KeyError:
            pass

        hst = self._calc_SFR(hst)
        hst = self._calc_outflow(hst)

        # Radiation variables
        if par['configure']['radps'] == 'ON':
            hst = self._calc_radiation(hst)
        try:
            hst['avir_neu_cl'] = -2.0*(hst['Ekin_HI_cl']+hst['Ekin_H2_cl']) \
                                 /(hst['Egrav_H2_cl']+hst['Egrav_HI_cl'])
        except KeyError:
            pass

        #hst.index = hst['time_code']

        self.hst = hst

        return hst

    def _calc_outflow(self, hst):

        u = self.u
        domain = self.domain
        vol = domain['Lx'].prod()

        # Obtain ejected mass by integrating mass flux
        if 'rho_out' in hst.columns:
            hst.rho_out *= vol*u.Msun/u.Myr # Outflow rate in Msun/Myr
            hst['Mof_dot'] = hst.rho_out
            hst['Mof'] = integrate.cumtrapz(hst['rho_out'], hst['time'], initial=0.0)

        for ph in ('H2','HI','HII'):
            c = f'rho_{ph}_out'
            if c in hst.columns:
                hst[c] *= vol*u.Msun/u.Myr
                hst[f'Mof_{ph}_dot'] = hst[c]
                hst[f'Mof_{ph}'] = integrate.cumtrapz(
                    hst[c], hst['time'], initial=0.0)

            c = f'rho_{ph}_out_cl'
            if c in hst.columns:
                hst[c] *= vol*u.Msun/u.Myr
                hst[f'Mof_cl_{ph}_dot'] = hst[c]
                hst[f'Mof_cl_{ph}'] = integrate.cumtrapz(
                    hst[c], hst['time'], initial=0.0)
        try:
            hst['Mof_cl'] = hst['Mof_cl_H2'] + hst['Mof_cl_HI'] + hst['Mof_cl_HII']
        except KeyError:
            pass

        return hst

    def _calc_SFR(self, hst):
        """Compute instantaneous SFR, SFR averaged over the past 1 Myr, 3Myr, etc.
        """

        # Instantaneous SFR
        hst['SFR'] = deriv_convolve(hst['Mstar'].values, hst['time'].values,
                                    gauss=True, fft=False, stddev=3.0)

        # Set any negative values to zero
        hst['SFR'][hst['SFR'] < 0.0] = 0.0

        if hst.time.max() > 1.0:
            hst_ = hst[hst.time < 1.0]
            winsize_1Myr = hst_.index.size
            winsize_3Myr = 3*winsize_1Myr
            hst['SFR_1Myr'] = hst.SFR.rolling(
                winsize_1Myr, min_periods=1, win_type='boxcar').mean()
            hst['SFR_3Myr'] = hst.SFR.rolling(
                winsize_3Myr, min_periods=1, win_type='boxcar').mean()
        else:
            self.logger.warning('Total time interval smaller than 1 Myr')
            #pass

        return hst

    def _calc_radiation(self, hst):

        par = self.par
        u = self.u
        domain = self.domain
        # total volume of domain (code unit)
        vol = domain['Lx'].prod()

        # Total/escaping luminosity in Lsun
        ifreq = dict()
        for f in ('PH','LW','PE'): #,'PE_unatt'):
            try:
                ifreq[f] = par['radps']['ifreq_{0:s}'.format(f)]
            except KeyError:
                pass

        for i in range(par['radps']['nfreq']):
            for k, v in ifreq.items():
                if i == v:
                    try:
                        hst[f'Ltot_{k}'] = hst[f'Ltot{i}']*vol*u.Lsun
                        hst[f'Lesc_{k}'] = hst[f'Lesc{i}']*vol*u.Lsun
                        try:
                            hst[f'Ldust_{k}'] = hst[f'Ldust{i}']*vol*u.Lsun
                        except KeyError:
                            self.logger.info('Ldust not found in hst')

                        hnu = (par['radps'][f'hnu_{k}']*au.eV).cgs.value
                        hst[f'Qtot_{k}'] = hst[f'Ltot_{k}'].values * \
                                           (ac.L_sun.cgs.value)/hnu
                        hst[f'Qesc_{k}'] = hst[f'Lesc_{k}'].values * \
                                           (ac.L_sun.cgs.value)/hnu
                        # Cumulative number of escaped photons
                        hst[f'Qtot_cum_{k}'] = \
                        integrate.cumtrapz(hst[f'Qtot_{k}'], hst.time*u.time.cgs.value, initial=0.0)
                        hst[f'Qesc_cum_{k}'] = \
                        integrate.cumtrapz(hst[f'Qesc_{k}'], hst.time*u.time.cgs.value, initial=0.0)
                        # Instantaneous escape fraction
                        hst[f'fesc_{k}'] = hst[f'Lesc_{k}']/hst[f'Ltot_{k}']
                        # Cumulative escape fraction
                        hst[f'fesc_cum_{k}'] = \
                        integrate.cumtrapz(hst[f'Lesc_{k}'], hst.time, initial=0.0)/\
                        integrate.cumtrapz(hst[f'Ltot_{k}'], hst.time, initial=0.0)
                        hst[f'fesc_cum_{k}'].fillna(value=0.0, inplace=True)
                    except KeyError as e:
                        raise e

        if 'Ltot_LW' in hst.columns and 'Ltot_PE' in hst.columns:
            hst['fesc_FUV'] = (hst['Lesc_PE'] + hst['Lesc_LW'])/(hst['Ltot_PE'] + hst['Ltot_LW'])
            hst['fesc_cum_FUV'] = \
                integrate.cumtrapz(hst['Lesc_PE'] + hst['Lesc_LW'], hst.time, initial=0.0)/\
                integrate.cumtrapz(hst['Ltot_PE'] + hst['Ltot_LW'], hst.time, initial=0.0)
            hst[f'fesc_cum_FUV'].fillna(value=0.0, inplace=True)

        return hst


def plt_hst_compare(sa, models=['B2S4'], ls=['-'], r=None, savefig=True):

    plt.rcParams['ytick.right'] = True

    fig, axes = plt.subplots(3, 1, figsize=(6, 14), sharex=True)
    axes = axes.flatten()
    # ax1t = axes[1].twinx()
    # L0 = 1e6
    models = np.atleast_1d(models)
    ls = np.atleast_1d(ls)

    for i, (mdl, ls_) in enumerate(zip(models, ls)):
        s = sa.set_model(mdl)
        M0 = s.par['problem']['M_cloud']
        if r is not None:
            h = r.loc[mdl]['hst']
        else:
            h = s.read_hst()

        x = h.time

        plt.sca(axes[0])
        #plt.plot(x, h.M_cl/M0, label=r'$M_{\rm cl}$', c='k', ls=ls_)
        #plt.plot(x, h.Mgas/M0, label=r'_nolegend_', c='lightgray', ls=ls_)
        plt.plot(x, h.Mstar/M0, label=r'$M_{*}$', c='g', ls=ls_)
        plt.plot(x, (h.MHI_cl+h.MH2_cl)/M0, label=r'$M_{\rm HI+H_2}$', c='k', ls=ls_)
        plt.plot(x, h.MHI_cl/M0, label=r'$M_{\rm HI}$', c='hotpink', ls=ls_)
        plt.plot(x, h.MH2_cl/M0, label=r'$M_{\rm H_2}$', c='crimson', ls=ls_)
        plt.plot(x, h.MHII_cl/M0, label=r'$M_{\rm H^+}$', c='y', ls=ls_)

        plt.sca(axes[1])
        plt.plot(x, h.Ltot_LW + h.Ltot_PE, label=r'$L_{\rm FUV}$', c='C0', ls=ls_)
        plt.plot(x, h.Ltot_PH, label=r'$L_{\rm LyC}$', c='C1', ls=ls_)

        plt.sca(axes[2])
        plt.plot(x, h.fesc_PH, label=r'$f_{\rm esc,LyC}$', c='C1', ls=ls_)
        plt.plot(x, h.fesc_FUV, label=r'$f_{\rm esc,FUV}$', c='C0', ls=ls_)
        plt.plot(x, h.fesc_cum_PH, label=r'$f_{\rm esc,LyC}^{\rm cum}$', c='C1', lw=3.5, ls=ls_)
        plt.plot(x, h.fesc_cum_FUV, label=r'$f_{\rm esc,FUV}^{\rm cum}$', c='C0', lw=3.5, ls=ls_)

    plt.sca(axes[0])
    plt.ylabel(r'mass/$M_0$')
    plt.legend(fontsize='small', loc=1, facecolor='whitesmoke', edgecolor='grey')
    plt.yscale('log')
    plt.ylim(0.01, 2)

    plt.sca(axes[1])
    plt.ylabel(r'luminosity [$L_{\odot}$]')
    plt.yscale('log')
    plt.ylim(1e4,1e7)
    plt.legend(fontsize='small', loc=1, facecolor='whitesmoke', edgecolor='grey')

    plt.sca(axes[2])
    plt.xlabel(r'time [Myr]')
    plt.ylabel(r'escape fraction')
    plt.legend(fontsize='small', loc=(0.03,0.5), facecolor='whitesmoke', edgecolor='grey')
    plt.ylim(0.0, 1.0)

    labels = ('(a)','(b)','(c)')
    locs = ((0.03, 0.92),(0.03, 0.92),(0.03, 0.92))
    for ax,label,loc in zip(axes,labels,locs):
        ax.set_xlim(left=0)
        ax.grid()
        ax.annotate(label, loc, xycoords='axes fraction', fontsize='large')

    plt.tight_layout()

    if savefig:
        savname = '/tigress/jk11/figures/GMC/paper/hst/hst-{0:s}.png'.\
                  format('-'.join(models))
        plt.savefig(savname, dpi=200, bbox_inches='tight')
        scp_to_pc(savname, target='GMC-MHD-Results')

    return fig
