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
from ..util.cloud import Cloud

np.seterr(divide='ignore', invalid='ignore')

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
        vol_cgs = vol*u.cm**3
        Myr_cgs = u.time
        nscalars = par['configure']['nscalars']

        if par['configure']['new_cooling'] == 'ON':
            newcool = True
        else:
            newcool =False

        try:
            if par['radps']['irayt'] == 1:
                rayt = True
            else:
                rayt = False
        except KeyError:
            rayt = True

        if par['configure']['sixray'] == 'ON':
            sixray = True
        else:
            sixray = False

        if par['configure']['gas'] == 'mhd':
            mhd = True
        else:
            mhd = False

        cl = Cloud(M=par['problem']['M_cloud'],
                   R=par['problem']['R_cloud'],
                   alpha_vir=par['problem']['alpha_vir'])

        iWind = par['feedback']['iWind']

        try:
            iPhot = par['radps']['iPhot']
        except KeyError:
            try:
                iPhot = par['radps']['iPhotIon']
            except KeyError:
                iPhot = True

        iRadp = par['radps']['apply_force']

        # Time in code unit
        hst['time_code'] = hst['time']
        # Time in Myr
        hst['time'] *= u.Myr
        # Time in freefall time
        hst['tau'] = hst['time']/cl.tff.to('Myr').value
        # Time step
        hst['dt_code'] = hst['dt']
        hst['dt'] *= u.Myr

        if newcool and (sixray or newcool):
            for c in ('dt_cool_min','dt_xH2_min','dt_xHII_min'):
                hst[c] *= vol*u.Myr

        ########
        # Mass #
        ########
        cols = ['M', 'M_cl', 'Mc', 'Mu', 'Mw', 'Mi', 'Mh', 'M_sp', 'M_sp_esc']
        if newcool:
            cols += ['M_cl_neu', 'M_H2', 'M_HI', 'M_H2_cl', 'M_HI_cl']
        if iWind:
            cols += ['M_s1']

        for c in cols:
            try:
                hst[c] *= vol*u.Msun
            except KeyError:
                self.logger.warning('[read_hst]: Column {0:s} not found'.format(c))
                continue

        ######################
        # Star particle mass #
        ######################
        # M_sp: total
        # M_sp_in: mass of sp currently in the domain
        # M_sp_esc: mass of sp escaped the domain
        hst['M_sp_in'] = hst['M_sp']
        hst['M_sp'] += hst['M_sp_esc']
        # Number of star particles and number of active star particles (active >= 0)
        hst['N_sp'] *= vol
        hst['N_sp_active'] *= vol

        hst = self._calc_SFR(hst)

        ################
        # Outflow mass #
        ################
        # Obtain ejected mass by integrating mass flux
        cols = ['M', 'M_cl']
        if newcool:
            cols += ['M_cl_neu', 'M_H2', 'M_HI', 'M_H2_cl', 'M_HI_cl']
        if iWind:
            cols += ['M_s1']

        for c in cols:
            try:
                hst['d'+c] *= vol*u.Msun/u.Myr # Outflow rate in Msun/Myr
                hst[c + '_of'] = integrate.cumtrapz(hst['d'+c], hst['time'], initial=0.0)
            except KeyError:
                self.logger.warning('[read_hst]: Column {0:s} not found'.format(c))
                continue

        ##########
        # Energy #
        ##########
        cols = ['E{}', 'E{}_cl']
        if newcool:
            cols += ['E{}_cl_neu', 'E{}_H2', 'E{}_HI', 'E{}_H2_cl', 'E{}_HI_cl']

        kind = ['thm','kin','grav','grav_vir']
        if mhd:
            kind += ['mag']

        # Convert energy unit [Msun*(km/s)**2]
        for c in cols:
            for k in kind:
                if c.format(k) == 'Egrav' or c.format(k) == 'Egrav_vir_cl':
                    continue
                try:
                    hst[c.format(k)] *= vol*u.Msun*(u.kms)**2
                except:
                    self.logger.warning('[read_hst]: Column {0:s} not found'.\
                                        format(c.format(k)))
                    continue

        # Mean velocity and kinetic energy
        if newcool:
            for i in range(1,4):
                hst[f'rhov{i}_cl_neu'] *= vol*u.Msun*u.kms
                hst[f'rhov{i}sq_cl_neu'] *= vol*u.Msun*u.kms
                hst[f'vmean{i}_cl_neu'] = hst['rhov1_cl_neu']/hst['M_cl_neu']

            hst['vmean_cl_neu'] = np.sqrt(hst['vmean1_cl_neu']**2 +
                                          hst['vmean2_cl_neu']**2 +
                                          hst['vmean3_cl_neu']**2)
            hst['Ekin_cl_neu0'] = hst['Ekin_cl_neu']
            hst['Ekin_cl_neu'] = hst['Ekin_cl_neu0'] - \
                (hst['vmean1_cl_neu']*hst[f'rhov1_cl_neu'] +
                 hst['vmean2_cl_neu']*hst[f'rhov2_cl_neu'] +
                 hst['vmean3_cl_neu']*hst[f'rhov3_cl_neu']) + 0.5*hst['M_cl_neu']*hst['vmean_cl_neu']**2

        ###################
        # Radial Momentum #
        ###################
        cols = ['pr', 'pr_cl', 'pr_xcm', 'pr_xcm_cl']
        if newcool:
            cols += ['pr_cl_neu', 'pr_H2', 'pr_HI', 'pr_H2_cl', 'pr_HI_cl',
                     'pr_xcm_cl_neu', 'pr_xcm_H2', 'pr_xcm_HI',
                     'pr_xcm_H2_cl', 'pr_xcm_HI_cl']

        for c in cols:
            try:
                hst[c] *= vol*u.Msun*u.kms
            except:
                self.logger.warning('[read_hst]: Column {0:s} not found'.\
                                    format(c.format(k)))
                continue


        ###########################
        # Outflow energy #
        ###########################
        cols = ['Ekin', 'Ethm']
        if iWind:
            cols += ['Ekin_s1', 'Ethm_s1']
        for c in cols:
            try:
                hst['d'+c] *= vol*u.Msun*(u.kms)**2/u.Myr
                hst[c+'_of'] = integrate.cumtrapz(hst['d'+c], hst['time'], initial=0.0)
            except KeyError:
                self.logger.warning('[read_hst]: Column {0:s} not found'.format(c))
                continue

        #################################################
        # Energy cooled away in wind polluted gas [erg] #
        #################################################
        if iWind:
            Myr_cgs = (1.0*au.Myr).to('s').value
            c = 'net_cool_s1'
            hst[c] *= vol_cgs
            hst[c+'_cumul'] = integrate.cumtrapz(hst[c], hst['time'], initial=0.0)*Myr_cgs

            # Conversion factor for energy rate
            # conv_Edot = (1.0*au.M_sun*(au.km/au.s)**2/au.Myr).cgs.value
            # hst['']conv_Edot*(h['dEthm_s1']+h['dEkin_s1'])

        ###########################
        # Outflow radial momentum #
        ###########################
        cols = ['pr', 'pr_cl', 'pr_xcm', 'pr_xcm_cl']
        if newcool:
            cols += ['pr_cl_neu', 'pr_H2', 'pr_HI', 'pr_H2_cl', 'pr_HI_cl',
                     'pr_xcm_cl_neu', 'pr_xcm_H2', 'pr_xcm_HI',
                     'pr_xcm_H2_cl', 'pr_xcm_HI_cl']

        for c in cols:
            try:
                hst['d'+c] *= vol*u.Msun*u.kms/u.Myr
                hst[c+'_of'] = integrate.cumtrapz(hst['d'+c], hst['time'], initial=0.0)
            except KeyError:
                self.logger.warning('[read_hst]: Column {0:s} not found'.format(c))
                continue

        ########################
        # Outward radial force #
        ########################
        cols = ['Fthm', 'Fgrav', 'Fcent']
        if rayt:
            cols += ['Frad']
        if mhd:
            cols += ['Fmagp','Fmagt']

        for c in cols:
            try:
                hst[c] *= vol*u.Msun*u.kms/u.Myr
                hst[c + '_int'] = integrate.cumtrapz(hst[c], hst['time'], initial=0.0)
            except KeyError:
                self.logger.warning('[read_hst]: Column {0:s} not found'.format(c))
                continue

        if rayt:
            hst = self._calc_radiation(hst)

    #         hst.rho_out *= vol*u.Msun/u.Myr # Outflow rate in Msun/Myr
    #         hst['Mof_dot'] = hst.rho_out
    #         hst['Mof'] = integrate.cumtrapz(hst['rho_out'], hst['time'], initial=0.0)

    #     for ph in ('H2','HI','HII'):
    #         c = f'rho_{ph}_out'
    #         if c in hst.columns:
    #             hst[c] *= vol*u.Msun/u.Myr
    #             hst[f'Mof_{ph}_dot'] = hst[c]
    #             hst[f'Mof_{ph}'] = integrate.cumtrapz(
    #                 hst[c], hst['time'], initial=0.0)

    #         c = f'rho_{ph}_out_cl'
    #         if c in hst.columns:
    #             hst[c] *= vol*u.Msun/u.Myr
    #             hst[f'Mof_cl_{ph}_dot'] = hst[c]
    #             hst[f'Mof_cl_{ph}'] = integrate.cumtrapz(
    #                 hst[c], hst['time'], initial=0.0)
    #     try:
    #         hst['Mof_cl'] = hst['Mof_cl_H2'] + hst['Mof_cl_HI'] + hst['Mof_cl_HII']
    #     except KeyError:
    #         pass


        if iWind:
            hst['wind_Minj'] *= vol*u.Msun
            hst['wind_Einj'] *= vol*u.erg
            hst['wind_pinj'] *= vol*u.Msun*u.kms
            hst['wind_Mdot'] *= vol*u.Msun/u.Myr
            hst['wind_Edot'] *= vol*u.erg/u.s
            hst['wind_pdot'] *= vol*u.Msun*u.kms/u.Myr

        return hst


    #     # Mass of (gas, gas, starpar, cold/intermediate/warm/hot temperature gas,
    #     #          molecular,atomic,ionized) in Msun
    #     for c in ('Mgas','Mcold','Minter','Mwarm','Mhot',
    #               'MH2','MHI','MH2_cl','MHI_cl','M_cl',
    #               'Mstar','Mstar_esc','Mstar_s0','Mstar_s1','Mstar_s2'):
    #         try:
    #             hst[c] *= vol*u.Msun
    #         except KeyError:
    #             self.logger.warning('[read_hst]: Column {0:s} not found'.format(c))
    #             continue

    #     for i in range(nscalars):
    #         try:
    #             hst[f'scalar{i}'] *= vol*u.Msun
    #         except KeyError:
    #             self.logger.warning('Nscalar {0:i}, but column {0:s} not found'.\
    #                                 format(nscalars,c))
    #             continue

    #     # Convert energy unit [Msun*(km/s)**2]
    #     for c in hst.columns:
    #         if 'Emag' in c or 'Ekin' in c or 'Egrav' in c:
    #             hst[c] *=  vol*u.Msun*(u.kms)**2

    #     # Velocity dispersion
    #     # (mass-weighted rms velocity magnitude)
    #     try:
    #         hst['vdisp_cl'] = np.sqrt(2.0*(hst['Ekin_H2_cl'] + hst['Ekin_HI_cl'])
    #                                   /(hst['MHI_cl'] + hst['MH2_cl']))
    #     except KeyError:
    #         self.logger.warning('Could not compute vdisp_cl due to KeyError')


    #     try:
    #         hst['MHII'] = hst['Mgas'] - hst['MHI'] - hst['MH2']
    #         hst['MHII_cl'] = hst['M_cl'] - hst['MHI_cl'] - hst['MH2_cl']

    #         # Volume
    #         hst['VHII'] = 1.0 - hst['VHI'] - hst['VH2']
    #         hst['VHII_cl'] = hst['V_cl'] - hst['VHI_cl'] - hst['VH2_cl']
    #     except KeyError:
    #         pass


    #     # Radiation variables
    #     if par['configure']['radps'] == 'ON':
    #         hst = self._calc_radiation(hst)
    #     try:
    #         hst['avir_neu_cl'] = -2.0*(hst['Ekin_HI_cl']+hst['Ekin_H2_cl']) \
    #                              /(hst['Egrav_H2_cl']+hst['Egrav_HI_cl'])
    #     except KeyError:
    #         pass

        #hst.index = hst['time_code']


    # def _calc_outflow(self, hst):

    #     u = self.u
    #     domain = self.domain
    #     vol = domain['Lx'].prod()

    #     # Obtain ejected mass by integrating mass flux
    #     if 'rho_out' in hst.columns:
    #         hst.rho_out *= vol*u.Msun/u.Myr # Outflow rate in Msun/Myr
    #         hst['Mof_dot'] = hst.rho_out
    #         hst['Mof'] = integrate.cumtrapz(hst['rho_out'], hst['time'], initial=0.0)

    #     for ph in ('H2','HI','HII'):
    #         c = f'rho_{ph}_out'
    #         if c in hst.columns:
    #             hst[c] *= vol*u.Msun/u.Myr
    #             hst[f'Mof_{ph}_dot'] = hst[c]
    #             hst[f'Mof_{ph}'] = integrate.cumtrapz(
    #                 hst[c], hst['time'], initial=0.0)

    #         c = f'rho_{ph}_out_cl'
    #         if c in hst.columns:
    #             hst[c] *= vol*u.Msun/u.Myr
    #             hst[f'Mof_cl_{ph}_dot'] = hst[c]
    #             hst[f'Mof_cl_{ph}'] = integrate.cumtrapz(
    #                 hst[c], hst['time'], initial=0.0)
    #     try:
    #         hst['Mof_cl'] = hst['Mof_cl_H2'] + hst['Mof_cl_HI'] + hst['Mof_cl_HII']
    #     except KeyError:
    #         pass

    #     return hst

    def _calc_SFR(self, hst):
        """Compute instantaneous SFR, SFR averaged over the past 1 Myr, 3Myr, etc.
        """

        # Instantaneous SFR
        hst['SFR'] = deriv_convolve(hst['M_sp'].values, hst['time'].values,
                                    gauss=True, fft=False, stddev=3.0)

        # Set any negative values to zero
        hst['SFR'][hst['SFR'] < 0.0] = 0.0
        if hst.time.max() > 1.0:
            hst_ = hst[hst.time < 1.0]
            winsize_1Myr = hst_.index.size
            hst['SFR_1Myr'] = hst.SFR.rolling(
                winsize_1Myr, min_periods=1, win_type='boxcar').mean()

        if hst.time.max() > 3.0:
            hst_ = hst[hst.time < 3.0]
            winsize_3Myr = 3*winsize_1Myr
            hst['SFR_3Myr'] = hst.SFR.rolling(
                winsize_3Myr, min_periods=1, win_type='boxcar').mean()

            # self.logger.warning('Failed to calculate SFR')
            # pass

        return hst

    def _calc_radiation(self, hst):

        par = self.par
        u = self.u
        domain = self.domain
        if self.config_time > pd.to_datetime('2022-05-01 00:00:00 -04:00'):
            vol = 1.0
        else:
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
                        hst[f'Ldust_{k}'] = hst[f'Ldust{i}']*vol*u.Lsun

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
                        # Instantaneous fraction
                        hst[f'fesc_{k}'] = hst[f'Lesc_{k}']/hst[f'Ltot_{k}']
                        hst[f'fdust_{k}'] = hst[f'Ldust_{k}']/hst[f'Ltot_{k}']
                        # Cumulative fraction
                        hst[f'fesc_cum_{k}'] = \
                        integrate.cumtrapz(hst[f'Lesc_{k}'], hst.time, initial=0.0)/\
                        integrate.cumtrapz(hst[f'Ltot_{k}'], hst.time, initial=0.0)
                        hst[f'fesc_cum_{k}'].fillna(value=0.0, inplace=True)
                        hst[f'fdust_cum_{k}'] = \
                        integrate.cumtrapz(hst[f'Ldust_{k}'], hst.time, initial=0.0)/\
                        integrate.cumtrapz(hst[f'Ltot_{k}'], hst.time, initial=0.0)
                        hst[f'fdust_cum_{k}'].fillna(value=0.0, inplace=True)
                    except KeyError as e:
                        raise e

        if 'Ltot_LW' in hst.columns and 'Ltot_PE' in hst.columns:
            hst['fesc_FUV'] = (hst['Lesc_PE'] + hst['Lesc_LW'])/(hst['Ltot_PE'] + hst['Ltot_LW'])
            hst['fesc_cum_FUV'] = \
                integrate.cumtrapz(hst['Lesc_PE'] + hst['Lesc_LW'], hst.time, initial=0.0)/\
                integrate.cumtrapz(hst['Ltot_PE'] + hst['Ltot_LW'], hst.time, initial=0.0)
            hst[f'fesc_cum_FUV'].fillna(value=0.0, inplace=True)

            hst['fdust_FUV'] = (hst['Ldust_PE'] + hst['Ldust_LW'])/(hst['Ltot_PE'] + hst['Ltot_LW'])
            hst['fdust_cum_FUV'] = \
                integrate.cumtrapz(hst['Ldust_PE'] + hst['Ldust_LW'], hst.time, initial=0.0)/\
                integrate.cumtrapz(hst['Ltot_PE'] + hst['Ltot_LW'], hst.time, initial=0.0)
            hst[f'fdust_cum_FUV'].fillna(value=0.0, inplace=True)

        hst['Ltot_FUV'] = hst['Ltot_LW'] + hst['Ltot_PE']
        hst['Ltot'] = hst['Ltot_PH'] + hst['Ltot_LW'] + hst['Ltot_PE']
        # Should we multipy Ltot_PE by 1.25?
        hst['Ltot_over_c'] = ((hst['Ltot_PH'] + hst['Ltot_LW'] + self.par['radps']['frad_PE_boost']*hst['Ltot_PE']).values*\
                              ac.L_sun/ac.c).to('Msun km s-1 Myr-1')
        hst['Ltot_over_c_int'] = cumtrapz(hst['Ltot_over_c'], hst['time'], initial=0.0)

        return hst


class PlotHst(object):
    """
    Plot time evolution of various quantities
    """

    def __init__(self, sa, df, models=None, suptitle=None, tlim=None, normed_x=False,
                 ls=['-','--',':','-.','--',':'],
                 lw=[1.5, 1.5, 1.5, 1.5, 3, 4],
                 subplots_kwargs=dict(nrows=1, ncols=2,
                                      figsize=(15,5), merge_last_row=False),
                 plt_vars_kwargs=dict(),
                 plt_vars=['mass', 'momentum'],
                 savfig=True, savdir=None):

        self.sa = sa
        self.df = df
        if models is None:
            self.models = sa.models
        else:
            self.models = models
        self.models = np.atleast_1d(self.models)
        self.linestyles = ['solid','dashed','dotted','dot-dashed','thick-dashed','thick-dotted']
        self.ls = ls
        self.lw = lw

        self.get_subplots(**subplots_kwargs)
        if suptitle is None:
            self.set_suptitle()
        if tlim is None:
            self.tlim = (0, 20)

        self.normed_x = normed_x
        if self.normed_x:
            self.col_time = 'tau'
        else:
            self.col_time = 'time_code'

        for i, v in enumerate(plt_vars):
            method = getattr(self, 'plt_' + v)
            try:
                method(self.axes[i], **(plt_vars_kwargs[v]))
            except KeyError:
                method(self.axes[i])

        if savfig:
            if savdir is None:
                savdir = osp.join('/tigress/jk11/figures/SF-CLOUD/hst')

            if not osp.exists(savdir):
                os.makedirs(savdir)

            plt.savefig(osp.join(savdir, 'hst-{0:s}.png'.format('-'.join(self.models))),
                        dpi=200)

    def get_subplots(self, nrows=3, ncols=2, figsize=(15,15), merge_last_row=False):

        self.fig, self.axes = plt.subplots(nrows, ncols,
                                           figsize=figsize, constrained_layout=True)
        self.axes = self.axes.flatten()
        if merge_last_row:
            # Last row is merged
            gs = self.axes[-2].get_gridspec()
            # remove the underlying axes
            for ax in self.axes[-2:]:
                ax.remove()
            self.axbig = self.fig.add_subplot(gs[-1,:])

    def set_suptitle(self):
        suptitle = ''
        if len(self.models) == 1:
            suptitle = 'Model: {0:s}'.format(self.models[0])
        else:
            for i,(ls, mdl) in enumerate(zip(self.linestyles, self.models)):
                if i > 0:
                    suptitle += '\n '

                suptitle += '\n {0:s}: {1:s}'.format(ls,mdl)

        self.fig.suptitle(suptitle, linespacing=0.7)

    def set_params(self, ax, models, setp_kwargs, kind, **kwargs):
        if ax is None:
            ax = plt.gca()
        if models is None:
            models = self.models

        if not self.normed_x:
            xlabel = 'time [code]'
        else:
            xlabel = r'${\rm time}/t_{\rm ff,0}$'

        M0 = self.df.loc[models[0]]['M']
        if kind == 'mass':
            if kwargs['normed_y']:
                ylabel = 'mass/$M_0$'
                ylim = (0.0, 1.1)
            else:
                ylabel = r'mass $[M_{\odot}]$'
                ylim = (0.0, 1.1e5*M0/1e5)

            setp_kwargs_def  = dict(xlabel=xlabel, ylabel=ylabel,
                                    yscale='linear', ylim=ylim)

        if kind == 'volume':
            setp_kwargs_def = dict(xlabel=xlabel, ylabel='volume fraction',
                                   yscale='linear', ylim=(0,1.2))
        if kind == 'dt':
            setp_kwargs_def = dict(xlabel=xlabel, ylabel='dt', yscale='log', ylim=(1e-6,1e-2))
        if kind == 'force':
            setp_kwargs_def = dict(xlabel=xlabel,
                                   ylabel=r'force $[M_{\odot}{\rm km}\,{\rm s}^{-1}\,{\rm Myr}^{-1}]$',
                                   yscale='log', ylim=(1e3,5e6))
        if kind == 'momentum':
            setp_kwargs_def = dict(xlabel=xlabel,
                                   ylabel=r'momentum $[M_{\odot}{\rm km}\,{\rm s}^{-1}]$',
                                   yscale='linear')
        if kind == 'luminosity':
            setp_kwargs_def = dict(xlabel=xlabel, ylabel=r'luminosity $[L_{\odot}]$',
                                   yscale='log')
        if kind == 'fesc':
            setp_kwargs_def = dict(xlabel=xlabel, ylabel=r'escape fraction',
                                   yscale='linear', ylim=(0,1.4))
        if setp_kwargs is not None:
            setp_kwargs_def.update(setp_kwargs)

        return ax, models, setp_kwargs_def

    def plt_mass(self, ax=None, models=None, setp_kwargs=None, normed_y=True,
                 plt_H2=True,plt_hot=True):
        ax, models, setp_kwargs = self.set_params(ax, models,
                        setp_kwargs, 'mass', **dict(normed_y=normed_y))
        plt.sca(ax)
        for i, (mdl,ls,lw) in enumerate(zip(models, self.ls, self.lw)):
            mhd = self.df.loc[mdl]['mhd']
            iWind = self.df.loc[mdl]['iWind']
            tff = self.df.loc[mdl]['tff']
            h = self.df.loc[mdl]['hst']
            x = h[self.col_time]
            if normed_y:
                M0 = self.df.loc[mdl]['M']
            else:
                M0 = 1.0

            plt.plot(x, h['M_cl']/M0, c='C0', ls=ls, lw=lw)
            plt.plot(x, h['M_sp']/M0, c='C1', ls=ls, lw=lw)
            plt.plot(x, (h['M_cl']-h['M_H2_cl']-h['M_HI_cl'])/M0, c='C3', ls=ls, lw=lw)
            plt.plot(x, h['M_cl_of']/M0, c='C5', ls=ls, lw=lw)
            plt.plot(x, (h['M_cl_of']-h['M_HI_cl_of']-h['M_H2_cl_of'])/M0, c='C7', ls=ls, lw=lw)

            if plt_H2:
                plt.plot(x, h['M_H2_cl']/M0, c='C4', ls=ls, lw=lw)
            if plt_hot:
                plt.plot(x, (h['Mi']+h['Mh'])/M0, c='C2', ls=ls, lw=lw)
            if iWind:
                plt.plot(x, h['wind_Minj']/M0, c='C8', ls=ls, lw=0.5)
                plt.plot(x, h['M_sp_s1']/M0, c='C2', ls=ls, lw=3)

            plt.plot(x, (h['M_cl'] + h['M_sp'] + h['M_cl_of'])/M0, c='grey', ls=ls, lw=lw)

        plt.setp(ax, **setp_kwargs)
        labels = [r'$M_{\rm cl}$',r'$M_{\ast}$',r'$M_{\rm HII,cl}$']
        labels += [r'$M_{\rm of,cl}$',r'$M_{\rm of,HII,cl}$']
        if plt_H2:
            labels += [r'$M_{\rm H_2,cl}$']
        if plt_hot:
            labels += [r'$M_{\rm hot}$']
        if iWind:
            labels.extend([r'$M_{\rm wind}$',r'$M_{\ast,{\rm wind}}$'])

        ax.legend(labels, loc=2)


    def plt_volume(self, ax=None, models=None, setp_kwargs=None):
        ax, models, setp_kwargs = self.set_params(ax, models,
                        setp_kwargs, 'volume')
        plt.sca(ax)
        for i, (mdl,ls,lw) in enumerate(zip(models, self.ls, self.lw)):
            mhd = self.df.loc[mdl]['mhd']
            iWind = self.df.loc[mdl]['iWind']
            tff = self.df.loc[mdl]['tff']
            h = self.df.loc[mdl]['hst']
            x = h[self.col_time]

            plt.plot(x, 1.0-h['V_HI']-h['V_H2'], c='C0', ls=ls, lw=1.5)
            plt.plot(x, h['Vi'] + h['Vh'], c='C1', ls=ls, lw=1.5)
            if iWind:
                plt.plot(x, h['Vf'], c='C2', ls=ls, lw=1.5)

        plt.setp(ax, **setp_kwargs)
        ax.legend([r'ionized',r'hot',r'free wind'])

    def plt_dt(self, ax=None, models=None, setp_kwargs=None, dtHII=True):
        ax, models, setp_kwargs = self.set_params(ax, models, setp_kwargs, 'dt')
        plt.sca(ax)
        for i, (mdl,ls,lw) in enumerate(zip(models, self.ls, self.lw)):
            tff = self.df.loc[mdl]['tff']
            h = self.df.loc[mdl]['hst']
            x = h[self.col_time]

            plt.plot(x, h['dt'], ls=ls, c='k', lw=lw)
            if dtHII:
                plt.plot(x, h['dt_xHII_min'], ls=ls, c='C0')

        plt.setp(ax, **setp_kwargs)
        ax.legend([r'$dt$',r'$dt_{\rm HII}$'])

    def plt_force(self, ax=None, models=None, setp_kwargs=None, plt_inj=True):
        ax, models, setp_kwargs = self.set_params(ax, models, setp_kwargs, 'force')
        plt.sca(ax)
        for i, (mdl,ls,lw) in enumerate(zip(models, self.ls, self.lw)):
            mhd = self.df.loc[mdl]['mhd']
            iWind = self.df.loc[mdl]['iWind']
            iRadp = self.df.loc[mdl]['iRadp']
            tff = self.df.loc[mdl]['tff']
            h = self.df.loc[mdl]['hst']
            x = h[self.col_time]

            plt.plot(x, h['Fthm'], ls=ls, c='C0', lw=lw)
            plt.plot(x, -h['Fgrav'], ls=ls, c='C2', lw=lw)
            plt.plot(x, h['Fcent'], ls=ls, c='C3', lw=lw)
            if iRadp:
                plt.plot(x, h['Frad'], ls=ls, c='C1', lw=lw)

            Ftot = h['Fthm'] + h['Fgrav']+ h['Fcent']
            if iRadp:
                Ftot += h['Frad']
            if mhd:
                Ftot += h['Fmagp'] + h['Fmagt']

            if plt_inj:
                if iRadp:
                    plt.plot(x, h['Ltot_over_c'], ls=ls, c='C1', lw=0.5)
                if iWind:
                    plt.plot(x, h['wind_pdot'], ls=ls, c='C0', lw=0.5)

            plt.plot(x, Ftot, ls=ls, c='k', lw=lw)

        plt.setp(ax, **setp_kwargs)
        labels = [r'$F_{\rm thm}$', r'$-F_{\rm grav}$',
                  r'$F_{\rm cent}$']
        if iRadp:
            labels += [r'$F_{\rm rad}$']
        if plt_inj:
            if iRadp:
                labels += [r'$L/c$']
            if iWind:
                labels += [r'$\dot{p}_{\rm w}$']

        labels += [r'$F_{\rm tot}$']

        ax.legend(labels, loc=2)

    def plt_momentum(self, ax=None, models=None, setp_kwargs=None, plt_inj=True):
        ax, models, setp_kwargs = self.set_params(ax, models, setp_kwargs, 'momentum')
        plt.sca(ax)
        for i, (mdl,ls,lw) in enumerate(zip(models, self.ls, self.lw)):
            mhd = self.df.loc[mdl]['mhd']
            iWind = self.df.loc[mdl]['iWind']
            iRadp = self.df.loc[mdl]['iRadp']
            tff = self.df.loc[mdl]['tff']
            h = self.df.loc[mdl]['hst']
            x = h[self.col_time]

            plt.plot(x, h['Fthm_int'], label='thm', ls=ls, c='C0')
            plt.plot(x, -h['Fgrav_int'], label='|grav|', ls=ls, c='C2')
            plt.plot(x, h['Fcent_int'], label='cent', ls=ls, c='C3')
            if iRadp:
                plt.plot(x, h['Frad_int'], label='rad', ls=ls, c='C1')

            Ftot_int = h['Fthm_int'] + h['Fcent_int'] + h['Fgrav_int']
            if iRadp:
                Ftot_int += h['Frad_int']
            # if mhd:
            #     Ftot_int += h['Fmagp_int'] + h['Fmagt_int']

            if plt_inj:
                if iRadp:
                    plt.plot(x, h['Ltot_over_c_int'], ls=ls, c='C1', lw=0.5)
                if iWind:
                    plt.plot(x, h['wind_pinj'], ls=ls, c='C0', lw=0.5)

            plt.plot(x, Ftot_int, ls=ls, c='grey', lw=3, alpha=0.7)
            plt.plot(x, h['pr'] + h['pr_of'] - h['pr'].iloc[0], ls=ls, c='k', label='tot')
            # plt.plot(x, h['pr_xcm'] + h['pr_xcm_of'] - h['pr_xcm'].iloc[0], lw=4,
            #          ls=ls, c='k', label='tot')

        plt.setp(ax, **setp_kwargs)
        labels = [r'$\int F_{\rm thm}dt$',
                  r'$-\int F_{\rm grav} dt$',
                  r'$\int F_{\rm cent}dt$',]
        if iRadp:
            labels += [r'$\int F_{\rm rad} dt$',]
        if plt_inj:
            if iRadp:
                labels += [r'$\int L/c dt$']
            if iWind:
                labels += [r'$\int \dot{p}_w dt$']

        labels += [r'$\int F_{\rm tot} dt$',
                   r'$\Delta p_{\rm r,box} + p_{\rm r,of}$',]

        ax.legend(labels, loc=2)


    def plt_luminosity(self, ax=None, models=None, setp_kwargs=None, plt_wind=True):

        ax, models, setp_kwargs = self.set_params(ax, models, setp_kwargs, 'luminosity')
        plt.sca(ax)
        for i, (mdl,ls,lw) in enumerate(zip(models, self.ls, self.lw)):
            mhd = self.df.loc[mdl]['mhd']
            iWind = self.df.loc[mdl]['iWind']
            tff = self.df.loc[mdl]['tff']
            h = self.df.loc[mdl]['hst']
            x = h[self.col_time]

            plt.plot(x, h['Ltot_PH'], ls=ls, c='C0')
            plt.plot(x, h['Ltot_FUV'], ls=ls, c='C1')
            if plt_wind and iWind:
                plt.plot(x, h['wind_Edot']/ac.L_sun.cgs.value, ls=ls, lw=0.5, c='k')

        plt.setp(ax, **setp_kwargs)
        labels = [r'$L_{\rm LyC}$', r'$L_{\rm FUV}$']
        if plt_wind and iWind:
            labels += [r'$L_{\rm wind}$']
        ax.legend(labels, loc=4, ncol=2)
        ax.set_ylim(ax.get_ylim()[1]*1e-4,ax.get_ylim()[1])

    def plt_fesc(self, ax=None, models=None, setp_kwargs=None,
                 plt_avg=True, plt_dust=True):
        ax, models, setp_kwargs = self.set_params(ax, models, setp_kwargs, 'fesc')
        plt.sca(ax)
        for i, (mdl,ls,lw) in enumerate(zip(models, self.ls, self.lw)):
            tff = self.df.loc[mdl]['tff']
            h = self.df.loc[mdl]['hst']
            x = h[self.col_time]

            plt.plot(x, h['fesc_PH'], ls=ls, lw=1.0, c='C0')
            plt.plot(x, h['fesc_FUV'], ls=ls, lw=1.0, c='C1')
            if plt_avg:
                plt.plot(x, h['fesc_cum_PH'], ls=ls, lw=3, c='C0')
                plt.plot(x, h['fesc_cum_FUV'], ls=ls, lw=3, c='C1')

            if plt_dust:
                plt.plot(x, h['fdust_PH'], ls=ls, lw=1.0, c='C0', alpha=0.35)
                plt.plot(x, h['fdust_FUV'], ls=ls, lw=1.0, c='C1', alpha=0.35)
                if plt_avg:
                    plt.plot(x, h['fdust_cum_PH'], ls=ls, lw=3, c='C0', alpha=0.35)
                    plt.plot(x, h['fdust_cum_FUV'], ls=ls, lw=3, c='C1', alpha=0.35)

        plt.setp(ax, **setp_kwargs)
        labels = [r'$f_{\rm esc,LyC}$', r'$f_{\rm esc,FUV}$']
        if plt_avg:
            labels += [r'$\langle f_{\rm esc,LyC} \rangle$',
                       r'$\langle f_{\rm esc,FUV} \rangle$']
        if plt_dust:
            labels += [r'$f_{\rm dust,LyC}$', r'$f_{\rm dust,FUV}$']
            if plt_avg:
                labels += [r'$\langle f_{\rm dust,LyC} \rangle$',
                           r'$\langle f_{\rm dust,FUV} \rangle$']

        ax.legend(labels, loc=2, ncol=2)


# ax = plt.gca()

# t1 = lambda x: x/tff
# t2 = lambda x: x*tff

# sax = ax.secondary_xaxis('top', functions=(t1,t2))
# ax.xaxis.set_ticks_position('bottom')
# sax.set_xlabel(r'$t/t_{\rm ff,0}$', labelpad=12)
