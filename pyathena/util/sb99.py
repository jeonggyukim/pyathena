import pathlib
import xarray as xr
import os.path as osp
import numpy as np
import astropy.units as au
import astropy.constants as ac
import pandas as pd
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d
from ..microphysics.dust_draine import DustDraine

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize, LogNorm

local = pathlib.Path(__file__).parent.absolute()

class SB99(object):
    
    def __init__(self, basename='/projects/EOSTRIKE/SB99/Z1_M1E6/output/',
                 prefix='Z1_M1E6', logM=6.0, tmax_Myr=100.0):
        """
        Parameters
        ----------
        logM : float
             Mass of a (coeval) cluster in log10 M/Msun
        tmax_Myr : float
             Maximum age in Myr

        Returns
        -------
        df : pandas DataFrame
        """
        
        self.logM = logM
        self.tmax_Myr = tmax_Myr
        self.basename = basename
        self.fnames = dict()
        self.fnames['snr'] = osp.join(self.basename, prefix + '.snr1')
        self.fnames['power'] = osp.join(self.basename, prefix + '.power1')
        self.fnames['spectrum'] = osp.join(self.basename, prefix + '.spectrum1')
        
        # self.dfs = self.read_sn()
        # self.dfw = self.read_wind()
        # self.rr = self.read_rad()
        
    def read_sn(self):
        """Function to read snr1 (supernova rate) output
        """

        names = ['time', 'SN_rate', 'Edot_SN', 'Einj_SN', 'SN_rate_IB',
                 'Edot_SN_IB','Einj_SN_IB', 'Mpgen_typ', 'Mpgen_min',
                 'Edot_tot', 'Einj_tot']
        
        df = pd.read_csv(self.fnames['snr'], names=names, skiprows=7, delimiter='\s+')
        for c in df.columns:
            if c == 'time' or c.startswith('Mpgen'):
                continue
            df[c] = 10.0**(df[c] - self.logM)

        df = df.rename(columns={'time': 'time_yr'})
        df['time_Myr'] = df['time_yr']*1e-6
        # df['time'] = df['time_Myr']

        # Move column
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]
        
        return df
    
    def read_wind(self):
        """Function to read power1 (stellar wind power) output
        """

        names = ['time','Edot_all','Edot_OB','Edot_RSG','Edot_LBV','Edot_WR','Einj_all',
                 'pdot_all','pdot_OB','pdot_RSG','pdot_LBV','pdot_WR']
        df = pd.read_csv(self.fnames['power'], names=names, skiprows=7, delimiter='\s+')
        for c in df.columns:
            if c == 'time':
                continue
            df[c] = 10.0**(df[c] - self.logM)

        df = df.rename(columns={'time': 'time_yr'})
        df['time_Myr'] = df['time_yr']*1e-6

        # Wind terminal velocity
        for v in ('all', 'OB','RSG', 'LBV', 'WR'):
            df['Vw_' + v] = (2.0*df['Edot_' + v]/df['pdot_' + v])/1e5
        
        # Move column
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]
    
        return df

    def read_rad(self):
        """Function to read SB99 spectrum data and mean dust opacity
        """
        
        eV_cgs = (1.0*au.eV).cgs.value
        hc_cgs = (ac.h*ac.c).cgs.value
        Lsun_cgs = (ac.L_sun).cgs.value

        d = DustDraine()
        df_dust = d.dfa['Rv31']
        f_Cext = interp1d(np.log10(df_dust['lwav']), np.log10(df_dust['Cext']))
        f_Cabs = interp1d(np.log10(df_dust['lwav']), np.log10(df_dust['Cext']*(1.0 - df_dust['albedo'])))

        df = pd.read_csv(self.fnames['spectrum'], skiprows=6, sep='\s+',
                           names=['time', 'wav', 'logf', 'logfstar', 'logfneb'])
        df = df.rename(columns={'time': 'time_yr'})
        df['time_Myr'] = df['time_yr']*1e-6

        # Move column
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]

        dfg = df.groupby('time_yr')
        ntime = len(dfg.groups.keys())
        nwav = len(dfg.groups[list(dfg.groups.keys())[0]])

        # Normalize by the cluster mass
        time = np.empty(ntime)
        wav = np.empty(nwav)
        logf = np.empty((ntime, nwav), dtype=float)

        # Luminosity
        L_tot = []
        L_LyC = [] # LyC
        L_LW = [] # LW
        L_PE = [] # PE
        L_OPT = [] # Optical + IR

        wav0 = 912.0
        wav1 = 1108.0
        wav2 = 2068.0
        wav3 = 200000.0

        for i, (time_, df_) in enumerate(dfg):
            if time_*1e-6 > self.tmax_Myr:
                continue

            time[i] = time_
            logf[i, :] = df_.logf
            wav = df_.wav
            idx0 = wav <= wav0
            idx1 = np.logical_and(wav <= wav1, wav > wav0)
            idx2 = np.logical_and(wav <= wav2, wav > wav1)
            idx3 = np.logical_and(wav <= wav3, wav > wav2)

            L_tot.append(simps(10.0**(df_.logf - self.logM), df_.wav)/Lsun_cgs)
            L_LyC.append(simps(10.0**(df_.logf[idx0] - self.logM), df_.wav[idx0])/Lsun_cgs)
            L_LW.append(simps(10.0**(df_.logf[idx1] - self.logM), df_.wav[idx1])/Lsun_cgs)
            L_PE.append(simps(10.0**(df_.logf[idx2] - self.logM), df_.wav[idx2])/Lsun_cgs)
            L_OPT.append(simps(10.0**(df_.logf[idx3] - self.logM), df_.wav[idx3])/Lsun_cgs)

            # wavelength in micron
            l = wav*1e-4
            J = 10.0**df_['logf']
            f_J = interp1d(np.log10(l), np.log10(J))

            if i == 0:
                w = dict()
                w['LyC'] = np.logspace(np.log10(l.min()), np.log10(wav0*1e-4), 1000)
                w['LW'] = np.logspace(np.log10(wav0*1e-4), np.log10(wav1*1e-4), 1000)
                w['PE'] = np.logspace(np.log10(wav1*1e-4), np.log10(wav2*1e-4), 1000)
                w['OPT'] = np.logspace(np.log10(wav2*1e-4), np.log10(1.0), 1000)

                Cext = dict()
                Cabs = dict()
                hnu = dict()
                for k in w.keys():
                    Cext[k] = []
                    Cabs[k] = []
                    hnu[k] = []

            for k in w.keys():
                Cext[k].append(simps(10.0**f_Cext(np.log10(w[k]))*10.0**f_J(np.log10(w[k]))*w[k], w[k])/ \
                               simps(10.0**f_J(np.log10(w[k]))*w[k], w[k]))
                Cabs[k].append(simps(10.0**f_Cabs(np.log10(w[k]))*10.0**f_J(np.log10(w[k]))*w[k], w[k])/ \
                               simps(10.0**f_J(np.log10(w[k]))*w[k], w[k]))
                hnu[k].append(simps(10.0**f_J(np.log10(w[k])), w[k])/ \
                               simps(10.0**f_J(np.log10(w[k]))*w[k], w[k])*hc_cgs/(1.0*au.eV).cgs.value*1e4)

        for k in w.keys():
            Cext[k] = np.array(Cext[k])
            Cabs[k] = np.array(Cabs[k])
            hnu[k] = np.array(hnu[k])

        if i == 0:
            w = dict()
            w['LyC'] = np.logspace(np.log10(l.min()), np.log10(wav0*1e-4), 1000)
            w['LW'] = np.logspace(np.log10(wav0*1e-4), np.log10(wav1*1e-4), 1000)
            w['PE'] = np.logspace(np.log10(wav1*1e-4), np.log10(wav2*1e-4), 1000)
            w['OPT'] = np.logspace(np.log10(wav2*1e-4), np.log10(wav3*1e-4), 1000)

        L_tot = np.array(L_tot)
        L_LyC = np.array(L_LyC)
        L_LW = np.array(L_LW)
        L_PE = np.array(L_PE)
        L_OPT = np.array(L_OPT)
        L_UV = L_LyC + L_PE + L_LW
        L_FUV = L_PE + L_LW
        time_Myr = time*1e-6
        
        L = dict()
        L['tot'] = np.array(L_tot)
        L['LyC'] = np.array(L_LyC)
        L['LW'] = np.array(L_LW)
        L['PE'] = np.array(L_PE)
        L['OPT'] = np.array(L_OPT)
        L['UV'] = L['LyC'] + L['LW'] + L['PE']
        L['FUV'] = L['LW'] + L['PE']


        # Momentum injection rate (Msun km/s / Myr / Msun)
        pdot = dict()
        for v in ('tot', 'LyC', 'LW', 'PE', 'OPT', 'UV', 'FUV'):
            pdot[v] = (((L[v]*au.L_sun/ac.c).to('g cm s-2')).to('Msun km s-1 Myr-1')).value
        
        # Luminosity-weighted effective timescale
        # (e-folding timescale if L is decaying exponentially)
        tdecay_lum = dict()
        for k in L.keys():
            tdecay_lum[k] = trapz(L[k]*time_Myr, time_Myr)/trapz(L[k], time_Myr)

        Cext_mean = dict()
        Cabs_mean = dict()
        hnu_mean = dict()

        # Luminosity-weighted average cross-section, photon energy
        for k in Cext.keys():
            Cext_mean[k] = np.average(Cext[k], weights=L[k])
            Cabs_mean[k] = np.average(Cabs[k], weights=L[k])
            hnu_mean[k] = np.average(hnu[k], weights=L[k])

        # Photoionization cross section, mean energy of photoelectrons
        from ..microphysics.photx import PhotX,get_sigma_pi_H2

        sigma_pi_H = []
        sigma_pi_H2 = []
        dhnu_H_LyC = []
        dhnu_H2_LyC = []
        hnu_LyC = []
        
        ph = PhotX()
        l_th_H = ph.get_Eth(1,1,unit='Angstrom') # threshold wavelength
        l_th_H2 = hc_cgs*1e8/(15.2*eV_cgs)
        for i, (time_, df_) in enumerate(dfg):
            #print(time_,self.tmax_Myr)
            if time_*1e-6 > self.tmax_Myr:
                continue
            
            idx0 = df_.wav <= l_th_H
            E_th = hc_cgs/(df_.wav[idx0]*1e-8)/eV_cgs
            sigma_pi_H_l = ph.get_sigma(1,1,E_th)
            sigma_pi_H2_l = get_sigma_pi_H2(E_th.values)
        
            Jl = 10.0**(df_.logf[idx0] - self.logM)
            l = df_[idx0].wav
            int_Jl_dl = simps(Jl, l)
            int_lJl_dl = simps(Jl*l, l)
            int_sigma_H_lJl_dl = simps(Jl*l*sigma_pi_H_l, l)
            int_sigma_H2_lJl_dl = simps(Jl*l*sigma_pi_H2_l, l)
            hnu_LyC.append(hc_cgs*1e8*int_Jl_dl/int_lJl_dl/eV_cgs)
            sigma_pi_H.append(int_sigma_H_lJl_dl/int_lJl_dl)
            sigma_pi_H2.append(int_sigma_H2_lJl_dl/int_lJl_dl)
            dhnu_H_LyC.append(1e8*hc_cgs*simps(Jl*l*sigma_pi_H_l*(1/l - 1/l_th_H), l)/int_sigma_H_lJl_dl/eV_cgs)
            dhnu_H2_LyC.append(1e8*hc_cgs*simps(Jl*l*sigma_pi_H2_l*(1/l - 1/l_th_H2), l)/int_sigma_H2_lJl_dl/eV_cgs)

        dhnu_H_LyC = np.array(dhnu_H_LyC)
        dhnu_H2_LyC = np.array(dhnu_H2_LyC)
        sigma_pi_H = np.array(sigma_pi_H)
        sigma_pi_H2 = np.array(sigma_pi_H2)
        
        r = dict(df=df, df_dust=df_dust,
                 time_yr=time, time_Myr=time_Myr,
                 wav=wav, logf=logf, logM=self.logM,
                 L=L, pdot=pdot, tdecay_lum=tdecay_lum,
                 wav0=wav0, wav1=wav1, wav2=wav2, wav3=wav3,
                 Cabs=Cabs, Cext=Cext, hnu=hnu,
                 hnu_LyC=hnu_LyC, dhnu_H_LyC=dhnu_H_LyC, dhnu_H2_LyC=dhnu_H2_LyC,
                 sigma_pi_H=sigma_pi_H, sigma_pi_H2=sigma_pi_H2,
                 Cabs_mean=Cabs_mean, Cext_mean=Cext_mean, hnu_mean=hnu_mean)

        return r


    @staticmethod
    def plt_spec_sigmad(rr, lambda_Llambda=False, plt_isrf=True, tmax=50.0, nstride=10):
        """Function to plot SB99 spectrum

        Parameters
        ----------
        lambda_Llambda : bool
            Plot lambda_Llambda instead of Llambda
        """

        if plt_isrf:
            fig, axes = plt.subplots(3, 2, figsize=(12, 12),
                                     gridspec_kw=dict(width_ratios=(0.98,0.02),
                                                      height_ratios=(1/3.0,1/3.0,1/3.0),
                                                      wspace=0.05, hspace=0.11),)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                                     gridspec_kw=dict(width_ratios=(0.98,0.02),
                                                      height_ratios=(0.65,0.35),
                                                      wspace=0.05, hspace=0.01),)

        # Dust opacity
        irow = 0
        plt.sca(axes[irow,0])
        plt.tick_params(right=False, which='both', axis='y')
        from pyathena.microphysics import dust_draine
        muH = (1.4*au.u).cgs.value
        d = dust_draine.DustDraine()
        df = d.dfa['Rv31']
        plt.semilogy(df.lwav*1e4, df.Cext/muH, c='k', label='ext')
        plt.semilogy(df.lwav*1e4, df.kappa_abs, c='k', ls='--', label='abs')
        plt.xlim(1e2,2068)
        plt.ylim(1e2,2.5e3)
        plt.ylabel(r'$\kappa_{\rm d}\;[{\rm cm}^2\,{\rm g}^{-1}]$')
        plt.legend()

        def kappa2sigma(x):
            return x*muH

        def sigma2kappa(x):
            return x/muH

        sax1 = plt.gca().secondary_yaxis('right', functions=(kappa2sigma,sigma2kappa))
        sax1.set_ylabel(r'$\sigma_{\rm d}\;[{\rm cm}^2\,{\rm H}^{-1}]$')
        # axes[1,0].tick_params(right=False, labelright=False)

        def l_to_hnu(x):
            return ac.h.cgs.value*ac.c.cgs.value/1e-8/(1.0*au.eV).cgs.value/x

        def hnu_to_l(x):
            return 1.0/(ac.h.cgs.value*ac.c.cgs.value/1e-8/(1.0*au.eV).cgs.value)/x

        axes[irow,0].tick_params(top=False, labeltop=False)
        sax0 = plt.gca().secondary_xaxis('top', functions=(l_to_hnu,hnu_to_l))
        tick_loc = np.array([6.0, 11.2, 13.6, 50.0])
        def ftick(x):
            return ["%.1f" % z for z in x]

        # secax.set_xlim(ax1.get_xlim())
        sax0.set_xticks(tick_loc)
        sax0.set_xticklabels(ftick(tick_loc))
        sax0.set_xlabel(r'$h\nu\;[{\rm eV}]$', labelpad=10)
        # secax.tick_params(axis='x', which='major', pad=15)

        ytext = 1.8e3
        plt.annotate('LyC', ((912+plt.gca().get_xlim()[0])*0.5,ytext),
                     xycoords='data', ha='center')
        plt.annotate('LW', ((912+1108)*0.5,ytext), xycoords='data', ha='center')
        plt.annotate('PE', ((1108+2068)*0.5,ytext), xycoords='data', ha='center')
        
        plt.sca(axes[irow,1])
        plt.axis('off')

        irow += 1
        plt.sca(axes[irow,0])
        norm = mpl.colors.Normalize(0.0, tmax)
        cmap = mpl.cm.jet_r
        dfg = rr['df'].groupby('time_Myr')
        logM = rr['logM']
        for i, (time_, df_) in enumerate(dfg):
            if time_ > tmax:
                continue
                
            if i % nstride == 0:
                print('{0:.1f}'.format(time_), end=' ')
                if lambda_Llambda:
                    plt.plot(df_.wav, df_.wav*10.0**(df_.logf - logM), 
                             c=cmap(norm(time_)))#, marker='o', ms=3)
                else:
                    plt.plot(df_.wav, 10.0**(df_.logf - logM), 
                             c=cmap(norm(time_)))#, marker='o', ms=3)

        plt.xlim(100, 2068)
        if lambda_Llambda:
            plt.ylim(1e31, 1e38)
        else:
            plt.ylim(1e28, 1e35)
        
        plt.yscale('log')
        plt.ylabel(r'$L_{\lambda}/M_{\ast}\;[{\rm erg}\,{\rm s}^{-1}\,\AA^{-1}\,M_{\odot}^{-1}]$')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, tmax))
        plt.colorbar(sm, cax=axes[irow,1], label=r'$t_{\rm age}\;[{\rm Myr}]$')

        for ax in axes[:,0]:
            plt.sca(ax)
            plt.axvspan(0,912, color='grey', ymin=0, alpha=0.1)
            plt.axvspan(912,1108, color='grey', ymin=0, alpha=0.15)
            plt.axvspan(1108,2068, color='grey', ymin=0, alpha=0.2)
            plt.xlabel(r'$\lambda\;[\AA]$')
            plt.xlim(1e2,2068)

        if plt_isrf:
            irow += 1
            plt.sca(axes[irow,1])
            plt.axis('off')
            plt_nuJnu_mid_plane_parallel(ax=axes[irow,0])

        return fig

    @staticmethod
    def plt_lum_evol(ax, rr, rw, rs, plt_sn=False):

        #pa.set_plt_fancy()
        plt.sca(ax)
        plt.plot(rr['time_Myr'], rr['L']['tot'], label=r'Bolometric', c='k')
        plt.plot(rr['time_Myr'], rr['L']['UV'], label=r'${\rm LyC+FUV}\;(<2068\,{\rm \AA})$', c='k', ls='--')
        plt.plot(rr['time_Myr'], rr['L']['LyC'], label=r'${\rm LyC}\;(<912\,{\rm \AA})$', c='C0')
        plt.plot(rr['time_Myr'], rr['L']['LW'], label=r'${\rm LW}\;(912$-$1108\,{\rm \AA})$', c='C1')
        plt.plot(rr['time_Myr'], rr['L']['PE'], label=r'${\rm PE}\;(1108$-$2068\,{\rm \AA})$', c='C2')
        plt.plot(rr['time_Myr'], rr['L']['OPT'], label=r'${\rm OPT}\;(2068$-$10000\,{\rm \AA})$', c='C3')
        plt.plot(rw['time_Myr'], rw['Edot_all']/(1.0*au.L_sun).cgs.value, c='C7', 
                 label=r'$L_{\rm w}/M_{\ast}$')
        if plt_sn:
            plt.plot(rs['time_Myr'], rs['Edot_SN']/(1.0*au.L_sun).cgs.value, c='C8', 
                     label=r'$L_{\rm sn}/M_{\ast}$')

        plt.yscale('log')
        plt.xlim(0, 20)
        plt.ylim(1e-1,2e3)
        plt.xlabel(r'$t_{\rm age}\;[{\rm Myr}]$')
        #plt.ylabel(r'$\Psi\,{\rm and}\,\Psi_w \;[L_{\odot}\,M_{\odot}^{-1}]$')
        plt.ylabel(r'$L/M_{\ast} \;[L_{\odot}\,M_{\odot}^{-1}]$')
        plt.legend(fontsize='small', loc=4)

        return ax
    
    @staticmethod
    def plt_pdot_evol(ax, rr, rw, rs):

        plt.sca(ax)
        plt.plot(rr['time_Myr'], (rr['L']['tot']*au.L_sun/ac.c/au.M_sun).to('km s-1 Myr-1'),
                 label=r'Bolometric', c='k')
        plt.plot(rr['time_Myr'], (rr['L']['LyC']*au.L_sun/ac.c/au.M_sun).to('km s-1 Myr-1'),
                 label=r'${\rm LyC}\;(<912\,{\rm \AA})$', c='C0', ls='-')
        plt.plot(rr['time_Myr'], (rr['L']['UV']*au.L_sun/ac.c/au.M_sun).to('km s-1 Myr-1'),
                 label=r'${\rm LyC+FUV}\;(<2068\,{\rm \AA})$', c='k', ls='--')
        plt.plot(rw['time_Myr'], (rw['pdot_all'].values*au.dyne/au.M_sun).to('km s-1 Myr-1'),
                 label=r'$\dot{p}_{\rm wind}/M_{\ast}$', c='C7')
        plt.xlim(0,20)
        plt.ylim(1e-1,5e1)
        plt.yscale('log')

        plt.xlabel(r'$t_{\rm age}\;[{\rm Myr}]$')
        plt.ylabel(r'$\dot{p}/M_{\ast} \;[{\rm km}\,{\rm s}^{-1}\,{\rm Myr}^{-1}]$')
        #plt.legend()
        
        return ax

    @staticmethod
    def plt_lum_cumul(ax, rr, rw, rs, normed=True, plt_sn=False):

        from scipy.integrate import cumulative_trapezoid
        integrate_L_cum = lambda L, t: cumulative_trapezoid((L*au.L_sun).cgs.value, 
                                                            (t*au.yr).cgs.value, initial=0.0)

        L_tot_cum = integrate_L_cum(rr['L']['tot'], rr['time_yr'])
        L_UV_cum = integrate_L_cum(rr['L']['UV'], rr['time_yr'])

        if normed:
            norm = L_tot_cum
        else:
            norm = 1.0
            
        plt.sca(ax)
        plt.plot(rr['time_Myr'], integrate_L_cum(rr['L']['LyC'], rr['time_yr'])/norm,
                 label='LyC', c='C0')
        plt.plot(rr['time_Myr'], integrate_L_cum(rr['L']['LW'], rr['time_yr'])/norm,
                 label='LW', c='C1')
        plt.plot(rr['time_Myr'], integrate_L_cum(rr['L']['PE'], rr['time_yr'])/norm,
                 label='PE', c='C2')
        plt.plot(rr['time_Myr'], integrate_L_cum(rr['L']['UV'], rr['time_yr'])/norm,
                 label=r'${\rm LyC+FUV}\;(<2068\,{\rm \AA})$', c='k', ls='--')
        plt.plot(rr['time_Myr'], integrate_L_cum(rr['L']['tot'], rr['time_yr'])/norm,
                 label='Bolometric', c='k')
        plt.plot(rw['time_Myr'], rw['Einj_all'], c='C7', label=r'$L_{\rm w}$')

        if plt_sn:
            plt.plot(rs['time_Myr'], rs['Einj_SN'], c='C8', label=r'$L_{\rm sn}$')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$t_{\rm age}\;[{\rm Myr}]$')
        if not normed:
            plt.ylabel(r'$\int (L/M_{\ast}) dt\;[{\rm erg}\,M_{\odot}^{-1}]$')
        else:
            plt.ylabel(r'$\int L dt/ \int L_{\rm UV}dt$')

        plt.xlim(3e-1, 5e1)
        plt.ylim(1e47, 2e51)
        # plt.legend()

        return ax

    @staticmethod
    def plt_pdot_cumul(ax, rr, rw, rs, normed=True, plt_sn=False):

        from scipy.integrate import cumulative_trapezoid
        integrate_pdot = lambda pdot, t: cumulative_trapezoid(
            pdot, t*au.Myr, initial=0.0)

        pdot_tot_cum = integrate_pdot(rr['L']['tot'], rr['time_Myr'])

        if normed:
            norm = pdot_tot_cum
        else:
            norm = 1.0
            
        plt.sca(ax)
        plt.plot(rr['time_Myr'], integrate_pdot(rr['pdot']['LyC'], rr['time_Myr'])/norm,
                 label='LyC', c='C0')
        # Skip PE and LW
        # plt.plot(rr['time_Myr'], integrate_pdot(rr['pdot']['LW'], rr['time_Myr'])/norm,
        #          label='LW', c='C1')
        # plt.plot(rr['time_Myr'], integrate_pdot(rr['pdot']['PE'], rr['time_Myr'])/norm,
        #          label='PE', c='C2')
        plt.plot(rr['time_Myr'], integrate_pdot(rr['pdot']['UV'], rr['time_Myr'])/norm,
                 label=r'${\rm LyC+FUV}\;(<2068\,{\rm \AA})$', c='k', ls='--')
        plt.plot(rr['time_Myr'], integrate_pdot(rr['pdot']['tot'], rr['time_Myr'])/norm,
                 label='Bolometric', c='k')

        # from cgs to astro units
        pdot_conv = (1.0*au.g*au.cm/au.s**2).to('Msun km s-1 Myr-1')
        plt.plot(rw['time_Myr'], integrate_pdot(rw['pdot_all']*pdot_conv,
                                                rw['time_Myr'])/norm,
                 c='C7', label=r'$L_{\rm w}$')

        # if plt_sn:
        #     plt.plot(rs['time_Myr'], rs['Einj_SN'], c='C8', label=r'$L_{\rm sn}$')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$t_{\rm age}\;[{\rm Myr}]$')
        if not normed:
            plt.ylabel(r'$\int (\dot{p}/M_{\ast}) dt\;[{\rm km}\,{\rm s}^{-1}]$')
        else:
            plt.ylabel(r'$\int \dot{p} dt/ \int \dot{p}_{\rm UV} dt$')

        plt.xlim(3e-1, 5e1)
        plt.ylim(5e-1,5e2)
        # plt.legend()

        return ax

    @staticmethod
    def plt_Edot_pdot_evol_cumul(rr, rw, rs, plt_sn=True, normed=False):
    
        fig, axes = plt.subplots(2,2,figsize=(12, 10), constrained_layout=True,
                                 gridspec_kw=dict(height_ratios=[0.5,0.5]))
        axes = axes.flatten()
        SB99.plt_lum_evol(axes[0], rr, rw, rs, plt_sn=plt_sn)
        SB99.plt_pdot_evol(axes[1], rr, rw, rs)
        SB99.plt_lum_cumul(axes[2], rr, rw, rs, normed=normed, plt_sn=plt_sn)
        SB99.plt_pdot_cumul(axes[3], rr, rw, rs, normed=normed, plt_sn=plt_sn)
        
        for ax in axes:
            ax.grid()
            #ax.set_xlim(0,50)
            #ax.set_xscale('linear')
    
        return fig


def plt_nuJnu_mid_plane_parallel(ax, Sigma_gas=10.0*au.M_sun/au.pc**2, plt_dr78=True):

    sb2 = SB99('/projects/EOSTRIKE/SB99/Z1_SFR1/output/', prefix='Z1_SFR1', logM=0.0)
    rr = sb2.read_rad()
    w = rr['wav'].values*1e-4
    d = DustDraine()
    dfdr = d.dfa['Rv31']
    f_Cext = interp1d(np.log10(dfdr['lwav']), np.log10(dfdr['Cext']),
                      bounds_error=False)
    f_Cabs = interp1d(np.log10(dfdr['lwav']), np.log10(dfdr['Cext']*(1.0 - dfdr['albedo'])),
                      bounds_error=False)
    
    Sigma_SFR = 2.5e-3
    Llambda = Sigma_SFR*10.0**rr['logf'][-1,:]*au.erg/au.s/au.angstrom
    Sigma = 10.0*au.M_sun/au.pc**2
    area = (1.0*au.kpc)**2
    muH = 1.4*au.u
    kappa_dust_ext = (10.0**f_Cext(np.log10(w))*au.cm**2/au.u).cgs
    kappa_dust_abs = (10.0**f_Cabs(np.log10(w))*au.cm**2/au.u).cgs
    tau_perp = (Sigma*kappa_dust_abs).to('').value
    
    from scipy.special import expn
    # Intensity at the midplane (see Ostriker et al. 2010)
    Jlambda = (Llambda/area/(4.0*np.pi*tau_perp)*
               (1.0 - expn(2, 0.5*tau_perp))).to('erg s-1 cm-2 angstrom-1')
    # Naive estimation without attenuation
    Jlambda0 = (Llambda/area/4.0).to('erg s-1 cm-2 angstrom-1')
    
    plt.sca(ax)
    l, = plt.loglog(rr['wav'], #rr['wav']*
                    Jlambda, label=r'SB99 + Ostriker et al. (2010)')
    plt.loglog(rr['wav'], #rr['wav']*
               Jlambda0, c=l.get_color(), alpha=0.5, ls='--', label=r'')
    
    if plt_dr78:
        from pyathena.util import rad_isrf
        wav2 = np.logspace(np.log10(912), np.log10(2068), 1000)*au.angstrom
        nu2 = (ac.c/wav2).to('Hz')
        E = (nu2*ac.h).to('eV')
        # Note that J_lambda = nuJnu/lambda
        plt.semilogy(wav2, rad_isrf.nuJnu_Dr78(E)/wav2, label='Draine (1978)')
        #plt.loglog(wav2, nu2*rad_isrf.Jnu_MMP83(wav2), c='C1', label='Mathis+83')

    plt.legend(loc=4)
    plt.xlim(1e2,2068)
    plt.xscale('linear')
    plt.ylim(1e-8,5e-6)
    plt.ylabel(r'$J_{\lambda}\;[{\rm erg}\,{\rm s}^{-1}\,{\rm cm}^{-2}\,{\rm sr}^{-1}\AA^{-1}]$')

    return None
