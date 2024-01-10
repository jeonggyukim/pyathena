import pathlib
import xarray as xr
import os
import os.path as osp
import numpy as np
import astropy.units as au
import astropy.constants as ac
import pandas as pd
from scipy.integrate import simps, trapz, cumulative_trapezoid
from scipy.interpolate import interp1d

from ..microphysics.dust_draine import DustDraine

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize, LogNorm

local = pathlib.Path(__file__).parent.absolute()

class SB99(object):

    def __init__(self, basedir='/projects/EOSTRIKE/SB99/Z014_M1E6_GenevaV00_dt01',
                 verbose=True):

        """
        Parameters
        ----------
        basedir : str
            directory name in which SB99 simulation results are located
        verbose : bool
            Print some useful parameters
        """

        self.basedir = basedir
        self.verbose = verbose

        self._find_files()
        self._read_params()

    def _find_files(self):

        ff = os.listdir(self.basedir)
        prefixes = []
        for f in ff:
            prefixes.append(f.split('.')[0])

        most_common = lambda lst: max(set(lst), key=lst.count)
        self.prefix = most_common(prefixes)

        self.files = dict()
        for f in ff:
            name = f.split('.')[-1][:-1]
            self.files[name] = osp.join(self.basedir, f)

        return None

    def _read_params(self):
        """Read some important parameters
        Full information contained in self.par
        """

        with open(self.files['output'], 'r') as fp:
            l = fp.readlines()
            l = [l_.strip() for l_ in l]
            l = list(filter(None, l))

        # Set cluster mass
        self.cont_SF = int(l[3]) > 0

        for i,l_ in enumerate(l):
            if l_.startswith('LAST GRID POINT:'):
                self.tmax_Myr = float(l[i+1])/1e6
            if l_.startswith('TIME STEP FOR PRINTING OUT THE SYNTHETIC SPECTRA:'):
                self.dt_Myr_spec = float(l[i+1])/1e6

        if not self.cont_SF:
            self.logM = np.log10(float(l[5]))
            if self.verbose:
                print('[SB99] Fixed mass', end=' ; ')
                print('logM:', self.logM)
        else:
            self.logM = 0.0
            self.SFR = float(l[7])
            if self.verbose:
                print('continuous SF')
                print('SFR:', self.SFR)

        self.par = l

    def read_sn(self):
        """Function to read snr1 (supernova rate) output
        """

        names = ['time', 'SN_rate', 'Edot_SN', 'Einj_SN', 'SN_rate_IB',
                 'Edot_SN_IB','Einj_SN_IB', 'Mpgen_typ', 'Mpgen_min',
                 'Edot_tot', 'Einj_tot']

        df = pd.read_csv(self.files['snr'], names=names, skiprows=7, delimiter='\s+')
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

    def read_yield(self):
        names = ['time','Mdot_H','Mdot_He','Mdot_C','Mdot_N',
                 'Mdot_O','Mdot_Mg','Mdot_Si','Mdot_S','Mdot_Fe',
                 'Mdot_wind','Mdot_sn','Mdot_wind_sn','M_tot']
        df = pd.read_csv(self.files['yield'], names=names,
                         skiprows=7, delimiter='\s+')

        # Normalize by cluster mass
        for c in df.columns:
            if c == 'time':
                continue
            df[c] = 10.0**(df[c] - self.logM)

        df = df.rename(columns={'time': 'time_yr'})
        df['time_Myr'] = df['time_yr']*1e-6
        # df.set_index('time_Myr', inplace=True)

        return df

    def read_wind(self):
        """Function to read power1 (stellar wind power) output
        """

        names = ['time','Edot_all','Edot_OB','Edot_RSG','Edot_LBV','Edot_WR','Einj_all',
                 'pdot_all','pdot_OB','pdot_RSG','pdot_LBV','pdot_WR']
        df = pd.read_csv(self.files['power'], names=names, skiprows=7, delimiter='\s+')

        # Normalize by cluster mass
        for c in df.columns:
            if c == 'time':
                continue
            df[c] = 10.0**(df[c] - self.logM)

        df = df.rename(columns={'time': 'time_yr'})
        df['time_Myr'] = df['time_yr']*1e-6

        # Edot and pdot are given in cgs units
        # Note that unit of pdot is converted later

        # Wind terminal velocity [km s-1]
        Vw_conv = (1.0*au.cm/au.s).to('km s-1')
        for v in ('all', 'OB','RSG', 'LBV', 'WR'):
            df['Vw_' + v] = (2.0*df['Edot_' + v]/df['pdot_' + v])*Vw_conv

        # Wind mass loss rate [Msun Myr-1 Msun-1]
        Mdot_conv = (1.0*au.g/au.s).to('Msun Myr-1').value
        for v in ('all', 'OB','RSG', 'LBV', 'WR'):
            df['Mdot_' + v] = df['pdot_' + v] / (df['Vw_' + v]/Vw_conv)*Mdot_conv

        # Momentum injection rate [Msun km s-1 Myr-1 Msun-1]
        pdot_conv = (1.0*au.g*au.cm/au.s**2).to('Msun km s-1 Myr-1').value
        for v in ('all', 'OB','RSG', 'LBV', 'WR'):
            df['pdot_' + v] = df['pdot_' + v]*pdot_conv

        # Time averaged energy, momentum injection rates \int_0^t q dt / \int_0^t dt
        for v in ('all', 'OB','RSG', 'LBV', 'WR'):
            df['Edot_' + v + '_avg'] = cumulative_trapezoid(df['Edot_'+v], x=df['time_Myr'], initial=0.0)/\
                    cumulative_trapezoid(np.repeat(1.0,len(df['time_Myr'])), x=df['time_Myr'], initial=0.0)
            df['pdot_' + v + '_avg'] = cumulative_trapezoid(df['pdot_'+v], x=df['time_Myr'], initial=0.0)/\
                    cumulative_trapezoid(np.repeat(1.0,len(df['time_Myr'])), x=df['time_Myr'], initial=0.0)
            df['Edot_' + v + '_avg'].iloc[0] = df['Edot_' + v + '_avg'].iloc[1]
            df['pdot_' + v + '_avg'].iloc[0] = df['pdot_' + v + '_avg'].iloc[1]

        # Move time to the first column
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
        f_Crpr = interp1d(np.log10(df_dust['lwav']),
                          np.log10(df_dust['Cext']*(1.0 - df_dust['albedo']) + \
                                   (1.0 - df_dust['cos'])*df_dust['Cext']*df_dust['albedo']))

        df = pd.read_csv(self.files['spectrum'], skiprows=6, sep='\s+',
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
        L_OPT = [] # Optical
        L_IR = [] # IR
        L_FUV = []

        wav0 = 912.0
        wav1 = 1108.0
        wav2 = 2068.0
        wav3 = 10000.0
        wav4 = 200000.0

        for i, (time_, df_) in enumerate(dfg):
            if time_*1e-6 > self.tmax_Myr:
                continue

            time[i] = time_
            logf[i, :] = df_.logf
            wav = df_.wav
            idx0 = wav <= wav0 # LyC
            idx1 = np.logical_and(wav <= wav1, wav > wav0) # LW
            idx2 = np.logical_and(wav <= wav2, wav > wav1) # PE
            idx3 = np.logical_and(wav <= wav3, wav > wav2) # OPT
            idx4 = np.logical_and(wav <= wav4, wav > wav3) # IR
            idx5 = np.logical_and(wav <= wav2, wav > wav0) # FUV

            L_tot.append(simps(10.0**(df_.logf - self.logM), df_.wav)/Lsun_cgs)
            L_LyC.append(simps(10.0**(df_.logf[idx0] - self.logM), df_.wav[idx0])/Lsun_cgs)
            L_LW.append(simps(10.0**(df_.logf[idx1] - self.logM), df_.wav[idx1])/Lsun_cgs)
            L_PE.append(simps(10.0**(df_.logf[idx2] - self.logM), df_.wav[idx2])/Lsun_cgs)
            L_OPT.append(simps(10.0**(df_.logf[idx3] - self.logM), df_.wav[idx3])/Lsun_cgs)
            L_IR.append(simps(10.0**(df_.logf[idx4] - self.logM), df_.wav[idx4])/Lsun_cgs)
            L_FUV.append(simps(10.0**(df_.logf[idx5] - self.logM), df_.wav[idx5])/Lsun_cgs)

            # wavelength in micron
            l = wav*1e-4
            J = 10.0**df_['logf']
            f_J = interp1d(np.log10(l), np.log10(J))

            if i == 0:
                w = dict()
                w['LyC'] = np.logspace(np.log10(l.min()), np.log10(wav0*1e-4), 1000)
                w['LW'] = np.logspace(np.log10(wav0*1e-4), np.log10(wav1*1e-4), 1000)
                w['PE'] = np.logspace(np.log10(wav1*1e-4), np.log10(wav2*1e-4), 1000)
                w['OPT'] = np.logspace(np.log10(wav2*1e-4), np.log10(wav3*1e-4), 1000)
                w['IR'] = np.logspace(np.log10(wav3*1e-4), np.log10(wav4*1e-4), 1000)
                w['FUV'] = np.logspace(np.log10(wav0*1e-4), np.log10(wav2*1e-4), 1000)

                Cext = dict()
                Cabs = dict()
                Crpr = dict()
                hnu = dict()
                for k in w.keys():
                    Cext[k] = []
                    Cabs[k] = []
                    Crpr[k] = []
                    hnu[k] = []

            for k in w.keys():
                Cext[k].append(simps(10.0**f_Cext(np.log10(w[k]))*10.0**f_J(np.log10(w[k]))*w[k], w[k])/ \
                               simps(10.0**f_J(np.log10(w[k]))*w[k], w[k]))
                Cabs[k].append(simps(10.0**f_Cabs(np.log10(w[k]))*10.0**f_J(np.log10(w[k]))*w[k], w[k])/ \
                               simps(10.0**f_J(np.log10(w[k]))*w[k], w[k]))
                Crpr[k].append(simps(10.0**f_Crpr(np.log10(w[k]))*10.0**f_J(np.log10(w[k]))*w[k], w[k])/ \
                               simps(10.0**f_J(np.log10(w[k]))*w[k], w[k]))
                hnu[k].append(simps(10.0**f_J(np.log10(w[k])), w[k])/ \
                               simps(10.0**f_J(np.log10(w[k]))*w[k], w[k])*hc_cgs/(1.0*au.eV).cgs.value*1e4)

        for k in w.keys():
            Cext[k] = np.array(Cext[k])
            Cabs[k] = np.array(Cabs[k])
            Crpr[k] = np.array(Crpr[k])
            hnu[k] = np.array(hnu[k])

        if i == 0:
            w = dict()
            w['LyC'] = np.logspace(np.log10(l.min()), np.log10(wav0*1e-4), 1000)
            w['LW'] = np.logspace(np.log10(wav0*1e-4), np.log10(wav1*1e-4), 1000)
            w['PE'] = np.logspace(np.log10(wav1*1e-4), np.log10(wav2*1e-4), 1000)
            w['OPT'] = np.logspace(np.log10(wav2*1e-4), np.log10(wav3*1e-4), 1000)
            w['IR'] = np.logspace(np.log10(wav3*1e-4), np.log10(wav4*1e-4), 1000)
            w['FUV'] = np.logspace(np.log10(wav0*1e-4), np.log10(wav2*1e-4), 1000)

        L_tot = np.array(L_tot)
        L_LyC = np.array(L_LyC)
        L_LW = np.array(L_LW)
        L_PE = np.array(L_PE)
        L_OPT = np.array(L_OPT)
        L_IR = np.array(L_IR)
        L_FUV = np.array(L_FUV)
        time_Myr = time*1e-6

        L = dict()
        L['tot'] = np.array(L_tot)
        L['LyC'] = np.array(L_LyC)
        L['LW'] = np.array(L_LW)
        L['PE'] = np.array(L_PE)
        L['OPT'] = np.array(L_OPT)
        L['IR'] = np.array(L_IR)
        L['FUV'] = np.array(L_FUV)
        L['UV'] = L['LyC'] + L['FUV']

        # Momentum injection rate (Msun km/s / Myr / Msun)
        pdot = dict()
        for v in ('tot', 'LyC', 'LW', 'PE', 'OPT', 'IR', 'UV', 'FUV'):
            pdot[v] = (((L[v]*au.L_sun/ac.c).to('g cm s-2')).to('Msun km s-1 Myr-1')).value

        # Luminosity-weighted effective timescale
        # (e-folding timescale if L is decaying exponentially)
        tdecay_lum = dict()
        tcumul_lum_50 = dict()
        tcumul_lum_90 = dict()
        for k in L.keys():
            tdecay_lum[k] = trapz(L[k]*time_Myr, time_Myr)/trapz(L[k], time_Myr)

            idx50 = L[k].cumsum()/L[k].cumsum()[-1] > 0.5
            tcumul_lum_50[k] = time_Myr[idx50][0]

            idx90 = L[k].cumsum()/L[k].cumsum()[-1] > 0.9
            tcumul_lum_90[k] = time_Myr[idx90][0]

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
                 tcumul_lum_50=tcumul_lum_50, tcumul_lum_90=tcumul_lum_90,
                 wav0=wav0, wav1=wav1, wav2=wav2, wav3=wav3,
                 Cabs=Cabs, Cext=Cext, Crpr=Crpr, hnu=hnu,
                 hnu_LyC=hnu_LyC, dhnu_H_LyC=dhnu_H_LyC, dhnu_H2_LyC=dhnu_H2_LyC,
                 sigma_pi_H=sigma_pi_H, sigma_pi_H2=sigma_pi_H2)

        # Photon luminosity
        r['Q'] = dict()
        Q_conv = (1.0*ac.L_sun/au.eV).cgs.value
        for k in r['hnu'].keys():
            r['Q'][k] = r['L'][k]/r['hnu'][k]*Q_conv

        # Compute time-averaged quantities: q_avg = \int_0^t q * weight dt / \int_0^t weight dt, where weight=1, L, Q
        for kk in ['L','Q']:
            r[kk+'_avg'] = dict()
            for k in r[kk].keys():
                r[kk+'_avg'][k] = cumulative_trapezoid(r[kk][k], x=r['time_Myr'], initial=0.0)/\
                                cumulative_trapezoid(np.repeat(1.0,len(r['time_Myr'])), x=r['time_Myr'], initial=0.0)
                r[kk+'_avg'][k][0] = r[kk+'_avg'][k][1]

        for kk in ['hnu','Cabs','Cext','Crpr']:
            r[kk+'_Lavg'] = dict()
            for k in r[kk].keys():
                r[kk+'_Lavg'][k] = cumulative_trapezoid(r[kk][k]*r['L'][k], x=r['time_Myr'], initial=0.0)/\
                                cumulative_trapezoid(r['L'][k], x=r['time_Myr'], initial=0.0)
                r[kk+'_Lavg'][k][0] = r[kk+'_Lavg'][k][1]

        for kk in ['hnu','Cabs','Cext','Crpr']:
            r[kk+'_Qavg'] = dict()
            for k in r[kk].keys():
                r[kk+'_Qavg'][k] = cumulative_trapezoid(r[kk][k]*r['Q'][k], x=r['time_Myr'], initial=0.0)/\
                                cumulative_trapezoid(r['Q'][k], x=r['time_Myr'], initial=0.0)
                r[kk+'_Qavg'][k][0] = r[kk+'_Qavg'][k][1]

        # Q-weighted average for quantities related to ionizing radiation
        for k in ['dhnu_H_LyC','dhnu_H2_LyC','sigma_pi_H','sigma_pi_H2']:
            r[k+'_Qavg'] = cumulative_trapezoid(r[k]*r['Q']['LyC'], x=r['time_Myr'], initial=0.0)/\
                cumulative_trapezoid(r['Q']['LyC'], x=r['time_Myr'], initial=0.0)
            r[k+'_Qavg'][0] = r[k+'_Qavg'][1]

        return r

    def read_quanta(self):
        """Function to read ionizing photon rate
        """
        names = ['time_yr', 'Q_HI', 'Lfrac_HI', 'Q_HeI', 'Lfrac_HeI',
                 'Q_HeII', 'Lfrac_HeII', 'logL']
        df = pd.read_csv(self.files['quanta'], names=names, skiprows=7, delimiter='\s+')
        df['time_Myr'] = df['time_yr']*1e-6

        # Normalize by cluster mass
        for c in df.columns:
            if c.startswith('time') or c.startswith('Lfrac'):
                continue
            df[c] = 10.0**(df[c] - self.logM)

        # for c in names[1:]:
        #     df[c] = 10.0**df[c]

        return df

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
        plt.ylabel(r'$\kappa_{\rm d}(\lambda)\;[{\rm cm}^2\,{\rm g}^{-1}]$')
        plt.legend()

        def kappa2sigma(x):
            return x*muH

        def sigma2kappa(x):
            return x/muH

        sax1 = plt.gca().secondary_yaxis('right', functions=(kappa2sigma,sigma2kappa))
        sax1.set_ylabel(r'$\sigma_{\rm d}(\lambda)\;[{\rm cm}^2\,{\rm H}^{-1}]$')
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
    def plt_lum_evol(ax, rr, rw, rs, lw=2, plt_sn=False):

        #pa.set_plt_fancy()
        plt.sca(ax)
        plt.plot(rr['time_Myr'], rr['L']['tot'], label=r'Bolometric', c='k', lw=lw)
        plt.plot(rr['time_Myr'], rr['L']['UV'], label=r'${\rm LyC+FUV}\;(<2068\,{\rm \AA})$', c='k', ls='--', lw=lw)
        plt.plot(rr['time_Myr'], rr['L']['LyC'], label=r'${\rm LyC}\;(<912\,{\rm \AA})$', c='C0', lw=lw)
        plt.plot(rr['time_Myr'], rr['L']['LW'], label=r'${\rm LW}\;(912$-$1108\,{\rm \AA})$', c='C1', lw=lw)
        plt.plot(rr['time_Myr'], rr['L']['PE'], label=r'${\rm PE}\;(1108$-$2068\,{\rm \AA})$', c='C2', lw=lw)
        plt.plot(rr['time_Myr'], rr['L']['OPT'], label=r'${\rm OPT}\;(2068$-$10000\,{\rm \AA})$', c='C3', lw=lw)
        plt.plot(rw['time_Myr'], rw['Edot_all']/(1.0*au.L_sun).cgs.value, c='C7',
                 label=r'$L_{\rm w}/M_{\ast}$', lw=lw)
        if plt_sn:
            plt.plot(rs['time_Myr'], rs['Edot_SN']/(1.0*au.L_sun).cgs.value, c='C8',
                     label=r'$L_{\rm sn}/M_{\ast}$', lw=lw)

        plt.yscale('log')
        plt.xlim(0, 20)
        plt.ylim(1e-1,2e3)
        plt.xlabel(r'$t_{\rm age}\;[{\rm Myr}]$')
        #plt.ylabel(r'$\Psi\,{\rm and}\,\Psi_w \;[L_{\odot}\,M_{\odot}^{-1}]$')
        plt.ylabel(r'$L/M_{\ast} \;[L_{\odot}\,M_{\odot}^{-1}]$')
        #plt.legend(fontsize='small', loc=4)

        return ax

    @staticmethod
    def plt_pdot_evol(ax, rr, rw, rs, lw=2):

        plt.sca(ax)
        plt.plot(rr['time_Myr'], (rr['L']['tot']*au.L_sun/ac.c/au.M_sun).to('km s-1 Myr-1'),
                 label=r'Bolometric', c='k', lw=lw)
        # plt.plot(rr['time_Myr'], (rr['L']['LyC']*au.L_sun/ac.c/au.M_sun).to('km s-1 Myr-1'),
        #          label=r'${\rm LyC}\;(<912\,{\rm \AA})$', c='C0', ls='-')
        plt.plot(rr['time_Myr'], (rr['L']['UV']*au.L_sun/ac.c/au.M_sun).to('km s-1 Myr-1'),
                 label=r'${\rm LyC+FUV}\;(<2068\,{\rm \AA})$', c='k', ls='--', lw=lw)
        plt.plot(rw['time_Myr'], rw['pdot_all'].values,
                 label=r'$\dot{p}_{\rm wind}/M_{\ast}$', c='C7', lw=lw)
        plt.xlim(0,20)
        plt.ylim(1e-1,5e1)
        plt.yscale('log')

        plt.xlabel(r'$t_{\rm age}\;[{\rm Myr}]$')
        plt.ylabel(r'$\dot{p}/M_{\ast} \;[{\rm km}\,{\rm s}^{-1}\,{\rm Myr}^{-1}]$')
        #plt.legend()

        return ax

    @staticmethod
    def plt_lum_cumul(ax, rr, rw, rs, normed=True, plt_sn=False, lw=2):

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
                 label='LyC', c='C0', lw=lw)
        plt.plot(rr['time_Myr'], integrate_L_cum(rr['L']['LW'], rr['time_yr'])/norm,
                 label='LW', c='C1', lw=lw)
        plt.plot(rr['time_Myr'], integrate_L_cum(rr['L']['PE'], rr['time_yr'])/norm,
                 label='PE', c='C2', lw=lw)
        plt.plot(rr['time_Myr'], integrate_L_cum(rr['L']['OPT'], rr['time_yr'])/norm,
                 label='OPT', c='C3', lw=lw)
        plt.plot(rr['time_Myr'], integrate_L_cum(rr['L']['UV'], rr['time_yr'])/norm,
                 label=r'${\rm LyC+FUV}\;(<2068\,{\rm \AA})$', c='k', ls='--', lw=lw)
        plt.plot(rr['time_Myr'], integrate_L_cum(rr['L']['tot'], rr['time_yr'])/norm,
                 label='Bolometric', c='k', lw=lw)
        plt.plot(rw['time_Myr'], rw['Einj_all'], c='C7', label=r'$L_{\rm w}$', lw=lw)

        if plt_sn:
            plt.plot(rs['time_Myr'], rs['Einj_SN'], c='C8', label=r'$L_{\rm sn}$', lw=lw)

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
    def plt_pdot_cumul(ax, rr, rw, rs, normed=False, plt_sn=False, lw=2):

        integrate_pdot = lambda pdot, t: cumulative_trapezoid(
            pdot, t*au.Myr, initial=0.0)

        pdot_tot_cum = integrate_pdot(rr['L']['tot'], rr['time_Myr'])

        if normed:
            norm = pdot_tot_cum
        else:
            norm = 1.0

        plt.sca(ax)
        # Skip LyC, PE, and LW
        # plt.plot(rr['time_Myr'], integrate_pdot(rr['pdot']['LyC'], rr['time_Myr'])/norm,
        #          label='LyC', c='C0')
        # plt.plot(rr['time_Myr'], integrate_pdot(rr['pdot']['LW'], rr['time_Myr'])/norm,
        #          label='LW', c='C1')
        # plt.plot(rr['time_Myr'], integrate_pdot(rr['pdot']['PE'], rr['time_Myr'])/norm,
        #          label='PE', c='C2')
        plt.plot(rr['time_Myr'], integrate_pdot(rr['pdot']['UV'], rr['time_Myr'])/norm,
                 label=r'${\rm LyC+FUV}\;(<2068\,{\rm \AA})$', c='k', ls='--', lw=lw)
        plt.plot(rr['time_Myr'], integrate_pdot(rr['pdot']['tot'], rr['time_Myr'])/norm,
                 label='Bolometric', c='k', lw=lw)

        # from cgs to astro units
        pdot_conv = 1.0 #(1.0*au.g*au.cm/au.s**2).to('Msun km s-1 Myr-1')
        plt.plot(rw['time_Myr'], integrate_pdot(rw['pdot_all']*pdot_conv,
                                                rw['time_Myr'])/norm,
                 c='C7', label=r'$L_{\rm w}$', lw=lw)

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
    def plt_Edot_pdot_evol_cumul(rr, rw, rs, plt_sn=True, lw=2, normed=False, fig=None, axes=None):
        if fig is None:
            fig, axes = plt.subplots(2,2,figsize=(12, 10), constrained_layout=True,
                                     gridspec_kw=dict(height_ratios=[0.5,0.5]))
            axes = axes.flatten()

        SB99.plt_lum_evol(axes[0], rr, rw, rs, plt_sn=plt_sn, lw=lw)
        SB99.plt_pdot_evol(axes[1], rr, rw, rs, lw=lw)
        SB99.plt_lum_cumul(axes[2], rr, rw, rs, normed=normed, plt_sn=plt_sn, lw=lw)
        SB99.plt_pdot_cumul(axes[3], rr, rw, rs, normed=normed, plt_sn=plt_sn, lw=lw)

        for ax in axes:
            ax.grid()
            #ax.set_xlim(0,50)
            #ax.set_xscale('linear')

        plt.legend([mpl.lines.Line2D([0],[0],c='k'),
                    mpl.lines.Line2D([0],[0],c='k',ls='--'),
                    mpl.lines.Line2D([0],[0],c='C0'),
                    mpl.lines.Line2D([0],[0],c='C1'),
                    mpl.lines.Line2D([0],[0],c='C2'),
                    mpl.lines.Line2D([0],[0],c='C3'),
                    mpl.lines.Line2D([0],[0],c='C7'),
                    mpl.lines.Line2D([0],[0],c='C8')],
                   [r'Bolometric',
                    r'${\rm LyC+FUV}\;(<2068\,{\rm \AA})$',
                    r'${\rm LyC}\;(<912\,{\rm \AA})$',
                    r'${\rm LW}\;(912$-$1108\,{\rm \AA})$',
                    r'${\rm PE}\;(1108$-$2068\,{\rm \AA})$',
                    r'${\rm OPT}\;(2068$-$10000\,{\rm \AA})$',
                    # r'$L_{\rm w}/M_{\ast}$',
                    # r'$L_{\rm sn}/M_{\ast}$'
                    r'Stellar winds',
                    r'Supernovae',
                    ], loc=4, fontsize='small')

        return fig, axes


def plt_nuJnu_mid_plane_parallel(ax,
                                 Sigma_gas=10.0*au.M_sun/au.pc**2, plt_dr78=True):

    sb = SB99('/projects/EOSTRIKE/SB99/Z014_SFR1_GenevaV00_logdt')
    rr = sb.read_rad()
    w = rr['wav'].values*1e-4
    d = DustDraine()
    dfdr = d.dfa['Rv31']
    f_Cext = interp1d(np.log10(dfdr['lwav']), np.log10(dfdr['Cext']),
                      bounds_error=False)
    f_Cabs = interp1d(np.log10(dfdr['lwav']), np.log10(dfdr['Cext']*(1.0 - dfdr['albedo'])),
                      bounds_error=False)

    print('max time', rr['time_Myr'][-1])
    Sigma_SFR = 2.5e-3
    Llambda_over_SFR = 10.0**rr['logf'][-1,:]*au.erg/au.s/au.angstrom
    Llambda = Sigma_SFR*10.0**rr['logf'][-1,:]*au.erg/au.s/au.angstrom
    area = (1.0*au.kpc)**2
    muH = 1.4*au.u
    kappa_dust_ext = (10.0**f_Cext(np.log10(w))*au.cm**2/au.g).cgs
    kappa_dust_abs = (10.0**f_Cabs(np.log10(w))*au.cm**2/au.g).cgs
    tau_perp = (Sigma_gas*kappa_dust_abs).to('').value

    from scipy.special import expn
    # Intensity at the midplane (see Ostriker et al. 2010)
    Jlambda = (Llambda/area/(4.0*np.pi*au.sr*tau_perp)*
               (1.0 - expn(2, 0.5*tau_perp))).to('erg s-1 cm-2 angstrom-1 sr-1')
    # Naive estimation without attenuation
    Jlambda0 = (Llambda/area/4.0/au.sr).to('erg s-1 cm-2 angstrom-1 sr-1')

    ww = rr['wav'].values
    idx = (ww > 912.0) & (ww < 2068.0)
    from scipy import integrate

    print(tau_perp)
    print('FUV emissivity per area [Lsun/pc^2]',
          integrate.trapz(Sigma_SFR*Llambda_over_SFR[idx],ww[idx]*au.angstrom).to('Lsun')/1e6)
    print('L_FUV_over_SFR/1e7',(integrate.trapz(Llambda_over_SFR[idx],ww[idx])*au.angstrom).to('Lsun')/1e7)
    print('J_FUV',integrate.trapz(Jlambda[idx],ww[idx]))
    print('J_FUV_unatt',integrate.trapz(Jlambda0[idx],ww[idx]))

    plt.sca(ax)
    # Show FUV only
    l, = plt.loglog(rr['wav'][idx], #rr['wav']*
                    Jlambda[idx], label=r'SB99 + Ostriker et al. (2010)')
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

    return rr

def print_lum_weighted_avg_quantities(rr, tmax=50.0):
    """Function to print some useful numbers for radiation
    """
    L_tot = rr['L']['tot']
    L_LyC = rr['L']['LyC']
    L_PE = rr['L']['PE']
    L_LW = rr['L']['LW']
    L_OPT = rr['L']['OPT']
    time = rr['time_yr']
    print('Luminosity-weighted timescale \int t*L dt/\int L dt:',rr['tdecay_lum'])
    print('Bolometric at t=0:', L_tot[0], L_tot.max(), time[L_tot==L_tot.max()][0]/1e6)
    print('Bolometric at maximum:', L_tot.max())
    print('Time at maximum of Bolometric:', time[L_tot==L_tot.max()][0]/1e6)

    print('Lyman Continuum')
    idx = L_LyC.cumsum()/L_LyC.cumsum()[-1] > 0.5
    print('- 50% of LyC photons are emitted in the first', time[idx][0]/1e6,'Myr')
    idx = L_LyC.cumsum()/L_LyC.cumsum()[-1] > 0.90
    print('- 90% of LyC photons are emitted in the first', time[idx][0]/1e6,'Myr')
    idx = L_LyC.cumsum()/L_LyC.cumsum()[-1] > 0.95
    print('- 95% of LyC photons are emitted in the first', time[idx][0]/1e6,'Myr')
    idx = L_LyC/L_LyC[0] < 0.5
    print('- 50% of the initial value at',time[idx][0]/1e6, 'Myr')

    print('FUV (LW+PE)')
    L_FUV = L_PE + L_LW
    idx = (L_FUV).cumsum()/L_FUV.cumsum()[-1] > 0.5
    print('- 50% of FUV photons are emitted in the first', time[idx][0]/1e6,'Myr')
    idx = (L_FUV).cumsum()/L_FUV.cumsum()[-1] > 0.9
    print('- 90% of FUV photons are emitted in the first', time[idx][0]/1e6,'Myr')
    idx = (L_FUV).cumsum()/L_FUV.cumsum()[-1] > 0.95
    print('- 95% of FUV photons are emitted in the first', time[idx][0]/1e6,'Myr')
    idx = L_FUV/L_FUV[0] < 0.5
    print('- 50% of the initial value at',time[idx][0]/1e6, 'Myr')

    idx = rr['time_Myr'] < tmax
    for k in ['LyC','LW','PE','OPT']:
        print(k, ':')
        print('Cabs Cext Crpr hnu : {0:5.2e}, {1:5.2e}, {2:5.2e} {3:5.2e}'.format(
            np.average(rr['Cabs'][k][idx], weights=rr['L'][k][idx]),
            np.average(rr['Cext'][k][idx], weights=rr['L'][k][idx]),
            np.average(rr['Crpr'][k][idx], weights=rr['L'][k][idx]),
            np.average(rr['hnu'][k][idx], weights=rr['L'][k][idx])
        ))
        if k == 'LyC':
            print('sigma_pi_H dhnu_H {0:5.2e}, {1:5.2e}'.format(
                np.average(rr['sigma_pi_H'][idx], weights=rr['L'][k][idx]),
                np.average(rr['dhnu_H_LyC'][idx], weights=rr['L'][k][idx])
                  ))
            print('sigma_pi_H2 dhnu_H2 {0:5.2e}, {1:5.2e}'.format(
                np.average(rr['sigma_pi_H2'][idx], weights=rr['L'][k][idx]),
                np.average(rr['dhnu_H2_LyC'][idx], weights=rr['L'][k][idx])
                  ))


def plt_cross_sections_hnu(rr):

    #rr = sb.read_rad()

    fig, axes = plt.subplots(3, 1, figsize=(6, 12),
                             sharex=True, gridspec_kw=dict(hspace=0.1))
    axes = axes.flatten()

    x = rr['time_Myr']

    # Dust extinction/absorption cross section
    plt.sca(axes[0])
    for k in ['LyC','LW','PE','OPT']:
        l1, = plt.plot(x, rr['Cext'][k], label=k)
        l2, = plt.plot(x, rr['Cabs'][k], c=l1.get_color(), ls='--')
    plt.ylabel(r'$\sigma_{\rm d}\;[{\rm cm}^{2}\,{\rm H}^{-1}]$')
    plt.ylim(0,3e-21)
    plt.grid(which='both')
    leg = plt.legend(loc=0)
    plt.gca().add_artist(leg)
    plt.legend([mpl.lines.Line2D([0],[0],c='k',ls='-'),
                mpl.lines.Line2D([0],[0],c='k',ls='--')], ['ext','abs'], loc=2)

    # Mean energy of photons
    plt.sca(axes[2])
    for k in ['LyC','LW','PE','OPT']:
        l1, = plt.plot(x, rr['hnu'][k])
        if k == 'LyC':
            plt.plot(x, rr['dhnu_H_LyC'], c=l1.get_color(), ls=':')
            plt.plot(x, rr['dhnu_H2_LyC'], c=l1.get_color(), ls='-.')

    plt.ylabel(r'photon energy $\;[{\rm eV}]$')
    plt.ylim(1,30)
    plt.yscale('log')
    plt.grid(which='both')
    # Mean energy of photoejcted electrons
    # plt.sca(axes[2])
    # plt.ylabel(r'$q_{\rm pi} \;[{\rm eV}]$')

    # Mean photoionization cross section
    plt.sca(axes[1])
    l1, = plt.plot(x, rr['sigma_pi_H'], ls=':', label=r'${\rm H}$')
    plt.plot(x, rr['sigma_pi_H2'], ls='-.', c=l1.get_color(), label=r'${\rm H}_2$')
    plt.ylabel(r'$\sigma_{\rm pi} \;[{\rm cm}^2]$')
    plt.xlabel(r'${\rm age}\;[{\rm Myr}]$')
    plt.xlim(0,20)
    plt.grid(which='both')
    plt.legend()

    return fig


def print_tbl_data(rr):
    bands = ['LyC','LW','PE','FUV','OPT']
    bands2 = ['LyC','LW','PE','FUV','OPT','tot']

    tbl_data = []
    tbl_data.append(r'\multicolumn{7}{c}{Timescales (Myr)} \\')
    tbl_data.append(r'\tableline')
    # t_decay
    tbl_data.append(r'(1) $t_{\rm decay}$ & ' +
                    r' & '.join([r'{0:.1f}'.format(rr['tdecay_lum'][b]) for b in bands2]) + r' \\')
    # t_cumul_50
    tbl_data.append(r'(2) $t_{\rm cumul,50\%}$ & ' +
                    r' & '.join([r'{0:.1f}'.format(rr['tcumul_lum_50'][b]) for b in bands2]) + r' \\')
    # t_cumul_90
    tbl_data.append(r'(3) $t_{\rm cumul,90\%}$ & ' +
                    r' & '.join([r'{0:.1f}'.format(rr['tcumul_lum_90'][b]) for b in bands2]) + r' \\')


    idx = np.where(rr['time_Myr'] > 20.0)[0][0]
    # idx = -1

    # Dust cross sections
    tbl_data.append(r'\tableline')
    # tbl_data.append(r'\multicolumn{7}{c}{Cross sections ($\sigma_{\rm d}/10^{-21}\cm^{2}$)} \\')
    tbl_data.append(r'\multicolumn{7}{c}{Cross sections ($\sigma_{\rm d}/10^{-21}\cm^{2}$, $\sigma_{\rm pi}/10^{-18}\cm^{2}$)} \\')
    tbl_data.append(r'\tableline')
    tbl_data.append(r'(4) $\langle \sigma_{\rm d,abs} \rangle$ & ' +
                    r' & '.join([r'{0:.2f}'.format(rr['Cabs_Lavg'][b][idx]/1e-21) for b in bands]) + r' & - \\')
    tbl_data.append(r'(5) $\langle \sigma_{\rm d,ext} \rangle$ & ' +
                    r' & '.join([r'{0:.2f}'.format(rr['Cext_Lavg'][b][idx]/1e-21) for b in bands]) + r' & - \\')
    tbl_data.append(r'(6) $\langle \sigma_{\rm d,pr} \rangle$ & ' +
                    r' & '.join([r'{0:.2f}'.format(rr['Crpr_Lavg'][b][idx]/1e-21) for b in bands]) + r' & - \\')
    # Photoionization cross sections
    tbl_data.append(r'(7) $\langle \sigma_{\rm pi,H} \rangle$ & ' +
                    r'{0:.1f}'.format(rr['sigma_pi_H_Qavg'][idx]/1e-18) + r' & - & - & - & - & - \\')
    tbl_data.append(r'(8) $\langle \sigma_{\rm pi,H_2} \rangle$ & ' +
                    r'{0:.1f}'.format(rr['sigma_pi_H2_Qavg'][idx]/1e-18) + r' & - & - & - & - & - \\')
    # Photon energies
    tbl_data.append(r'\tableline')
    tbl_data.append(r'\multicolumn{7}{c}{Photon Energy (${\rm eV}$)} \\')
    tbl_data.append(r'\tableline')
    tbl_data.append(r'(9) $\langle h\nu \rangle$ & ' +
                    r' & '.join([r'{0:.1f}'.format(rr['hnu_Qavg'][b][idx]) for b in bands]) + r' \\')
    # Mean energy of photoionization of H
    tbl_data.append(r'(10) $\langle q_{\rm pi,H} \rangle$ & ' +
                    r'{0:.1f}'.format(rr['dhnu_H_LyC_Qavg'][idx]) + r' & - & - & - & - & - \\')
    # Mean energy of photoionization of H2
    tbl_data.append(r'(11) $\langle q_{\rm pi,H_2} \rangle$ & ' +
                    r'{0:.1f}'.format(rr['dhnu_H2_LyC_Qavg'][idx]) + r' & - & - & - & - & - \\')

    for td in tbl_data:
        print(td)


from pyathena.microphysics.dust_draine import DustDraine
from pyathena.util.sb99 import SB99
from scipy.interpolate import interp1d
from scipy.special import expn
from scipy import integrate


def get_ISRF_SB99_plane_parallel(Sigma_gas=10.0*au.M_sun/au.pc**2,
                                 Sigma_SFR=2.5e-3*au.M_sun/au.kpc**2/au.yr,
                                 age_Myr=0.99e3,
                                 Z_dust=1.0, dust_kind='Rv31', Z_star=0.014, verbose=True):

    Z_gas = Z_dust
    Z_star_str = '{0:03d}'.format(int(Z_star*1000))
    model = '/projects/EOSTRIKE/SB99/Z{0:s}_SFR1_GenevaV00_logdt_10Gyr'.format(Z_star_str)

    sb = SB99(model, verbose=verbose)
    rr = sb.read_rad()

    if sb.cont_SF:
        SFR = sb.SFR*au.M_sun/au.yr
    else:
        print('SB99 does not assume continuous SF!')
        raise

    d = DustDraine()
    if dust_kind in ['Rv31', 'Rv31', 'Rv55', 'LMCavg', 'SMCbar']:
        dfdr = d.dfa[dust_kind]
    else:
        print('dust_kind {0:s} not supported'.format(dust_kind))
        raise

    # Cross sections
    f_Cabs = interp1d(np.log10(dfdr['lwav']),
                      np.log10(Z_dust*dfdr['K_abs']/d.GTD['Rv31']),
                      bounds_error=False)

    # wavelength in micron
    w_micron = rr['wav'].values*1e-4
    w_angstrom = rr['wav'].values

    # Luminosity per SFR and area at maximum time
    #idx = -1
    idx = np.where(rr['time_Myr'] > age_Myr)[0][0]
    Llambda_per_SFR = 10.0**rr['logf'][idx, :]*au.erg/au.s/au.angstrom/SFR
    Llambda_per_area = Sigma_SFR*Llambda_per_SFR

    muH = (1.4 - 0.02*Z_gas)*au.u

    #kappa_dust_ext = Z_dust*(10.0**f_Cext(np.log10(w_micron))*au.cm**2/au.u).cgs
    kappa_dust_abs = Z_dust*(10.0**f_Cabs(np.log10(w_micron))*au.cm**2/au.g).cgs

    # Wavelength-dependent perpendicular dust optical depth
    tau_perp = (Sigma_gas*kappa_dust_abs).to('').value

    # Naive estimation without attenuation
    Jlambda_unatt = (Llambda_per_area/(4.0*np.pi*au.sr)).to('erg s-1 cm-2 angstrom-1 sr-1')
    # Intensity at the midplane (see Ostriker et al. 2010)
    Jlambda = Jlambda_unatt/tau_perp*(1.0 - expn(2, 0.5*tau_perp))

    ###################################
    # Wavelength integrated quantities
    ###################################
    w_bdry = np.array([0,912,2068,10000])
    band = np.array(['LyC','FUV','OPT'])
    nband = len(band)

    J_unatt = dict()
    J = dict()
    L_per_area = dict()
    L_per_SFR = dict()
    for i in range(nband):
        b = band[i]
        idx = (w_angstrom > w_bdry[i]) & (w_angstrom <= w_bdry[i+1])
        # Naive estimate of mean intensity (L/area/4pi)
        J_unatt[b] = integrate.trapz(Jlambda_unatt[idx],
                                     w_angstrom[idx]*au.angstrom)
        # Mean intensity with dust attenuation
        J[b] = integrate.trapz(Jlambda[idx],
                               w_angstrom[idx]*au.angstrom)
        # Luminosity per unit area
        L_per_area[b] = (integrate.trapz(Llambda_per_area[idx],
                                         w_angstrom[idx]*au.angstrom)).to('Lsun kpc-2')
        L_per_SFR[b] = (integrate.trapz(Llambda_per_SFR[idx],
                                        w_angstrom[idx])*au.angstrom).to('Lsun Msun-1 yr')


    r = dict()
    r['sb'] = sb
    r['sb_rad'] = rr
    r['Z_star'] = Z_star
    r['Z_star_str'] = Z_star_str
    r['Z_dust'] = Z_dust
    r['SFR'] = SFR
    r['Sigma_gas'] = Sigma_gas
    r['Sigma_SFR'] = Sigma_SFR
    r['w_micron'] = w_micron
    r['w_angstrom'] = w_angstrom
    r['tau_perp'] = tau_perp
    r['Jlambda_unatt'] = Jlambda_unatt
    r['Jlambda'] = Jlambda
    r['J_unatt'] = J_unatt
    r['J'] = J
    r['L_per_area'] = L_per_area
    r['L_per_SFR'] = L_per_SFR

    r['Jlambda'] = (Llambda_per_area/(4.0*np.pi*au.sr*tau_perp)*
                    (1.0 - expn(2, 0.5*tau_perp))).to('erg s-1 cm-2 angstrom-1 sr-1')

    idx = (w_angstrom > 912.0) & (w_angstrom < 2068.0)
    # Mean FUV intensity (naive estimate)
    r['J_FUV_unatt'] = integrate.trapz(Jlambda_unatt[idx],
                                  w_angstrom[idx]*au.angstrom)
    # Mean FUV intensity
    r['J_FUV'] = integrate.trapz(Jlambda[idx],
                            w_angstrom[idx]*au.angstrom)
    # FUV luminosity per unit area
    r['Sigma_FUV'] = (integrate.trapz(Llambda_per_area[idx],
                                      w_angstrom[idx]*au.angstrom)).to('Lsun kpc-2')
    r['L_FUV_per_SFR'] = (integrate.trapz(Llambda_per_SFR[idx],
                           w_angstrom[idx])*au.angstrom).to('Lsun Msun-1 yr')


    # wavelength in Angstrom

#     print('Z_dust',Z_dust)
#     #print('correction factor:',1/tau_perp*(1.0 - expn(2, 0.5*tau_perp)))
#     print('Llambda_over_SFR',Llambda_over_SFR)

#     r = dict()
#     # r['Z_star'] = Z_star
#     r['Z_star'] = Z_star
#     r['Z_dust'] = Z_dust
#     r['SFR'] = SFR
#     r['Sigma_gas'] = Sigma_gas
#     r['Sigma_SFR'] = Sigma_SFR
#     r['w_angstrom'] = w_angstrom
#     r['Jlambda_unatt'] = Jlambda_unatt
#     r['Jlambda'] = Jlambda
#     r['Sigma_FUV'] = Sigma_FUV
#     r['J_FUV_unatt'] = J_FUV_unatt
#     r['J_FUV'] = J_FUV
#     r['tau_perp'] = tau_perp
#     r['L_FUV_per_SFR'] = L_FUV_per_SFR

#     r['sb'] = sb
#     r['rr'] = rr

    if verbose:
        print('Z_star, Z_dust', Z_star, Z_dust)
        print('Sigma_FUV : {:g}'.format(r['Sigma_FUV']))
        print('L_FUV_per_SFR : {:g}'.format(r['L_FUV_per_SFR']))
        print('J_FUV_unatt: {:g}'.format(r['J_FUV_unatt']))
        print('J_FUV : {:g}'.format(r['J_FUV']))
        print('Overall correction factor : {:g}'.format(r['J_FUV']/r['J_FUV_unatt']))

    return r


def write_sb99_output_for_tigress(sb):

    rr = sb.read_rad()
    ry = sb.read_yield()
    rw = sb.read_wind()
    rs = sb.read_sn()

    from scipy.interpolate import interp1d

    # Set age array (MUST BE THE SAME AS THE AGE ARRAYS IN Athena-TIGRESS)
    dage_sn = 0.2
    age_sn = np.arange(0,40.2,dage_sn)

    dage_rad = 0.2
    age_rad = np.arange(0,40.2,dage_rad)

    # Separate age array for wind output to keep consistency with old TIGRESS version
    dage_wind = 0.1
    age_wind = np.arange(0,50.0,dage_wind) + 0.01
    age_wind = np.insert(age_wind, 0, 0.0)

    f_Psi_LW = interp1d(rr['time_Myr'], rr['L']['LW'],
                        bounds_error=False, fill_value='extrapolate')
    f_Psi_PE = interp1d(rr['time_Myr'], rr['L']['PE'],
                        bounds_error=False, fill_value='extrapolate')
    f_Xi_LyC = interp1d(rr['time_Myr'], rr['Q']['LyC'],
                        bounds_error=False, fill_value='extrapolate')

    # Set small numbers to zero
    rs['SN_rate'].loc[(rs['SN_rate'] < 1e-35)] = 0.0
    f_SN_rate = interp1d(rs['time_Myr'], rs['SN_rate']*1e6,
                         bounds_error=False, fill_value='extrapolate')


    f_log_Edot_wind = interp1d(rw['time_Myr'], np.log10(rw['Edot_all']),
                               bounds_error=False, fill_value='extrapolate')
    # Note that the mass loss rate is multiplied by 1e6 (from yr-1 to Myr-1)
    f_log_Mdot_wind = interp1d(ry['time_Myr'], np.log10(ry['Mdot_wind']) + sb.logM,
                               bounds_error=False, fill_value='extrapolate')

    res_rad = dict()
    res_wind = dict()
    res_sn = dict()

    res_sn['age_Myr'] = age_sn
    res_sn['SN_rate'] = f_SN_rate(age_sn)

    res_rad['age_Myr'] = age_rad
    res_rad['Xi_LyC'] = f_Xi_LyC(age_rad)
    res_rad['Psi_LW'] = f_Psi_LW(age_rad)
    res_rad['Psi_PE'] = f_Psi_PE(age_rad)

    res_wind['age_Myr'] = age_wind
    res_wind['log_Edot_wind'] = f_log_Edot_wind(age_wind)
    res_wind['log_Mdot_wind'] = f_log_Mdot_wind(age_wind)

    df_sn = pd.DataFrame.from_dict(res_sn)
    df_rad = pd.DataFrame.from_dict(res_rad)
    df_wind = pd.DataFrame.from_dict(res_wind)

    import json
    from tabulate import tabulate

    # https://stackoverflow.com/questions/27988356/save-fixed-width-text-file-from-pandas-dataframe
    def to_fwf(df, fname):
        content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain", floatfmt='.10e')
        open(fname, "w").write(content)

    pd.DataFrame.to_fwf = to_fwf

    def get_name(sb):
        basename = osp.basename(sb.basedir.strip('/'))
        name = ''
        for b in basename.split('_'):
            if b.startswith('Z'):
                name += b + '_'
            if b.startswith('Geneva'):
                name += b + '_'

        return name.rstrip('_')

    writedir1 = '/projects/EOSTRIKE/SB99/Tables-Athena-TIGRESS'
    writedir2 = '/home/jk11/Athena-TIGRESS/dat/'

    name = get_name(sb)
    for writedir in (writedir1, writedir2):
        print('Write to {0:s}'.format(writedir))
        # Write SB99 input parameters (list) to a text file
        # Can be read with json.load
        with open(osp.join(writedir, name + '_par.txt'), 'w') as file:
             file.write(json.dumps(sb.par))

        # Write
        df_sn.to_fwf(osp.join(writedir, name + '_sn.txt'))
        df_wind.to_fwf(osp.join(writedir, name + '_wind.txt'))
        df_rad.to_fwf(osp.join(writedir, name + '_rad.txt'))

        # df_sn.to_csv(osp.join(writedir, name + '_sn.txt'),
        # index=False, sep=r' ', float_format='%g', header=False)
        # df_wind.to_csv(osp.join(writedir, name + '_wind.txt'),
        # index=False, sep=r' ', float_format='%g', header=False)
        # df_rad.to_csv(osp.join(writedir, name + '_rad.txt'),
        # index=False, sep=r' ', float_format='%g', header=False)

    return sb

if __name__ == '__main__':

    models = dict(
        # Non-rotating models
        #Z001_GenevaV00='/projects/EOSTRIKE/SB99/Z001_M1E6_GenevaV00_dt01/',
        Z002_GenevaV00='/projects/EOSTRIKE/SB99/Z002_M1E6_GenevaV00_dt01/',
        #Z008_GenevaV00='/projects/EOSTRIKE/SB99/Z008_M1E6_GenevaV00_dt01/',
        Z014_GenevaV00='/projects/EOSTRIKE/SB99/Z014_M1E6_GenevaV00_dt01/',
        #Z040_GenevaV00='/projects/EOSTRIKE/SB99/Z040_M1E6_GenevaV00_dt01/',

        # Models with rotation
        #Z001_GenevaV40='/projects/EOSTRIKE/SB99/Z001_M1E6_GenevaV40_dt01/',
        Z002_GenevaV40='/projects/EOSTRIKE/SB99/Z002_M1E6_GenevaV40_dt01/',
        #Z008_GenevaV40='/projects/EOSTRIKE/SB99/Z008_M1E6_GenevaV40_dt01/',
        Z014_GenevaV40='/projects/EOSTRIKE/SB99/Z014_M1E6_GenevaV40_dt01/',
        #Z040_GenevaV40='/projects/EOSTRIKE/SB99/Z040_M1E6_GenevaV40_dt01/',
    )

    for mdl in models.keys():
        print(mdl)
        sb = sb99.SB99(models[mdl])
        sb = write_sb99_output_for_tigress(sb)
        print(sb.par[25], sb.par[-7])
        # break
