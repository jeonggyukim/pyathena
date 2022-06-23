# read_hst.py

import os
import os.path as osp
import glob
import xarray as xr
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
import astropy.units as au
import astropy.constants as ac

from ..io.read_hst import read_hst
from ..load_sim import LoadSim

class Hst:

    @LoadSim.Decorators.check_pickle_hst
    def read_hst(self, savdir=None, force_override=False):
        """Function to read hst and convert quantities to convenient units
        """

        par = self.par
        u = self.u
        domain = self.domain

        # volume of resolution element (code unit)
        dvol = domain['dx'].prod()
        # total volume of domain (code unit)
        vol = domain['Lx'].prod()
        # Area of domain (code unit)
        LxLy = domain['Lx'][0]*domain['Lx'][1]
        # Lz
        Lz = domain['Lx'][2]

        Omega = self.par['problem']['Omega']
        if Omega>0:
            time_orb = 2*np.pi/Omega*u.Myr # Orbital time in Myr
        else:
            time_orb = 1.0
        try:
            if self.par['configure']['new_cooling'] == 'ON':
                newcool = True
            else:
                newcool = False
        except KeyError:
            newcool = False
        nscalars = self.par['configure']['nscalars']


        hst = read_hst(self.files['hst'], force_override=force_override)

        h = pd.DataFrame()

        if self.par['configure']['gas'] == 'mhd':
            mhd = True
        else:
            mhd = False

        # Time in code unit
        h['time_code'] = hst['time']
        # Time in Myr
        h['time'] = h['time_code']*u.Myr
        h['time_orb'] = h['time']/time_orb
        # Time step
        h['dt_code'] = hst['dt']
        h['dt'] = hst['dt']*u.Myr

        # if par['configure']['new_cooling'] == 'ON' and \
        #    (par['configure']['radps'] == 'ON' or par['configure']['sixray'] == 'ON'):
        #     for c in ('dt_cool_min','dt_xH2_min','dt_xHII_min'):
        #         hst[c] *= u.Myr*vol

        # Total gas mass in Msun
        h['mass'] = hst['mass']*vol*u.Msun
        h['mass_sp'] = hst['msp']*vol*u.Msun
        for i in range(nscalars):
            h[f'mass{i}'] = hst[f'scalar{i}']*vol*u.Msun

        if newcool:
            h['M_HI'] = h['mass{0:d}'.format(nscalars - 3)]
            h['Sigma_HI'] = h['M_HI']/(LxLy*u.pc**2)
            h['M_H2'] = 2.0*h['mass{0:d}'.format(nscalars - 2)]
            h['Sigma_H2'] = h['M_H2']/(LxLy*u.pc**2)
            h['M_HII'] = h['mass'] - h['M_H2'] - h['M_HI']
            h['Sigma_HII'] = h['M_HII']/(LxLy*u.pc**2)

        # Total outflow mass
        if self.test_phase_sep_hst():
            self.logger.info("[read_hst]: Reading phase separated history files...")
            hph = self.read_hst_phase_all()
            hw = self.read_hst_phase(iph=0)
            h['massflux_lbd_d'] = hw['Fzm_lower']*u.mass_flux
            h['massflux_ubd_d'] = hw['Fzm_upper']*u.mass_flux
            h['mass_out'] = (hw['Fzm_upper_dt']-hw['Fzm_lower_dt'])*LxLy*u.Msun
        else:
            h['massflux_lbd_d'] = hst['F3_lower']*u.mass_flux
            h['massflux_ubd_d'] = hst['F3_upper']*u.mass_flux
            h['mass_out'] = integrate.cumtrapz(hst['F3_upper'] - hst['F3_lower'], hst['time'], initial=0.0)
            h['mass_out'] = h['mass_out']*LxLy*u.Msun

        # Mass surface density in Msun/pc^2
        h['Sigma_gas'] = h['mass']/(LxLy*u.pc**2)
        h['Sigma_sp'] = h['mass_sp']/(LxLy*u.pc**2)
        h['Sigma_out'] = h['mass_out']/(LxLy*u.pc**2)

        # Calculate (cumulative) SN ejecta mass
        # JKIM: only from clustered type II(?)
        try:
            sn = read_hst(self.files['sn'], force_override=force_override)
            t_ = np.array(hst['time'])
            Nsn, snbin = np.histogram(sn.time, bins=np.concatenate(([t_[0]], t_)))
            h['mass_snej'] = Nsn.cumsum()*self.par['feedback']['MejII'] # Mass of SN ejecta [Msun]
            h['Sigma_snej'] = h['mass_snej']/(LxLy*u.pc**2)
        except KeyError:
            pass

        # H mass/surface density in Msun
        #h['M_gas'] = h['mass']/u.muH
        #h['Sigma_gas'] = h['M_gas']/(LxLy*u.pc**2)

        if self.test_phase_sep_hst():
            h['H'] = np.sqrt(hw['H2']/hw['mass'])
            for ph in ['c','u','w1','w2','h1','h2']:
                if ph == 'h1':
                    phlist = ['WH' + tail for tail in ['MM','IM','NM']]
                elif ph == 'h2':
                    phlist = ['H' + tail for tail in ['MM','IM','NM']]
                else:
                    phlist = [ph.upper() + tail for tail in ['MM','IM','NM']]
                h['mf_{}'.format(ph)] = hph['mass'].sel(phase=phlist).sum(dim='phase')/hw['mass']
                h['vf_{}'.format(ph)] = hph['vol'].sel(phase=phlist).sum(dim='phase')/hw['vol']
                h['H_{}'.format(ph)] = \
                    np.sqrt(hph['H2'].sel(phase=phlist).sum(dim='phase')/hph['mass'].sel(phase=phlist).sum(dim='phase'))
            h['mf_w'] = h['mf_w1'] + h['mf_w2']
            h['vf_w'] = h['vf_w1'] + h['vf_w2']
            h['H_w'] = np.sqrt((h['H_w1']**2*h['mf_w1'] +
                                h['H_w2']**2*h['mf_w2'])/
                                (h['mf_w']))
            h['mf_2p'] = h['mf_c'] + h['mf_u'] + h['mf_w']
            h['vf_2p'] = h['vf_c'] + h['vf_u'] + h['vf_w']
            h['H_2p'] = np.sqrt((h['H_c']**2*h['mf_c'] +
                                 h['H_u']**2*h['mf_u'] +
                                 h['H_w']**2*h['mf_w'])/
                                 (h['mf_2p']))
        else:
            # Mass, volume fraction, scale height
            h['H'] = np.sqrt(hst['H2'] / hst['mass'])
            for ph in ['c','u','w','h1','h2']:
                h['mf_{}'.format(ph)] = hst['M{}'.format(ph)]/hst['mass']
                h['vf_{}'.format(ph)] = hst['V{}'.format(ph)]
                h['H_{}'.format(ph)] = \
                    np.sqrt(hst['H2{}'.format(ph)] / hst['M{}'.format(ph)])
            #print(h['mf_c'])
            #h['Vmid_2p'] = hst['Vmid_2p']

            # mf, vf, H of thermally bistable (cold + unstable + warm) medium
            h['mf_2p'] = h['mf_c'] + h['mf_u'] + h['mf_w']
            h['vf_2p'] = h['vf_c'] + h['vf_u'] + h['vf_w']
            h['H_2p'] = np.sqrt((hst['H2c'] + hst['H2u'] + hst['H2w']) / \
                                (hst['Mc'] + hst['Mu'] + hst['Mw']))

        # Kinetic and magnetic energy
        h['KE'] = hst['x1KE'] + hst['x2KE'] + hst['x3KE']
        if mhd:
            h['ME'] = hst['x1ME'] + hst['x2ME'] + hst['x3ME']

        # hst['x2KE'] = hst['x2dke']
        for ax in ('1','2','3'):
            Ekf = 'x{}KE'.format(ax)
            if ax == '2':
                Ekf = 'x2dke'
            # Mass weighted velocity dispersion??
            h['v{}'.format(ax)] = np.sqrt(2*hst[Ekf]/hst['mass'])
            if mhd:
                h['vA{}'.format(ax)] = \
                    np.sqrt(2*hst['x{}ME'.format(ax)]/hst['mass'])
            if self.test_phase_sep_hst():
                phlist = ['C' + tail for tail in ['MM','IM','NM']]
                phlist += ['U' + tail for tail in ['MM','IM','NM']]
                phlist += ['W1' + tail for tail in ['MM','IM','NM']]
                phlist += ['W2' + tail for tail in ['MM','IM','NM']]
                m2p = hph['mass'].sel(phase=phlist).sum(dim='phase')
                ke2p = hph['x{}KE'.format(ax)].sel(phase=phlist).sum(dim='phase')
                h['v{}_2p'.format(ax)] = np.sqrt(2*ke2p/m2p)
            else:
                h['v{}_2p'.format(ax)] = \
                    np.sqrt(2*hst['x{}KE_2p'.format(ax)]/hst['mass']/h['mf_2p'])

        if self.test_phase_sep_hst():
            h['cs'] = np.sqrt(hw['P']/hw['mass'])
            h['Pth_mid'] = hst['Pth_mid']*u.pok
            h['Pth_mid_2p'] = hst['Pth_mid_2p']*u.pok/hst['Vmid_2p']
            h['Pturb_mid'] = hst['Pturb_mid']*u.pok
            h['Pturb_mid_2p'] = hst['Pturb_mid_2p']*u.pok/hst['Vmid_2p']
        else:
            h['cs'] = np.sqrt(hst['P']/hst['mass'])
            h['Pth_mid'] = hst['Pth']*u.pok
            h['Pth_mid_2p'] = hst['Pth_2p']*u.pok/hst['Vmid_2p']
            h['Pturb_mid'] = hst['Pturb']*u.pok
            h['Pturb_mid_2p'] = hst['Pturb_2p']*u.pok/hst['Vmid_2p']

        # Midplane number density
        h['nmid'] = hst['nmid']
        h['nmid_2p'] = hst['nmid_2p']/hst['Vmid_2p']

        # Star formation rate per area [Msun/kpc^2/yr]
        h['sfr10'] = hst['sfr10']
        h['sfr40'] = hst['sfr40']
        h['sfr100'] = hst['sfr100']

        try:
            if par['configure']['radps'] == 'ON':
                radps = True
            else:
                radps = False
        except KeyError:
            radps = False

        if radps:
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
                            h[f'Ltot_{k}'] = hst[f'Ltot{i}']*vol*u.Lsun
                            h[f'Lesc_{k}'] = hst[f'Lesc{i}']*vol*u.Lsun
                            if par['radps']['eps_extinct'] > 0.0:
                                h[f'Leps_{k}'] = hst[f'Leps{i}']*vol*u.Lsun
                            try:
                                h[f'Ldust_{k}'] = hst[f'Ldust{i}']*vol*u.Lsun
                            except KeyError:
                                self.logger.info('Ldust not found in hst')

                            hnu = (par['radps'][f'hnu_{k}']*au.eV).cgs.value
                            h[f'Qtot_{k}'] = h[f'Ltot_{k}'].values * \
                                               (ac.L_sun.cgs.value)/hnu
                            h[f'Qesc_{k}'] = h[f'Lesc_{k}'].values * \
                                               (ac.L_sun.cgs.value)/hnu
                            # Cumulative number of escaped photons
                            h[f'Qtot_cum_{k}'] = \
                            integrate.cumtrapz(h[f'Qtot_{k}'], h.time*u.time.cgs.value, initial=0.0)
                            h[f'Qesc_cum_{k}'] = \
                            integrate.cumtrapz(h[f'Qesc_{k}'], h.time*u.time.cgs.value, initial=0.0)
                            # Instantaneous escape fraction
                            h[f'fesc_{k}'] = h[f'Lesc_{k}']/h[f'Ltot_{k}']
                            # Cumulative escape fraction
                            h[f'fesc_cum_{k}'] = \
                            integrate.cumtrapz(h[f'Lesc_{k}'], h.time, initial=0.0)/\
                            integrate.cumtrapz(h[f'Ltot_{k}'], h.time, initial=0.0)
                            h[f'fesc_cum_{k}'].fillna(value=0.0, inplace=True)
                        except KeyError as e:
                            pass
                            #raise e

            if 'Ltot_LW' in hst.columns and 'Ltot_PE' in hst.columns:
                h['fesc_FUV'] = (hst['Lesc_PE'] + hst['Lesc_LW'])/(hst['Ltot_PE'] + hst['Ltot_LW'])
                h['fesc_cum_FUV'] = \
                    integrate.cumtrapz(hst['Lesc_PE'] + hst['Lesc_LW'], hst.time, initial=0.0)/\
                    integrate.cumtrapz(hst['Ltot_PE'] + hst['Ltot_LW'], hst.time, initial=0.0)
                h[f'fesc_cum_FUV'].fillna(value=0.0, inplace=True)

        try:
            h['xi_CR0'] = hst['xi_CR0']
        except KeyError:
            pass

        h.index = h['time_code']

        self.hst = h

        # SN data
        try:
            if osp.exists(self.files['sn']):
                sn = read_hst(self.files['sn'],force_override=force_override)
                snr = get_snr(sn['time']*self.u.Myr,hst['time']*self.u.Myr)

                self.sn = sn
                self.snr = snr/LxLy
        except:
            pass

        return h

    def test_phase_sep_hst(self):
        fhstph = glob.glob(self.files["hst"].replace(".hst", ".phase*.hst"))
        fhstph.sort()
        nphase = len(fhstph)
        return nphase>0

    def read_hst_phase(self, iph=0):
        if iph == 0:
            f = self.files["hst"].replace(".hst", ".whole.hst")
        else:
            f = self.files["hst"].replace(".hst", ".phase{}.hst".format(iph))

        h = read_hst(f)
        h.index = h["time"]
        return h.to_xarray()

    def read_hst_phase_all(self):
        fhstph = glob.glob(self.files["hst"].replace(".hst", ".phase*.hst"))

        fhstph.sort()

        nphase = len(fhstph)

        hstph = xr.Dataset()
        for iph in range(1, nphase+1):
            phname = self.get_hst_phase_name(iph)
            hstph[phname] = self.read_hst_phase(iph).to_array()
        hstph = hstph.to_array("phase").to_dataset("variable")

        return hstph

    def get_hst_phase_name(self, iph):
        if iph == 0: return "whole"

        thphase = ["C", "U", "W1", "W2", "WH", "H"]
        chphase = ["I", "M", "N"]

        nthph = len(thphase)
        nchph = len(chphase)
        if iph > nthph*nchph:
            raise KeyError("{} phases are defined, but iph={} is given".format(nthph*nchph,iph))
        return thphase[(iph-1) % nthph] + chphase[(iph-1) // nthph] + "M"

def plt_hst_compare(sa, models=None, read_hst_kwargs=dict(savdir=None, force_override=False),
                    c=['k', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5'],
                    ncol=3,
                    lw=[2,2,2,2,2,2,2],
                    column=['Sigma_gas', 'Sigma_sp', 'Sigma_out',
                            'sfr10', 'sfr40', 'dt', 'xi_CR0',
                            'Pturb_mid', 'Pturb_mid_2p',
                            'Pth_mid', 'Pth_mid_2p',
                            'H_c','H_u','H_w','H_2p',
                            'v3_2p','nmid_2p',
                            'mf_c','mf_u','mf_w'],
                    xlim=None,
                    ylim='R8',
                    figsize=None,
                   ):

    if ylim == 'R8':
        ylim=dict(Sigma_gas=(5,13),
                  Sigma_sp=(0,4),
                  Sigma_out=(0.01,10),
                  sfr10=(1e-4,4e-2),
                  sfr40=(1e-4,4e-2),
                  dt=(1e-4,1e-2),
                  xi_CR0=(1e-17,1e-15),
                  Pturb_mid=(1e3,1e5),
                  Pturb_mid_2p=(1e3,1e5),
                  Pth_mid=(1e3,1e5),
                  Pth_mid_2p=(1e3,1e5),
                  H=(0,1000),
                  H_c=(0,300),
                  H_u=(0,300),
                  H_w=(0,1000),
                  H_2p=(0,1000),
                  v3_2p=(0,20.0),
                  nmid_2p=(1e-1,1e1),
                  Vmid_2p=(1e-2,1.0),
                  mf_c=(1e-2,1.0),
                  mf_u=(1e-2,1.0),
                  mf_w=(1e-1,1.0),)

    elif ylim == 'LGR4':
        ylim=dict(Sigma_gas=(0,70),
                  Sigma_sp=(0,30),
                  Sigma_out=(0,10),
                  sfr10=(1e-3,5e-1),
                  sfr40=(1e-3,5e-1),
                  dt=(1e-4,1e-2),
                  xi_CR0=(1e-17,1e-14),
                  Pturb_mid=(1e3,1e6),
                  Pturb_mid_2p=(1e3,1e6),
                  Pth_mid=(1e4,1e6),
                  Pth_mid_2p=(1e4,1e6),
                  H=(0,1000),
                  H_c=(0,400),
                  H_u=(0,400),
                  H_w=(0,1000),
                  H_2p=(0,1000),
                  v3_2p=(0,40.0),
                  nmid_2p=(1e-1,1e2),
                  Vmid_2p=(1e-2,1.0),
                  mf_c=(1e-2,1.0),
                  mf_u=(1e-2,1.0),
                  mf_w=(1e-1,1.0),)


    ylabel = dict(Sigma_gas=r'$\Sigma_{\rm gas}\;[M_{\odot}\,{\rm pc}^{-2}]$',
                  Sigma_sp=r'$\Sigma_{\rm *,formed}\;[M_{\odot}\,{\rm pc}^{-2}]$',
                  Sigma_out=r'$\Sigma_{\rm of}\;[M_{\odot}\,{\rm pc}^{-2}]$',
                  sfr10=r'$\Sigma_{\rm SFR,10Myr}\;[M_{\odot}\,{\rm pc}^{-2}]$',
                  sfr40=r'$\Sigma_{\rm SFR,40Myr}\;[M_{\odot}\,{\rm pc}^{-2}]$',
                  dt=r'${\rm d}t_{\rm mhd}\;[{\rm Myr}]$',
                  xi_CR0=r'$\xi_{\rm CR,0}\;[{\rm s}^{-1}]$',
                  Pturb_mid_2p=r'$P_{\rm turb,mid,2p}\;[{\rm cm}^{-3}\,{\rm K}]$',
                  Pturb_mid=r'$P_{\rm turb,mid}\;[{\rm cm}^{-3}\,{\rm K}]$',
                  Pth_mid_2p=r'$P_{\rm thm,mid,2p}\;[{\rm cm}^{-3}\,{\rm K}]$',
                  Pth_mid=r'$P_{\rm thm,mid}\;[{\rm cm}^{-3}\,{\rm K}]$',
                  H=r'$H\;[{\rm pc}]$',
                  H_c=r'$H_{\rm c}\;[{\rm pc}]$',
                  H_u=r'$H_{\rm u}\;[{\rm pc}]$',
                  H_w=r'$H_{\rm w}\;[{\rm pc}]$',
                  H_2p=r'$H_{\rm 2p}\;[{\rm pc}]$',
                  v3_2p=r'$v_{z,2p}\;[{\rm km}\,{\rm s}^{-1}]$',
                  nmid_2p=r'$n_{{\rm H,mid,2p}}$',
                  Vmid_2p=r'$f_{V,{\rm mid,2p}}$',
                  mf_c=r'$f_{M,{\rm c}}$',
                  mf_u=r'$f_{M,{\rm u}}$',
                  mf_w=r'$f_{M,{\rm w}}$',
                 )

    yscale = dict(Sigma_gas='linear',
                  Sigma_sp='linear',
                  Sigma_out='log',
                  sfr10='log',
                  sfr40='log',
                  dt='log',
                  xi_CR0='log',
                  Pturb_mid_2p='log',
                  Pturb_mid='log',
                  Pth_mid_2p='log',
                  Pth_mid='log',
                  H='linear',
                  H_c='linear',
                  H_u='linear',
                  H_w='linear',
                  H_2p='linear',
                  v3_2p='linear',
                  nmid_2p='log',
                  Vmid_2p='log',
                  mf_c='log',
                  mf_u='log',
                  mf_w='log',
                 )

    if models is None:
        models = sa.models

    nc = ncol
    nr = round(len(column)/nc)
    if figsize is None:
        figsize=(6*nc, 4*nr)

    fig, axes = plt.subplots(nr, nc, figsize=figsize,
                             constrained_layout=True)
    axes = axes.flatten()

    for i,mdl in enumerate(models):
        s = sa.set_model(mdl)
        print(mdl)
        h = s.read_hst(**read_hst_kwargs)
        for j,(ax,col) in enumerate(zip(axes,column)):
            if j == 0:
                label = mdl
            else:
                label = '_nolegend_'
            try:
                ax.plot(h['time'], h[col], c=c[i], lw=lw[i], label=label)
            except KeyError:
                pass

    for j,(ax,col) in enumerate(zip(axes,column)):
        ax.set(xlabel=r'${\rm time}\,[{\rm Myr}]$', ylabel=ylabel[col],
               yscale=yscale[col], ylim=ylim[col])

        if xlim is not None:
            ax.set(xlim=xlim)

    axes[0].legend(loc='best', fontsize='small')

    return fig


def get_snr(sntime,time,tbin='auto',snth=100.):
    import xarray as xr
    snt = sntime.to_numpy()
    t = time.to_numpy()
    if tbin is 'auto':
        tbin=0.
        dtbin=0.1
        snrmean=0.
        while (tbin < 40) & (snrmean < snth):
            tbin += dtbin
            idx=np.less(snt[np.newaxis,:],t[:,np.newaxis]) & \
            np.greater(snt[np.newaxis,:],(t[:,np.newaxis]-tbin))
            snr=idx.sum(axis=1)
            snrmean=snr.mean()

        snr = snr/tbin
    else:
        idx=np.less(snt[np.newaxis,:],t[:,np.newaxis]) & \
        np.greater(snt[np.newaxis,:],(t[:,np.newaxis]-tbin))
        snr=idx.sum(axis=1)/tbin
    snr=xr.DataArray(snr,coords=[time],dims=['time'])
    return snr
