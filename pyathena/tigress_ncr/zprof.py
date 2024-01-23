# zprof.py

import os
import os.path as osp

import xarray as xr
import numpy as np
import astropy.units as au
import astropy.constants as ac
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from ..load_sim import LoadSim
from ..io.read_zprof import read_zprof_all, ReadZprofBase
from ..classic.utils import texteffect

class Zprof(ReadZprofBase):

    @LoadSim.Decorators.check_netcdf_zprof
    def _read_zprof(self, phase='whole', savdir=None, force_override=False):
        """Function to read zprof and convert quantities to convenient units.
        """

        ds = read_zprof_all(osp.dirname(self.files['zprof'][0]),
                            self.problem_id, phase=phase, savdir=savdir,
                            force_override=force_override)
        u = self.u
        # Divide all variables by total area Lx*Ly
        domain = self.domain
        dxdy = domain['dx'][0]*domain['dx'][1]
        LxLy = domain['Lx'][0]*domain['Lx'][1]

        ds = ds/LxLy

        # Rename time to time_code and use physical time in Myr as dimension
        ds = ds.rename(dict(time='time_code'))
        ds = ds.assign_coords(time=ds.time_code*self.u.Myr)
        ds = ds.assign_coords(z_kpc=ds.z*self.u.kpc)
        ds = ds.swap_dims(dict(time_code='time'))

        ds['pok'] = ds['P']*(u.energy_density/ac.k_B).cgs.value
        ds['Pturbok'] = 2.0*ds['Ek3']*(u.energy_density/ac.k_B).cgs.value

        try:
            ds['Jrad_LW'] = ds['Erad1']*u.energy_density*ac.c.cgs.value/(4.0*np.pi)
            ds['Jrad_PE'] = ds['Erad2']*u.energy_density*ac.c.cgs.value/(4.0*np.pi)
        except KeyError:
            pass

        # ds._set_attrs(ds.domain)

        return ds

    def load_zprof_for_sfr(self):
        """Process zprof data for pressure/weight analysis
        """
        if hasattr(self,'zp_pw'):
            return self.zp_pw
        zp = self.read_zprof(phase=['2p','h','whole'])

        zplist=[]
        for ph in zp:
            zplist.append(zp[ph].expand_dims('phase').assign_coords(phase=[ph]))

        zp=xr.concat(zplist,dim='phase')
        rename_dict = {'h':'hot'}
        zp = zp.to_array().to_dataset('phase').rename(rename_dict)
        zp = zp.to_array('phase').to_dataset('variable')

        dm=self.domain
        u=self.u
        momzp=xr.Dataset()
        momzp['Pthm']=(zp['P']*u.pok)
        momzp['Ptrb']=(2.0*zp['Ek3']*u.pok)
        if 'B1' in zp:
            momzp['dPimag']=((zp['dPB1']+zp['dPB2']-zp['dPB3'])*u.pok)
            momzp['oPimag']=0.5*(zp['B1']**2+zp['B2']**2-zp['B3']**2)*u.pok
            momzp['Pimag'] = momzp['dPimag'] + momzp['oPimag']
        else:
            momzp['dPimag']=xr.zeros_like(zp['P'])
            momzp['oPimag']=xr.zeros_like(zp['P'])
            momzp['Pimag']=xr.zeros_like(zp['P'])

        momzp['Ptot']=momzp['Pthm']+momzp['Ptrb']+momzp['Pimag']

        momzp['uWext']=(zp['dWext'].sel(z=slice(0,dm['re'][2]))[:,::-1].\
                        cumsum(dim='z')[:,::-1]*u.pok*dm['dx'][2])
        momzp['lWext']=(-zp['dWext'].sel(z=slice(dm['le'][2],0)).\
                        cumsum(dim='z')*u.pok*dm['dx'][2])
        momzp['Wext']=(momzp['uWext'].fillna(0.)+momzp['lWext'].fillna(0.))
        if 'dWsg' in zp:
            momzp['uWsg']=(zp['dWsg'].sel(z=slice(0,dm['re'][2]))[:,::-1].\
                           cumsum(dim='z')[:,::-1]*u.pok*dm['dx'][2])
            momzp['lWsg']=(-zp['dWsg'].sel(z=slice(dm['le'][2],0)).\
                           cumsum(dim='z')*u.pok*dm['dx'][2])
            momzp['Wsg']=(momzp['uWsg'].fillna(0.)+momzp['lWsg'].fillna(0.))
        else:
            momzp['Wsg']=xr.zeros_like(zp['P'])
        momzp['W']=momzp['Wext']+momzp['Wsg']
        momzp['A']=zp['A']

        self.zp_pw = momzp
        return momzp

    def load_zprof_for_wind(self):
        """Process zprof data for pressure/weight analysis
        """
        if hasattr(self,'zp_flux'):
            return self.zp_flux
        zp = self.read_zprof(phase=['2p','h','whole'])

        zplist=[]
        for ph in zp:
            zplist.append(zp[ph].expand_dims('phase').assign_coords(phase=[ph]))

        zp=xr.concat(zplist,dim='phase')
        rename_dict = {'h':'hot'}
        zp = zp.to_array().to_dataset('phase').rename(rename_dict)
        zp = zp.to_array('phase').to_dataset('variable')

        dm=self.domain
        u=self.u
        flux=xr.Dataset()
        flux['mass_net']=((zp['pFzd']-zp['mFzd'])*u.mass_flux)
        flux['massp']=(zp['pFzd']*u.mass_flux)
        flux['massm']=(zp['mFzd']*u.mass_flux)
        flux['mom_kin']=(zp['pFzM3']*u.momentum_flux)
        try:
            flux['mom_th']=(zp['pP']*u.momentum_flux)
        except KeyError:
            flux['mom_th']=(zp['P']*u.momentum_flux)
        flux['mom']=flux['mom_kin']+flux['mom_th']
        flux['energy_kin']=(zp['pFzE1']+zp['pFzE2']+zp['pFzE3'])*u.energy_flux
        flux['energy_th']=zp['pFzP']*u.energy_flux
        flux['energy']=flux['energy_kin']+flux['energy_th']
        flux['A']=zp['pA']
        flux['d']=zp['pd']

        self.zp_flux = flux
        return flux

    def load_zprof_for_phase(self,thermal_only=False):
        if hasattr(self,'zp_th'):
            if thermal_only:
                return self.zp_th
            else:
                return self.zp_th, self.zp_ch

        # thermal phase
        zpth = self.read_zprof(phase=['c','u','w','h1','h2','whole'])
        # chemical/thermal phase
        if not thermal_only:
            zpch = self.read_zprof(phase=['mol','CNM','UNM','WNM',
                'pi','h1','h2','whole'])

        def _to_dset(zp):
            zplist=[]
            for ph in zp:
                if self.test_newcool():
                    ns = self.par['configure']['nscalars']
                    sHI = 's{}'.format(ns-2)
                    sH2 = 's{}'.format(ns-1)
                    zpdata = zp[ph][['A','d',sHI,sH2]]
                    zpdata = zpdata.rename({sHI:'dHI',sH2:'dH2'})
                    zpdata['dHII'] = zpdata['d']-(zpdata['dHI']+2.0*zpdata['dH2'])
                else:
                    zpdata = zp[ph][['A','d']]
                zplist.append(zpdata.expand_dims('phase').assign_coords(phase=[ph]))
            zp=xr.concat(zplist,dim='phase')
            if 'mol' in zp.phase:
                rename_dict = {'mol':'MOL','pi':'WIM','h1':'WHIM','h2':'HIM'}
                zp = zp.to_array().to_dataset('phase').rename(rename_dict)
                zp = zp.to_array('phase').to_dataset('variable')
            return zp

        zpth = _to_dset(zpth)
        if not thermal_only: zpch = _to_dset(zpch)

        self.zp_th = zpth
        if not thermal_only: self.zp_ch = zpch

        if thermal_only:
            return zpth
        else:
            return zpth, zpch


def plt_zprof_compare(sa, models=None, phase=['c','u','w','h1','h2'], field='d',
                      norm=dict(c=LogNorm(1e-4,1e1),
                                u=LogNorm(1e-4,1e1),
                                w=LogNorm(1e-5,1e0),
                                h1=LogNorm(1e-5,1e-1),
                                h2=LogNorm(1e-5,1e-2)),
                      tmax=None,
                      figsize=None,
                      read_zprof_kwargs=None, verbose=False):
    """
    Plot space-time diagram using zprof of different models

    Parameters
    ----------
    sa : instance of LoadSimAll

    models : list of str
        Model names
    phase : list of str
        Phases to compare
    """

    nc = len(models)
    nr = len(phase)
    if figsize is None:
        figsize = (7*nc,6*nr)

    zpa = dict()
    fig, axes = plt.subplots(nr, nc, figsize=figsize, constrained_layout=True)
    for ir,mdl in enumerate(models):
        s = sa.set_model(mdl, verbose=verbose)
        print(mdl)
        zpa[mdl] = s.read_zprof(phase=phase, **read_zprof_kwargs)
        for ic, ph in enumerate(phase):
            zp = zpa[mdl][ph]
            if tmax is None:
                tmax = zp['time'].max()
            extent = (0, tmax, s.domain['le'][2], s.domain['re'][2])
            zpd = zp[field].where(zp['time'] < tmax).dropna('time')
            zpd.plot.imshow(ax=axes[ic,ir], norm=norm[ph], extent=extent)
            axes[ic,ir].text(0.07, 0.9, ph, **texteffect(fontsize='xx-large'),
                             ha='center', transform=axes[ic,ir].transAxes, color='k')

    plt.suptitle('  '.join(models))

    return fig, zpa

def plt_zprof_avg_compare(sa, models=None, phase=['whole','c','u','w','h1','h2'],
                          ls = ['-','--',':','-.'],
                          field='d',
                          xlim=dict(whole=None,w=None,h1=None,h2=None,
                                    c=(-500,500),u=(-500,500)),
                          ylim=dict(d=dict(whole=(1e-4,1e1),
                                           c=(1e-4,1e1),
                                           u=(1e-4,1e1),
                                           w=(1e-4,1e1),
                                           h1=(1e-5,1e-1),
                                           h2=(1e-5,1e-1)),
                                    ),
                          trange=None, figsize=None,
                          read_zprof_kwargs=None, verbose=False):
    """
    Compare time-averaged z-profiles of different models

    Parameters
    ----------
    sa : instance of LoadSimAll

    models : list of str
        Model names
    phase : list of str
        Phases to compare
    """

    if len(phase) > 5:
        nr = 2
        nc = round(len(phase)/2)
        if figsize is None:
            figsize = (4*nc,4*nr)
    else:
        nc = len(phase)
        nr = 1
        if figsize is None:
            figsize = (3.5*nc,6*nr)

    zpa = dict()
    fig, axes = plt.subplots(nr, nc, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()
    for i,mdl in enumerate(models):
        print(mdl)
        s = sa.set_model(mdl, verbose=verbose)
        zpa[mdl] = s.read_zprof(phase=phase, **read_zprof_kwargs)
        for j,ph in enumerate(phase):
            zp = zpa[mdl][ph]
            if trange is not None:
                zpd = zp[field].where(np.logical_and(zp['time'] >= trange[0],
                                                     zp['time'] < trange[1])).dropna('time')
            else:
                zpd = zp[field]
            axes[j].semilogy(zpd['z'], zpd.mean(dim='time'), ls=ls[i], label=mdl)
            axes[j].set(ylim=ylim[field][ph], title=ph)

    for j,ph in enumerate(phase):
        if xlim[ph] is not None:
            axes[j].set_xlim(*xlim[ph])

    axes[0].legend(fontsize='small')
    plt.suptitle('  '.join(models))

    return fig, zpa

def plot_stacked_bar(frac,x='time',**kwargs):
    """A convenient wrapper for stacked bar graph plot for pressure/weight
    """
    # setup colors and labels using seaborn
    import seaborn as sns
    cp_br=sns.diverging_palette(250,20,n=8)
    cp_mag=sns.diverging_palette(200,60,n=4)
    c_2p = cp_br[:2]
    c_2p += cp_mag[:2]
    c_hot = cp_br[6:][::-1]
    c_hot += cp_mag[2:][::-1]

    varname = dict(Pthm = r'P_{{\rm thm}}',
                   Ptrb = r'P_{{\rm trb}}',
                   dPimag = r'\delta\Pi_{{\rm mag}}',
                   oPimag = r'\overline{{\Pi}}_{{\rm mag}}',
                   Wsg = r'W_{{\rm sg}}',
                   Wext = r'W_{{\rm ext}}',
                  )
    color_idx = dict(Pthm=0, Ptrb=1, dPimag=2, oPimag=3, Wsg=0, Wext=1)
    colors = {'hot':c_hot,'2p':c_2p}

    # plot using fill_between
    f0 = 0
    for varph in frac.varph:
        try:
            var,ph = varph.values.item()
            label = r'${:s}^{{\rm {:s}}}$'.format(varname[var],ph)
        except KeyError:
            ph,var = varph.values.item()
            label = r'${:s}^{{\rm {:s}}}$'.format(varname[var],ph)
        c = colors[ph][color_idx[var]]
        if x == 'time':
            plt.fill_between(frac.time,f0,f0+frac.sel(varph=varph),label=label,color=c)
        elif x == 'model':
            plt.bar(frac.model,frac.sel(varph=varph),bottom=f0,
                    label=label,color=c,**kwargs)

        f0 += frac.sel(varph=varph).data.copy()

def plot_pressure_fraction(ptdata,ph='2p',rolling=None,normed=True):
    """Plot time evolution of pressure contributions as a stacked bar graph
    """

    if rolling is None:
        pt=ptdata
    else:
        pt=ptdata.rolling(time=rolling,center=True).mean()

    # cacluate fractions
    Ptot = pt.sel(variable='Ptot',phase=ph).drop_vars(['variable','phase'])
    frac = pt.sel(variable=['Pthm','Ptrb','dPimag','oPimag'],phase=[ph])\
             .stack(varph = ['phase','variable'])
    if normed: frac = frac/Ptot

    # plot
    plot_stacked_bar(frac)

    # decorate
    plt.legend(loc=2,bbox_to_anchor=(1.01,1))
    plt.xlabel('time [Myr]')
    plt.ylabel(r'$P_{\rm comp}^{\rm ph}/P_{\rm tot}$')
    if normed: plt.ylim(0,1)

def plot_weight_fraction(wtdata,rolling=None,normed=True):
    """Plot time evolution of pressure contributions as a stacked bar graph
    """

    if rolling is None:
        wt=wtdata
    else:
        wt=wtdata.rolling(time=rolling,center=True).mean()
    # cacluate fractions
    Wtot = wt.sel(variable='W',phase='whole').drop_vars(['variable','phase'])
    frac = wt.sel(variable=['Wsg','Wext'],phase=['2p','hot'])\
             .stack(varph = ['phase','variable'])
    if normed: frac = frac/Wtot

    # plot
    plot_stacked_bar(frac)

    # decorate
    plt.legend(loc=2,bbox_to_anchor=(1.01,1))
    plt.xlabel('time [Myr]')
    plt.ylabel(r'$W_{\rm comp}^{\rm ph}/W$')
    if normed: plt.ylim(0,1)

def plt_pressure_weight_tevol(sim,rolling=None,normed=True):
    pw = sim.load_zprof_for_sfr()

    area = pw['A'].sel(z=slice(-10,10)).mean(dim='z')
    area_frac = area/area.sel(phase='whole')

    wtevol=pw.to_array().sel(variable=['W','Wsg','Wext'],z=slice(-10,10)).mean(dim='z')
    wtevol = wtevol#/area_frac

    da_p = pw.to_array().sel(variable=['Ptot','Ptrb','Pthm','Pimag','dPimag','oPimag'])
    pmid_tevol=da_p.sel(z=slice(-10,10)).mean(dim='z')
    pzmax_tevol=da_p.sel(z=(-1000,1000),method='nearest').mean(dim='z')
    ptevol=pmid_tevol-pzmax_tevol
    ptevol = ptevol/area_frac

    fig,axes = plt.subplots(3,1,figsize=(10,10),gridspec_kw=dict(wspace=1),sharex=True)

    # time evolution of midplane values
    plt.sca(axes[0])
    plt.plot(wtevol.time,wtevol.sel(variable='W',phase='whole'),label=r'$W$')
    plt.plot(ptevol.time,ptevol.sel(variable='Ptot',phase='whole'),label=r'$P_{\rm tot}$')
    #plt.plot(wtevol.time,wtevol.sel(variable='W',phase='2p'),label=r'$W^{\rm 2p}$')
    plt.plot(ptevol.time,ptevol.sel(variable='Ptot',phase='2p'),\
            label=r'$f_V^{\rm 2p}\cdot P_{\rm tot}^{\rm 2p}$')
    plt.legend(loc=2,bbox_to_anchor=(1.01,1))
    plt.ylabel(r'$P, W\,[k_B\, {\rm cm^{-3}\, K}]$')

    # pressure fraction
    plt.sca(axes[1])
    plot_pressure_fraction(ptevol,rolling=rolling,normed=normed)
    plt.xlabel('')
    #plt.legend(loc=2,ncol=2,bbox_to_anchor=(1.01,1))

    # weight fraction
    plt.sca(axes[2])
    plot_weight_fraction(wtevol,rolling=rolling,normed=normed)

    # pressure fraction
    #plt.sca(axes[3])
    #plot_pressure_fraction(ptevol,rolling=rolling,ph='hot')
    #plt.xlabel('')
    #plt.legend(loc=2,ncol=1,bbox_to_anchor=(1.01,1))


    return fig

def calc_phase_fraction(zpdata,x='time',var='fmass',zcut='H',z0=0,rolling=None):
    """Plot time evolution of filling factors of different phases
    """
    phs = list(zpdata.phase.data)

    z = zpdata.z
    d = zpdata.sel(variable='d')

    if z0 == 0:
        if zcut == 'H':
            zcut = np.sqrt((d*z**2).sel(phase=phs[:-3]).sum(dim=['z','phase']) \
                          /d.sel(phase=phs[:-3]).sum(dim=['z','phase']))
        phdata = zpdata.where(np.abs(z)<zcut).sum(dim='z')
    else:
        phdata = zpdata.sel(z=slice(z0-100,z0+100)).sum(dim='z')

    if x == 'time':
        if rolling is None:
            pht=phdata
        else:
            pht=phdata.rolling(time=rolling,center=True).mean()
    else:
        pht=phdata

    # cacluate fractions
    if var == 'fmass':
        tot = pht.sel(phase='whole',variable='d').drop_vars(['phase'])
        frac = pht.sel(phase=phs[:-1],variable='d')/tot
    elif var == 'fvol':
        tot = pht.sel(phase='whole',variable='A').drop_vars(['phase'])
        frac = pht.sel(phase=phs[:-1],variable='A')/tot

    if (var == 'fmass') & ('MOL' in phs) & ('dH2' in zpdata.coords['variable']):
        # recalculate mass fraction using xH2 and xHII
        fH = pht.sel(phase='whole',variable=['dHI','dH2','dHII'])
        fH = fH/tot

        # get fmol and fpi from scalars
        fmol = 2.0*fH.sel(variable='dH2').drop_vars(['phase','variable'])
        fpi = fH.sel(variable='dHII').drop_vars(['phase','variable'])

        # recacluate CNM and WNM
        fHI = pht.sel(variable='dHI')/tot
        fcnm = fHI.sel(phase='CNM')
        funm = fHI.sel(phase='UNM')
        fwnm = fHI.sel(phase='WNM')

        # other phases
        fh1 = frac.drop_vars('variable').sel(phase=['WHIM']).sum(dim='phase')
        fh2 = frac.drop_vars('variable').sel(phase=['HIM']).sum(dim='phase')

        # save back to frac
        frac_new = xr.Dataset()

        frac_new['MOL']=fmol
        frac_new['CNM']=fcnm
        frac_new['UNM']=funm
        frac_new['WNM']=fwnm
        frac_new['WIM']=fpi
        frac_new['WHIM']=fh1
        frac_new['HIM']=fh2

        frac = frac_new.to_array('phase')

        total = frac.sum(dim='phase')
        if (1-total).std() > 0.01:
            print("Warning: phase fraction doesn't sum up to 1 -- sum={}".format(total.mean()))

        frac /= total

    return frac

def plot_phase_fraction(frac,x='time',zcut='H',z0=0,rolling=None):
    # plot
    import seaborn as sns
    import cmasher.cm as cma
    colors=sns.color_palette(cma.pride(np.linspace(0.0,1.0,8)))
    if len(frac.phase) == 5:
        colors = colors[1:4]+colors[5:-1]
    else:
        colors = colors[:-1]

    # plot using fill_between
    f0 = 0
    for c,ph in zip(colors,frac.phase):
        if x=='time':
            plt.fill_between(frac.time,f0,f0+frac.sel(phase=ph),label=ph.data,color=c)
        elif x=='model':
            plt.bar(frac.model,frac.sel(phase=ph),bottom=f0,label=ph.data,color=c)
        f0 += frac.sel(phase=ph).data.copy()

    # decorate
    if x=='time': plt.xlabel('time [Myr]')
    plt.ylim(0,1)

def plt_phase_balance(sim,**z_kwargs):
    if sim.test_newcool():
        zpth,zpch = sim.load_zprof_for_phase()
        fig,axes = plt.subplots(2,2,figsize=(12,8))
        zplist = [zpth,zpch]
    else:
        zpth = sim.load_zprof_for_phase(thermal_only = True)
        fig,axes = plt.subplots(2,1,figsize=(8,6),squeeze=False)
        axes = axes.T
        zplist = [zpth]

    for zpdata,axes_ in zip(zplist,axes):
        for var,ax in zip(['fvol','fmass'],axes_):
            plt.sca(ax)
            frac = calc_phase_fraction(zpdata.to_array(),var=var,**z_kwargs)
            plot_phase_fraction(frac,**z_kwargs)
            plt.ylabel(var)

    for ax in axes[:,-1]:
        plt.sca(ax)
        plt.legend(loc=2,bbox_to_anchor=(1.01,1))

    return fig

def plt_upsilon_comparison(sa,trange=slice(300,500),torb=None,norm=False):
    Upsilon_unit = (ac.k_B*au.K/au.cm**3/ac.M_sun*ac.kpc**2*au.yr).to('km/s').value
    Pstack = xr.Dataset()
    Wstack = xr.Dataset()
    Upsilon = xr.Dataset()

    for m in sa.models:
        s = sa.set_model(m)
        if torb is not None:
            to = 2*np.pi/s.par['problem']['Omega']*s.u.Myr
            t0 = to*torb[0]
            t1 = to*torb[1]
            trange = slice(t0,t1)
        h = s.read_hst()
        sfr = h.set_index('time')['sfr10'].to_xarray()
        sfr = sfr.sel(time=trange).mean().data
        H2p = h.set_index('time')['H_2p'].to_xarray()
        H2p = H2p.sel(time=trange).mean().data
        pw = s.load_zprof_for_sfr()

        print(m,sfr,H2p)
        da_p = pw.to_array().sel(variable=['Pthm','Ptrb','dPimag','oPimag'])
        pmid_tevol=da_p.sel(z=slice(-10,10)).mean(dim='z')
        area = pw['A'].sel(z=slice(-10,10)).mean(dim='z')
        area_frac = area/area.sel(phase='whole')
        pzmax_tevol=da_p.sel(z=(-4*H2p,4*H2p),method='nearest').mean(dim='z')
        ptevol=pmid_tevol-pzmax_tevol
        ptevol=ptevol/area_frac

        Pstack[m] = ptevol.sel(time=trange,phase=['2p'])\
                          .stack(varph=['phase','variable']).mean(dim='time')

        Upsilon[m] = Pstack[m]/sfr*Upsilon_unit
        if norm:
            Ptot = Pstack[m].sum(dim='varph')
            Pstack[m] = Pstack[m]/Ptot

        wtevol=pw.to_array().sel(variable=['Wsg','Wext'],z=slice(-10,10)).mean(dim='z')
        Wstack[m] = wtevol.sel(time=trange,phase=['2p'])\
                          .stack(varph=['phase','variable']).mean(dim='time')

    fig,axes = plt.subplots(2,1,figsize=(8,6))
    plt.sca(axes[0])
    plot_stacked_bar(Pstack.to_array('model'),x='model',width=-0.4,align='edge')
    plot_stacked_bar(Wstack.to_array('model'),x='model',width=0.4,align='edge')
    plt.legend(loc=2,bbox_to_anchor=(1.01,1))
    if norm:
        plt.ylabel(r'$P_{\rm comp}^{\rm ph}/P_{\rm tot}$')
    else:
        plt.ylabel(r'$P/k_B\,[{\rm cm^{-3} K}]$')

    plt.sca(axes[1])
    plot_stacked_bar(Upsilon.to_array('model'),x='model')
    plt.ylabel(r'$\Upsilon\,[{\rm km/s}]$')

    return fig

def plt_phase_comparison(sa,thermal_only=False,
        trange=slice(300,500),torb=None,**zkwargs):
    thstack_fvol = xr.Dataset()
    thstack_fmass = xr.Dataset()
    chstack_fvol = xr.Dataset()
    chstack_fmass = xr.Dataset()

    for m in sa.models:
        s = sa.set_model(m)
        if torb is not None:
            to = 2*np.pi/s.par['problem']['Omega']*s.u.Myr
            t0 = to*torb[0]
            t1 = to*torb[1]
            trange = slice(t0,t1)

        if thermal_only:
            zpth = s.load_zprof_for_phase(thermal_only=thermal_only)
            fvol = calc_phase_fraction(zpth.to_array(),var='fvol',**zkwargs)
            fmass = calc_phase_fraction(zpth.to_array(),var='fmass',**zkwargs)
            thstack_fvol[m] = fvol.sel(time=trange).mean(dim='time')
            thstack_fmass[m] = fmass.sel(time=trange).mean(dim='time')
        else:
            if s.test_newcool():
                zpth, zpch = s.load_zprof_for_phase(thermal_only=thermal_only)
                fvol = calc_phase_fraction(zpth.to_array(),var='fvol',**zkwargs)
                fmass = calc_phase_fraction(zpth.to_array(),var='fmass',**zkwargs)
                thstack_fvol[m] = fvol.sel(time=trange).mean(dim='time')
                thstack_fmass[m] = fmass.sel(time=trange).mean(dim='time')
                fvol = calc_phase_fraction(zpch.to_array(),var='fvol',**zkwargs)
                fmass = calc_phase_fraction(zpch.to_array(),var='fmass',**zkwargs)
                chstack_fvol[m] = fvol.sel(time=trange).mean(dim='time')
                chstack_fmass[m] = fmass.sel(time=trange).mean(dim='time')

    fraclist = [thstack_fvol.to_array('model'),thstack_fmass.to_array('model')]
    if thermal_only:
        fig,axes = plt.subplots(2,1,figsize=(8,6),squeeze=False)
        axes = axes.T
    else:
        fig,axes = plt.subplots(2,2,figsize=(12,8))
        fraclist += [chstack_fvol.to_array('model'),chstack_fmass.to_array('model')]

    for frac,ax in zip(fraclist,axes.flat):
        plt.sca(ax)
        plot_phase_fraction(frac,x='model')

    for ax in axes[:,-1]:
        plt.sca(ax)
        plt.legend(loc=2,bbox_to_anchor=(1.01,1))

    return fig
