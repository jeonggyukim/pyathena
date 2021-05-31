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

        self.zp_pw = momzp
        return momzp

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
            zpch = self.read_zprof(phase=['mol','CNM','UNM','WNM','pi','h1','h2','whole'])

        def _to_dset(zp):
            zplist=[]
            for ph in zp:
                zpdata = zp[ph][['A','d']]
                zplist.append(zpdata.expand_dims('phase').assign_coords(phase=[ph]))
            zp=xr.concat(zplist,dim='phase')
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

def plot_stacked_bar(frac,x='time'):
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
    colors = {'h':c_hot,'2p':c_2p}

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
            plt.bar(frac.model,frac.sel(varph=varph),bottom=f0,label=label,color=c)

        f0 += frac.sel(varph=varph).data.copy()

def plot_pressure_fraction(ptdata,rolling=None):
    """Plot time evolution of pressure contributions as a stacked bar graph
    """

    if rolling is None:
        pt=ptdata
    else:
        pt=ptdata.rolling(time=rolling,center=True).mean()

    # cacluate fractions
    Ptot = pt.sel(variable='Ptot',phase='whole').drop_vars(['variable','phase'])
    frac = pt.sel(variable=['Pthm','Ptrb','dPimag','oPimag'],phase=['2p','h'])\
             .stack(varph = ['variable','phase'])/Ptot

    # plot
    plot_stacked_bar(frac)

    # decorate
    plt.legend(loc=2,bbox_to_anchor=(1.01,1))
    plt.xlabel('time [Myr]')
    plt.ylabel(r'$P_{\rm comp}^{\rm ph}/P_{\rm tot}$')
    plt.ylim(0,1)

def plot_weight_fraction(wtdata,rolling=None):
    """Plot time evolution of pressure contributions as a stacked bar graph
    """

    if rolling is None:
        wt=wtdata
    else:
        wt=wtdata.rolling(time=rolling,center=True).mean()
    # cacluate fractions
    Wtot = wt.sel(variable='W',phase='whole').drop_vars(['variable','phase'])
    frac = wt.sel(variable=['Wsg','Wext'],phase=['2p','h'])\
             .stack(varph = ['variable','phase'])/Wtot

    # plot
    plot_stacked_bar(frac)

    # decorate
    plt.legend(loc=2,bbox_to_anchor=(1.01,1))
    plt.xlabel('time [Myr]')
    plt.ylabel(r'$W_{\rm comp}^{\rm ph}/W$')
    plt.ylim(0,1)

def plt_pressure_weight_tevol(sim,rolling=None):
    pw = sim.load_zprof_for_sfr()

    wtevol=pw.to_array().sel(variable=['W','Wsg','Wext'],z=slice(-10,10)).mean(dim='z')

    da_p = pw.to_array().sel(variable=['Ptot','Ptrb','Pthm','Pimag','dPimag','oPimag'])
    pmid_tevol=da_p.sel(z=slice(-10,10)).mean(dim='z')
    pzmax_tevol=da_p.sel(z=(-2000,2000),method='nearest').mean(dim='z')
    ptevol=pmid_tevol-pzmax_tevol

    fig,axes = plt.subplots(3,1,figsize=(10,10),gridspec_kw=dict(wspace=1),sharex=True)

    # time evolution of midplane values
    plt.sca(axes[0])
    plt.plot(wtevol.time,wtevol.sel(variable='W',phase='whole'),label=r'$W$')
    plt.plot(ptevol.time,ptevol.sel(variable='Ptot',phase='whole'),label=r'$P_{\rm tot}$')
    plt.legend(loc=2,bbox_to_anchor=(1.01,1))
    plt.ylabel(r'$P, W\,[k_B\, {\rm cm^{-3}\, K}]$')

    # pressure fraction
    plt.sca(axes[1])
    plot_pressure_fraction(ptevol,rolling=rolling)
    plt.xlabel('')
    plt.legend(loc=2,ncol=2,bbox_to_anchor=(1.01,1))

    # weight fraction
    plt.sca(axes[2])
    plot_weight_fraction(wtevol,rolling=rolling)

    return fig

def plot_phase_fraction(zpdata,x='time',var='fmass',zcut='H',z0=0,rolling=None):
    """Plot time evolution of filling factors of different phases
    """
    phs = list(zpdata.phase.data)

    z = zpdata.z
    A = zpdata.A
    d = zpdata.d

    if z0 == 0:
        if zcut == 'H':
            zcut = np.sqrt((d*z**2).sel(phase=phs[:-3]).sum(dim=['z','phase']) \
                          /d.sel(phase=phs[:-3]).sum(dim=['z','phase']))
        if var == 'fmass':
            phdata = d.where(np.abs(z)<zcut).sum(dim='z')
        else:
            phdata = A.where(np.abs(z)<zcut).sum(dim='z')
    else:
        if var == 'fmass':
            phdata = d.sel(z=slice(z0-100,z0+100)).sum(dim='z')
        else:
            phdata = A.sel(z=slice(z0-100,z0+100)).sum(dim='z')


    if x == 'time':
        if rolling is None:
            pht=phdata
        else:
            pht=phdata.rolling(time=rolling,center=True).mean()


    # cacluate fractions
    tot = pht.sel(phase='whole').drop_vars(['phase'])
    frac = pht.sel(phase=phs[:-1])/tot

    # plot
    import seaborn as sns
    import cmasher.cm as cma
    colors=sns.color_palette(cma.pride(np.linspace(0.0,1.0,8)))
    if len(phs) == 6:
        colors = colors[1:4]+colors[5:-1]
    else:
        colors = colors[:-1]
    # plot using fill_between
    f0 = 0
    for c,ph in zip(colors,frac.phase):
        if x=='time':
            plt.fill_between(frac.time,f0,f0+frac.sel(phase=ph),label=ph.data,color=c)
            f0 += frac.sel(phase=ph).data.copy()
        elif x=='model':
            y = frac.mean(dim='time')
            plt.var(frac.model,y,bottom=f0,label=ph.data,color=c)
            f0 += y.data.copy()

    # decorate

    if x=='time': plt.xlabel('time [Myr]')
    plt.ylabel(var)
    plt.ylim(0,1)

def plt_phase_balance(sim,**z_kwargs):
    zpth,zpch = sim.load_zprof_for_phase()

    fig,axes = plt.subplots(2,2,figsize=(12,8))

    for zpdata,axes_ in zip([zpth,zpch],axes):
        for var,ax in zip(['fvol','fmass'],axes_):
            plt.sca(ax)
            plot_phase_fraction(zpdata,var=var,**z_kwargs)
            plt.xlim(100,600)
    for ax in axes[:,-1]:
        plt.sca(ax)
        plt.legend(loc=2,bbox_to_anchor=(1.01,1))

    return fig

def plt_upsilon_comparison(sa,trange=slice(300,500)):
    Upsilon_unit = (ac.k_B*au.K/au.cm**3/ac.M_sun*ac.kpc**2*au.yr).to('km/s').value
    Pstack = xr.Dataset()
    Upsilon = xr.Dataset()
    for m in sa.models:
        s = sa.set_model(m)
        h = s.read_hst()
        sfr = h.set_index('time')['sfr10'].to_xarray()
        pw = s.load_zprof_for_sfr()

        print(m,sfr.sel(time=trange).mean().data)
        da_p = pw.to_array().sel(variable=['Pthm','Ptrb','dPimag','oPimag'])
        pmid_tevol=da_p.sel(z=slice(-10,10)).mean(dim='z')
        pzmax_tevol=da_p.sel(z=(-2000,2000),method='nearest').mean(dim='z')
        ptevol=pmid_tevol-pzmax_tevol

        Pstack[m] = ptevol.sel(time=trange,phase=['2p','h'])\
                          .stack(varph=['phase','variable']).mean(dim='time')
        Upsilon[m] = Pstack[m]/sfr.sel(time=trange).mean()*Upsilon_unit

    fig,axes = plt.subplots(2,1,figsize=(8,10))
    plt.sca(axes[0])
    plot_stacked_bar(Pstack.to_array('model'),x='model')
    plt.legend(loc=2,bbox_to_anchor=(1.01,1))
    plt.ylabel(r'$P/k_B\,[{\rm cm^{-3} K}]$')

    plt.sca(axes[1])
    plot_stacked_bar(Upsilon.to_array('model'),x='model')
    plt.ylabel(r'$\Upsilon\,[{\rm km/s}]$')

    return fig

def plt_phase_comparison(sa,thermal_only=False,trange=slice(300,500)):
    thstack = xr.Dataset()
    chstack = xr.Dataset()
    for m in sa.models:
        s = sa.set_model(m)
        if thermal_only:
            zpth = s.load_zprof_for_phase(thermal_only=thermal_only)
            thstack[m] = zpth.to_array().sel(time=trange)
        else:
            zpth, zpch = s.load_zprof_for_phase(thermal_only=thermal_only)
            thstack[m] = zpth.to_array().sel(time=trange)
            chstack[m] = zpch.to_array().sel(time=trange)

    if thermal_only:
        fig,axes = plt.subplots(2,1,figsize=(8,10),squeeze=False)
        zplist = [thstack]
    else:
        fig,axes = plt.subplots(2,2,figsize=(12,8))
        zplist = [thstack,chstack]

    for zpdata,axes_ in zip(zplist,axes):
        for var,ax in zip(['fvol','fmass'],axes_):
            plt.sca(ax)
            plot_phase_fraction(zpdata.to_array('model'),var=var,x='model',**z_kwargs)

    for ax in axes[:,-1]:
        plt.sca(ax)
        plt.legend(loc=2,bbox_to_anchor=(1.01,1))

    return fig
