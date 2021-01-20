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
