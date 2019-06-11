"""
*This module is not intended to be used as a script*
Putting a script inside a module's directory is considered as an antipattern
(see rejected PEP 3122).
You are encouraged to write a seperate script that executes the functions in
this module. - SMOON
"""
import os
import time
import os.path as osp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from astropy import units as au

from ..classic.cooling import coolftn
from ..io.read_hst import read_hst

def plt_proj_density(s, num, dat=None, savfig=True):

    if dat is None:
        ds = s.load_vtk(num=num)
        dat = ds.get_field(field='density', as_xarray=True)
    else:
        ds = s.load_vtk(num=num)
       
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    dat['surface_density'] = (dat['density']*s.u.Msun/s.u.pc**3
            *ds.domain['dx'][2]*s.u.pc).sum(dim='z')
    dat['surface_density'].plot.imshow(ax=ax, norm=mpl.colors.LogNorm(),
            cmap='pink_r', vmin=1e0, vmax=1e4)
    ax.set_aspect('equal')
    plt.suptitle('{0:s}, time: {1:.1f} Myr'.format(s.name, ds.domain['time']*s.u.Myr))
    
    if savfig:
        savdir = osp.join('./figures-proj')
        if not os.path.exists(savdir):
            os.makedirs(savdir)
        plt.savefig(osp.join(savdir, 'proj-density.{0:s}.{1:04d}.png'
            .format(s.name, ds.num)),bbox_inches='tight')
    
    return plt.gcf()

def plt_all(s, num, dat=None, savfig=True):
   
    # load vtk and hst files
    if dat is None:
        ds = s.load_vtk(num=num)
        dat = ds.get_field(field=['density','pressure'],
                as_xarray=True)
    else:
        ds = s.load_vtk(num=num)
    hst = read_hst(s.files['hst'])
    time = ds.domain['time']*s.u.Myr
    axis_idx = dict(x=0, y=1, z=2)
    
    # prepare variables to be plotted
    mH = (1.00784*au.u).cgs.value
    dat['nH'] = (dat['density']*s.u.density.value)/(s.u.muH*mH)
    dat['pok'] = dat['pressure']*s.u.pok
    # T_1 = (p/k) / (rho m_p) is the temperature assuming mu=1
    # TODO dat['temperature'] should be computed from the tabulated mu.
    dat['T1'] = dat['pok']/(dat['nH']*s.u.muH)
    dat['surface_density_xy'] = (dat['density']*ds.domain['dx'][axis_idx['z']]).sum(dim='z')*s.u.Msun/s.u.pc**2
    dat['surface_density_xz'] = (dat['density']*ds.domain['dx'][axis_idx['z']]).sum(dim='y')*s.u.Msun/s.u.pc**2

    vol = (ds.domain['Lx'][0]*ds.domain['Lx'][1]*ds.domain['Lx'][2])
    hst['time'] *= s.u.Myr
    hst['mass'] *= (vol*s.u.Msun)
    hst['Mw'] *= (vol*s.u.Msun)
    hst['Mu'] *= (vol*s.u.Msun)
    hst['Mc'] *= (vol*s.u.Msun)
    hst['msp'] *= (vol*s.u.Msun)
    hst['sfr10'] *= (ds.domain['Lx'][axis_idx['x']]
            *ds.domain['Lx'][axis_idx['y']]*s.u.pc**2)/1e6
    hst['sfr40'] *= (ds.domain['Lx'][axis_idx['x']]
            *ds.domain['Lx'][axis_idx['y']]*s.u.pc**2)/1e6
    hst['sfr100'] *= (ds.domain['Lx'][axis_idx['x']]
            *ds.domain['Lx'][axis_idx['y']]*s.u.pc**2)/1e6

    fig = plt.figure(figsize=(40,18))
    gs = GridSpec(3,5,figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1:,0])
    ax3 = fig.add_subplot(gs[0,1])
    ax4 = fig.add_subplot(gs[1:,1])
    ax5 = fig.add_subplot(gs[0,2])
    ax6 = fig.add_subplot(gs[1:,2])
    ax7 = fig.add_subplot(gs[0,3])
    ax8 = fig.add_subplot(gs[0,4])
    ax9 = fig.add_subplot(gs[1,3:])
    ax10 = fig.add_subplot(gs[2,3:])

    (dat['nH'].interp(z=0)).plot.imshow(ax=ax1, norm=mpl.colors.LogNorm(),
            cmap='viridis', vmin=1e0, vmax=1e4)
    (dat['nH'].interp(y=0)).plot.imshow(ax=ax2, norm=mpl.colors.LogNorm(),
            cmap='viridis', vmin=1e-2, vmax=1e4)
    dat['surface_density_xy'].plot.imshow(ax=ax3, norm=mpl.colors.LogNorm(),
            cmap='pink_r', vmin=1e0, vmax=1e4)
    dat['surface_density_xz'].plot.imshow(ax=ax4, norm=mpl.colors.LogNorm(),
            cmap='pink_r', vmin=1e-2, vmax=1e5)
    (dat['T1'].interp(z=0)).plot.imshow(ax=ax5, norm=mpl.colors.LogNorm(),
            cmap='coolwarm', vmin=1e1, vmax=1e7)
    (dat['T1'].interp(y=0)).plot.imshow(ax=ax6, norm=mpl.colors.LogNorm(),
            cmap='coolwarm', vmin=1e1, vmax=1e7)
    histnP,xedgnP,yedgnP = np.histogram2d(
            np.log10(np.array(dat['nH']).flatten()),
            np.log10(np.array(dat['pok']).flatten()), bins=100,
            range=[[-3,5],[2,8]],density=True,
            weights=np.array(dat['nH']).flatten())
    histnT,xedgnT,yedgnT = np.histogram2d(
            np.log10(np.array(dat['nH']).flatten()),
            np.log10(np.array(dat['T1']).flatten()), bins=100,
            range=[[-3,5],[1,7]],density=True,
            weights=np.array(dat['nH']).flatten())
    ax7.imshow(histnP.T, origin='lower', norm=mpl.colors.LogNorm(),
            extent=[xedgnP[0], xedgnP[-1], yedgnP[0], yedgnP[-1]], cmap='Greys')
    ax8.imshow(histnT.T, origin='lower', norm=mpl.colors.LogNorm(),
            extent=[xedgnT[0], xedgnT[-1], yedgnT[0], yedgnT[-1]], cmap='Greys')
    ax7.set_xlabel(r'$n_{\rm H}\,[{\rm cm}^{-3}]$')
    ax7.set_ylabel(r'$P/k_{\rm B}\,[{\rm K\,cm^{-3}}]$')
    ax8.set_xlabel(r'$n_{\rm H}\,[{\rm cm}^{-3}]$')
    ax8.set_ylabel(r'$T\,[{\rm K}]$')

    ax9.semilogy(hst['time'], hst['sfr10'], 'r-', label='sfr10')
    ax9.semilogy(hst['time'], hst['sfr40'], 'g-', label='sfr40')
    ax9.semilogy(hst['time'], hst['sfr100'], 'm-', label='sfr100')
    ax9.set_xlabel("time ["+r"${\rm Myr}$"+"]")
    ax9.set_ylabel("star formation rate ["+r"$M_\odot\,{\rm yr}^{-1}$"+"]")
    ax9.set_ylim(1e-1,1e1)
    ax9.plot([time,time],[1e-1,1e1],'y-',lw=5)
    ax9.legend()

    ax10.semilogy(hst['time'], hst['Mc'], 'b-', label=r"$M_c$")
    ax10.semilogy(hst['time'], hst['Mu'], 'g-', label=r"$M_u$")
    ax10.semilogy(hst['time'], hst['Mw'], 'r-', label=r"$M_w$")
    ax10.semilogy(hst['time'], hst['mass'], 'k-', label=r"$M_{\rm tot}$")
    ax10.semilogy(hst['time'], hst['msp'], 'k--', label=r"$M_{\rm sp}$")
    ax10.set_xlabel("time ["+r"${\rm Myr}$"+"]")
    ax10.set_ylabel("mass ["+r"${M_\odot}$"+"]")
    ax10.set_ylim(1e6,1e8)
    ax10.plot([time,time],[1e6,1e8],'y-',lw=5)
    ax10.legend()

    for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]:
        ax.set_aspect('equal')

    plt.suptitle('{0:s}, time: {1:.1f} Myr'.format(s.name, time))
    
    if savfig:
        savdir = osp.join('./figures-all')
        if not os.path.exists(savdir):
            os.makedirs(savdir)
        plt.savefig(osp.join(savdir, 'all.{0:s}.{1:04d}.png'
            .format(s.name, ds.num)),bbox_inches='tight')
    
    return plt.gcf()

def mass_conservation(s, savfig=True):
    # load history file
    hst = read_hst(s.files['hst'])
    Lx=ds.domain['Lx'][0]
    Ly=ds.domain['Lx'][1]
    Lz=ds.domain['Lx'][2]
    vol = Lx*Ly*Lz

    time = hst['time']*s.u.Myr
    Mtot = (hst['mass']+hst['msp'])*vol*s.u.Msun
    outflow = ((-hst['F1_lower']+hst['F1_upper'])*Ly*Lz
              +(-hst['F2_lower']+hst['F2_upper'])*Lx*Lz
              +(-hst['F3_lower']+hst['F3_upper'])*Lx*Ly)*s.u.Msun/s.u.Myr
    plt.plot(time,(Mtot + (outflow)*time)/Mtot[0])
    plt.plot(time,np.ones(len(time)))
    return plt.gcf()
