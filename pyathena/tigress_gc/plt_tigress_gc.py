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
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from astropy import units as au

from ..classic.cooling import coolftn
from ..io.read_hst import read_hst

def plt_proj_density(s, num, fig, savfig=True):
    """
    Create density projection
    """
    ax = fig.add_subplot(111)

    ds = s.load_vtk(num=num)
    dat = ds.get_field(field='density', as_xarray=True)
    dat['surface_density'] = (dat['density']*s.u.Msun/s.u.pc**3
            *ds.domain['dx'][2]*s.u.pc).sum(dim='z')
    dat['surface_density'].plot.imshow(ax=ax, norm=mpl.colors.LogNorm(),
            cmap='pink_r', vmin=1e0, vmax=1e4)
    ax.set_aspect('equal')
    fig.suptitle('{0:s}, time: {1:.1f} Myr'.format(s.name, ds.domain['time']*s.u.Myr))
    
    if savfig:
        savdir = osp.join('./figures-proj')
        if not os.path.exists(savdir):
            os.makedirs(savdir)
        fig.savefig(osp.join(savdir, 'proj-density.{0:s}.{1:04d}.png'
            .format(s.name, ds.num)),bbox_inches='tight')

def plt_all(s, num, fig, savfig=True):
    """
    Create large plot including density slice, density projection, temperature
    slice, phase diagram, star formation rate, and mass fractions.
    """
    # load vtk and hst files
    ds = s.load_vtk(num=num)
    dat = ds.get_field(field=['density','pressure'], as_xarray=True)
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

    gs = GridSpec(3,5,figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1:,0], sharex=ax1)
    ax3 = fig.add_subplot(gs[0,1])
    ax4 = fig.add_subplot(gs[1:,1], sharex=ax3)
    ax5 = fig.add_subplot(gs[0,2])
    ax6 = fig.add_subplot(gs[1:,2], sharex=ax5)
    ax7 = fig.add_subplot(gs[0,3])
    ax8 = fig.add_subplot(gs[0,4])
    ax9 = fig.add_subplot(gs[1,3:])
    ax10 = fig.add_subplot(gs[2,3:], sharex=ax9)
    cax1 = make_axes_locatable(ax1).append_axes('right', size='5%', pad=0.05)
    cax2 = make_axes_locatable(ax2).append_axes('right', size='5%', pad=0.05)
    cax3 = make_axes_locatable(ax3).append_axes('right', size='5%', pad=0.05)
    cax4 = make_axes_locatable(ax4).append_axes('right', size='5%', pad=0.05)
    cax5 = make_axes_locatable(ax5).append_axes('right', size='5%', pad=0.05)
    cax6 = make_axes_locatable(ax6).append_axes('right', size='5%', pad=0.05)

    (dat['nH'].interp(z=0)).plot.imshow(ax=ax1, norm=mpl.colors.LogNorm(),
            cmap='viridis', vmin=1e0, vmax=1e4, cbar_ax=cax1)
    (dat['nH'].interp(y=0)).plot.imshow(ax=ax2, norm=mpl.colors.LogNorm(),
            cmap='viridis', vmin=1e-2, vmax=1e4, cbar_ax=cax2)
    dat['surface_density_xy'].plot.imshow(ax=ax3, norm=mpl.colors.LogNorm(),
            cmap='pink_r', vmin=1e0, vmax=1e4, cbar_ax=cax3)
    dat['surface_density_xz'].plot.imshow(ax=ax4, norm=mpl.colors.LogNorm(),
            cmap='pink_r', vmin=1e-2, vmax=1e5, cbar_ax=cax4)
    (dat['T1'].interp(z=0)).plot.imshow(ax=ax5, norm=mpl.colors.LogNorm(),
            cmap='coolwarm', vmin=1e1, vmax=1e7, cbar_ax=cax5)
    (dat['T1'].interp(y=0)).plot.imshow(ax=ax6, norm=mpl.colors.LogNorm(),
            cmap='coolwarm', vmin=1e1, vmax=1e7, cbar_ax=cax6)
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

    for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
        ax.set_aspect('equal')

    fig.suptitle('{0:s}, time: {1:.1f} Myr'.format(s.name, time), fontsize=30, x=.5, y=.93)
    
    if savfig:
        savdir = osp.join('./figures-all')
        if not os.path.exists(savdir):
            os.makedirs(savdir)
        fig.savefig(osp.join(savdir, 'all.{0:s}.{1:04d}.png'
            .format(s.name, ds.num)),bbox_inches='tight')

def mass_flux(s, fig):
    """
    Check mass conservation of TIGRESS-GC simulation by measuring the net
    outflow rate and the total mass of the simulation box.
    Note that the nozzle inflow is already taken into account in the
    hst['F2_lower'] and hst['F2_upper'].

    ISSUE: We cannot check mass conservation accurately unless we use very 
    short time step between the history dump, because we have to time-integrate
    surface flux numerically - SMOON
    """
    # load history file
    hst = read_hst(s.files['hst'])
    Lx=s.par['domain1']['x1max']-s.par['domain1']['x1min']
    Ly=s.par['domain1']['x2max']-s.par['domain1']['x2min']
    Lz=s.par['domain1']['x3max']-s.par['domain1']['x3min']
    vol = Lx*Ly*Lz

    time = hst['time']*s.u.Myr
    sfr = hst['sfr10']*Lx*Ly*s.u.pc**2/1e6
    Mtot = (hst['mass']+hst['msp'])*vol*s.u.Msun
    inflow = np.ones(len(time))
    xflux = (-hst['F1_lower']+hst['F1_upper'])*Ly*Lz*s.u.Msun/s.u.Myr/1e6
    yflux = (-hst['F2_lower']+hst['F2_upper'])*Lx*Lz*s.u.Msun/s.u.Myr/1e6 + inflow
    zflux = (-hst['F3_lower']+hst['F3_upper'])*Lx*Ly*s.u.Msun/s.u.Myr/1e6

    ax = fig.add_subplot(111)
    ax.semilogy(time, sfr, 'k-', label='SFR')
    ax.semilogy(time, xflux, 'b-', label='x1flux')
    ax.semilogy(time, yflux, 'r-', label='x2flux')
    ax.semilogy(time, zflux, 'g-', label='x3flux')
    ax.semilogy(time, inflow, 'k--', label='inflow rate')
    ax.semilogy(time, sfr+xflux+yflux+zflux, 'k:', label='SFR+fluxes')

def mass_conservation(s, fig):
    # load history file
    hst = read_hst(s.files['hst'])
    Lx=s.par['domain1']['x1max']-s.par['domain1']['x1min']
    Ly=s.par['domain1']['x2max']-s.par['domain1']['x2min']
    Lz=s.par['domain1']['x3max']-s.par['domain1']['x3min']
    vol = Lx*Ly*Lz

    time = hst['time']*s.u.Myr
    sfr = hst['sfr10']*Lx*Ly*s.u.pc**2/1e6
    Mtot = (hst['mass']+hst['msp'])*vol*s.u.Msun
    Msp_left = hst['msp_left']*vol*s.u.Msun
    xflux = (-hst['F1_lower']+hst['F1_upper'])*Ly*Lz*s.u.Msun/s.u.Myr
    yflux = (-hst['F2_lower']+hst['F2_upper'])*Lx*Lz*s.u.Msun/s.u.Myr
    zflux = (-hst['F3_lower']+hst['F3_upper'])*Lx*Ly*s.u.Msun/s.u.Myr
    outflow = xflux+yflux+zflux

    ax.semilogy(time, Mtot, 'k-')
    ax.semilogy(time, Mtot[0]-outflow*time-Msp_left, 'k--')
    #ax.plot(time,(Mtot + (outflow)*time)/Mtot[0])
