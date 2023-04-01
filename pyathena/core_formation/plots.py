"""Collection of plotting scripts

Recommended function signature:
    def plot_something(s, ds, ax=None)
"""

# python modules
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import numpy as np
import xarray as xr
from pandas import read_csv
from pathlib import Path
import yt
import pickle

# pythena modules
from pyathena.core_formation import tools


def plot_tcoll_cores(s, pid, hw=0.25):
    # Load the progenitor iso of the particle pid
    fname = Path(s.basedir, 'tcoll_cores.p')
    with open(fname, 'rb') as handle:
        tcoll_cores = pickle.load(handle)
    iso = tcoll_cores[pid]

    # Load hdf5 snapshot at t = t_coll
    num = s.nums_tcoll[pid]
    ds = s.load_hdf5(num, load_method='pyathena')

    # load leaf dict at t = t_coll
    fname = Path(s.basedir, 'fiso.{:05d}.p'.format(num))
    with open(fname, 'rb') as handle:
        leaf_dict = pickle.load(handle)['leaf_dict']

    # Find the location of the core
    xc, yc, zc = tools.get_coords_iso(ds, iso)

    # Calculate radial profile
    rprf = tools.calculate_radial_profiles(ds, (xc, yc, zc), 2*hw)

    # create figure
    fig = plt.figure(figsize=(28, 21))
    gs = gridspec.GridSpec(3, 4, wspace=0.2, hspace=0.15)

    xaxis = dict(z='x', x='y', y='z')
    yaxis = dict(z='y', x='z', y='x')
    zaxis = dict(z='z', x='x', y='y')
    xlim = dict(z=(xc-hw, xc+hw),
                x=(yc-hw, yc+hw),
                y=(zc-hw, zc+hw))
    ylim = dict(z=(yc-hw, yc+hw),
                x=(zc-hw, zc+hw),
                y=(xc-hw, xc+hw))
    zlim = dict(z=(zc-hw, zc+hw),
                x=(xc-hw, xc+hw),
                y=(yc-hw, yc+hw))
    xlabel = dict(z=r'$x$', x=r'$y$', y=r'$z$')
    ylabel = dict(z=r'$y$', x=r'$z$', y=r'$x$')

    for i, prj_axis in enumerate(['z','x','y']):
        # 1. projections
        plt.sca(fig.add_subplot(gs[i,0]))
        img = plot_projection(s, ds, axis=prj_axis, add_colorbar=False)
        rec = plt.Rectangle((xlim[prj_axis][0], ylim[prj_axis][0]), 2*hw, 2*hw, fill=False, ec='r')
        plt.gca().add_artist(rec)
        plt.xlabel(xlabel[prj_axis])
        plt.ylabel(ylabel[prj_axis])

        # 2. zoom-in projections
        d = ds.sel({xaxis[prj_axis]:slice(*xlim[prj_axis]),
                    yaxis[prj_axis]:slice(*ylim[prj_axis]),
                    zaxis[prj_axis]:slice(*zlim[prj_axis])})
        plt.sca(fig.add_subplot(gs[i,1]))
        plot_projection(s, d, axis=prj_axis, add_colorbar=False)
        plt.xlim(xlim[prj_axis])
        plt.ylim(ylim[prj_axis])
        plt.xlabel(xlabel[prj_axis])
        plt.ylabel(ylabel[prj_axis])

        # 3. zoom-in projections for individual iso
        # load selected iso
        rho_ = tools.apply_fiso_mask(ds.dens, leaf_dict, iso, fill_value=0)
        Mcore = (rho_*s.domain['dx'].prod()).sum().data[()]
        Vcore = ((rho_>0).sum()*s.domain['dx'].prod())
        Rcore = (3*Vcore/(4*np.pi))**(1./3.)
        ds_iso = xr.Dataset(dict(dens=rho_))
        ds_iso = ds_iso.sel({xaxis[prj_axis]:slice(*xlim[prj_axis]),
                             yaxis[prj_axis]:slice(*ylim[prj_axis]),
                             zaxis[prj_axis]:slice(*zlim[prj_axis])})
        # load other isos
        leaf_dict_without_iso = {k: v for k, v in leaf_dict.items() if k != iso}
        rho_ = tools.apply_fiso_mask(ds.dens, leaf_dict_without_iso, fill_value=0)
        ds_bkgr = xr.Dataset(dict(dens=rho_))
        ds_bkgr = ds_bkgr.sel({xaxis[prj_axis]:slice(*xlim[prj_axis]),
                               yaxis[prj_axis]:slice(*ylim[prj_axis]),
                               zaxis[prj_axis]:slice(*zlim[prj_axis])})
        # plot
        plt.sca(fig.add_subplot(gs[i,2]))
        plot_projection(s, ds_bkgr, axis=prj_axis, add_colorbar=False, alpha=0.5, cmap='Greys')
        plot_projection(s, ds_iso, axis=prj_axis, add_colorbar=False)
        plt.xlim(xlim[prj_axis])
        plt.ylim(ylim[prj_axis])
        plt.xlabel(xlabel[prj_axis])
        plt.ylabel(ylabel[prj_axis])

    # 4. radial profiles
    # density
    plt.sca(fig.add_subplot(gs[0,3]))
    plt.loglog(rprf.r, rprf.rho, 'k-+')
    rhoLP = s.get_rhoLP(rprf.r)
    plt.loglog(rprf.r, rhoLP, 'k--')
    plt.axvline(Rcore, ls=':', c='k')
    plt.ylim(1e0, rhoLP[0])
    plt.xlabel(r'$r/L_{J,0}$')
    plt.ylabel(r'$\rho/\rho_0$')
    # annotations
    plt.text(0.5, 0.9, r'$t = {:.2f}$'.format(ds.Time)+r'$\,t_{J,0}$',
             transform=plt.gca().transAxes, backgroundcolor='w')
    plt.text(0.5, 0.8, r'$M = {:.2f}$'.format(Mcore)+r'$\,M_{J,0}$',
             transform=plt.gca().transAxes, backgroundcolor='w')
    plt.text(0.5, 0.7, r'$R = {:.2f}$'.format(Rcore)+r'$\,L_{J,0}$',
             transform=plt.gca().transAxes, backgroundcolor='w')
    # velocity
    plt.sca(fig.add_subplot(gs[1,3]))
    plt.semilogx(rprf.r, rprf.vel1, marker='+', label=r'$v_r$')
    plt.semilogx(rprf.r, rprf.vel2, marker='+', label=r'$v_\theta$')
    plt.semilogx(rprf.r, rprf.vel3, marker='+', label=r'$v_\phi$')
    plt.ylim(-3, 3)
    plt.xlabel(r'$r/L_{J,0}$')
    plt.ylabel(r'$v/c_s$')
    plt.legend()
    # velocity dispersion
    plt.sca(fig.add_subplot(gs[2,3]))
    plt.loglog(rprf.r, rprf.vel1_std, marker='+', label=r'$\sigma_r$')
    plt.loglog(rprf.r, rprf.vel2_std, marker='+', label=r'$\sigma_\theta$')
    plt.loglog(rprf.r, rprf.vel3_std, marker='+', label=r'$\sigma_\phi$')
    x0 = rprf.r[1]
    y0 = (rprf.vel1_std[1] + rprf.vel2_std[1] + rprf.vel3_std[1])/3
    plt.plot(rprf.r, y0*(rprf.r/x0)**0.5, 'k--')
    plt.plot(rprf.r, y0*(rprf.r/x0)**1, 'k--')
    plt.ylim(2e-1, 2e1)
    plt.xlabel(r'$r/L_{J,0}$')
    plt.ylabel(r'$\sigma/c_s$')
    plt.legend()

    return fig


def plot_sinkhistory(s, ds, pds):
    # find end time
    ds_end = s.load_hdf5(s.nums[-1], load_method='yt')
    tend = ds_end.current_time

    # create figure
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, wspace=0, hspace=0)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[0,2])
    ax3 = fig.add_subplot(gs[1,:])

    # plot projections
    for ax, axis in zip((ax0,ax1,ax2),('z','x','y')):
        plot_projection(s, ds, ax=ax, axis=axis, add_colorbar=False)
        ax.set_xticks([])
        ax.set_yticks([])
    ax0.plot(pds.x1, pds.x2, '*', color='b', ms=8, mew=0.5, alpha=0.7)
    ax1.plot(pds.x2, pds.x3, '*', color='b', ms=8, mew=0.5, alpha=0.7)
    ax2.plot(pds.x3, pds.x1, '*', color='b', ms=8, mew=0.5, alpha=0.7)

    # plot particle history
    plt.sca(ax3)
    for pid in s.pids:
        phst = s.load_parhst(pid)
        time = phst.time
        mass = phst.mass
        tslc = time < ds.current_time
        plt.plot(time[tslc], mass[tslc])
    plt.axvline(ds.current_time, linestyle=':', color='k', linewidth=0.5)
    plt.xlim(0.19, tend)
    plt.ylim(1e-2, 1e1)
    plt.yscale('log')
    plt.xlabel(r'$t/t_\mathrm{J,0}$')
    plt.ylabel(r'$M_*/M_\mathrm{J,0}$')
    return fig


def plot_projection(s, ds, field='dens', axis='z',
                    vmin=1e-1, vmax=2e2, cmap='pink_r', alpha=1,
                    ax=None, cax=None,
                    add_colorbar=True, transpose=False):
    # some domain informations
    xmin, ymin, zmin = s.domain['le']
    xmax, ymax, zmax = s.domain['re']
    Lx, Ly, Lz = s.domain['Lx']

    if isinstance(ds, xr.Dataset):
        # Reset the domain information, for the case when
        # ds is a part of the whole domain.
        xmin = ds.x[0] - 0.5*ds.dx
        ymin = ds.y[0] - 0.5*ds.dy
        zmin = ds.z[0] - 0.5*ds.dz
        xmax = ds.x[-1] + 0.5*ds.dx
        ymax = ds.y[-1] + 0.5*ds.dy
        zmax = ds.z[-1] + 0.5*ds.dz
        Lx = xmax - xmin
        Ly = ymax - ymin
        Lz = zmax - zmin

    wh = dict(zip(('x','y','z'), ((Ly, Lz), (Lz, Lx), (Lx, Ly))))
    extent = dict(zip(('x','y','z'), ((ymin,ymax,zmin,zmax),
                                      (zmin,zmax,xmin,xmax),
                                      (xmin,xmax,ymin,ymax))))
    permutations = dict(z=('y','x'), y=('x','z'), x=('z','y'))
    field_dict_yt = dict(dens=('athena_pp', 'dens'))
    field_dict_pyathena = dict(dens='dens')

    if isinstance(ds, yt.frontends.athena_pp.AthenaPPDataset):
        # create projection using yt
        fld = field_dict_yt[field]
        prj = ds.proj(fld, axis)
        prj = prj.to_frb(width=wh[axis][0], height=wh[axis][1], resolution=800)
        prj = np.array(prj[fld])
    elif isinstance(ds, xr.Dataset):
        fld = field_dict_pyathena[field]
        prj = ds[fld].integrate(axis).transpose(*permutations[axis]).to_numpy()
    else:
        TypeError("ds must be either yt or xarray dataset")

    if ax is not None:
        plt.sca(ax)
    if transpose:
        prj = prj.T
    img = plt.imshow(prj, norm=LogNorm(vmin, vmax), origin='lower',
                     extent=extent[axis], cmap=cmap, alpha=alpha)
    if add_colorbar:
        plt.colorbar(cax=cax)
    return img


def plot_Pspec(s, ds, ax=None):
    """Requires load_method='pyathena'"""
    # prepare frequencies
    from scipy.fft import fftn, fftfreq, fftshift, ifftn
    from pyathena.util.transform import groupby_bins
    Lx, Ly, Lz = s.domain['Lx']
    Nx, Ny, Nz = s.domain['Nx']
    dx, dy, dz = s.domain['Lx']/(s.domain['Nx'])
    kx = fftshift(2*np.pi*fftfreq(Nx, dx))
    ky = fftshift(2*np.pi*fftfreq(Ny, dy))
    kz = fftshift(2*np.pi*fftfreq(Nz, dz))
    # Do FFTs
    for axis in [1,2,3]:
        vel = ds['mom'+str(axis)]/ds.dens
        ds['vel'+str(axis)] = vel
        vhat = fftn(vel.data, vel.shape)*dx*dy*dz
        vhat = fftshift(vhat)
        ds['vhat'+str(axis)] = xr.DataArray(
            vhat,
            coords=dict(kz=('z',kz), ky=('y',ky), kx=('x',kx)),
            dims=('z','y','x'))
    # Calculate 3D power spectrum
    Pk = np.abs(ds.vhat1)**2 + np.abs(ds.vhat2)**2 + np.abs(ds.vhat3)**2
    # Set radial wavenumbers
    Pk.coords['k'] = np.sqrt(Pk.kz**2 + Pk.ky**2 + Pk.kx**2)
    kmin = np.sqrt((2*np.pi/Lx)**2 + (2*np.pi/Ly)**2 + (2*np.pi/Lz)**2)
    kmax = np.sqrt((2*np.pi/(2*dx))**2 + (2*np.pi/(2*dy))**2 + (2*np.pi/(2*dz))**2)
    lmin, lmax = 2*np.pi/kmax, 2*np.pi/kmin
    # Perform spherical average
    Pk = groupby_bins(Pk, 'k', np.linspace(kmin, kmax, min(Nx, Ny, Nz)//2))

    # Plot results
    if ax is not None:
        plt.sca(ax)

    Pk.plot(marker='+')
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(Pk.k, Pk[0]*(Pk.k/Pk.k.min())**(-4), 'b--', label=r'$n=-4$')
    plt.xlim(kmin, kmax)
    plt.xlabel(r'$k/L_J^{-1}$')
    plt.ylim(1e-7, 1e3)
    plt.ylabel(r'$P(k)=|\mathbf{v}(k)|^2$')
    plt.legend(loc='upper right')

    plt.twiny()
    plt.xscale('log')
    xticks = np.array([1e-1, 1e0, 1e1, 1e2])
    plt.gca().set_xticks(1/xticks)
    plt.gca().set_xticklabels(xticks)
    plt.xlim(1/lmax, 1/lmin)
    plt.xlabel(r'$l/L_J$')


def plot_PDF(s, ds, ax=None):
    """Requires load_method='pyathena'"""
    def gaussian(x, mu, sig):
        return np.exp(-(x - mu)**2 / (2*sig**2))/np.sqrt(2*np.pi*sig**2)

    rho = np.sort(ds.dens.data.flatten()).astype('float64')
    x = np.log(rho)
    mu_v = x.mean()
    mu_m = np.average(x, weights=rho)
    sig_v = x.std()
    sig_m = np.sqrt(np.average((x-mu_m)**2, weights=rho))

    if ax is not None:
        plt.sca(ax)

    # volume weighting
    fm, edges = np.histogram(np.log(rho), bins=100, density=True)
    centers = 0.5*(edges[1:] + edges[:-1])
    plt.stairs(fm, np.exp(edges), color='r')
    plt.plot(np.exp(centers), gaussian(centers, mu_v, sig_v), 'r:', lw=1)

    # mass weighting
    fm, edges = np.histogram(np.log(rho), weights=rho, bins=100, density=True)
    centers = 0.5*(edges[1:] + edges[:-1])
    plt.stairs(fm, np.exp(edges), color='b')
    plt.plot(np.exp(centers), gaussian(centers, mu_m, sig_m), 'b:', lw=1)

    # annotations
    plt.text(0.03, 0.9, r'$\mu = {:.2f}$'.format(mu_v),
             transform=plt.gca().transAxes, fontsize=15, c='r')
    plt.text(0.03, 0.8, r'$\sigma = {:.2f}$'.format(sig_v),
             transform=plt.gca().transAxes, fontsize=15, c='r')
    plt.text(0.03, 0.7, r'$\sigma^2/2 = {:.2f}$'.format(sig_v**2/2),
             transform=plt.gca().transAxes, fontsize=15, c='r')
    plt.text(0.03, 0.6, r'$e^\mu = {:.2f}$'.format(np.exp(mu_v)),
             transform=plt.gca().transAxes, fontsize=15, c='r')
    plt.text(0.72, 0.9, r'$\mu = {:.2f}$'.format(mu_m),
             transform=plt.gca().transAxes, fontsize=15, c='b')
    plt.text(0.72, 0.8, r'$\sigma = {:.2f}$'.format(sig_m),
             transform=plt.gca().transAxes, fontsize=15, c='b')
    plt.text(0.72, 0.7, r'$\sigma^2/2 = {:.2f}$'.format(sig_m**2/2),
             transform=plt.gca().transAxes, fontsize=15, c='b')
    plt.text(0.72, 0.6, r'$e^\mu = {:.2f}$'.format(np.exp(mu_m)),
             transform=plt.gca().transAxes, fontsize=15, c='b')
    plt.xscale('log')
    plt.xlabel(r'$\rho/\rho_0$')
    plt.ylabel('volume or mass fraction')
    plt.title(r'$t = {:.2f}$'.format(ds.Time))
    plt.xlim(1e-4, 1e4)
    plt.ylim(0, 0.3)
    plt.xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]);
