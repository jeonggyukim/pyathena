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
from pyathena.core_formation import tes
from grid_dendro import dendrogram
from grid_dendro import energy

def plot_energies(s, ds, leaves, nid, ax=None):
    if ax is not None:
        plt.sca(ax)
    prims = dict(rho=ds.dens.to_numpy(),
                 vel1=(ds.mom1/ds.dens).to_numpy(),
                 vel2=(ds.mom2/ds.dens).to_numpy(),
                 vel3=(ds.mom3/ds.dens).to_numpy(),
                 prs=s.cs**2*ds.dens.to_numpy(),
                 phi=ds.phigas.to_numpy())
    reff, engs = energy.calculate_cumulative_energies(prims, s.dV, leaves, nid)
    plt.plot(reff, engs['ethm'], label='thermal')
    plt.plot(reff, engs['ekin'], label='kinetic')
    plt.plot(reff, engs['egrv'], label='gravitational')
    plt.plot(reff, engs['etot'], label='total')
    plt.axhline(0, linestyle=':')
    plt.legend(loc='lower left')

def plot_central_density_evolution(s, ax=None):
    rho_crit_KM05 = tools.get_rhocrit_KM05(s.sonic_length)
    if ax is not None:
        plt.sca(ax)

    for pid in s.pids:
        rprf = s.rprofs[pid]
        rprf.rho.isel(r=0).plot(label=f'par{pid}')
        plt.yscale('log')
        plt.ylim(1e1, s.get_rhoLP(0.5*s.dx))
    plt.axhline(rho_crit_KM05, linestyle='--')
    plt.text(s.rprofs[1].t.min(), rho_crit_KM05, r"$\rho_\mathrm{crit, KM05}$", fontsize=18,
             ha='left', va='bottom')
    plt.text(s.rprofs[1].t.min(), 14.1*rho_crit_KM05, r"$14.1\rho_\mathrm{crit, KM05}$", fontsize=18,
             ha='left', va='bottom')
    plt.axhline(14.1*rho_crit_KM05, linestyle='--', lw=1)
    plt.legend(loc=(1.01, 0))
    plt.ylabel(r'$\rho_c/\rho_0$')
    plt.xlabel(r'$t/t_J$')
    plt.title(s.basename)

def plot_core_evolution(s, pid, num, hw=0.25, emin=None, emax=None, rmax=None):
    # Load the progenitor GRID-core of this particle.
    if num > s.nums_tcoll[pid]:
        raise ValueError("num must be smaller than num_tcoll")
    core = s.cores[pid].loc[num]

    # Load hdf5 snapshot at t = t_coll
    ds = s.load_hdf5(num, load_method='pyathena')

    # Load leaf dict at t = t_coll
    leaves = s.load_leaves(num)

    # Find the location of the core
    xc, yc, zc = tools.get_coords_node(ds, core.nid)

    # Calculate radial profile
    rprf = s.rprofs[pid].sel(t=ds.Time, method='nearest')

    # Create figure
    fig = plt.figure(figsize=(35, 21))
    gs = gridspec.GridSpec(3, 5, wspace=0.2, hspace=0.15)

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
        # 1. Projections
        plt.sca(fig.add_subplot(gs[i,0]))
        img = plot_projection(s, ds, axis=prj_axis, add_colorbar=False)
        rec = plt.Rectangle((xlim[prj_axis][0], ylim[prj_axis][0]), 2*hw, 2*hw, fill=False, ec='r')
        plt.gca().add_artist(rec)
        plt.xlabel(xlabel[prj_axis])
        plt.ylabel(ylabel[prj_axis])

        # 2. Zoom-in projections
        d, _ = tools.recenter_dataset(ds, (xc, yc, zc))
        d = d.sel(x=slice(-hw, hw), y=slice(-hw, hw), z=slice(-hw, hw))
        plt.sca(fig.add_subplot(gs[i,1]))
        plot_projection(s, d, axis=prj_axis, add_colorbar=False)
        plt.xlim(-hw, hw)
        plt.ylim(-hw, hw)
        plt.xlabel(xlabel[prj_axis])
        plt.ylabel(ylabel[prj_axis])

        # 3. Zoom-in projections for individual core
        # Load selected core
        rho_ = dendrogram.filter_by_node(ds.dens, leaves, core.nid, fill_value=0)
        ds_core = xr.Dataset(data_vars=dict(dens=rho_), attrs=ds.attrs)
        ds_core, _ = tools.recenter_dataset(ds_core, (xc, yc, zc))
        ds_core = ds_core.sel(x=slice(-hw, hw), y=slice(-hw, hw), z=slice(-hw, hw))

        # Load other cores
        other_cores = {k: v for k, v in leaves.items() if k != core.nid}
        rho_ = dendrogram.filter_by_node(ds.dens, other_cores, fill_value=0)
        ds_others = xr.Dataset(data_vars=dict(dens=rho_), attrs=ds.attrs)
        ds_others, _ = tools.recenter_dataset(ds_others, (xc, yc, zc))
        ds_others = ds_others.sel(x=slice(-hw, hw), y=slice(-hw, hw), z=slice(-hw, hw))

        # Plot
        plt.sca(fig.add_subplot(gs[i,2]))
        plot_projection(s, ds_others, axis=prj_axis, add_colorbar=False, alpha=0.5, cmap='Greys')
        plot_projection(s, ds_core, axis=prj_axis, add_colorbar=False)
        plt.xlim(-hw, hw)
        plt.ylim(-hw, hw)
        plt.xlabel(xlabel[prj_axis])
        plt.ylabel(ylabel[prj_axis])

    # 4. Radial profiles
    # Density
    plt.sca(fig.add_subplot(gs[0,3]))
    plt.loglog(rprf.r, rprf.rho, 'k-+')
    rhoLP = s.get_rhoLP(rprf.r)
    plt.loglog(rprf.r, rhoLP, 'k--')

    # overplot critical tes
    rhoc = rprf.rho.isel(r=0).data[()]
    rhoe = core.edge_density
    if not np.isnan(rhoe):
        ts = tes.TESe(p=core.pindex, xi_s=np.sqrt(rhoe)*core.sonic_radius)
        u0 = np.log(rhoc/rhoe)
        xi_max = ts.get_radius(u0)
        xi_min = np.sqrt(rhoe)*rprf.r[0]
        xi = np.logspace(np.log10(xi_min), np.log10(xi_max))
        u, du = ts.solve(xi, u0)
        plt.plot(xi/np.sqrt(rhoe), rhoe*np.exp(u), 'r--', lw=1)

    # overplot critical BE
    ts = tes.TESe()
    u_c, r_c, m_c = ts.get_crit()
    rhoe = rhoc*np.exp(-u_c)
    xi_min = np.sqrt(rhoe)*rprf.r[0]
    xi = np.logspace(np.log10(xi_min), np.log10(r_c))
    u, du = ts.solve(xi, u_c)
    plt.plot(xi/np.sqrt(rhoe), rhoe*np.exp(u), 'b:', lw=1)

    plt.axvline(core.radius, ls=':', c='k')
    plt.axvline(core.critical_radius, ls='--', c='k')
    plt.xlim(rprf.r[0]/2, 2*hw)
    plt.ylim(1e0, rhoLP[0])
    plt.xlabel(r'$r/L_{J,0}$')
    plt.ylabel(r'$\rho/\rho_0$')
    # Annotations
    plt.text(0.5, 0.9, r'$t = {:.3f}$'.format(ds.Time)+r'$\,t_{J,0}$',
             transform=plt.gca().transAxes, backgroundcolor='w')
    plt.text(0.5, 0.8, r'$M = {:.2f}$'.format(core.mass)+r'$\,M_{J,0}$',
             transform=plt.gca().transAxes, backgroundcolor='w')
    plt.text(0.5, 0.7, r'$R = {:.2f}$'.format(core.radius)+r'$\,L_{J,0}$',
             transform=plt.gca().transAxes, backgroundcolor='w')
    # Velocities
    plt.sca(fig.add_subplot(gs[1,3]))
    plt.plot(rprf.r, rprf.vel1, marker='+', label=r'$v_r$')
    plt.plot(rprf.r, rprf.vel2, marker='+', label=r'$v_\theta$')
    plt.plot(rprf.r, rprf.vel3, marker='+', label=r'$v_\phi$')
    plt.axvline(core.radius, ls=':', c='k')
    plt.axvline(core.critical_radius, ls='--', c='k')
    plt.axhline(0, ls=':')
    plt.xlim(0, hw)
    plt.ylim(-2.5, 1.5)
    plt.xlabel(r'$r/L_{J,0}$')
    plt.ylabel(r'$\left<v\right>/c_s$')
    plt.legend()

    # Velocity dispersions
    plt.sca(fig.add_subplot(gs[2,3]))
    plt.loglog(rprf.r, np.sqrt(rprf.vel1_sq), marker='+', label=r'$v_r$')
    plt.loglog(rprf.r, np.sqrt(rprf.vel2_sq), marker='+', label=r'$v_\theta$')
    plt.loglog(rprf.r, np.sqrt(rprf.vel3_sq), marker='+', label=r'$v_\phi$')
    plt.plot(rprf.r, (rprf.r/(s.sonic_length/2))**0.5, 'k--')
    plt.plot(rprf.r, (rprf.r/(s.sonic_length/2))**1, 'k--')

    # overplot linear fit
    if not np.isnan(core.sonic_radius):
        plt.plot(rprf.r, (rprf.r/core.sonic_radius)**(core.pindex), 'r--', lw=1)

    plt.axvline(core.radius, ls=':', c='k')
    plt.axvline(core.critical_radius, ls='--', c='k')
    plt.xlim(rprf.r[0], 2*hw)
    plt.ylim(1e-1, 1e1)
    plt.xlabel(r'$r/L_{J,0}$')
    plt.ylabel(r'$\left<v^2\right>^{1/2}/c_s$')
    plt.legend()

    # 5. Energies
    plt.sca(fig.add_subplot(gs[0,4]))
    plot_energies(s, ds, leaves, core.nid)
    if emin is not None and emax is not None:
        plt.ylim(emin, emax)
    if rmax is not None:
        plt.xlim(0, rmax)

    # 6. Accelerations
    plt.sca(fig.add_subplot(gs[1,4]))
    plot_forces(s, rprf)
    plt.title('')
    plt.axvline(core.radius, ls=':', c='k')
    plt.axvline(core.critical_radius, ls='--', c='k')
    plt.xlim(0, hw)
    plt.legend(fontsize=15)

    return fig


def plot_forces(s, rprf, ax=None, cumulative=False, xlim=(0, 0.2), ylim=(-20, 50), ylabel='acceleration'):
    peff = rprf.rho*(rprf.vel1_sq_mw + s.cs**2)
    stress = rprf.rho*(-2*rprf.vel1_sq_mw + rprf.vel2_sq_mw + rprf.vel3_sq_mw)

    if ax is not None:
        plt.sca(ax)

    if cumulative:
        istart = 2
        slicer = slice(istart, rprf.dims['r'])

        column_density = rprf.rho.isel(r=slicer).cumulative_integrate('r')
        f_p = (peff - peff.isel(r=istart)) / column_density
        f_geo = (stress/rprf.r).isel(r=slicer).cumulative_integrate('r') / column_density
        f_g = (rprf.rho*(rprf.ggas1_mw+rprf.gstar1_mw)).isel(r=slicer).cumulative_integrate('r') / column_density
        fnet = f_g - f_p - f_geo

        f_p.plot(marker='+', label='pressure')
        (f_p + f_geo).plot(marker='o', label='pressure + geometric')
        f_g.plot(marker='x', label='gravity')
        fnet.plot(marker='+', label='net radial force')

        plt.axhline(0, linestyle=':')
        plt.ylabel(ylabel)
        plt.xlim(xlim)
        plt.ylim(ylim)
    else:
        f_pthm = -rprf.grad_pthm1 / rprf.rho
        f_ptrb = -rprf.grad_ptrb1 / rprf.rho
        f_geo = stress / rprf.r / rprf.rho
        f_grav = rprf.ggas1_mw + rprf.gstar1_mw
        f_net = f_grav + f_pthm + f_ptrb + f_geo

        f_pthm.plot(lw=1, color='orange', label=r'$-\partial_r P_\mathrm{thm}$')
        f_ptrb.plot(lw=1, color='deepskyblue', label=r'$-\partial_r P_\mathrm{trb}$')
        f_geo.plot(lw=1, color='limegreen', label=r'$f_\mathrm{geo}$')
        (f_pthm + f_ptrb + f_geo).plot(marker='+', color='blue', lw=1, label='total')
        (-f_grav).plot(marker='x', ls='--', color='red', lw=1, label='gravity')
        (-f_net).plot(marker='*', color='k', lw=1, label='net inward force')

        # Overplot -GM/r^2
        Mr = (4*np.pi*rprf.rho*rprf.r**2).cumulative_integrate('r')
        gr = s.G*Mr/rprf.r**2
        gr.plot(color='r', lw=1, ls='--')

        plt.axhline(0, linestyle=':')
        plt.ylabel(ylabel)
        plt.xlim(xlim)
        plt.ylim(ylim)

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
        xmin = ds.x[0] - 0.5*s.dx
        ymin = ds.y[0] - 0.5*s.dy
        zmin = ds.z[0] - 0.5*s.dz
        xmax = ds.x[-1] + 0.5*s.dx
        ymax = ds.y[-1] + 0.5*s.dy
        zmax = ds.z[-1] + 0.5*s.dz
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


def plot_Pspec(s, ds, ax=None, ax_twin=None):
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
    if ax_twin is not None:
        plt.sca(ax_twin)
    else:
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
