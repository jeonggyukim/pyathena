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
import yt

# pythena modules
from pyathena.core_formation import tools
from pyathena.core_formation import tes
from grid_dendro import energy


def plot_projection(s, ds, field='dens', axis='z', op='sum',
                    vmin=1e-1, vmax=2e2, cmap='pink_r', alpha=1,
                    ax=None, cax=None, noplot=False,
                    add_colorbar=True, transpose=False):
    """Plot projection of the selected variable along the given axis.

    Parameters
    ----------
    s : LoadSimCoreFormation
        Object containing simulation metadata.
    ds : yt.frontends.athena_pp.AthenaPPDataset or xarray.Dataset
        Object containing fluid variables.
    field : str, optional
        Variable to plot.
    axis : str, optional
        Axis to project.
    vmin : float, optional
        Minimum color range.
    vmax : float, optional
        Maximum color range.
    cmap : str, optional
        Color map.
    alpha : float, optional
        Transparency.
    ax : matplotlib.axes, optional
        Axes to draw contours.
    cax : matplotlib.axes, optional
        Axes to draw color bar.
    add_colorbar : bool, optional
        If true, add color bar.
    transpose : bool, optional
        If true, transpose x and y axis.
    """
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

    wh = dict(zip(('x', 'y', 'z'), ((Ly, Lz), (Lz, Lx), (Lx, Ly))))
    extent = dict(zip(('x', 'y', 'z'), ((ymin, ymax, zmin, zmax),
                                        (zmin, zmax, xmin, xmax),
                                        (xmin, xmax, ymin, ymax))))
    permutations = dict(z=('y', 'x'), y=('x', 'z'), x=('z', 'y'))
    field_dict_yt = dict(dens=('athena_pp', 'dens'))
    field_dict_pyathena = dict(dens='dens', mask='mask')

    if isinstance(ds, yt.frontends.athena_pp.AthenaPPDataset):
        # create projection using yt
        fld = field_dict_yt[field]
        prj = ds.proj(fld, axis)
        prj = prj.to_frb(width=wh[axis][0], height=wh[axis][1], resolution=800)
        prj = np.array(prj[fld])
    elif isinstance(ds, xr.Dataset):
        fld = field_dict_pyathena[field]
        if op == 'sum':
            prj = ds[fld].integrate(axis).transpose(*permutations[axis])
        elif op == 'max':
            prj = ds[fld].max(axis).transpose(*permutations[axis])
        if noplot:
            return prj
        else:
            prj = prj.to_numpy()
    else:
        TypeError("ds must be either yt or xarray dataset")

    if ax is not None:
        plt.sca(ax)
    if transpose:
        prj = prj.T
        extent = {k: v[2:] + v[0:2] for k, v in extent.items()}
    img = plt.imshow(prj, norm=LogNorm(vmin, vmax), origin='lower',
                     extent=extent[axis], cmap=cmap, alpha=alpha)
    if add_colorbar:
        plt.colorbar(cax=cax)
    return img


def plot_energies(s, ds, gd, nid, ax=None):
    if ax is not None:
        plt.sca(ax)
    data = dict(rho=ds.dens.to_numpy(),
                vel1=(ds.mom1/ds.dens).to_numpy(),
                vel2=(ds.mom2/ds.dens).to_numpy(),
                vel3=(ds.mom3/ds.dens).to_numpy(),
                prs=s.cs**2*ds.dens.to_numpy(),
                phi=ds.phigas.to_numpy(),
                dvol=s.dV)
    reff, engs = energy.calculate_cumulative_energies(gd, data, nid)
    plt.plot(reff, engs['ethm'], label='thermal')
    plt.plot(reff, engs['ekin'], label='kinetic')
    plt.plot(reff, engs['egrv'], label='gravitational')
    plt.plot(reff, engs['etot'], label='total')
    plt.axhline(0, linestyle=':')
    plt.legend(loc='lower left')


def plot_grid_dendro_contours(s, gd, nodes, coords, axis='z', color='k',
                              lw=0.5, ax=None, transpose=False, recenter=None,
                              select=None):
    """Draw contours at the boundary of GRID-dendro objects

    Parameters
    ----------
    s : LoadSimCoreFormation
        Object containing simulation metadata.
    gd : grid_dendro.dendrogram.Dendrogram
        GRID-dendro dendrogram instance.
    nodes : int or array of ints
        ID of selected GRID-dendro nodes
    coords : xarray.core.coordinates.DatasetCoordinates
        xarray coordinate instance.
    axis : str, optional
        Axis to project.
    color : str, optional
        Contour line color.
    lw : float, optional
        Contour line width.
    ax : matplotlib.axes, optional
        Axes to draw contours.
    transpose : bool, optional
        If true, transpose x and y axis.
    recenter : tuple, optional
        New (x, y, z) coordinates of the center.
    select : dict, optional
        Selected region to slice data. If recenter is True, the selected
        region is understood in the recentered coordinates.
    """
    # some domain informations
    xmin, xmax = coords['x'].min(), coords['x'].max()
    ymin, ymax = coords['y'].min(), coords['y'].max()
    zmin, zmax = coords['z'].min(), coords['z'].max()
    dims = coords.dims
    extent = dict(zip(('x', 'y', 'z'), ((ymin, ymax, zmin, zmax),
                                        (zmin, zmax, xmin, xmax),
                                        (xmin, xmax, ymin, ymax))))
    permutations = dict(z=('y', 'x'), y=('x', 'z'), x=('z', 'y'))

    if isinstance(nodes, (int, np.int32, np.int64)):
        nodes = [nodes,]
    for nd in nodes:
        mask = xr.DataArray(np.ones([dims['z'], dims['y'], dims['x']], dtype=bool), coords=coords)
        mask = gd.filter_data(mask, nd, fill_value=0)
        if recenter is not None:
            mask, _ = tools.recenter_dataset(mask, recenter)
        if select is not None:
            mask = mask.sel(select)

        mask = mask.max(dim=axis)

        if mask.max() == 0 or mask.min() == 1:
            # If a core is outside the selected region, or the
            # selected region is entirely contained in a core,
            # no contour can be drawn.
            continue

        mask = mask.transpose(*permutations[axis])

        if ax is not None:
            plt.sca(ax)
        if transpose:
            mask = mask.T
            extent = {k: v[2:] + v[0:2] for k, v in extent.items()}

        mask.plot.contour(levels=[0.5], linewidths=lw, colors=color,
                          add_labels=False)
    plt.xlim(extent[axis][0], extent[axis][1])
    plt.ylim(extent[axis][2], extent[axis][3])


def plot_core_evolution(s, pid, num, hw=0.25, emin=None, emax=None, rmax=None):
    # Load the progenitor GRID-core of this particle.
    if num > s.tcoll_cores.loc[pid].num:
        raise ValueError("num must be smaller than num_tcoll")
    core = s.cores[pid].loc[num]

    # Load hdf5 snapshot at t = t_coll
    ds = s.load_hdf5(num, load_method='pyathena')

    # Load leaf dict at t = t_coll
    gd = s.load_dendrogram(num)

    # Find the location of the core
    xc, yc, zc = tools.get_coords_node(s, core.nid)

    # Calculate radial profile
    rprf = s.rprofs[pid].sel(num=num)

    # Create figure
    fig = plt.figure(figsize=(28, 21))
    gs = gridspec.GridSpec(3, 4, wspace=0.23, hspace=0.15)

    xlim = dict(z=(xc-hw, xc+hw),
                x=(yc-hw, yc+hw),
                y=(zc-hw, zc+hw))
    ylim = dict(z=(yc-hw, yc+hw),
                x=(zc-hw, zc+hw),
                y=(xc-hw, xc+hw))
    xlabel = dict(z=r'$x$', x=r'$y$', y=r'$z$')
    ylabel = dict(z=r'$y$', x=r'$z$', y=r'$x$')

    for i, prj_axis in enumerate(['z', 'x', 'y']):
        # 1. Projections
        plt.sca(fig.add_subplot(gs[i, 0]))
        plot_projection(s, ds, axis=prj_axis, add_colorbar=False)

#        # Overplot contours of leaves
#        plot_grid_dendro_contours(s, gd, gd.leaves, ds.coords, axis=prj_axis,
#                                  color='tab:gray')
#        # Overplot contours of parents of leaves, excluding trunk.
#        nodes = list(gd.leaves)
#        for nd in gd.leaves:
#            nodes.append(gd.parent[nd])
#        nodes = set(nodes)
#        if gd.trunk in nodes:
#            nodes.remove(gd.trunk)
#        plot_grid_dendro_contours(s, gd, nodes, ds.coords, axis=prj_axis,
#                                  color='tab:gray')

        rec = plt.Rectangle((xlim[prj_axis][0], ylim[prj_axis][0]),
                            2*hw, 2*hw, fill=False, ec='r')
        plt.gca().add_artist(rec)
        plt.xlabel(xlabel[prj_axis])
        plt.ylabel(ylabel[prj_axis])

        # 2. Zoom-in projections
        sel = dict(x=slice(-hw, hw), y=slice(-hw, hw), z=slice(-hw, hw))
        d, _ = tools.recenter_dataset(ds, (xc, yc, zc))
        d = d.sel(sel)
        plt.sca(fig.add_subplot(gs[i, 1]))
        plot_projection(s, d, axis=prj_axis, add_colorbar=False)

        parent = gd.parent[core.nid]
        grandparent = gd.parent[parent]
        nodes = list(gd.descendants[grandparent])
        nodes.append(grandparent)
        plot_grid_dendro_contours(s, gd, nodes, ds.coords, axis=prj_axis,
                                  recenter=(xc, yc, zc), select=sel,
                                  color='tab:gray')
        # Overplot critical radius
        if not np.isnan(core.critical_radius):
            crcl = plt.Circle((0, 0), core.critical_radius, fill=False,
                              ls='--', color='tab:gray')
            plt.gca().add_artist(crcl)
        if (not np.isnan(core.sonic_radius)) and core.sonic_radius < s.Lbox:
            crcl = plt.Circle((0, 0), core.sonic_radius, fill=False,
                              ls=':', color='tab:gray')
            plt.gca().add_artist(crcl)
        plt.xlim(-hw, hw)
        plt.ylim(-hw, hw)
        plt.xlabel(xlabel[prj_axis])
        plt.ylabel(ylabel[prj_axis])

    # 4. Radial profiles
    # Density
    plt.sca(fig.add_subplot(gs[0, 2]))
    plt.loglog(rprf.r, rprf.rho, 'k-+')
    rhoLP = s.get_rhoLP(rprf.r)
    plt.loglog(rprf.r, rhoLP, 'k--')

    # overplot critical tes
    rhoc = rprf.rho.isel(r=0).data[()]
    LJ_c = 1.0/np.sqrt(rhoc)
    xi_min = rprf.r.isel(r=0).data[()]/LJ_c
    if not np.isnan(core.critical_radius):
        ts = tes.TESc(p=core.pindex, xi_s=core.sonic_radius/LJ_c)
        xi_max = core.critical_radius/LJ_c
        xi = np.logspace(np.log10(xi_min), np.log10(xi_max))
        u, du = ts.solve(xi)
        plt.plot(xi*LJ_c, rhoc*np.exp(u), 'r--', lw=1)

    # overplot critical BE
    ts = tes.TESc()
    xi_max = ts.get_crit()
    xi = np.logspace(np.log10(xi_min), np.log10(xi_max))
    u, du = ts.solve(xi)
    plt.plot(xi*LJ_c, rhoc*np.exp(u), 'b:', lw=1)

    # overplot radius of the parent core
    nid = gd.parent[core.nid]
    vol = len(gd.get_all_descendant_cells(nid))*s.dV
    rparent = (3*vol/(4*np.pi))**(1./3.)
    plt.axvline(core.radius, c='tab:gray', lw=1)
    plt.axvline(rparent, c='tab:gray', lw=1)
    plt.axvline(core.critical_radius, ls='--', c='tab:gray')
    plt.axvline(core.sonic_radius, ls=':', c='tab:gray')
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
    plt.sca(fig.add_subplot(gs[1, 2]))
    plt.plot(rprf.r, rprf.vel1_mw, marker='+', label=r'$v_r$')
    plt.plot(rprf.r, rprf.vel2_mw, marker='+', label=r'$v_\theta$')
    plt.plot(rprf.r, rprf.vel3_mw, marker='+', label=r'$v_\phi$')
    ln1 = plt.axvline(core.radius, c='tab:gray', lw=1)
    plt.axvline(rparent, c='tab:gray', lw=1)
    ln2 = plt.axvline(core.critical_radius, ls='--', c='tab:gray')
    ln3 = plt.axvline(core.sonic_radius, ls=':', c='tab:gray')
    plt.axhline(0, ls=':')
    plt.xlim(0, hw)
    plt.ylim(-2.5, 1.5)
    plt.xlabel(r'$r/L_{J,0}$')
    plt.ylabel(r'$\left<v\right>/c_s$')
    lgd = plt.legend([ln1, ln2, ln3], [r'$R_\mathrm{tidal}$',
                                       r'$R_\mathrm{crit}$',
                                       r'$R_\mathrm{sonic}$'],
                     loc='lower right')
    plt.legend(loc='upper right')
    plt.gca().add_artist(lgd)

    # Velocity dispersions
    plt.sca(fig.add_subplot(gs[2, 2]))
    plt.loglog(rprf.r, np.sqrt(rprf.dvel1_sq_mw), marker='+', label=r'$v_r$')
    plt.loglog(rprf.r, np.sqrt(rprf.dvel2_sq_mw), marker='+',
               label=r'$v_\theta$')
    plt.loglog(rprf.r, np.sqrt(rprf.dvel3_sq_mw), marker='+',
               label=r'$v_\phi$')
    plt.plot(rprf.r, (rprf.r/(s.sonic_length/2))**0.5, 'k--')
    plt.plot(rprf.r, (rprf.r/(s.sonic_length/2))**1, 'k--')

    # overplot linear fit
    if not np.isnan(core.sonic_radius):
        plt.plot(rprf.r, (rprf.r/core.sonic_radius)**(core.pindex), 'r--',
                 lw=1)

    plt.axvline(core.radius, c='tab:gray', lw=1)
    plt.axvline(rparent, c='tab:gray', lw=1)
    plt.axvline(core.critical_radius, ls='--', c='tab:gray')
    plt.axvline(core.sonic_radius, ls=':', c='tab:gray')
    plt.xlim(rprf.r[0], 2*hw)
    plt.ylim(1e-1, 1e1)
    plt.xlabel(r'$r/L_{J,0}$')
    plt.ylabel(r'$\left<v^2\right>^{1/2}/c_s$')
    plt.legend(loc='lower right')

    # 5. Energies
    plt.sca(fig.add_subplot(gs[0, 3]))
    plot_energies(s, ds, gd, core.nid)
    if emin is not None and emax is not None:
        plt.ylim(emin, emax)
    if rmax is not None:
        plt.xlim(0, rmax)

    # 6. Accelerations
    plt.sca(fig.add_subplot(gs[1, 3]))
    plot_forces(s, rprf)
    plt.title('')
    plt.axvline(core.radius, c='tab:gray', lw=1)
    plt.axvline(rparent, c='tab:gray', lw=1)
    plt.axvline(core.critical_radius, ls='--', c='tab:gray')
    plt.axvline(core.sonic_radius, ls=':', c='tab:gray')
    plt.xlim(0, hw)
    plt.legend(ncol=3, fontsize=15, loc='upper right')

    return fig


def plot_forces(s, rprf, ax=None, xlim=(0, 0.2), ylim=(-20, 50)):
    acc = tools.get_accelerations(rprf)

    if ax is not None:
        plt.sca(ax)

    acc.thm.plot(lw=1, color='tab:orange', label=r'$f_\mathrm{thm}$')
    acc.trb.plot(lw=1, color='tab:blue', label=r'$f_\mathrm{trb}$')
    acc.ani.plot(lw=1, color='tab:green', label=r'$f_\mathrm{aniso}$')
    acc.cen.plot(lw=1, color='tab:olive', label=r'$f_\mathrm{cen}$')
    (-acc.grv).plot(marker='x', ls='--', color='tab:red', lw=1,
                    label=r'$-f_\mathrm{grav}$')
    (-acc.dvdt_lagrange).plot(marker='+', color='k', lw=1,
                              label=r'$-(dv/dt)_L$')
    (-acc.dvdt_euler).plot(color='tab:gray', lw=1, label=r'$-(dv/dt)_E$')

    # Overplot -GM/r^2
    Mr = (4*np.pi*rprf.rho*rprf.r**2).cumulative_integrate('r')
    gr = s.G*Mr/rprf.r**2
    gr.plot(color='tab:red', lw=1, ls='--')

    plt.axhline(0, linestyle=':')
    plt.ylabel('acceleration')
    plt.xlim(xlim)
    plt.ylim(ylim)


def plot_diagnostics(s, pid, normalize_time=True):
    """Create four-row plot showing history of core properties

    Parameters
    ----------
    s : LoadSimCoreFormation
        Simulation metadata
    pid : int
        Unique particle ID.
    normalize_time : bool, optional
        Flag to use normalized time (t-tcoll)/tff
    """
    fig, axs = plt.subplots(4, 1, figsize=(7, 15), sharex='col', gridspec_kw=dict(hspace=0.1))

    # Load cores
    cores = s.cores[pid].sort_index()
    tcoll = s.tcoll_cores.loc[pid].time
    tff = np.sqrt(3*np.pi/(32*cores.iloc[-1].mean_density))
    if normalize_time:
        time = (cores.time - tcoll) / tff
    else:
        time = cores.time

    # Calculate total forces acting on a core
    rprofs = s.rprofs[pid]
    ftot_lagrange, ftot_euler, fgrv = [], [], []
    fthm, ftrb, fcen, fani, fadv = [], [], [], [], []
    for num, core in cores.iterrows():
        rprf = rprofs.sel(num=num).sel(r=slice(0, core.envelop_tidal_radius))
        dmdr = 4*np.pi*rprf.r**2*rprf.rho
        fthm.append((rprf.thm*dmdr).integrate('r').data[()])
        ftrb.append((rprf.trb*dmdr).integrate('r').data[()])
        fcen.append((rprf.cen*dmdr).integrate('r').data[()])
        fani.append((rprf.ani*dmdr).integrate('r').data[()])
        fadv.append((rprf.adv*dmdr).integrate('r').data[()])
        fgrv.append((-rprf.grv*dmdr).integrate('r').data[()])
    fthm = np.array(fthm)
    ftrb = np.array(ftrb)
    fcen = np.array(fcen)
    fani = np.array(fani)
    fadv = np.array(fadv)
    fgrv = np.array(fgrv)

#    # Calculate edge density (TODO: replace this with more correct rho_e calculation)
#    rhoe = []
#    for num, core in cores.iterrows():
#        rhoe.append(rprofs.sel(num=num).rho.interp(r=core.envelop_tidal_radius).data[()])
#    rhoe = np.array(rhoe)

    # Do plotting
    plt.sca(axs[0])
    plt.plot(time, (fthm+ftrb+fcen+fani-fgrv)/fgrv, c='k')
    plt.plot(time, (fthm+ftrb+fcen+fani-fgrv-fadv)/fgrv, c='k', alpha=0.5)
    plt.ylim(-1, 1)
    plt.ylabel(r'$(F_\mathrm{p, eff} - F_\mathrm{grv}) / F_\mathrm{grv}$')
    if pid in s.good_cores:
        plt.title('{}, core {}'.format(s.basename, pid))
    else:
        plt.title('{}, core {}'.format(s.basename, pid)+r'$^*$')
    plt.twinx()
    plt.plot(time, fthm, lw=1, c='cyan', label=r'$F_\mathrm{thm}$')
    plt.plot(time, ftrb, lw=1, c='gray', label=r'$F_\mathrm{trb}$')
    plt.plot(time, fcen, lw=1, c='olive', label=r'$F_\mathrm{cen}$')
    plt.plot(time, fgrv, lw=1, c='pink', label=r'$F_\mathrm{grv}$')
    plt.plot(time, fani, lw=1, c='brown', ls=':', label=r'$F_\mathrm{ani}$')
    plt.plot(time, fadv, lw=1, c='purple', ls=':', label=r'$F_\mathrm{adv}$')
    plt.yscale('log')
    plt.ylim(1e-1, 1e2)
    plt.legend(loc='center left', bbox_to_anchor=(1.08, 0.5))

    plt.sca(axs[1])
    plt.plot(time, cores.envelop_tidal_radius, c='tab:blue', label=r'$R_\mathrm{tidal}$')
    plt.plot(time, cores.radius, c='tab:blue', ls='--', lw=1)
    plt.plot(time, cores.sonic_radius, c='tab:green', label=r'$R_\mathrm{sonic}$')
    plt.plot(time, cores.critical_radius, c='tab:red', label=r'$R_\mathrm{crit,c}$')
    plt.plot(time, cores.critical_radius_e, c='tab:red', ls='--', label=r'$R_\mathrm{crit,e}$')
    plt.yscale('log')
    plt.ylim(1e-2, 1e0)
    plt.ylabel(r'$R/L_{J,0}$')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.sca(axs[2])
    plt.plot(time, cores.envelop_tidal_mass, c='tab:blue', label=r'$M_\mathrm{tidal}$')
    plt.plot(time, cores.mass, c='tab:blue', ls='--', lw=1)
    plt.plot(time, cores.critical_mass, c='tab:red', label=r'$M_\mathrm{crit,c}$')
    plt.plot(time, cores.critical_mass_e, c='tab:red', ls='--', label=r'$M_\mathrm{crit,e}$')
    plt.yscale('log')
    plt.ylim(1e-3, 1e1)
    plt.ylabel(r'$M/M_{J,0}$')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.sca(axs[3])
    plt.plot(time, cores.center_density, c='tab:blue', ls='-', label=r'$\rho_c$')
    plt.plot(time, cores.mean_edge_density, c='tab:blue', ls='--', label=r'$\rho_e$')
    plt.plot(time, cores.mean_density, c='tab:blue', ls=':', label=r'$\overline{\rho}_\mathrm{tidal}$')
    plt.yscale('log')
    plt.ylabel(r'$\rho/\rho_0$')
    plt.legend(loc='upper left', bbox_to_anchor=(1.12, 1))
    plt.ylim(1e0, 1e4)
    plt.twinx()
    plt.semilogy(time, cores.center_density/cores.mean_edge_density, lw=1, ls='-',  c='tab:gray')
    plt.ylabel(r'$\rho_c/\rho_e$', c='tab:gray')
    plt.ylim(1e0, 1e2)

    if normalize_time:
        plt.xlim(-2, 0)
        plt.xlabel(r'$(t - t_\mathrm{coll})/t_\mathrm{ff}(\overline{\rho}_\mathrm{coll})$')
    else:
        plt.xlabel(r'$t/t_{J,0}$')
    for ax in axs:
        ax.grid()
    return fig


##### DEPRECATED #####


def plot_sinkhistory(s, ds, pds):
    # find end time
    ds_end = s.load_hdf5(s.nums[-1], load_method='yt')
    tend = ds_end.current_time

    # create figure
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, wspace=0, hspace=0)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :])

    # plot projections
    for ax, axis in zip((ax0, ax1, ax2), ('z', 'x', 'y')):
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
    plt.xlim(s.tcoll_cores.time.iloc[0], tend)
    plt.ylim(1e-2, 1e1)
    plt.yscale('log')
    plt.xlabel(r'$t/t_\mathrm{J,0}$')
    plt.ylabel(r'$M_*/M_\mathrm{J,0}$')
    return fig


def plot_Pspec(s, ds, ax=None, ax_twin=None):
    """Requires load_method='pyathena'"""
    # prepare frequencies
    from scipy.fft import fftn, fftfreq, fftshift
    from pyathena.util.transform import groupby_bins
    Lx, Ly, Lz = s.domain['Lx']
    Nx, Ny, Nz = s.domain['Nx']
    dx, dy, dz = s.domain['Lx']/(s.domain['Nx'])
    kx = fftshift(2*np.pi*fftfreq(Nx, dx))
    ky = fftshift(2*np.pi*fftfreq(Ny, dy))
    kz = fftshift(2*np.pi*fftfreq(Nz, dz))
    # Do FFTs
    for axis in [1, 2, 3]:
        vel = ds['mom'+str(axis)]/ds.dens
        ds['vel'+str(axis)] = vel
        vhat = fftn(vel.data, vel.shape)*dx*dy*dz
        vhat = fftshift(vhat)
        ds['vhat'+str(axis)] = xr.DataArray(
            vhat,
            coords=dict(kz=('z', kz), ky=('y', ky), kx=('x', kx)),
            dims=('z', 'y', 'x'))
    # Calculate 3D power spectrum
    Pk = np.abs(ds.vhat1)**2 + np.abs(ds.vhat2)**2 + np.abs(ds.vhat3)**2
    # Set radial wavenumbers
    Pk.coords['k'] = np.sqrt(Pk.kz**2 + Pk.ky**2 + Pk.kx**2)
    kmin = np.sqrt((2*np.pi/Lx)**2 + (2*np.pi/Ly)**2 + (2*np.pi/Lz)**2)
    kmax = np.sqrt((2*np.pi/(2*dx))**2 + (2*np.pi/(2*dy))**2
                   + (2*np.pi/(2*dz))**2)
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
    plt.text(s.rprofs[1].t.min(), rho_crit_KM05, r"$\rho_\mathrm{crit, KM05}$",
             fontsize=18, ha='left', va='bottom')
    plt.text(s.rprofs[1].t.min(), 14.1*rho_crit_KM05,
             r"$14.1\rho_\mathrm{crit, KM05}$", fontsize=18,
             ha='left', va='bottom')
    plt.axhline(14.1*rho_crit_KM05, linestyle='--', lw=1)
    plt.legend(loc=(1.01, 0))
    plt.ylabel(r'$\rho_c/\rho_0$')
    plt.xlabel(r'$t/t_J$')
    plt.title(s.basename)


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
    plt.xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])


def plot_mass(s, pid, ax=None, normalized=True):
    if ax is not None:
        plt.sca(ax)
    cores = s.cores[pid]
    tcoll_core = cores.iloc[-1]
    tcoll = s.tcoll_cores.loc[pid].time
    tff = np.sqrt(3*np.pi/(32*tcoll_core.mean_density))

    if normalized:
        plt.plot((cores.time - tcoll) / tff, cores.envelop_tidal_mass, c='tab:blue', ls='-', label=r'$M_\mathrm{tidal}$')
        plt.plot((cores.time - tcoll) / tff, cores.mass, c='tab:blue', ls='--')
        plt.plot((cores.time - tcoll) / tff, cores.critical_mass, c='tab:orange', label=r'$M_\mathrm{crit}$')
        plt.xlim(-2, 0)
        plt.xlabel(r'$(t-t_\mathrm{coll})/t_\mathrm{ff}(\overline{\rho}_\mathrm{coll})$')
    else:
        plt.plot(cores.time, cores.envelop_tidal_mass, c='tab:blue', ls='-', label=r'$M_\mathrm{tidal}$')
        plt.plot(cores.time, cores.mass, c='tab:blue', ls='--')
        plt.plot(cores.time, cores.critical_mass, c='tab:orange', label=r'$M_\mathrm{crit}$')
        plt.xlabel(r'$t/t_{J,0}$')

    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.ylim(1e-3, 1e1)
    plt.ylabel(r'$M/M_{J,0}$')
    plt.title('{}, core {}'.format(s.basename, pid))

    plt.twinx()
    if normalized:
        plt.semilogy((cores.time - tcoll) / tff, cores.center_density, '--', color='tab:brown')
    else:
        plt.semilogy(cores.time, cores.center_density, '--', color='tab:brown')
    plt.ylabel(r'$\rho_c/\rho_0$', color='tab:brown')
    plt.ylim(1e1, 1e4)


def plot_radii(s, pid, ax=None, normalized=True):
    """Plot Rtidal, Rcrit, and Rsonic

    Parameters
    ----------
    s : LoadSimCoreFormation
        Simulation metadata of a selected model
    pid : int
        Unique particle ID
    ax : matplotlib.axes.Axes, optional
        Axes to draw plot
    """
    if ax is not None:
        plt.sca(ax)
    cores = s.cores[pid]
    tcoll_core = cores.iloc[-1]
    tcoll = s.tcoll_cores.loc[pid].time
    tff = np.sqrt(3*np.pi/(32*tcoll_core.mean_density))

    if normalized:
        plt.plot((cores.time - tcoll) / tff, cores.envelop_tidal_radius, c='tab:blue', ls='-', label=r'$R_\mathrm{tidal}$')
        plt.plot((cores.time - tcoll) / tff, cores.radius, c='tab:blue', ls='--')
        plt.plot((cores.time - tcoll) / tff, cores.critical_radius, c='tab:orange', label=r'$R_\mathrm{crit}$')
        plt.plot((cores.time - tcoll) / tff, cores.sonic_radius, c='tab:green', label=r'$R_\mathrm{sonic}$')
        plt.xlim(-2, 0)
        plt.xlabel(r'$(t-t_\mathrm{coll})/t_\mathrm{ff}(\overline{\rho}_\mathrm{coll})$')
    else:
        plt.plot(cores.time, cores.envelop_tidal_radius, c='tab:blue', ls='-', label=r'$R_\mathrm{tidal}$')
        plt.plot(cores.time, cores.radius, c='tab:blue', ls='--')
        plt.plot(cores.time, cores.critical_radius, c='tab:orange', label=r'$R_\mathrm{crit}$')
        plt.plot(cores.time, cores.sonic_radius, c='tab:green', label=r'$R_\mathrm{sonic}$')
        plt.xlabel(r'$t/t_{J,0}$')

    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.ylim(1e-2, 2e0)
    plt.ylabel(r'$R/L_{J,0}$')
    plt.title('{}, core {}'.format(s.basename, pid))

    plt.twinx()
    if normalized:
        plt.semilogy((cores.time - tcoll) / tff, cores.center_density, '--', color='tab:brown')
    else:
        plt.semilogy(cores.time, cores.center_density, '--', color='tab:brown')
    plt.ylabel(r'$\rho_c/\rho_0$', color='tab:brown')
    plt.ylim(1e1, 1e4)


def plot_radii_all(sa, models, ax=None, ncells_min=10):
    """Plot radii of all cores in one axes

    See Also
    --------
    plot_radii
    """
    if ax is None:
        ax = plt.gca()
    ax2 = ax.twinx()
    for mdl in models:
        s = sa.set_model(mdl)
        for pid in s.pids:
            if tools.test_isolated_core(s, pid) and tools.test_resolved_core(s, pid, ncells_min):
                cores = s.cores[pid]
                tcoll_core = cores.iloc[-1]
                tcoll = s.tcoll_cores.loc[pid].time

                # select time and length unit
                time_unit = np.sqrt(3*np.pi/(32*tcoll_core.mean_density))
                length_unit = tcoll_core.envelop_tidal_radius

                # plot lines
                plt.sca(ax)
                l1, = plt.plot((cores.time - tcoll) / time_unit, cores.envelop_tidal_radius / length_unit,
                         c='tab:blue', lw=1, label=r'$R_\mathrm{tidal}$', alpha=0.6)
                l2, = plt.plot((cores.time - tcoll) / time_unit, cores.critical_radius / length_unit,
                         c='tab:orange', lw=1, label=r'$R_\mathrm{crit}$', alpha=0.6)

                ax2.semilogy((cores.time - tcoll) / time_unit, cores.center_density / s.get_rhoLP(s.dx/2),
                         c='tab:brown', lw=1, ls='--', alpha=0.6)

    plt.sca(ax)
    plt.yscale('log')
    plt.legend((l1, l2), (r'$R_\mathrm{tidal}$', r'$R_\mathrm{crit}$'), loc='upper right')
    plt.xlim(-2, 0)
    plt.xticks(np.arange(-2, 0.01, 0.5))
    plt.ylim(1e-1, 1e1)
    plt.xlabel(r'$(t-t_\mathrm{coll})/t_\mathrm{ff}(\overline{\rho}_\mathrm{coll})$')
    plt.ylabel(r'$R/R_\mathrm{tidal, coll}$')
    ax2.set_ylim(1e-4, 1e0)
    ax2.set_ylabel(r'$\rho_c/\rho_\mathrm{LP}$', color='tab:brown')


def plot_force_and_mass_ratios(sa, models, ax=None, ncells_min=10, pid_reject=None):
    if ax is None:
        ax = plt.gca()
    ax2 = ax.twinx()

    if pid_reject is None:
        pid_reject = set()

    times, fratios, mratios = [], [], []
    for mdl in models:
        s = sa.set_model(mdl)
        for pid in s.pids:
            if tools.test_isolated_core(s, pid) and tools.test_resolved_core(s, pid, ncells_min):
                if mdl in pid_reject:
                    if pid in pid_reject[mdl]:
                        continue
                cores = s.cores[pid]
                rprof = s.rprofs[pid]
                ftot_lagrange, ftot_euler, fgrv = [], [], []
                for num, rtidal in zip(cores.index, cores.envelop_tidal_radius):
                    rprf = rprof.sel(num=num).sel(r=slice(0, rtidal))
                    dm = 4*np.pi*rprf.r**2*rprf.rho
                    ftot_lagrange.append((rprf.dvdt_lagrange*dm).integrate('r').data[()])
                    ftot_euler.append((rprf.dvdt_euler*dm).integrate('r').data[()])
                    fgrv.append((rprf.grv*dm).integrate('r').data[()])
                ftot_lagrange = np.array(ftot_lagrange)
                ftot_euler = np.array(ftot_euler)
                fgrv = np.array(fgrv)

                tcoll_core = cores.iloc[-1]
                tcoll = s.tcoll_cores.loc[pid].time
                tff = np.sqrt(3*np.pi/(32*tcoll_core.mean_density))

                # plot lines
                t = (cores.time.values - tcoll) / tff
                fratio = ftot_lagrange/np.abs(fgrv)
                mratio = (cores.envelop_tidal_mass/cores.critical_mass).values
                ax.plot(t, fratio, c='tab:blue', lw=1, alpha=0.6)
                ax2.semilogy(t, mratio, c='tab:red', lw=1, alpha=0.6)
                times.append(t)
                fratios.append(fratio)
                mratios.append(mratio)

    plt.sca(ax)
    plt.xlim(-2, 0)
    plt.xticks(np.arange(-2, 0.01, 0.5))
    plt.ylim(-0.5, 0.5)
    plt.axhline(0, ls='--', c='k', lw=1)
    plt.xlabel(r'$(t-t_\mathrm{coll})/t_\mathrm{ff}(\overline{\rho}_\mathrm{coll})$')
    plt.ylabel(r'$(F_\mathrm{p,eff} - F_\mathrm{grv})/F_\mathrm{grv}$', c='tab:blue')
    ax2.set_ylim(1e-1, 1e1)
    ax2.set_ylabel(r'$M_\mathrm{tidal}/M_\mathrm{crit}$', c='tab:red')
    return times, fratios, mratios, ax2


def plot_force_imbalance(s, pid, ax=None, normalized=True):
    cores = s.cores[pid]
    rprof = s.rprofs[pid]
    ftot_lagrange, ftot_euler, fgrv = [], [], []
    for num, rtidal in zip(cores.index, cores.envelop_tidal_radius):
        rprf = rprof.sel(num=num).sel(r=slice(0, rtidal))
        dm = 4*np.pi*rprf.r**2*rprf.rho
        ftot_lagrange.append((rprf.dvdt_lagrange*dm).integrate('r').data[()])
        ftot_euler.append((rprf.dvdt_euler*dm).integrate('r').data[()])
        fgrv.append((rprf.grv*dm).integrate('r').data[()])
    ftot_lagrange = np.array(ftot_lagrange)
    ftot_euler = np.array(ftot_euler)
    fgrv = np.array(fgrv)

    tcoll_core = cores.iloc[-1]
    tcoll = s.tcoll_cores.loc[pid].time
    tff = np.sqrt(3*np.pi/(32*tcoll_core.mean_density))

    if ax is not None:
        plt.sca(ax)

    if normalized:
        plt.plot((cores.time - tcoll) / tff, ftot_lagrange/np.abs(fgrv),
                 c='tab:blue')
        plt.plot((cores.time - tcoll) / tff, ftot_euler/np.abs(fgrv), c='tab:blue',
                 alpha=0.5)
        plt.xlim(-2, 0)
        plt.xlabel(r'$(t-t_\mathrm{coll})/t_\mathrm{ff}(\overline{\rho}_\mathrm{coll})$')
    else:
        plt.plot(cores.time, ftot_lagrange/np.abs(fgrv), c='tab:blue')
        plt.plot(cores.time, ftot_euler/np.abs(fgrv), c='tab:blue', alpha=0.5)
        plt.xlabel(r'$t/t_{J,0}$')

    plt.ylim(-0.5, 0.5)
    plt.axhline(0, ls='--', c='k', lw=1)
    plt.ylabel(r'$F_\mathrm{net}/F_\mathrm{grv}$', c='tab:blue')

    plt.twinx()

    if normalized:
        plt.semilogy((cores.time - tcoll) / tff, cores.envelop_tidal_radius/cores.critical_radius,
                     c='tab:red', lw=1, label=r'$R_\mathrm{tidal}/R_\mathrm{crit}$')
        plt.semilogy((cores.time - tcoll) / tff, cores.sonic_radius/cores.critical_radius,
                     c='tab:orange', lw=1,
                     label=r'$R_\mathrm{sonic}/R_\mathrm{crit}$')
    else:
        plt.semilogy(cores.time, cores.envelop_tidal_radius/cores.critical_radius,
                     c='tab:red', lw=1, label=r'$R_\mathrm{tidal}/R_\mathrm{crit}$')
        plt.semilogy(cores.time, cores.sonic_radius/cores.critical_radius,
                     c='tab:orange', lw=1,
                     label=r'$R_\mathrm{sonic}/R_\mathrm{crit}$')

    plt.ylim(1e-1, 1e1)
    plt.ylabel("Radius ratios")
    plt.legend()


def plot_force_imbalance_all(sa, models, ax=None, ncells_min=10, pid_reject=None):
    if ax is None:
        ax = plt.gca()
    ax2 = ax.twinx()

    if pid_reject is None:
        pid_reject = set()

    times, fratios, rratios = [], [], []
    for mdl in models:
        s = sa.set_model(mdl)
        for pid in s.pids:
            if tools.test_isolated_core(s, pid) and tools.test_resolved_core(s, pid, ncells_min):
                if mdl in pid_reject:
                    if pid in pid_reject[mdl]:
                        continue
                cores = s.cores[pid]
                rprof = s.rprofs[pid]
                ftot_lagrange, ftot_euler, fgrv = [], [], []
                for num, rtidal in zip(cores.index, cores.envelop_tidal_radius):
                    rprf = rprof.sel(num=num).sel(r=slice(0, rtidal))
                    dm = 4*np.pi*rprf.r**2*rprf.rho
                    ftot_lagrange.append((rprf.dvdt_lagrange*dm).integrate('r').data[()])
                    ftot_euler.append((rprf.dvdt_euler*dm).integrate('r').data[()])
                    fgrv.append((rprf.grv*dm).integrate('r').data[()])
                ftot_lagrange = np.array(ftot_lagrange)
                ftot_euler = np.array(ftot_euler)
                fgrv = np.array(fgrv)

                tcoll_core = cores.iloc[-1]
                tcoll = s.tcoll_cores.loc[pid].time
                tff = np.sqrt(3*np.pi/(32*tcoll_core.mean_density))

                # plot lines
                t = (cores.time.values - tcoll) / tff
                fratio = ftot_lagrange/np.abs(fgrv)
                rratio = (cores.envelop_tidal_radius/cores.critical_radius).values
                ax.plot(t, fratio, c='tab:blue', lw=1, alpha=0.6)
                ax2.semilogy(t, rratio, c='tab:red', lw=1, alpha=0.6)
                times.append(t)
                fratios.append(fratio)
                rratios.append(rratio)

    plt.sca(ax)
    plt.xlim(-2, 0)
    plt.xticks(np.arange(-2, 0.01, 0.5))
    plt.ylim(-0.5, 0.5)
    plt.axhline(0, ls='--', c='k', lw=1)
    plt.xlabel(r'$(t-t_\mathrm{coll})/t_\mathrm{ff}(\overline{\rho}_\mathrm{coll})$')
    plt.ylabel(r'$(F_\mathrm{p,eff} - F_\mathrm{grv})/F_\mathrm{grv}$', c='tab:blue')
    ax2.set_ylim(1e-1, 1e1)
    ax2.set_ylabel(r'$R_\mathrm{tidal}/R_\mathrm{crit}$', c='tab:red')
    return times, fratios, rratios, ax2


def plot_rcrit_vs_rcore(sa, models, pid_sel=None, pid_exclude=None,
                        e1=0.7, e2=0.4, ncells_min=10):
    fig, axs = plt.subplots(2,2,figsize=(11,10))
    for mdl in models:
        s = sa.set_model(mdl)
        if pid_sel is None:
            pids = s.pids
        else:
            pids = pid_sel
        for pid in pids:
            if tools.test_isolated_core(s, pid) and tools.test_resolved_core(s, pid, ncells_min):
                cores = s.cores[pid]
                cprops_crit = tools.get_critical_core_props(s, pid, e1, e2)
                plt.sca(axs[0,0])
                plt.scatter(cores.radius, cores.critical_radius, marker='+')

                plt.sca(axs[0,1])
                plt.scatter(cprops_crit.radius, cprops_crit.critical_radius, marker='+')

                plt.sca(axs[1,0])
                plt.scatter(cores.sonic_radius, cores.critical_radius, marker='+')

                plt.sca(axs[1,1])
                plt.scatter(cprops_crit.sonic_radius, cprops_crit.critical_radius, marker='+')

    for ax in axs.flat:
        ax.set_aspect('equal')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$R_\mathrm{crit}/L_{J,0}$')
        ax.set_xlim(1e-2, 1e0)
        ax.set_ylim(1e-2, 1e0)
        ax.plot([1e-2, 1e0], [1e-2, 1e0], '--', c='tab:gray')
    for ax in axs[0,:]:
        ax.set_xlabel(r'$R_\mathrm{tidal}/L_{J,0}$')
    for ax in axs[1,:]:
        ax.set_xlabel(r'$R_\mathrm{sonic}/L_{J,0}$')
    return fig
