"""Module containing functions that are not generally reusable"""

# python modules
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import subprocess
import pickle
import glob
import logging

# pyathena modules
from pyathena.core_formation import plots
from pyathena.core_formation import tools
from pyathena.core_formation import config
from pyathena.util import uniform, transform
from grid_dendro import dendrogram


def combine_partab(s, ns=None, ne=None, partag="par0", remove=False,
                   include_last=False):
    """Combine particle .tab output files.

    Parameters
    ----------
    s : LoadSimCoreFormation
        LoadSimCoreFormation instance.
    ns : int, optional
        Starting snapshot number.
    ne : int, optional
        Ending snapshot number.
    partag : str, optional
        Particle tag (<particle?> in the input file).
    remove : str, optional
        If True, remove the block? per-core outputs after joining.
    include_last : bool
        If false, do not process last .tab file, which might being written
        by running Athena++ process.
    """
    script = "/home/sm69/tigris/vis/tab/combine_partab.sh"
    outid = "out{}".format(s.partab_outid)
    block0_pattern = '{}/{}.block0.{}.?????.{}.tab'.format(s.basedir,
                                                           s.problem_id, outid,
                                                           partag)
    file_list0 = sorted(glob.glob(block0_pattern))
    if not include_last:
        file_list0 = file_list0[:-1]
    if len(file_list0) == 0:
        print("Nothing to combine", flush=True)
        return
    if ns is None:
        ns = int(file_list0[0].split('/')[-1].split('.')[3])
    if ne is None:
        ne = int(file_list0[-1].split('/')[-1].split('.')[3])
    nblocks = 1
    for axis in [1, 2, 3]:
        nblocks *= ((s.par['mesh'][f'nx{axis}']
                    // s.par['meshblock'][f'nx{axis}']))
    if partag not in s.partags:
        raise ValueError("Particle {} does not exist".format(partag))
    subprocess.run([script, s.problem_id, outid, partag, str(ns), str(ne)],
                   cwd=s.basedir)

    if remove:
        joined_pattern = '{}/{}.{}.?????.{}.tab'.format(s.basedir,
                                                        s.problem_id, outid,
                                                        partag)
        joined_files = set(glob.glob(joined_pattern))
        block0_files = {f.replace('block0.', '') for f in file_list0}
        if block0_files.issubset(joined_files):
            print("All files are joined. Remove block* files", flush=True)
            file_list = []
            for fblock0 in block0_files:
                for i in range(nblocks):
                    file_list.append(fblock0.replace(
                        outid, "block{}.{}".format(i, outid)))
            file_list.sort()
            for f in file_list:
                Path(f).unlink()
        else:
            print("Not all files are joined", flush=True)


def critical_tes(s, pid, num, overwrite=False):
    """Calculates and saves critical tes associated with each core.

    Parameters
    ----------
    s : LoadSimCoreFormation
        LoadSimCoreFormation instance.
    pid : int
        Particle id.
    num : int
        Snapshot number
    overwrite : str, optional
        If true, overwrites the existing pickle file.
    """
    # Check if file exists
    ofname = Path(s.savdir, 'critical_tes',
                  'critical_tes.par{}.{:05d}.p'.format(pid, num))
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[critical_tes] file already exists. Skipping...')
        return

    if num not in s.rprofs[pid].num:
        msg = (f"Radial profile for num={num} does not exist. "
                "Cannot calculate critical_tes. Skipping...")
        logging.warning(msg)
        return

    msg = '[critical_tes] processing model {} pid {} num {}'
    print(msg.format(s.basename, pid, num))

    # Load the radial profile
    rprf = s.rprofs[pid].sel(num=num)
    core = s.cores[pid].loc[num]

    # Calculate critical TES
    critical_tes = tools.calculate_critical_tes(s, rprf, core)
    critical_tes['num'] = num

    # write to file
    if ofname.exists():
        ofname.unlink()
    with open(ofname, 'wb') as handle:
        pickle.dump(critical_tes, handle, protocol=pickle.HIGHEST_PROTOCOL)


def core_tracking(s, pid, protostellar=False, overwrite=False):
    """Loops over all sink particles and find their progenitor cores

    Finds a unique grid-dendro leaf at each snapshot that is going to collapse.
    For each sink particle, back-traces the evolution of its progenitor cores.
    Pickles the resulting data.

    Parameters
    ----------
    s : LoadSimCoreFormation
        LoadSimCoreFormation instance.
    pid : int
        Particle ID
    protostellar : bool
        If True, track cores including the protostellar phase.
    overwrite : str, optional
        If true, overwrites the existing pickle file.
    """
    # Check if file exists
    ofname = Path(s.savdir, 'cores', 'cores.par{}.p'.format(pid))
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[core_tracking] file already exists. Skipping...')
        return

    # Perform protostellar core tracking only for resolved cores
    # to save resources.
    if protostellar:
        cores = tools.track_protostellar_cores(s, pid)
    else:
        cores = tools.track_cores(s, pid)
    cores.to_pickle(ofname, protocol=pickle.HIGHEST_PROTOCOL)


def radial_profile(s, num, pids, overwrite=False, full_radius=False, days_overwrite=30):
    """Calculates and pickles radial profiles of all cores.

    Parameters
    ----------
    s : LoadSimCoreFormation
        LoadSimCoreFormation instance.
    pid : int
        Particle id.
    num : int
        Snapshot number
    overwrite : str, optional
        If true, overwrites the existing pickle file.
    """

    pids_skip = []
    for pid in pids:
        cores = s.cores[pid]
        if num not in cores.index:
            pids_skip.append(pid)
            continue
        ofname = Path(s.savdir, 'radial_profile',
                      'radial_profile.par{}.{:05d}.nc'.format(pid, num))
        if ofname.exists():
            if overwrite:
                creation_time = ofname.stat().st_ctime
                creation_date = datetime.datetime.fromtimestamp(creation_time)
                current_time = datetime.datetime.now()
                if (current_time - creation_date).days < days_overwrite:
                    pids_skip.append(pid)
                else:
                    pass
            else:
                pids_skip.append(pid)

    pids_to_process = sorted(set(pids) - set(pids_skip))

    if len(pids_to_process) == 0:
        msg = ("[radial_profile] Every core alreay has radial profiles at "
               f"num = {num}. Skipping...")
        print(msg)
        return

    msg = ("[radial_profile] Start reading snapshot at "
           f"num = {num}.")
    print(msg)
    # Load the snapshot
    ds0 = s.load_hdf5(num, quantities=['dens','phi','mom1','mom2','mom3'])
    ds0 = ds0.transpose('z', 'y', 'x')

    # Loop through cores
    for pid in pids_to_process:
        cores = s.cores[pid]
        if num not in cores.index:
            # This snapshot `num` does not contain any image of the core `pid`
            # Continue to the next core.
            continue

        # Create directory and check if a file already exists
        ofname = Path(s.savdir, 'radial_profile',
                      f'radial_profile.par{pid}.{num:05d}.nc')
        ofname.parent.mkdir(exist_ok=True)
        if ofname.exists() and not overwrite:
            msg = (f"[radial_profile] A file already exists for pid = {pid} "
                   f", num = {num}. Continue to the next core")
            print(msg)
            continue

        msg = (f"[radial_profile] processing model {s.basename}, "
               f"pid {pid}, num {num}")
        print(msg)

        core = cores.loc[num]

        if full_radius:
            rmax = 0.5*s.Lbox
        else:
            rmax = min(0.5*s.Lbox, 3*cores.loc[:cores.attrs['numcoll']].tidal_radius.max())

        # Find the location of the core
        center = tools.get_coords_node(s, core.leaf_id)
        center = dict(zip(['x', 'y', 'z'], center))

        # Roll the data such that the core is at the center of the domain
        ds, center = tools.recenter_dataset(ds0, center)

        # Calculate the angular momentum vector within the tidal radius.
        x = ds.x - center['x']
        y = ds.y - center['y']
        z = ds.z - center['z']
        r = np.sqrt(x**2 + y**2 + z**2)
        lx = (y*ds.mom3 - z*ds.mom2).where(r < core.tidal_radius).sum().data[()]*s.dV
        ly = (z*ds.mom1 - x*ds.mom3).where(r < core.tidal_radius).sum().data[()]*s.dV
        lz = (x*ds.mom2 - y*ds.mom1).where(r < core.tidal_radius).sum().data[()]*s.dV
        lvec = np.array([lx, ly, lz])

        # Calculate radial profile
        rprf = tools.calculate_radial_profile(s, ds, list(center.values()), rmax, lvec)
        rprf = rprf.expand_dims(dict(t=[ds.Time,]))
        rprf['lx'] = xr.DataArray([lx,], dims='t')
        rprf['ly'] = xr.DataArray([ly,], dims='t')
        rprf['lz'] = xr.DataArray([lz,], dims='t')

        # write to file
        if ofname.exists():
            ofname.unlink()
        rprf.to_netcdf(ofname)


def lagrangian_props(s, pid, method=1, overwrite=False):
    # Check if file exists
    ofname = Path(s.savdir, 'cores', f'lprops_ver{method}.par{pid}.p')
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[lagrangian_props] file already exists. Skipping...')
        return

    s.select_cores(method)
    cores = s.cores[pid]
    rprofs = s.rprofs[pid]
    print(f'[lagrangian_props] Calculate Lagrangian props for core {pid} with version {method}')
    lprops = tools.calculate_lagrangian_props(s, cores, rprofs)
    lprops.to_pickle(ofname, protocol=pickle.HIGHEST_PROTOCOL)


def observables(s, pid, num, overwrite=False):
    # Check if file exists
    ofname = Path(s.savdir, 'cores',
                  'observables.par{}.{:05d}.p'.format(pid, num))
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[observables] file already exists. Skipping...')
        return

    if num not in s.rprofs[pid].num:
        msg = (f"Radial profile for num={num} does not exist. "
                "Cannot calculate observables. Skipping...")
        logging.warning(msg)
        return

    msg = '[observables] processing model {} pid {} num {}'
    print(msg.format(s.basename, pid, num))

    # Load the radial profile
    rprf = s.rprofs[pid].sel(num=num)
    core = s.cores[pid].loc[num]

    # Calculate observables
    obsprops_3d = tools.calculate_observables(s, core, rprf, core.tidal_radius0, '3d')
    obsprops_itr = tools.calculate_observables(s, core, rprf, core.tidal_radius0, 'iterative')
    obsprops_2d_wholebox = tools.calculate_observables(s, core, rprf, core.tidal_radius0, '2d_wholebox')
    observables = dict(three_d        = obsprops_3d,
                       iterative      = obsprops_itr,
                       two_d_wholebox = obsprops_2d_wholebox)

    # write to file
    if ofname.exists():
        ofname.unlink()
    with open(ofname, 'wb') as handle:
        pickle.dump(observables, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_grid(s, num, overwrite=False):
    """Run GRID-dendro

    Parameters
    ----------
    s : LoadSimCoreFormation
        Simulation metadata.
    num : int
        Snapshot number.
    """
    # Check if file exists
    ofname = Path(s.savdir, 'GRID',
                  'dendrogram.{:05d}.p'.format(num))
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[run_grid] file already exists. Skipping...')
        return

    # Load data and construct dendrogram
    print('[run_grid] processing model {} num {}'.format(s.basename, num))
    ds = s.load_hdf5(num, quantities=['phi',],
                     load_method='pyathena').transpose('z', 'y', 'x')
    phi = ds.phi.to_numpy()
    gd = dendrogram.Dendrogram(phi, verbose=False)
    gd.construct()

    # Write to file
    with open(ofname, 'wb') as handle:
        pickle.dump(gd, handle, protocol=pickle.HIGHEST_PROTOCOL)


def prune(s, num, overwrite=False):
    """Prune GRID-dendro

    Parameters
    ----------
    s : LoadSimCoreFormation
        Simulation metadata.
    num : int
        Snapshot number.
    """
    # Check if file exists
    ofname = Path(s.savdir, 'GRID',
                  'dendrogram.pruned.{:05d}.p'.format(num))
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[prune] file already exists. Skipping...')
        return

    # Load original dendrogram and prune it
    print('[prune] processing model {} num {}'.format(s.basename, num))
    gd = s.load_dendro(num, pruned=False)
    gd.prune()

    # Write to file
    with open(ofname, 'wb') as handle:
        pickle.dump(gd, handle, protocol=pickle.HIGHEST_PROTOCOL)


def resample_hdf5(s, level=0):
    """Resamples AMR output into uniform resolution.

    Reads a HDF5 file with a mesh refinement and resample it to uniform
    resolution amounting to a given refinement level.

    Resampled HDF5 file will be written as
        {basedir}/uniform/{problem_id}.level{level}.?????.athdf

    Args:
        s: pyathena.LoadSim instance
        level: Refinement level to resample. root level=0.
    """
    ifname = Path(s.basedir, '{}.out2'.format(s.problem_id))
    odir = Path(s.basedir, 'uniform')
    odir.mkdir(exist_ok=True)
    ofname = odir / '{}.level{}'.format(s.problem_id, level)
    kwargs = dict(start=s.nums[0],
                  end=s.nums[-1],
                  stride=1,
                  input_filename=ifname,
                  output_filename=ofname,
                  level=level,
                  m=None,
                  x=None,
                  quantities=None)
    uniform.main(**kwargs)


def plot_core_evolution(s, pid, num, method=1, overwrite=False, rmax=None):
    """Creates multi-panel plot for t_coll core properties

    Parameters
    ----------
    s : LoadSimCoreFormation
        Simulation metadata.
    pid : int
        Unique ID of a selected particle.
    num : int
        Snapshot number.
    overwrite : str, optional
        If true, overwrite output files.
    """
    fname = Path(s.savdir, 'figures', "{}.par{}.ver{}.{:05d}.png".format(
        config.PLOT_PREFIX_CORE_EVOLUTION, pid, method, num))
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        print('[plot_core_evolution] file already exists. Skipping...')
        return
    print(f'[plot_core_evolution] processing model {s.basename} pid {pid} num {num}, ver{method}')
    s.select_cores(method)
    fig = plots.plot_core_evolution(s, pid, num, rmax=rmax)
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_mass_radius(s, pid, overwrite=False):
    fig = plt.figure()
    ax = fig.add_subplot()
    for num in s.cores[pid].index:
        msg = '[plot_mass_radius] processing model {} pid {} num {}'
        msg = msg.format(s.basename, pid, num)
        print(msg)
        fname = Path(s.savdir, 'figures', "{}.par{}.{:05d}.png".format(
            config.PLOT_PREFIX_MASS_RADIUS, pid, num))
        fname.parent.mkdir(exist_ok=True)
        if fname.exists() and not overwrite:
            print('[plot_mass_radius] file already exists. Skipping...')
            return
        plots.mass_radius(s, pid, num, ax=ax)
        fig.savefig(fname, bbox_inches='tight', dpi=200)
        ax.cla()


def plot_sink_history(s, num, overwrite=False):
    """Creates multi-panel plot for sink particle history

    Args:
        s: pyathena.LoadSim instance
    """
    fname = Path(s.savdir, 'figures', "{}.{:05d}.png".format(
                 config.PLOT_PREFIX_SINK_HISTORY, num))
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        print('[plot_sink_history] file already exists. Skipping...')
        return
    ds = s.load_hdf5(num, quantities=['dens',], load_method='pyathena')
    pds = s.load_partab(num)
    fig = plots.plot_sinkhistory(s, ds, pds)
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_core_structure(s, pid, overwrite=False):
    rmax = s.cores[pid].tidal_radius.max()
    for num in s.cores[pid].index:
        fname = Path(s.savdir, 'figures', "core_structure.par{}.{:05d}.png".format(pid, num))
        if fname.exists() and not overwrite:
            print('[plot_core_structure] file already exists. Skipping...')
            return
        msg = '[plot_core_structure] processing model {} pid {} num {}'
        msg = msg.format(s.basename, pid, num)
        print(msg)
        fig = plots.core_structure(s, pid, num, rmax=rmax)
        fig.savefig(fname, bbox_inches='tight', dpi=200)
        plt.close(fig)


def plot_diagnostics(s, pid, overwrite=False):
    """Creates diagnostics plots for a given model

    Save projections in {basedir}/figures for all snapshots.

    Parameters
    ----------
    s : LoadSimCoreFormation
        LoadSim instance
    pid : int
        Particle ID
    overwrite : bool, optional
        Flag to overwrite
    """
    fname = Path(s.savdir, 'figures',
                 'diagnostics_normalized.par{}.png'.format(pid))
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        print('[plot_diagnostics] file already exists. Skipping...')
        return

    msg = '[plot_diagnostics] model {} pid {}'
    print(msg.format(s.basename, pid))

    fig = plots.plot_diagnostics(s, pid, normalize_time=True)
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)

    fname = Path(s.savdir, 'figures', 'diagnostics.par{}.png'.format(pid))
    if fname.exists() and not overwrite:
        return
    fig = plots.plot_diagnostics(s, pid, normalize_time=False)
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_radial_profile_at_tcrit(s, nrows=5, ncols=6, overwrite=False):
    fname = Path(s.savdir, 'figures', 'radial_profile_at_tcrit.png')
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        print('[plot_radial_profile_at_tcrit] file already exists. Skipping...')
        return

    msg = '[plot_radial_profile_at_tcrit] Processing model {}'
    print(msg.format(s.basename))

    if len(s.good_cores()) > nrows*ncols:
        raise ValueError("Number of good cores {} exceeds the number of panels.".format(len(s.good_cores())))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), sharex=True,
                            gridspec_kw={'hspace':0.05, 'wspace':0.12})
    for pid, ax in zip(s.good_cores(), axs.flat):
        plots.radial_profile_at_tcrit(s, pid, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.text(0.6, 0.86, f"pid {pid}", transform=ax.transAxes)
        cores = s.cores[pid]
        nc = cores.attrs['numcrit']
        core = cores.loc[nc]
        ax.text(0.6, 0.73, "{:.2f} tff".format(core.tnorm1),
                transform=ax.transAxes)
    for ax in axs[:, 0]:
        ax.set_ylabel(r'$\rho/\rho_0$')
    for ax in axs[-1, :]:
        ax.set_xlabel(r'$r/R_\mathrm{tidal}$')
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)


def calculate_linewidth_size(s, num, seed=None, pid=None, overwrite=False, ds=None):
    if seed is not None and pid is not None:
        raise ValueError("Provide either seed or pid, not both")
    elif seed is not None:
        # Check if file exists
        ofname = Path(s.savdir, 'linewidth_size',
                      'linewidth_size.{:05d}.{}.nc'.format(num, seed))
        ofname.parent.mkdir(exist_ok=True)
        if ofname.exists() and not overwrite:
            print('[linewidth_size] file already exists. Skipping...')
            return

        msg = '[linewidth_size] processing model {} num {} seed {}'
        print(msg.format(s.basename, num, seed))

        if ds is None:
            ds = s.load_hdf5(num, quantities=['dens', 'mom1', 'mom2', 'mom3'])
            ds['vel1'] = ds.mom1/ds.dens
            ds['vel2'] = ds.mom2/ds.dens
            ds['vel3'] = ds.mom3/ds.dens

        rng = np.random.default_rng(seed)
        i, j, k = rng.integers(low=0, high=511, size=(3))
        origin = (ds.x.isel(x=i).data[()],
                  ds.y.isel(y=j).data[()],
                  ds.z.isel(z=k).data[()])
    elif pid is not None:
        if num not in s.cores[pid].index:
            print(f'[linewidth_size] {num} is not in the snapshot list of core {pid}')
            return

        # Check if file exists
        ofname = Path(s.savdir, 'linewidth_size',
                      'linewidth_size.{:05d}.par{}.nc'.format(num, pid))
        ofname.parent.mkdir(exist_ok=True)
        if ofname.exists() and not overwrite:
            print('[linewidth_size] file already exists. Skipping...')
            return

        msg = '[linewidth_size] processing model {} num {} pid {}'
        print(msg.format(s.basename, num, pid))

        lid = s.cores[pid].loc[num].leaf_id
        origin = tools.get_coords_node(s, lid)

        if ds is None:
            ds = s.load_hdf5(num, quantities=['dens', 'mom1', 'mom2', 'mom3'])
            ds['vel1'] = ds.mom1/ds.dens
            ds['vel2'] = ds.mom2/ds.dens
            ds['vel3'] = ds.mom3/ds.dens
    else:
        raise ValueError("Provide either seed or pid")

    d, origin = tools.recenter_dataset(ds, origin)
    d.coords['r'] = np.sqrt((d.z - origin[2])**2 + (d.y - origin[1])**2 + (d.x - origin[0])**2)

    rmax = s.Lbox/2
    nmax = np.floor(rmax/s.dx) + 1
    edges = np.insert(np.arange(s.dx/2, (nmax + 1)*s.dx, s.dx), 0, 0)
    d = d.sel(x=slice(origin[0] - edges[-1], origin[0] + edges[-1]),
              y=slice(origin[1] - edges[-1], origin[1] + edges[-1]),
              z=slice(origin[2] - edges[-1], origin[2] + edges[-1]))

    rprf = {}
    for k in ['vel1', 'vel2', 'vel3']:
        rprf[k] = transform.groupby_bins(d[k], 'r', edges, cumulative=True)
        rprf[k+'_sq'] = transform.groupby_bins(d[k]**2, 'r', edges, cumulative=True)
        rprf['d'+k] = np.sqrt(rprf[k+'_sq'] - rprf[k]**2)
    rprf['rho'] = transform.groupby_bins(d['dens'], 'r', edges, cumulative=True)
    rprf = xr.Dataset(rprf)

    # write to file
    if ofname.exists():
        ofname.unlink()
    rprf.to_netcdf(ofname)


def calculate_go15_core_mass(s, overwrite=False):
    """Calculate core mass using the definition of GO15

    Core mass is defined as the enclosed mass within the largest closed contour
    at t_coll
    """
    fname = Path(s.savdir) / 'mcore_go15.p'
    if fname.exists():
        if not overwrite:
            return
        else:
            fname.unlink()
    mcore = {}
    for pid in s.pids:
        cores = s.cores[pid]
        ncoll = cores.attrs['numcoll']
        ds = s.load_hdf5(ncoll, quantities=['dens'])
        gd = s.load_dendro(ncoll)
        lid = cores.loc[ncoll].leaf_id
        if np.isnan(lid):
            mcore[pid] = np.nan
        else:
            rho = gd.filter_data(ds.dens, lid, drop=True)
            mcore[pid] = (rho*s.dV).sum()
    with open(fname, 'wb') as f:
        pickle.dump(mcore, f)


def plot_pdfs(s, num, overwrite=False):
    """Creates density PDF and velocity power spectrum for a given model

    Save figures in {basedir}/figures for all snapshots.

    Args:
        s: pyathena.LoadSim instance
    """
    fname = Path(s.savdir, 'figures', "{}.{:05d}.png".format(
        config.PLOT_PREFIX_PDF_PSPEC, num))
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        print('[plot_pdfs] file already exists. Skipping...')
        return
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    ax1_twiny = axs[1].twiny()

    ds = s.load_hdf5(num, quantities=['dens', 'mom1', 'mom2', 'mom3'],
                     load_method='pyathena')
    plots.plot_PDF(s, ds, axs[0])
    plots.plot_Pspec(s, ds, axs[1], ax1_twiny)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def compare_projection(s1, s2, odir=Path("/tigress/sm69/public_html/files")):
    """Creates two panel plot comparing density projections

    Save projections in {basedir}/figures for all snapshots.

    Args:
        s1: pyathena.LoadSim instance
        s2: pyathena.LoadSim instance
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    nums = list(set(s1.nums) & set(s2.nums))
    odir = odir / "{}_{}".format(s1.basename, s2.basename)
    odir.mkdir(exist_ok=True)
    for num in nums:
        for ax, s in zip(axs, [s1, s2]):
            ds = s.load_hdf5(num, load_method='yt')
            plots.plot_projection(s, ds, ax=ax, add_colorbar=False)
            ax.set_title(r'$t={:.3f}$'.format(ds.current_time.value),
                         fontsize=16)
        fname = odir / "Projection_z_dens.{:05d}.png".format(num)
        fig.savefig(fname, bbox_inches='tight', dpi=200)
        for ax in axs:
            ax.cla()
