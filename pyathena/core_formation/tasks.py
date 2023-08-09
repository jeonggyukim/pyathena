"""Module containing functions that are not generally reusable"""

# python modules
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import subprocess
import pickle
import glob

# pyathena modules
from pyathena.core_formation import plots
from pyathena.core_formation import tools
from pyathena.core_formation import config
from pyathena.util import uniform
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


def core_tracking(s, pid, overwrite=False):
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
    overwrite : str, optional
        If true, overwrites the existing pickle file.
    """
    # Check if file exists
    ofname = Path(s.savdir, 'cores', 'cores.par{}.p'.format(pid))
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[core_tracking] file already exists. Skipping...')
        return

    cores = tools.track_cores(s, pid)
    cores.to_pickle(ofname, protocol=pickle.HIGHEST_PROTOCOL)


def radial_profile(s, pid, num, overwrite=False, rmax=None):
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
    # Check if file exists
    ofname = Path(s.savdir, 'radial_profile',
                  'radial_profile.par{}.{:05d}.nc'.format(pid, num))
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[radial_profile] file already exists. Skipping...')
        return

    msg = '[radial_profile] processing model {} pid {} num {}'
    print(msg.format(s.basename, pid, num))

    # Load the snapshot and the core id
    ds = s.load_hdf5(num,
                     quantities=['dens','phi','mom1','mom2','mom3'],
                     load_method='pyathena')
    ds = ds.transpose('z', 'y', 'x')
    core = s.cores[pid].loc[num]

    if rmax is None:
        rmax = 0.5*s.Lbox

    # Find the location of the core
    center = tools.get_coords_node(s, core.leaf_id)

    # Roll the data such that the core is at the center of the domain
    ds, center = tools.recenter_dataset(ds, center)

    # Calculate the angular momentum vector within the tidal radius.
    x = ds.x - center[0]
    y = ds.y - center[1]
    z = ds.z - center[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    lx = (y*ds.mom3 - z*ds.mom2).where(r < core.tidal_radius).sum().data[()]*s.dV
    ly = (z*ds.mom1 - x*ds.mom3).where(r < core.tidal_radius).sum().data[()]*s.dV
    lz = (x*ds.mom2 - y*ds.mom1).where(r < core.tidal_radius).sum().data[()]*s.dV
    lvec = np.array([lx, ly, lz])

    # Calculate radial profile
    rprf = tools.calculate_radial_profile(s, ds, center, rmax, lvec)
    rprf = rprf.expand_dims(dict(t=[ds.Time,]))
    rprf['lx'] = xr.DataArray([lx,], dims='t')
    rprf['ly'] = xr.DataArray([ly,], dims='t')
    rprf['lz'] = xr.DataArray([lz,], dims='t')

    # write to file
    if ofname.exists():
        ofname.unlink()
    rprf.to_netcdf(ofname)


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


def plot_core_evolution(s, pid, num, overwrite=False,
                        emin=None, emax=None, rmax=None):
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
    msg = '[plot_core_evolution] processing model {} pid {} num {}'
    msg = msg.format(s.basename, pid, num)
    print(msg)
    fname = Path(s.savdir, 'figures', "{}.par{}.{:05d}.png".format(
        config.PLOT_PREFIX_TCOLL_CORES, pid, num))
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        return

    fig = plots.plot_core_evolution(s, pid, num, emin=emin, emax=emax,
                                    rmax=rmax)
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_sink_history(s, num, overwrite=False):
    """Creates multi-panel plot for sink particle history

    Args:
        s: pyathena.LoadSim instance
    """
    fname = Path(s.savdir, 'figures', "{}.{:05d}.png".format(
                 config.PLOT_PREFIX_SINK_HISTORY, num))
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        return
    ds = s.load_hdf5(num, load_method='yt')
    pds = s.load_partab(num)
    fig = plots.plot_sinkhistory(s, ds, pds)
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_core_structure(s, pid, overwrite=False):
    rmax = s.cores[pid].tidal_radius.max()
    for num in s.cores[pid].index:
        fname = Path(s.savdir, 'figures', "core_structure.par{}.{:05d}.png".format(pid, num))
        if fname.exists() and not overwrite:
            return
        msg = '[plot_core_evolution] processing model {} pid {} num {}'
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
    msg = '[plot_diagnostics] model {} pid {}'
    print(msg.format(s.basename, pid))
    fname = Path(s.savdir, 'figures',
                 'diagnostics_normalized.par{}.png'.format(pid))
    if fname.exists() and not overwrite:
        return
    fig = plots.plot_diagnostics(s, pid, normalize_time=True)
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)

    fname = Path(s.savdir, 'figures', 'diagnostics.par{}.png'.format(pid))
    if fname.exists() and not overwrite:
        return
    fig = plots.plot_diagnostics(s, pid, normalize_time=False)
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_pdfs(s, overwrite=False):
    """Creates density PDF and velocity power spectrum for a given model

    Save figures in {basedir}/figures for all snapshots.

    Args:
        s: pyathena.LoadSim instance
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    ax1_twiny = axs[1].twiny()
    for num in s.nums:
        fname = Path(s.savdir, 'figures', "{}.{:05d}.png".format(
            config.PLOT_PREFIX_PDF_PSPEC, num))
        fname.parent.mkdir(exist_ok=True)
        if fname.exists() and not overwrite:
            continue
        ds = s.load_hdf5(num, load_method='pyathena')
        plots.plot_PDF(s, ds, axs[0])
        plots.plot_Pspec(s, ds, axs[1], ax1_twiny)
        fig.tight_layout()
        fig.savefig(fname, bbox_inches='tight')
        for ax in axs:
            ax.cla()
        ax1_twiny.cla()
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
