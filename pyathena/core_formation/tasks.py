"""Module containing functions that are not generally reusable"""

# python modules
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess
import pickle
import glob
import numpy as np
import xarray as xr
import pandas as pd

# pyathena modules
from pyathena.core_formation import plots
from pyathena.core_formation import tools
from pyathena.core_formation import config
from pyathena.util import uniform
from grid_dendro import dendrogram
from grid_dendro import energy


def save_critical_tes(s, pids=None, overwrite=False):
    """Calculates and saves critical tes associated with each core"""
    if pids is None:
        pids = s.pids
    elif isinstance(pids, int):
        pids = [pids,]
    for pid in pids:
        for use_vel in ['disp', 'total']:
            for fixed_slope in [False, True]:
                # Check if file exists
                suffix = "vel{}".format(use_vel)
                if fixed_slope:
                    suffix += "_fixed_slope"
                ofname = Path(s.basedir, 'cores',
                              'critical_tes_{}.par{}.p'.format(suffix, pid))
                ofname.parent.mkdir(exist_ok=True)
                if ofname.exists() and not overwrite:
                    continue

                critical_tes = pd.DataFrame(columns=('num',
                                                     'center_density',
                                                     'edge_density',
                                                     'critical_radius',
                                                     'pindex',
                                                     'sonic_radius'),
                                            dtype=object).set_index('num')
                for num, _ in s.cores[pid].iterrows():
                    msg = '[save_critical_tes] processing model {} pid {} num {}'
                    msg = msg.format(s.basename, pid, num)
                    print(msg)
                    rprf = s.rprofs[pid].sel(num=num)
                    critical_tes.loc[num] = tools.calculate_critical_tes(s, rprf, use_vel, fixed_slope)

                # write to file
                critical_tes.to_pickle(ofname, protocol=pickle.HIGHEST_PROTOCOL)


def find_and_save_cores(s, pid, overwrite=False):
    """Loops over all sink particles and find their progenitor cores

    Finds a unique grid-dendro leaf at each snapshot that is going to collapse.
    For each sink particle, back-traces the evolution of its progenitor cores.
    Pickles the resulting data.

    Parameters
    ----------
    s : LoadSimCoreFormation
        LoadSimCoreFormation instance.
    pid : int
        Particle id.
    overwrite : str, optional
        If true, overwrites the existing pickle file.
    """
    def _get_node_distance(ds, nd1, nd2):
        pos1 = tools.get_coords_node(ds, nd1)
        pos2 = tools.get_coords_node(ds, nd2)
        dst = tools.get_periodic_distance(pos1, pos2, s.Lbox)
        return dst

    # Check if file exists
    ofname = Path(s.basedir, 'cores', 'cores.par{}.p'.format(pid))
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        continue

    # start from t = t_coll

    # Load data
    num = s.tcoll_cores.loc[pid].num
    ds = s.load_hdf5(num, load_method='pyathena')
    gd = s.load_dendrogram(num)

    # Calculate position, mass, and radius of the core
    nid_old = tools.find_tcoll_core(s, pid)
    pos_old = tools.get_coords_node(ds, nid_old)
    rho = gd.filter_data(ds.dens, nid_old)
    Mcore_old = (rho*s.dV).sum().data[()]
    Vcore = ((rho > 0).sum()*s.dV).data[()]
    Rcore_old = (3*Vcore/(4*np.pi))**(1./3.)

    # Add t_coll core to a list of progenitor cores
    cores = pd.DataFrame(dict(num=[num,],
                              time=[ds.Time,],
                              nid=[nid_old,],
                              radius=[Rcore_old,],
                              mass=[Mcore_old,]),
                         dtype=object).set_index("num")

    for num in np.arange(num-1, config.GRID_NUM_START-1, -1):
        msg = '[find_and_save_cores] processing model {} pid {} num {}'
        msg = msg.format(s.basename, pid, num)
        print(msg)
        # loop backward in time to find all preimages of the t_coll core
        ds = s.load_hdf5(num, load_method='pyathena')
        gd = s.load_dendrogram(num)

        # find closeast leaf to the previous preimage
        dst = {leaf: _get_node_distance(ds, leaf, nid_old)
               for leaf in gd.leaves}
        dst_min = np.min(list(dst.values()))
        for k, v in dst.items():
            if v == dst_min:
                nid = k

        # Calculate position, mass, and radius of the core and
        # check if this core is really the same core in different time
        pos = tools.get_coords_node(ds, nid)
        rho = gd.filter_data(ds.dens, nid)
        Mcore = (rho*s.dV).sum().data[()]
        Vcore = ((rho > 0).sum()*s.dV).data[()]
        Rcore = (3*Vcore/(4*np.pi))**(1./3.)

        # Relative errors in position, mass, and radius.
        # Note that the normalization is the maximum of current or
        # previous core;
        # This is to account for situation where a bud is suddenly merged
        # leading to sudden change in the core radius and mass.
        fdst = tools.get_periodic_distance(pos_old, pos, s.Lbox)\
            / max(Rcore, Rcore_old)
        fmass = np.abs(Mcore - Mcore_old) / max(Mcore, Mcore_old)
        frds = np.abs(Rcore - Rcore_old) / max(Rcore, Rcore_old)

        # If relative errors are more than 100%, this core is unlikely the
        # same core at previous timestep. Stop backtracing.
        if fdst > 1 or fmass > 1 or frds > 1:
            break

        # Add this core to list of progenitor cores
        cores.loc[num] = dict(nid=nid, time=ds.Time, radius=Rcore,
                              mass=Mcore)

        # Save core properties
        nid_old = nid
        pos_old = pos
        Mcore_old = Mcore
        Rcore_old = Rcore

    # write to file
    cores = cores.sort_values('num')
    cores.to_pickle(ofname, protocol=pickle.HIGHEST_PROTOCOL)


def save_radial_profiles(s, pids=None, overwrite=False):
    """Calculates and saves radial profiles of all cores"""
    rmax = 0.5*s.Lbox
    if pids is None:
        pids = s.pids
    elif isinstance(pids, int):
        pids = [pids,]
    for pid in pids:
        # Check if file exists
        ofname = Path(s.basedir, 'cores',
                      'radial_profile.par{}.nc'.format(pid))
        ofname.parent.mkdir(exist_ok=True)
        if ofname.exists() and not overwrite:
            continue

        time, rprf = [], []
        for num, core in s.cores[pid].iterrows():
            msg = '[save_radial_profiles] processing model {} pid {} num {}'
            msg = msg.format(s.basename, pid, num)
            print(msg)
            # Load the snapshot and the core id
            ds = s.load_hdf5(num, load_method='pyathena')
            ds = ds.transpose('z', 'y', 'x')

            # Find the location of the core
            center = tools.get_coords_node(ds, core.nid)

            # Roll the data such that the core is at the center of the domain
            ds, center = tools.recenter_dataset(ds, center)

            # Calculate radial profile
            time.append(ds.Time)
            rprf.append(tools.calculate_radial_profiles(s, ds, center, rmax))

        # Concatenate in time.
        rprf = xr.concat(rprf, dim=pd.Index(time, name='t'),
                         combine_attrs='drop_conflicts')
        rprf = rprf.assign_coords(dict(num=('t', s.cores[pid].index)))
        # When writing to netcdf and read, num is dropped from index list.
        rprf = rprf.set_xindex('num')

        # write to file
        rprf.to_netcdf(ofname)


def run_GRID(s, num, overwrite=False):
    # Check if file exists
    ofname = Path(s.basedir, 'GRID', 'dendrogram.{:05d}.p'.format(num))
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        print('[run_GRID] file already exists. Skipping...')
        return

    # Load data and construct dendrogram
    print('[run_GRID] processing model {} num {}'.format(s.basename, num))
    ds = s.load_hdf5(num, load_method='pyathena').transpose('z', 'y', 'x')
    gd = dendrogram.Dendrogram(ds.phigas.to_numpy(), verbose=False)
    gd.construct()
    gd.prune()

    # Write to file
    with open(ofname, 'wb') as handle:
        pickle.dump(gd, handle, protocol=pickle.HIGHEST_PROTOCOL)


def combine_partab(s, ns=None, ne=None, partag="par0", remove=False):
    script = "/home/sm69/tigris/vis/tab/combine_partab.sh"
    outid = "out{}".format(s.partab_outid)
    block0_pattern = '{}/{}.block0.{}.?????.{}.tab'.format(s.basedir,
                                                           s.problem_id, outid,
                                                           partag)
    file_list0 = sorted(glob.glob(block0_pattern))
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
            file_pattern = '{}/{}.block*.{}.?????.{}.tab'
            file_pattern = file_pattern.format(s.basedir, s.problem_id, outid,
                                               partag)
            file_list = sorted(glob.glob(file_pattern))
            for f in file_list:
                Path(f).unlink()
        else:
            print("Not all files are joined", flush=True)


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


def make_plots_core_evolution(s, pids=None, overwrite=False):
    """Creates multi-panel plot for t_coll core properties

    Args:
        s: pyathena.LoadSim instance
    """
    if pids is None:
        pids = s.pids
    elif isinstance(pids, int):
        pids = [pids,]
    for pid in pids:
        # Read snapshot at t=t_coll and set plot limits
        num = s.tcoll_cores.loc[pid].num
        ds = s.load_hdf5(num, load_method='pyathena')
        gd = s.load_dendrogram(num)
        core = s.cores[pid].loc[num]
        data = dict(rho=ds.dens.to_numpy(),
                     vel1=(ds.mom1/ds.dens).to_numpy(),
                     vel2=(ds.mom2/ds.dens).to_numpy(),
                     vel3=(ds.mom3/ds.dens).to_numpy(),
                     prs=s.cs**2*ds.dens.to_numpy(),
                     phi=ds.phigas.to_numpy(),
                     dvol=s.dV)
        reff, engs = energy.calculate_cumulative_energies(gd, data, core.nid)
        emax = tools.roundup(max(engs['ekin'].max(), engs['ethm'].max()), 1)
        emin = tools.rounddown(engs['egrv'].min(), 1)
        rmax = tools.roundup(reff.max(), 2)

        # Now, loop through cores and make plots
        for num, core in s.cores[pid].iterrows():
            msg = '[make_plots_core_evolution] processing model {} pid {} num {}'
            msg = msg.format(s.basename, pid, num)
            print(msg)
            fname = Path(s.basedir, 'figures', "{}.par{}.{:05d}.png".format(
                config.PLOT_PREFIX_TCOLL_CORES, pid, num))
            fname.parent.mkdir(exist_ok=True)
            if fname.exists() and not overwrite:
                continue
            fig = plots.plot_core_evolution(s, pid, num, emin=emin, emax=emax,
                                            rmax=rmax)
            fig.savefig(fname, bbox_inches='tight', dpi=200)
            plt.close(fig)


def make_plots_sinkhistory(s, overwrite=False):
    """Creates multi-panel plot for sink particle history

    Args:
        s: pyathena.LoadSim instance
    """
    for num in s.nums:
        fname = Path(s.basedir, 'figures', "{}.{:05d}.png".format(
            config.PLOT_PREFIX_SINK_HISTORY, num))
        fname.parent.mkdir(exist_ok=True)
        if fname.exists() and not overwrite:
            continue
        ds = s.load_hdf5(num, load_method='yt')
        pds = s.load_partab(num)
        fig = plots.plot_sinkhistory(s, ds, pds)
        fig.savefig(fname, bbox_inches='tight', dpi=200)
        plt.close(fig)


def make_plots_projections(s, overwrite=False):
    """Creates density projections for a given model

    Save projections in {basedir}/figures for all snapshots.

    Args:
        s: pyathena.LoadSim instance
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.05)
    for num in s.nums:
        fname = Path(s.basedir, 'figures',
                     "Projection_z_dens.{:05d}.png".format(num))
        if fname.exists() and not overwrite:
            continue
        ds = s.load_hdf5(num, load_method='yt')
        plots.plot_projection(s, ds, ax=ax, cax=cax)
        ax.set_title(r'$t={:.3f}$'.format(ds.current_time.value), fontsize=16)
        fig.savefig(fname, bbox_inches='tight', dpi=200)
        ax.cla()
        cax.cla()
    plt.close(fig)


def make_plots_PDF_Pspec(s, overwrite=False):
    """Creates density PDF and velocity power spectrum for a given model

    Save figures in {basedir}/figures for all snapshots.

    Args:
        s: pyathena.LoadSim instance
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    ax1_twiny = axs[1].twiny()
    for num in s.nums:
        fname = Path(s.basedir, 'figures', "{}.{:05d}.png".format(
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


def make_plots_central_density_evolution(s, overwrite=False):
    """Creates plot showing central density evolution for each cores

    Save figures in {basedir}/figures for all snapshots.

    Args:
        s: pyathena.LoadSim instance
    """
    fname = Path(s.basedir, 'figures',
                 "{}.png".format(config.PLOT_PREFIX_RHOC_EVOLUTION))
    fname.parent.mkdir(exist_ok=True)
    if fname.exists() and not overwrite:
        return
    plots.plot_central_density_evolution(s)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
