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


def save_tcoll_cores(s, overwrite=False):
    """Loop over all sink particles and find their associated t_coll cores"""
    def _get_distance(ds, nd1, nd2):
        x, y, z = tools.get_coords_node(ds, nd1)
        x0, y0, z0 = tools.get_coords_node(ds, nd2)
        rds = np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
        return rds

    # Check if file exists
    ofname = Path(s.basedir, 'tcoll_cores', 'grid_dendro_nodes.p')
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        return

    tcoll_cores = dict()
    for pid in s.pids:
        # start from t = t_coll
        core_old = tools.find_tcoll_core(s, pid)
        num = s.nums_tcoll[pid]
        tcoll_cores[pid] = {num: core_old}
        ds = s.load_hdf5(num, load_method='pyathena')
        leaves = s.load_leaves(num)
        x_old, y_old, z_old = tools.get_coords_node(ds, core_old)
        rho = dendrogram.filter_by_node(ds.dens, leaves, core_old)
        Mcore_old = (rho*s.dV).sum().data[()]
        Vcore = ((rho>0).sum()*s.dV).data[()]
        Rcore_old = (3*Vcore/(4*np.pi))**(1./3.)

        for num in np.arange(num-1, config.GRID_NUM_START-1, -1):
            # loop backward in time to find all preimages of the t_coll cores
            ds = s.load_hdf5(num, load_method='pyathena')
            leaves = s.load_leaves(num)

            # find closeast leaf to the previous preimage
            dst = {leaf: _get_distance(ds, leaf, core_old) for leaf in leaves}
            dst_min = np.min(list(dst.values()))
            for k, v in dst.items():
                if v == dst_min:
                    core = k

            # Check if this core is really the same core in different time
            x, y, z = tools.get_coords_node(ds, core)
            rho = dendrogram.filter_by_node(ds.dens, leaves, core)
            Mcore = (rho*s.dV).sum().data[()]
            Vcore = ((rho>0).sum()*s.dV).data[()]
            Rcore = (3*Vcore/(4*np.pi))**(1./3.)
            # Relative error in position, normalized to the previous core radius
            fdst = np.sqrt((x - x_old)**2 + (y - y_old)**2 + (z - z_old)**2) / Rcore_old
            # Relative errors in mass and radius.
            fmass = np.abs(Mcore - Mcore_old) / Mcore_old
            frds = np.abs(Rcore - Rcore_old) / Rcore_old
            # If relative errors are more than 100%, this core is unlikely the
            # same core at previous timestep. Stop backtracing.
            if fdst > 1 or fmass > 1 or frds > 1:
                break

            tcoll_cores[pid][num] = core
            core_old = core
            x_old, y_old, z_old = x, y, z
            Mcore_old = Mcore
            Rcore_old = Rcore


    # write to file
    with open(ofname, 'wb') as handle:
        pickle.dump(tcoll_cores, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_radial_profiles_tcoll_cores(s, overwrite=False):
    rmax = 0.5*s.Lbox
    for pid in s.pids:
        # Check if file exists
        ofname = Path(s.basedir, 'tcoll_cores', 'radial_profile.par{}.p'.format(pid))
        ofname.parent.mkdir(exist_ok=True)
        if ofname.exists() and not overwrite:
            continue

        time, rprf = [], []
        for num, core in s.tcoll_cores[pid].items():
            # Load the snapshot and the core id
            ds = s.load_hdf5(num, load_method='pyathena').transpose('z','y','x')

            # Find the location of the core
            xc, yc, zc = tools.get_coords_node(ds, core)

            # Roll the data such that the core is at the center of the domain
            shape = np.array(list(ds.dims.values()), dtype=int)
            hNz, hNy, hNx = shape >> 1
            ishift = hNx - np.where(ds.x.data==xc)[0][0]
            jshift = hNy - np.where(ds.y.data==yc)[0][0]
            kshift = hNz - np.where(ds.z.data==zc)[0][0]
            ds = ds.roll(x=ishift, y=jshift, z=kshift)
            xc = ds.x.isel(x=hNx).data[()]
            yc = ds.y.isel(y=hNy).data[()]
            zc = ds.z.isel(z=hNz).data[()]

            # Calculate radial profile
            time.append(ds.Time)
            rprf.append(tools.calculate_radial_profiles(ds, (xc, yc, zc), rmax))
        rprf = xr.concat(rprf, dim=pd.Index(time, name='t'),
                         combine_attrs='drop_conflicts')

        # write to file
        with open(ofname, 'wb') as handle:
            pickle.dump(rprf, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_GRID(s, overwrite=False):
    cs = s.par['hydro']['iso_sound_speed']

    # Assume uniform grid
    dx = (s.par['mesh']['x1max'] - s.par['mesh']['x1min'])/s.par['mesh']['nx1']
    dy = (s.par['mesh']['x2max'] - s.par['mesh']['x2min'])/s.par['mesh']['nx2']
    dz = (s.par['mesh']['x3max'] - s.par['mesh']['x3min'])/s.par['mesh']['nx3']
    dV = dx*dy*dz

    for num in s.nums[config.GRID_NUM_START:]:
        # Check if file exists
        print('processing model {} num {}'.format(s.basename, num))
        ofname = Path(s.basedir, 'GRID', 'leaves.{:05d}.p'.format(num))
        ofname.parent.mkdir(exist_ok=True)
        if ofname.exists() and not overwrite:
            continue

        # Load data and construct dendrogram
        ds = s.load_hdf5(num, load_method='pyathena').transpose('z','y','x')
        grd = dendrogram.Dendrogram(ds.phigas.to_numpy())
        grd.construct()
        grd.prune()

        # Write to file
        with open(ofname, 'wb') as handle:
            pickle.dump(grd.leaves, handle, protocol=pickle.HIGHEST_PROTOCOL)


def combine_partab(s, ns=None, ne=None, partag="par0", remove=False):
    script = "/home/sm69/tigris/vis/tab/combine_partab.sh"
    outid = "out{}".format(s.partab_outid)
    block0_pattern = '{}/{}.block0.{}.?????.{}.tab'.format(s.basedir,
                                                         s.problem_id, outid,
                                                         partag)
    file_list0 = sorted(glob.glob(block0_pattern))
    if len(file_list0) == 0:
        print("Nothing to combine")
        return
    if ns is None:
        ns = int(file_list0[0].split('/')[-1].split('.')[3])
    if ne is None:
        ne = int(file_list0[-1].split('/')[-1].split('.')[3])
    nblocks = 1
    for axis in [1,2,3]:
        nblocks *= ((s.par['mesh'][f'nx{axis}']
                    // s.par['meshblock'][f'nx{axis}']))
    if not partag in s.partags:
        raise ValueError("Particle {} does not exist".format(partag))
    subprocess.run([script, s.problem_id, outid, partag, str(ns), str(ne)],
                   cwd=s.basedir)

    if remove:
        joined_pattern = '{}/{}.{}.?????.{}.tab'.format(s.basedir,
                                                      s.problem_id, outid,
                                                      partag)
        joined_files = set(glob.glob(joined_pattern))
        if {f.replace('block0.', '') for f in file_list0}.issubset(joined_files):
            print("All files are joined. Remove block* files")
            file_pattern = '{}/{}.block*.{}.?????.{}.tab'.format(s.basedir,
                                                                 s.problem_id, outid,
                                                                 partag)
            file_list = sorted(glob.glob(file_pattern))
            for f in file_list:
                Path(f).unlink()
        else:
            print("Not all files are joined")


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
    fig, axs = plt.subplots(1,2,figsize=(14,7))
    nums = list(set(s1.nums) & set(s2.nums))
    odir = odir / "{}_{}".format(s1.basename, s2.basename)
    odir.mkdir(exist_ok=True)
    for num in nums:
        for ax, s in zip(axs, [s1, s2]):
            ds = s.load_hdf5(num, load_method='yt')
            plots.plot_projection(s, ds, ax=ax, add_colorbar=False)
            ax.set_title(r'$t={:.3f}$'.format(ds.current_time.value), fontsize=16)
        fname = odir / "Projection_z_dens.{:05d}.png".format(num)
        fig.savefig(fname, bbox_inches='tight', dpi=200)
        for ax in axs:
            ax.cla()


def make_plots_tcoll_cores(s):
    """Creates multi-panel plot for t_coll core properties

    Args:
        s: pyathena.LoadSim instance
    """
    for pid in s.pids:
        for num in s.tcoll_cores[pid]:
            fig = plots.plot_tcoll_cores(s, pid, num)
            odir = Path(s.basedir, 'figures')
            odir.mkdir(exist_ok=True)
            fname = odir / "{}.par{}.{:05d}.png".format(
                    config.PLOT_PREFIX_TCOLL_CORES, pid, num)
            fig.savefig(fname, bbox_inches='tight', dpi=200)
            plt.close(fig)


def make_plots_sinkhistory(s):
    """Creates multi-panel plot for sink particle history

    Args:
        s: pyathena.LoadSim instance
    """
    for num in s.nums:
        ds = s.load_hdf5(num, load_method='yt')
        pds = s.load_partab(num)
        fig = plots.plot_sinkhistory(s, ds, pds)
        odir = Path(s.basedir, 'figures')
        odir.mkdir(exist_ok=True)
        fname = odir / "{}.{:05d}.png".format(config.PLOT_PREFIX_SINK_HISTORY, num)
        fig.savefig(fname, bbox_inches='tight', dpi=200)
        plt.close(fig)


def make_plots_projections(s):
    """Creates density projections for a given model

    Save projections in {basedir}/figures for all snapshots.

    Args:
        s: pyathena.LoadSim instance
    """
    fig, ax = plt.subplots(figsize=(8,8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.05)
    for num in s.nums:
        ds = s.load_hdf5(num, load_method='yt')
        plots.plot_projection(s, ds, ax=ax, cax=cax)
        odir = Path(s.basedir, 'figures')
        odir.mkdir(exist_ok=True)
        fname = odir / "Projection_z_dens.{:05d}.png".format(num)
        ax.set_title(r'$t={:.3f}$'.format(ds.current_time.value), fontsize=16)
        fig.savefig(fname, bbox_inches='tight', dpi=200)
        ax.cla()
        cax.cla()


def make_plots_PDF_Pspec(s):
    """Creates density PDF and velocity power spectrum for a given model

    Save figures in {basedir}/figures for all snapshots.

    Args:
        s: pyathena.LoadSim instance
    """
    fig, axs = plt.subplots(1,2,figsize=(12,6))
    ax1_twiny = axs[1].twiny()
    for num in s.nums:
        ds = s.load_hdf5(num, load_method='pyathena')
        plots.plot_PDF(s, ds, axs[0])
        plots.plot_Pspec(s, ds, axs[1], ax1_twiny)
        fig.tight_layout()
        odir = Path(s.basedir, 'figures')
        odir.mkdir(exist_ok=True)
        fname = odir / "{}.{:05d}.png".format(config.PLOT_PREFIX_PDF_PSPEC, num)
        fig.savefig(fname, bbox_inches='tight')
        for ax in axs:
            ax.cla()
        ax1_twiny.cla()
