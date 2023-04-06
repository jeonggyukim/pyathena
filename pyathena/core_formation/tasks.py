# python modules
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess
import pickle
import glob
import xarray as xr
import pandas as pd

# pyathena modules
from pyathena.core_formation import plots
from pyathena.core_formation import tools
from pyathena.util import uniform
from grid_dendro import dendrogram


def save_tcoll_cores(s):
    """Loop over all sink particles and find their associated t_coll cores
    """
    tcoll_cores = dict()
    for pid in s.pids:
        tcoll_cores[pid] = tools.find_tcoll_core(s, pid)

    # write to file
    ofname = Path(s.basedir, 'tcoll_cores', 'grid_dendro_nodes.p')
    ofname.parent.mkdir(exist_ok=True)
    with open(ofname, 'wb') as handle:
        pickle.dump(tcoll_cores, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_radial_profiles_tcoll_cores(s, overwrite=False):
    # Load the progenitor GRID-cores of each particles
    fname = Path(s.basedir, 'tcoll_cores', 'grid_dendro_nodes.p')
    with open(fname, 'rb') as handle:
        nodes = pickle.load(handle)

    rmax = 0.5*s.domain['Lx'][0]
    rprf = []
    for pid in s.pids:
        node = nodes[pid]

        # Load hdf5 snapshot at t = t_coll
        num = s.nums_tcoll[pid]
        ds = s.load_hdf5(num, load_method='pyathena')

        # Find the location of the core
        xc, yc, zc = tools.get_coords_node(ds, node)

        # Calculate radial profile
        rprf.append(tools.calculate_radial_profiles(ds, (xc, yc, zc), rmax))
    rprf = xr.concat(rprf, dim=pd.Index(s.pids, name='pid'), combine_attrs='drop_conflicts')

    # write to file
    ofname = Path(s.basedir, 'tcoll_cores', 'radial_profile.p')
    ofname.parent.mkdir(exist_ok=True)
    if ofname.exists() and not overwrite:
        return
    else:
        with open(ofname, 'wb') as handle:
            pickle.dump(rprf, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_GRID(s, overwrite=False):
    cs = s.par['hydro']['iso_sound_speed']

    # Assume uniform grid
    dx = (s.par['mesh']['x1max'] - s.par['mesh']['x1min'])/s.par['mesh']['nx1']
    dy = (s.par['mesh']['x2max'] - s.par['mesh']['x2min'])/s.par['mesh']['nx2']
    dz = (s.par['mesh']['x3max'] - s.par['mesh']['x3min'])/s.par['mesh']['nx3']
    dV = dx*dy*dz

    # Load data
    for num in s.nums[50:]:
        print('processing model {} num {}'.format(s.basename, num))
        ds = s.load_hdf5(num, load_method='pyathena').transpose('z','y','x')

        grd = dendrogram.Dendrogram(ds.phi.to_numpy())
        grd.construct()
        grd.prune()

        # write to file
        ofname = Path(s.basedir, 'GRID', 'leaves.{:05d}.p'.format(num))
        ofname.parent.mkdir(exist_ok=True)
        if ofname.exists() and not overwrite:
            continue
        with open(ofname, 'wb') as handle:
            pickle.dump(grd.leaves, handle, protocol=pickle.HIGHEST_PROTOCOL)


def combine_partab(s, ns=None, ne=None, partag="par0", remove=False):
    script = "/home/sm69/tigris/vis/tab/combine_partab.sh"
    if ns is None:
        ns = s.nums_partab[partag][0]
    if ne is None:
        ne = s.nums_partab[partag][-1]
    nblocks = 1
    for axis in [1,2,3]:
        nblocks *= (s.par['mesh'][f'nx{axis}'] // s.par['meshblock'][f'nx{axis}'])
    outid = "out{}".format(s.partab_outid)
    if not partag in s.partags:
        raise ValueError("Particle {} does not exist".format(partag))
    subprocess.run([script, s.problem_id, outid, partag, str(ns), str(ne)], cwd=s.basedir)

    if remove:
        file_pattern = '{}/{}.{}.?????.{}.tab'.format(s.basedir, s.problem_id, outid, partag)
        file_list = glob.glob(file_pattern)
        num_files = len(file_list)
        if (num_files == (ne - ns + 1)):
            subprocess.run(["find", ".", "-name",
                            '{}.block*.{}.?????.{}.tab'.format(s.problem_id, outid, partag),
                            "-delete"], cwd=s.basedir)


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
        fig = plots.plot_tcoll_cores(s, pid)
        odir = Path(s.basedir, 'figures')
        odir.mkdir(exist_ok=True)
        fname = odir / "tcoll_cores.par{}.png".format(pid)
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
        fname = odir / "sink_history.{:05d}.png".format(num)
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
        fname = odir / "PDF_Pspecs.{:05d}.png".format(num)
        fig.savefig(fname, bbox_inches='tight')
        for ax in axs:
            ax.cla()
        ax1_twiny.cla()
