# python modules
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess
import pickle
import glob

# pyathena modules
import pyathena as pa
from pyathena.core_formation.plots import *
from pyathena.util import uniform
from fiso.fiso_tree import construct_tree, calc_leaf
from fiso.tree_bound import compute


models = dict(M10J4P0N256='/scratch/gpfs/sm69/cores/M10.J4.P0.N256',
              M10J4P0N512='/scratch/gpfs/sm69/cores/M10.J4.P0.N512',
              M5J2P0N256='/scratch/gpfs/sm69/cores/M5.J2.P0.N256',
              M5J2P0N512='/scratch/gpfs/sm69/cores/M5.J2.P0.N512',
              )
sa = pa.LoadSimCoreFormationAll(models)


def find_tcoll_cores(mdl):
    """Loop over all sink particles and find their associated t_coll cores
    """
    s = sa.set_model(mdl)
    tcoll_cores = dict()
    for pid in s.pids[:30]:
        num = s.nums_tcoll[pid]

        # load fiso dict at t = t_coll
        fname = Path(s.basedir, 'fiso.{:05d}.p'.format(num))
        with open(fname, 'rb') as handle:
            fiso_dicts = pickle.load(handle)
            leaf_dict = fiso_dicts['leaf_dict']

        # find t_coll core associated with this pid
        rsq_max = 3
        while pid not in tcoll_cores:
            for iso in leaf_dict:
                k, j, i = np.unravel_index(iso, s.domain['Nx'], order='C')
                i0, j0, k0 = (np.array((s.xp0[pid], s.yp0[pid], s.zp0[pid]))
                              - s.domain['le']) // s.domain['dx']
                rsq = (k-k0)**2 + (j-j0)**2 + (k-k0)**2
                if rsq <= rsq_max:
                    if pid in tcoll_cores:
                        raise ValueError("More than one potential t_coll cores are found near this sink particle. Reduce the threshold")
                    tcoll_cores[pid] = iso
            rsq_max += 1
            if rsq_max > 25:
                print(pid)
                raise ValueError("Cannot find a t_coll core within 5*dx from the sink particle")
    ofname = Path(s.basedir, 'tcoll_cores.p')
    with open(ofname, 'wb') as handle:
        pickle.dump(fiso_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_fiso(mdl, overwrite=False):
    s = sa.set_model(mdl)
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

        # pack data for fiso input
        rho = ds.dens
        phi = ds.phi
        prs = cs**2*rho
        vx = ds.mom1/rho
        vy = ds.mom2/rho
        vz = ds.mom3/rho
        bpressure = 0.0*rho
        data = [rho.data, phi.data, prs.data, bpressure.data, vx.data, vy.data, vz.data]

        # Construct isocontours using fiso
        iso_dict, iso_label, iso_list, eic_list = construct_tree(phi.data, 'periodic')
        leaf_dict = calc_leaf(iso_dict, iso_list, eic_list)
        hpr_dict, hbr_dict = compute(data, iso_dict, iso_list, eic_list)
        # remove empty HBRs
        hbr_dict = {key: value for key, value in hbr_dict.items() if len(value)>0}

        fiso_dicts = dict(iso_dict=iso_dict, iso_label=iso_label,
                          iso_list=iso_list, eic_list=eic_list,
                          leaf_dict=leaf_dict, hpr_dict=hpr_dict, hbr_dict=hbr_dict)
        ofname = Path(s.basedir, 'fiso.{:05d}.p'.format(num))
        if ofname.exists() and not overwrite:
            continue
        with open(ofname, 'wb') as handle:
            pickle.dump(fiso_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)


def combine_partab(mdl, ns=None, ne=None, partag="par0", remove=False):
    script = "/home/sm69/tigris/vis/tab/combine_partab.sh"
    s = sa.set_model(mdl)
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
        file_pattern = '{}.{}.?????.{}.tab'.format(s.problem_id, outid, partag)
        file_list = glob.glob(file_pattern)
        num_files = len(file_list)
        if (num_files == (ne - ns + 1)):
            subprocess.run(["find", ".", "-name",
                            '{}.block*.{}.?????.{}.tab'.format(s.problem_id, outid, partag),
                            "-delete"], cwd=s.basedir)

def resample_hdf5(mdl, level=0):
    """Resamples AMR output into uniform resolution.

    Reads a HDF5 file with a mesh refinement and resample it to uniform
    resolution amounting to a given refinement level.

    Resampled HDF5 file will be written as
        {basedir}/uniform/{problem_id}.level{level}.?????.athdf

    Args:
        mdl: Model name.
        level: Refinement level to resample. root level=0.
    """
    s = sa.set_model(mdl)
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

def compare_projection(mdl1, mdl2, odir=Path("/tigress/sm69/public_html/files")):
    """Creates two panel plot comparing density projections

    Save projections in {basedir}/figures for all snapshots.

    Args:
        mdl1: Model name
        mdl2: Model name
    """
    fig, axs = plt.subplots(1,2,figsize=(14,7))
    s1 = sa.set_model(mdl1)
    s2 = sa.set_model(mdl2)
    nums = list(set(s1.nums) & set(s2.nums))
    odir = odir / "{}_{}".format(mdl1, mdl2)
    odir.mkdir(exist_ok=True)
    for num in nums:
        for ax, mdl in zip(axs, [mdl1, mdl2]):
            s = sa.set_model(mdl)
            ds = s.load_hdf5(num, load_method='yt')
            plot_projection(s, ds, ax=ax, add_colorbar=False)
            ax.set_title(r'$t={:.3f}$'.format(ds.current_time.value), fontsize=16)
        fname = odir / "Projection_z_dens.{:05d}.png".format(num)
        fig.savefig(fname, bbox_inches='tight', dpi=200)
        for ax in axs:
            ax.cla()

def create_sinkhistory(mdl):
    s = sa.set_model(mdl)
    for num in s.nums:
        ds = s.load_hdf5(num, load_method='yt')
        pds = s.load_partab(num)
        fig = plot_sinkhistory(s, ds, pds)
        odir = Path(s.basedir, 'figures')
        odir.mkdir(exist_ok=True)
        fname = odir / "sink_history.{:05d}.png".format(num)
        fig.savefig(fname, bbox_inches='tight', dpi=200)
        plt.close(fig)

def create_projections(mdl):
    """Creates density projections for a given model

    Save projections in {basedir}/figures for all snapshots.

    Args:
        mdl: Model name
    """
    fig, ax = plt.subplots(figsize=(8,8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.05)
    s = sa.set_model(mdl)
    for num in s.nums:
        ds = s.load_hdf5(num, load_method='yt')
        plot_projection(s, ds, ax=ax, cax=cax)
        odir = Path(s.basedir, 'figures')
        odir.mkdir(exist_ok=True)
        fname = odir / "Projection_z_dens.{:05d}.png".format(num)
        ax.set_title(r'$t={:.3f}$'.format(ds.current_time.value), fontsize=16)
        fig.savefig(fname, bbox_inches='tight', dpi=200)
        ax.cla()
        cax.cla()

def create_PDF_Pspec(mdl):
    """Creates density PDF and velocity power spectrum for a given model

    Save figures in {basedir}/figures for all snapshots.

    Args:
        mdl: Model name
    """
    s = sa.set_model(mdl)
    fig, axs = plt.subplots(1,2,figsize=(12,6))
    for num in s.nums:
        ds = s.load_hdf5(num, load_method='pyathena')
        plot_PDF(s, ds, axs[0])
        plot_Pspec(s, ds, axs[1])
        fig.tight_layout()
        odir = Path(s.basedir, 'figures')
        odir.mkdir(exist_ok=True)
        fname = odir / "PDF_Pspecs.{:05d}.png".format(num)
        fig.savefig(fname, bbox_inches='tight')
        for ax in axs:
            ax.cla()


if __name__ == "__main__":
    models = ['M5J2P0N256']
    for mdl in models:
        # combine output files
        combine_partab(mdl, remove=True)

        # make plots
        create_sinkhistory(mdl)
        create_projections(mdl)
        create_PDF_Pspec(mdl)

        # make movie
        s = sa.set_model(mdl)
        srcdir = Path(s.basedir, "figures")
        plot_prefix = ["sink_history", "PDF_Pspecs"]
        for prefix in plot_prefix:
            subprocess.run(["make_movie", "-p", prefix, "-s", srcdir, "-d", srcdir])
            subprocess.run(["mv", "{}/{}.mp4".format(srcdir, prefix),
                "/tigress/sm69/public_html/files/{}.{}.mp4".format(mdl, prefix)])

        # other works
        run_fiso(mdl, overwrite=True)
        find_tcoll_cores(mdl)
        resample_hdf5(mdl)
