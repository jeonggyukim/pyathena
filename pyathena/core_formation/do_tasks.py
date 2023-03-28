import yt
from pathlib import Path
import pyathena as pa
import matplotlib.pyplot as plt
from pyathena.core_formation.plots import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyathena.util import uniform
from pyathena.io.athena_read import partab
import subprocess
from fiso.fiso_tree import construct_tree, calc_leaf
from fiso.tree_bound import compute
import pickle

models = dict(N256='/scratch/gpfs/sm69/cores/without_sinkpar/M10.P0.N256',
              N256_niter0='/scratch/gpfs/sm69/cores/without_sinkpar/M10.P0.N256.niter0',
              N256_roe_fofc='/scratch/gpfs/sm69/cores/without_sinkpar/M10.P0.N256.roe.fofc',
              N256_amr='/scratch/gpfs/sm69/cores/without_sinkpar/M10.P0.N256.amr',
              N256_amr_lmax2='/scratch/gpfs/sm69/cores/without_sinkpar/M10.P0.N256.amr.lmax2',
              N256_amr_TL='/scratch/gpfs/sm69/cores/without_sinkpar/M10.P0.N256.amr.TL',
              N1024='/scratch/gpfs/sm69/cores/without_sinkpar/M10.P0.N1024',
              largebox='/scratch/gpfs/sm69/cores/without_sinkpar/M10.P0.N512.samesonic',
              smallbox='/scratch/gpfs/sm69/cores/without_sinkpar/M10.P0.N256.samesonic.L2',
              M5='/scratch/gpfs/sm69/cores/without_sinkpar/M5.P0.N256',
              M10J4P0N256='/scratch/gpfs/sm69/cores/M10.J4.P0.N256',
              M10J4P0N256_multiple_mblock_per_rank='/scratch/gpfs/sm69/cores/M10.J4.P0.N256.multiple_mblock_per_rank',
              M10J4P0N512='/scratch/gpfs/sm69/cores/M10.J4.P0.N512',
              )
sa = pa.LoadSimCoreFormationAll(models)


def construct_fiso_tree(mdl):
    s = sa.set_model(mdl)
    cs = s.par['hydro']['iso_sound_speed']

    # Assume uniform grid
    dx = (s.par['mesh']['x1max'] - s.par['mesh']['x1min'])/s.par['mesh']['nx1']
    dy = (s.par['mesh']['x2max'] - s.par['mesh']['x2min'])/s.par['mesh']['nx2']
    dz = (s.par['mesh']['x3max'] - s.par['mesh']['x3min'])/s.par['mesh']['nx3']
    dV = dx*dy*dz

    # Load data
    for num in s.nums[60:180]:
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
        iso_dict, iso_label, iso_list, eic_list = construct_tree(phi.data)
        leaf_dict = calc_leaf(iso_dict, iso_list, eic_list)
        hpr_dict, hbr_dict = compute(data, iso_dict, iso_list, eic_list)
        # remove empty HBRs
        hbr_dict = {key: value for key, value in hbr_dict.items() if len(value)>0}

        fiso_dicts = dict(iso_dict=iso_dict, iso_label=iso_label,
                          iso_list=iso_list, eic_list=eic_list,
                          leaf_dict=leaf_dict, hpr_dict=hpr_dict, hbr_dict=hbr_dict)
        ofname = Path(s.basedir, 'fiso.{:05d}.p'.format(num))
        with open(ofname, 'wb') as handle:
            pickle.dump(fiso_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)


def combine_partab(mdl, ns, ne, partag="par0", remove=False):
    script = "/home/sm69/tigris/vis/tab/combine_partab.sh"
    s = sa.set_model(mdl)
    nblocks = 1
    for axis in [1,2,3]:
        nblocks *= (s.par['mesh'][f'nx{axis}'] // s.par['meshblock'][f'nx{axis}'])
    outid = "out{}".format(s.partab_outid)
    if not partag in s.partags:
        raise ValueError("Particle {} does not exist".format(partag))
    subprocess.run([script, s.problem_id, outid, partag, str(nblocks), str(ns), str(ne)], cwd=s.basedir)
    if remove:
        subprocess.run(["find", ".", "-name",
                        '{}.block*.{}.*.{}.tab'.format(s.problem_id, outid, partag),
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
        fpartab = Path(s.basedir, "{}.out3.{:05d}.par0.tab".format(s.problem_id, num))
        pds = partab(fpartab)
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
    models = ['M10J4P0N512']
    for mdl in models:
#        construct_fiso_tree(mdl)
        combine_partab(mdl, 0, 180)
#        create_sinkhistory(mdl)

    # make movie and move mp4 to public
#    plot_prefix = ["Projection_z_dens", "PDF_Pspecs", "sink_history"]
#    plot_prefix = ["sink_history"]
#    for mdl in models:
#        s = sa.set_model(mdl)
#        srcdir = Path(s.basedir, "figures")
#        for prefix in plot_prefix:
#            subprocess.run(["make_movie", "-p", prefix, "-s", srcdir, "-d", srcdir])
#            subprocess.run(["mv", "{}/{}.mp4".format(srcdir, prefix),
#                "/tigress/sm69/public_html/files/{}.{}.mp4".format(mdl, prefix)])
#
#    compare_projection("N256", "largebox")
#    compare_projection("N256", "smallbox")

###        resample_hdf5(mdl)
#        create_projections(mdl)
#        try:
#            create_PDF_Pspec(mdl)
#        except:
#            pass
