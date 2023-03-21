import yt
from pathlib import Path
import pyathena as pa
import matplotlib.pyplot as plt
from pyathena.core_formation.plots import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyathena.util import uniform
from pyathena.io.athena_read import partab
import subprocess

models = dict(N256 = '/scratch/gpfs/sm69/cores/M10.P0.N256',
              N256_niter0 = '/scratch/gpfs/sm69/cores/M10.P0.N256.niter0',
              N256_roe_fofc = '/scratch/gpfs/sm69/cores/M10.P0.N256.roe.fofc',
              N256_amr = '/scratch/gpfs/sm69/cores/M10.P0.N256.amr',
              N256_amr_lmax2 = '/scratch/gpfs/sm69/cores/M10.P0.N256.amr.lmax2',
              N256_amr_TL = '/scratch/gpfs/sm69/cores/M10.P0.N256.amr.TL',
              N1024 = '/scratch/gpfs/sm69/cores/M10.P0.N1024',
              largebox = '/scratch/gpfs/sm69/cores/M10.P0.N512.samesonic',
              smallbox = '/scratch/gpfs/sm69/cores/M10.P0.N256.samesonic.L2',
              M5 = '/scratch/gpfs/sm69/cores/M5.P0.N256',
              sink256 = '/scratch/gpfs/sm69/cores/M10.P0.N256.sink.merge',
              sink512 = '/scratch/gpfs/sm69/cores/M10.P0.N512.sink.merge',
              )
sa = pa.LoadSimCoreFormationAll(models)

def combine_partab(mdl, outid="out3", parid="par0"):
    s = sa.set_model(mdl)
    script = "/home/sm69/tigris/vis/tab/combine_partab.sh"
    subprocess.run([script, s.problem_id, outid, parid], cwd=s.basedir)
    subprocess.run(["find", ".", "-name",
                    '{}.block*.{}.*.{}.tab'.format(s.problem_id, outid, parid),
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

    models = ['sink256']
    for mdl in models:
        combine_partab(mdl)
        create_sinkhistory(mdl)

    # make movie and move mp4 to public
#    plot_prefix = ["Projection_z_dens", "PDF_Pspecs", "sink_history"]
    plot_prefix = ["sink_history"]
    for mdl in models:
        s = sa.set_model(mdl)
        srcdir = Path(s.basedir, "figures")
        for prefix in plot_prefix:
            subprocess.run(["make_movie", "-p", prefix, "-s", srcdir, "-d", srcdir])
            subprocess.run(["mv", "{}/{}.mp4".format(srcdir, prefix),
                "/tigress/sm69/public_html/files/{}.{}.mp4".format(mdl, prefix)])

#    compare_projection("N256", "largebox")
#    compare_projection("N256", "smallbox")

###        resample_hdf5(mdl)
#        create_projections(mdl)
#        try:
#            create_PDF_Pspec(mdl)
#        except:
#            pass
