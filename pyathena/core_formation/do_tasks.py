from pathlib import Path
import subprocess
import pyathena as pa
from pyathena.core_formation.tasks import *

if __name__ == "__main__":
    # load all models
    models = dict(M10J4P0N256='/scratch/gpfs/sm69/cores/M10.J4.P0.N256',
                  M10J4P0N512='/scratch/gpfs/sm69/cores/M10.J4.P0.N512',
                  M5J2P0N256='/scratch/gpfs/sm69/cores/M5.J2.P0.N256',
                  M5J2P0N512='/scratch/gpfs/sm69/cores/M5.J2.P0.N512',
                  )
    sa = pa.LoadSimCoreFormationAll(models)

    # select some models
    models = ['M5J2P0N256']
    for mdl in models:
        s = sa.set_model(mdl)

        # combine output files
        combine_partab(s, remove=True)

        # make plots
        create_sinkhistory(s)
        create_projections(s)
        create_PDF_Pspec(s)

        # make movie
        srcdir = Path(s.basedir, "figures")
        plot_prefix = ["sink_history", "PDF_Pspecs"]
        for prefix in plot_prefix:
            subprocess.run(["make_movie", "-p", prefix, "-s", srcdir, "-d", srcdir])
            subprocess.run(["mv", "{}/{}.mp4".format(srcdir, prefix),
                "/tigress/sm69/public_html/files/{}.{}.mp4".format(mdl, prefix)])

        # other works
        run_fiso(s, overwrite=True)
        find_tcoll_cores(s, overwrite=True)
        save_radial_profiles_tcoll_cores(s, overwrite=True)
        resample_hdf5(s)
