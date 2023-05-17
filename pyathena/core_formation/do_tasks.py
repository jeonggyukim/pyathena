from pathlib import Path
import argparse
import subprocess
import pyathena as pa
from pyathena.core_formation.tasks import *
from pyathena.core_formation.config import *

if __name__ == "__main__":
    # load all models
    models = dict(M10J2P0N512='/scratch/gpfs/sm69/cores/M10.J2.P0.N512',
                  M5J2P0N256='/scratch/gpfs/sm69/cores/M5.J2.P0.N256',
                  M5J2P0N512='/scratch/gpfs/sm69/cores/M5.J2.P0.N512',
                  M5J2P1N256='/scratch/gpfs/sm69/cores/M5.J2.P1.N256',
                  M5J2P1N512='/scratch/gpfs/sm69/cores/M5.J2.P1.N512',
                  M5J2P2N256='/scratch/gpfs/sm69/cores/M5.J2.P2.N256',
                  M5J2P2N512='/scratch/gpfs/sm69/cores/M5.J2.P2.N512')
    sa = pa.LoadSimCoreFormationAll(models)

    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs='+', type=str,
                        help="List of models to process")
    parser.add_argument("-g", "--overwrite_grid", action="store_true",
                        help="Overwrite GRID-dendro output")
    parser.add_argument("-t", "--overwrite_cores", action="store_true",
                        help="Overwrite prestellar cores")
    parser.add_argument("-r", "--overwrite_radial_profiles", action="store_true",
                        help="Overwrite radial profiles of prestellar cores")
    parser.add_argument("-c", "--overwrite_critical_tes", action="store_true",
                        help="Overwrite critical turbulent equilibrium spheres for "
                             "prestellar cores")
    parser.add_argument("-pt", "--overwrite_plots_core_evolution", action="store_true",
                        help="Overwrite core evolution plots")
    parser.add_argument("-ps", "--overwrite_sink_history", action="store_true",
                        help="Overwrite sink history plots")
    parser.add_argument("-pp", "--overwrite_PDF_Pspecs", action="store_true",
                        help="Overwrite PDF_Pspecs plots")
    parser.add_argument("-pr", "--overwrite_rhoc_evolution", action="store_true",
                        help="Overwrite central density evolution plots")
    parser.add_argument("--pids", nargs='+', type=int,
                        help="List of particle ids to process")
    args = parser.parse_args()

    # Select models
    for mdl in args.models:
        s = sa.set_model(mdl)

        # Combine output files.
        print(f"Combine partab files for model {mdl}", flush=True)
        combine_partab(s, remove=True)

        # Run GRID-dendro.
        print(f"run GRID-dendro for model {mdl}", flush=True)
        run_GRID(s, overwrite=args.overwrite_grid)

        # Find t_coll cores and save their GRID-dendro node ID's.
        print(f"find t_coll cores for model {mdl}", flush=True)
        find_and_save_cores(s, pids=args.pids, overwrite=args.overwrite_cores)
        try:
            s._load_cores()
        except FileNotFoundError:
            pass

        # Calculate radial profiles of t_coll cores and pickle them.
        print(f"calculate and save radial profiles of t_coll cores for model {mdl}", flush=True)
        save_radial_profiles(s, pids=args.pids, overwrite=args.overwrite_radial_profiles)
        try:
            s._load_radial_profiles()
        except FileNotFoundError:
            pass

        # Find critical tes
        print(f"find critical tes for t_coll cores for model {mdl}", flush=True)
        save_critical_tes(s, pids=args.pids, overwrite=True)
        try:
            s._load_critical_tes()
        except FileNotFoundError:
            pass

        # Resample AMR data into uniform grid
#        print(f"resample AMR to uniform for model {mdl}", flush=True)
#        resample_hdf5(s)

        # make plots
        print(f"draw t_coll cores plots for model {mdl}", flush=True)
        make_plots_core_evolution(s, pids=args.pids, overwrite=args.overwrite_plots_core_evolution)

        print(f"draw sink history plots for model {mdl}", flush=True)
        make_plots_sinkhistory(s, overwrite=args.overwrite_sink_history)

        print(f"draw PDF-power spectrum plots for model {mdl}", flush=True)
        make_plots_PDF_Pspec(s, overwrite=args.overwrite_PDF_Pspecs)

        print(f"draw central density evolution plot for model {mdl}", flush=True)
        make_plots_central_density_evolution(s, overwrite=args.overwrite_rhoc_evolution)

#        print(f"draw projection plots for model {mdl}", flush=True)
#        make_plots_projections(s)

        # make movie
        print(f"create movies for model {mdl}", flush=True)
        srcdir = Path(s.basedir, "figures")
        plot_prefix = [PLOT_PREFIX_SINK_HISTORY, PLOT_PREFIX_PDF_PSPEC]
        for prefix in plot_prefix:
            subprocess.run(["make_movie", "-p", prefix, "-s", srcdir, "-d", srcdir])
#            subprocess.run(["mv", "{}/{}.mp4".format(srcdir, prefix),
#                "/tigress/sm69/public_html/files/{}.{}.mp4".format(mdl, prefix)])
        for pid in s.pids:
            prefix = "{}.par{}".format(PLOT_PREFIX_TCOLL_CORES, pid)
            subprocess.run(["make_movie", "-p", prefix, "-s", srcdir, "-d", srcdir])
#            subprocess.run(["mv", "{}/{}.mp4".format(srcdir, prefix),
#                "/tigress/sm69/public_html/files/{}.{}.mp4".format(mdl, prefix)])
