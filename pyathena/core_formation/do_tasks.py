from pathlib import Path
import argparse
import subprocess
from multiprocessing import Pool
import pyathena as pa
from pyathena.core_formation.tasks import *
from pyathena.core_formation.config import *

if __name__ == "__main__":
    # load all models
    models = dict(M10J2P0N512='/scratch/gpfs/sm69/cores/M10.J2.P0.N512',
                  M10J4P0N512='/scratch/gpfs/sm69/cores/M10.J4.P0.N512',
                  M10J4P1N512='/scratch/gpfs/sm69/cores/M10.J4.P1.N512',
                  M10J4P2N512='/scratch/gpfs/sm69/cores/M10.J4.P2.N512',
                  M10J4P0N1024='/scratch/gpfs/sm69/cores/M10.J4.P0.N1024',
                  M10J4P1N1024='/scratch/gpfs/sm69/cores/M10.J4.P1.N1024',
                  M10J4P2N1024='/scratch/gpfs/sm69/cores/M10.J4.P2.N1024',
                  M75J3P0N512='/scratch/gpfs/sm69/cores/M7.5.J3.P0.N512',
                  M75J3P1N512='/scratch/gpfs/sm69/cores/M7.5.J3.P1.N512',
                  M75J3P2N512='/scratch/gpfs/sm69/cores/M7.5.J3.P2.N512',
                  M5J2P0N256='/scratch/gpfs/sm69/cores/M5.J2.P0.N256',
                  M5J2P1N256='/scratch/gpfs/sm69/cores/M5.J2.P1.N256',
                  M5J2P2N256='/scratch/gpfs/sm69/cores/M5.J2.P2.N256',
                  M5J2P0N512='/scratch/gpfs/sm69/cores/M5.J2.P0.N512',
                  M5J2P1N512='/scratch/gpfs/sm69/cores/M5.J2.P1.N512',
                  M5J2P2N512='/scratch/gpfs/sm69/cores/M5.J2.P2.N512',
                  M5J2P3N512='/scratch/gpfs/sm69/cores/M5.J2.P3.N512',
                  M5J2P4N512='/scratch/gpfs/sm69/cores/M5.J2.P4.N512')
    sa = pa.LoadSimCoreFormationAll(models)

    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs='+', type=str,
                        help="List of models to process")
    parser.add_argument("--pids", nargs='+', type=int,
                        help="List of particle ids to process")

    parser.add_argument("--np", type=int, default=1, help="Number of processors")

    parser.add_argument("-j", "--join-partab", action="store_true",
                        help="Join partab files")
    parser.add_argument("-f", "--join-partab-full", action="store_true",
                        help="Join partab files including last output")
    parser.add_argument("-g", "--grid-dendro", action="store_true",
                        help="Run GRID-dendro")
    parser.add_argument("-c", "--core-tracking", action="store_true",
                        help="Eulerian core tracking")
    parser.add_argument("-r", "--radial-profile", action="store_true",
                        help="Calculate radial profiles of each cores")
    parser.add_argument("-t", "--critical-tes", action="store_true",
                        help="Calculate critical TES of each cores")
    parser.add_argument("-p", "--make-plots", action="store_true",
                        help="Create various plots")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="Overwrite everything")
    parser.add_argument("-m", "--make-movie", action="store_true",
                        help="Create movies")

    args = parser.parse_args()

    # Select models
    for mdl in args.models:
        s = sa.set_model(mdl)

        # Combine output files.
        if args.join_partab:
            print(f"Combine partab files for model {mdl}", flush=True)
            include_last = True if args.join_partab_full else False
            combine_partab(s, remove=True, include_last=include_last)

        # Run GRID-dendro.
        if args.grid_dendro:
            print(f"Run GRID-dendro for model {mdl}", flush=True)
            def wrapper(num):
                run_GRID(s, num, overwrite=args.overwrite)
            with Pool(args.np) as p:
                p.map(wrapper, s.nums[GRID_NUM_START:], 1)

        # Find t_coll cores and save their GRID-dendro node ID's.
        if args.core_tracking:
            print(f"find t_coll cores for model {mdl}", flush=True)
            def wrapper(pid):
                find_and_save_cores(s, pid, overwrite=args.overwrite)
            with Pool(args.np) as p:
                p.map(wrapper, s.pids, 1)
            try:
                s._load_cores()
            except FileNotFoundError:
                pass

        # Calculate radial profiles of t_coll cores and pickle them.
        if args.radial_profile:
            msg = "calculate and save radial profiles of t_coll cores for model {}"
            print(msg.format(mdl), flush=True)
            def wrapper(pid):
                save_radial_profiles(s, pid, overwrite=args.overwrite)
            with Pool(args.np) as p:
                p.map(wrapper, s.pids, 1)
            try:
                s._load_radial_profiles()
            except FileNotFoundError:
                pass

        # Find critical tes
        if args.critical_tes:
            print(f"find critical tes for t_coll cores for model {mdl}", flush=True)
            def wrapper(pid):
                save_critical_tes(s, pid, overwrite=args.overwrite)
            with Pool(args.np) as p:
                p.map(wrapper, s.pids, 1)
            try:
                s._load_cores()
            except FileNotFoundError:
                pass

        # Resample AMR data into uniform grid
#        print(f"resample AMR to uniform for model {mdl}", flush=True)
#        resample_hdf5(s)

        # make plots
        if args.make_plots:
            print(f"draw t_coll cores plots for model {mdl}", flush=True)
            def wrapper(pid):
                make_plots_core_evolution(s, pids=pid,
                                          overwrite=args.overwrite)
            with Pool(args.np) as p:
                p.map(wrapper, s.pids, 1)

            print(f"draw sink history plots for model {mdl}", flush=True)
            make_plots_sinkhistory(s, overwrite=args.overwrite)

            print(f"draw PDF-power spectrum plots for model {mdl}", flush=True)
            make_plots_PDF_Pspec(s, overwrite=args.overwrite)

            print(f"draw central density evolution plot for model {mdl}", flush=True)
            make_plots_central_density_evolution(s, overwrite=args.overwrite)

        # make movie
        if args.make_movie:
            print(f"create movies for model {mdl}", flush=True)
            srcdir = Path(s.basedir, "figures")
            plot_prefix = [PLOT_PREFIX_SINK_HISTORY, PLOT_PREFIX_PDF_PSPEC]
            for prefix in plot_prefix:
                subprocess.run(["make_movie", "-p", prefix, "-s", srcdir, "-d", srcdir])
            for pid in s.pids:
                prefix = "{}.par{}".format(PLOT_PREFIX_TCOLL_CORES, pid)
                subprocess.run(["make_movie", "-p", prefix, "-s", srcdir, "-d", srcdir])
