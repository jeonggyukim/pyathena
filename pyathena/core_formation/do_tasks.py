from pathlib import Path
import numpy as np
import argparse
import subprocess
from multiprocessing import Pool
import pyathena as pa
from pyathena.core_formation import config, tasks, tools

if __name__ == "__main__":
    # load all models
    models = {f"M5J2P{iseed}N512": f"/scratch/gpfs/sm69/cores/M5.J2.P{iseed}.N512"
          for iseed in range(0, 25)}
    for iseed in range(0, 6):
        models[f"M10J4P{iseed}N1024"] = f"/scratch/gpfs/sm69/cores/M10.J4.P{iseed}.N1024"
    models['M10J4P3N1024newsink'] = "/scratch/gpfs/sm69/cores/M10.J4.P3.N1024.newsink"
    models['M5J2P3N512newsink'] = "/scratch/gpfs/sm69/cores/M5.J2.P3.N512.newsink"
    models['M5J2P0N512dfloor'] = "/scratch/gpfs/sm69/cores/M5.J2.P0.N512.dfloor3"
    sa = pa.LoadSimCoreFormationAll(models)

    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs='+', type=str,
                        help="List of models to process")
    parser.add_argument("--pids", nargs='+', type=int,
                        help="List of particle ids to process")
    parser.add_argument("--np", type=int, default=1,
                        help="Number of processors")
    parser.add_argument("--combine-partab", action="store_true",
                        help="Join partab files")
    parser.add_argument("--combine-partab-full", action="store_true",
                        help="Join partab files including last output")
    parser.add_argument("-g", "--run-grid", action="store_true",
                        help="Run GRID-dendro")
    parser.add_argument("-c", "--core-tracking", action="store_true",
                        help="Eulerian core tracking")
    parser.add_argument("-r", "--radial-profile", action="store_true",
                        help="Calculate radial profiles of each cores")
    parser.add_argument("-t", "--critical-tes", action="store_true",
                        help="Calculate critical TES of each cores")
    parser.add_argument("--linewidth-size", action="store_true",
                        help="Calculate linewidth-size relation")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="Overwrite everything")
    parser.add_argument("-m", "--make-movie", action="store_true",
                        help="Create movies")
    parser.add_argument("--plot-core-evolution", action="store_true",
                        help="Create core evolution plots")
    parser.add_argument("--plot-sink-history", action="store_true",
                        help="Create sink history plots")
    parser.add_argument("--plot-pdfs", action="store_true",
                        help="Create density pdf and velocity power spectrum")
    parser.add_argument("--plot-diagnostics", action="store_true",
                        help="Create diagnostics plot for each core")
    parser.add_argument("--pid-start", type=int)
    parser.add_argument("--pid-end", type=int)

    args = parser.parse_args()

    # Select models
    for mdl in args.models:
        s = sa.set_model(mdl, force_override=True)

        if args.pid_start is not None and args.pid_end is not None:
            pids = np.arange(args.pid_start, args.pid_end+1)
        else:
            pids = s.pids
        if args.pids:
            pids = args.pids
        pids = sorted(list(set(s.pids) & set(pids)))

        # Combine output files.
        if args.combine_partab:
            print(f"Combine partab files for model {mdl}")
            tasks.combine_partab(s, remove=True, include_last=False)

        if args.combine_partab_full:
            print(f"Combine all partab files for model {mdl}")
            tasks.combine_partab(s, remove=True, include_last=True)

        # Run GRID-dendro.
        if args.run_grid:
            def wrapper(num):
                tasks.run_grid(s, num, overwrite=args.overwrite)
            print(f"Run GRID-dendro for model {mdl}")
            with Pool(args.np) as p:
                p.map(wrapper, s.nums[config.GRID_NUM_START:], 1)

        # Find t_coll cores and save their GRID-dendro node ID's.
        if args.core_tracking:
            def wrapper(pid):
                tasks.core_tracking(s, pid, overwrite=args.overwrite)
            print(f"Find t_coll cores for model {mdl}")
            with Pool(args.np) as p:
                p.map(wrapper, s.pids)

        # Calculate radial profiles of t_coll cores and pickle them.
        if args.radial_profile:
            msg = "calculate and save radial profiles of t_coll cores for"\
                  " model {}"
            print(msg.format(mdl))
            for pid in pids:
#                rmax = s.cores[pid].tidal_radius.max()
                rmax = None
                def wrapper(num):
                    tasks.radial_profile(s, pid, num,
                                         overwrite=args.overwrite,
                                         rmax=rmax)
                with Pool(args.np) as p:
                    p.map(wrapper, s.cores[pid].index)

                # Remove combined rprofs which will be outdated.
                fname = Path(s.savdir, 'radial_profile',
                             'radial_profile.par{}.nc'.format(pid))
                if fname.exists():
                    fname.unlink()

        # Find critical tes
        if args.critical_tes:
            print(f"find critical tes for t_coll cores for model {mdl}")
            for pid in pids:
                def wrapper(num):
                    tasks.critical_tes(s, pid, num, overwrite=args.overwrite)
                with Pool(args.np) as p:
                    p.map(wrapper, s.cores[pid].index)

        # Resample AMR data into uniform grid
#        print(f"resample AMR to uniform for model {mdl}")
#        tasks.resample_hdf5(s)

        # Calculate radial profiles of t_coll cores and pickle them.
        if args.linewidth_size:
            for num in [30, 40, 50, 60]:
                ds = s.load_hdf5(num, quantities=['dens', 'mom1', 'mom2', 'mom3'])
                ds['vel1'] = ds.mom1/ds.dens
                ds['vel2'] = ds.mom2/ds.dens
                ds['vel3'] = ds.mom3/ds.dens
                def wrapper(seed):
                    tasks.calculate_linewidth_size(s, num, seed, overwrite=args.overwrite, ds=ds)
                with Pool(args.np) as p:
                    p.map(wrapper, np.arange(1000))

        # make plots
        if args.plot_core_evolution:
            print(f"draw t_coll cores plots for model {mdl}")
            for pid in pids:
                # Read snapshot at t=t_coll and set plot limits
                num = s.tcoll_cores.loc[pid].num
                if len(s.cores[pid]) == 0:
                    continue
                core = s.cores[pid].loc[num]
                rprf = s.rprofs[pid].sel(num=num)
                rprf = tools.calculate_cumulative_energies(s, rprf, core)
                rprf = rprf.sel(r=slice(0, core.tidal_radius))
                emax = tools.roundup(max(rprf.ekin.max(), rprf.ethm.max()), 1)
                emin = tools.rounddown(rprf.egrv.min(), 1)

                def wrapper(num):
                    tasks.plot_core_evolution(s, pid, num,
                                              overwrite=args.overwrite,
                                              emin=emin, emax=emax)
                with Pool(args.np) as p:
                    p.map(wrapper, s.cores[pid].index)

        if args.plot_sink_history:
            def wrapper(num):
                tasks.plot_sink_history(s, num, overwrite=args.overwrite)
            print(f"draw sink history plots for model {mdl}")
            with Pool(args.np) as p:
                p.map(wrapper, s.nums)

        if args.plot_pdfs:
            def wrapper(num):
                tasks.plot_pdfs(s, num, overwrite=args.overwrite)
            print(f"draw PDF-power spectrum plots for model {mdl}")
            with Pool(args.np) as p:
                p.map(wrapper, s.nums)

        if args.plot_diagnostics:
            print(f"draw diagnostics plots for model {mdl}")
            for pid in pids:
                tasks.plot_diagnostics(s, pid, overwrite=args.overwrite)

        # make movie
        if args.make_movie:
            print(f"create movies for model {mdl}")
            srcdir = Path(s.savdir, "figures")
            plot_prefix = [config.PLOT_PREFIX_PDF_PSPEC, config.PLOT_PREFIX_SINK_HISTORY]
            for prefix in plot_prefix:
                subprocess.run(["make_movie", "-p", prefix, "-s", srcdir, "-d",
                                srcdir])
            for pid in pids:
                prefix = "{}.par{}".format(config.PLOT_PREFIX_TCOLL_CORES, pid)
                subprocess.run(["make_movie", "-p", prefix, "-s", srcdir, "-d",
                                srcdir])
