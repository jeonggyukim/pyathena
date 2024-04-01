from pathlib import Path
import numpy as np
import argparse
import subprocess
from multiprocessing import Pool
import pyathena as pa
from pyathena.core_formation import config, tasks, tools

if __name__ == "__main__":
    # load all models
    m1 = {f"M5J2P{iseed}N512": f"/scratch/gpfs/sm69/cores/hydro/M5.J2.P{iseed}.N512" for iseed in range(0, 40)}
    m2 = {f"M10J4P{iseed}N1024": f"/scratch/gpfs/sm69/cores/hydro/M10.J4.P{iseed}.N1024" for iseed in range(0, 7)}
    m3 = {f"M5J2P{iseed}N256": f"/scratch/gpfs/sm69/cores/hydro/M5.J2.P{iseed}.N256" for iseed in range(0, 1)}
    models = {**m1, **m2, **m3}

    # MHD models
    for iseed in [1,]:
        models[f"M5J2B2P{iseed}N512"] = f"/scratch/gpfs/sm69/cores/mhd/M5.J2.B2.P{iseed}.N512"
        models[f"M10J4B2P{iseed}N1024"] = f"/scratch/gpfs/sm69/cores/mhd/M10.J4.B2.P{iseed}.N1024"

    # Experimental
    models['M3J4P1N1024'] = "/tigress/sm69/cores/hydro/M3.J4.P1.N1024"
    models['M30J4P1N1024'] = "/tigress/sm69/cores/hydro/M30.J4.P1.N1024"
    models['M15J2P1N1024'] = "/tigress/sm69/cores/hydro/M15.J2.P1.N1024"
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
    parser.add_argument("--prune", action="store_true",
                        help="Prune dendrogram")
    parser.add_argument("--track-cores", action="store_true",
                        help="Perform reverse core tracking (prestellar phase)")
    parser.add_argument("--track-protostellar-cores", action="store_true",
                        help="Perform forward core tracking (protostellar phase)")
    parser.add_argument("-r", "--radial-profile", action="store_true",
                        help="Calculate radial profiles of each cores")
    parser.add_argument("-t", "--critical-tes", action="store_true",
                        help="Calculate critical TES of each cores")
    parser.add_argument("--lagrangian-props", action="store_true",
                        help="Calculate Lagrangian properties of cores")
    parser.add_argument("--observables", action="store_true",
                        help="Calculate observable properties of cores")
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

        # Run GRID-dendro.
        if args.prune:
            def wrapper(num):
                tasks.prune(s, num, overwrite=args.overwrite)
            print(f"Run GRID-dendro for model {mdl}")
            with Pool(args.np) as p:
                p.map(wrapper, s.nums[config.GRID_NUM_START:], 1)

        # Find t_coll cores and save their GRID-dendro node ID's.
        if args.track_cores:
            def wrapper(pid):
                tasks.core_tracking(s, pid, overwrite=args.overwrite)
            print(f"Perform core tracking for model {mdl}")
            with Pool(args.np) as p:
                p.map(wrapper, pids)

        # Perform forward core tracking (only for good cores).
        if args.track_protostellar_cores:
            def wrapper(pid):
                tasks.core_tracking(s, pid, protostellar=True, overwrite=args.overwrite)
            print(f"Perform protostellar core tracking for model {mdl}")
            # Only select resolved cores.
            pids = sorted(set(pids) & set(s.good_cores()))
            with Pool(args.np) as p:
                p.map(wrapper, pids)

        # Calculate radial profiles of t_coll cores and pickle them.
        if args.radial_profile:
            msg = ("calculate and save radial profiles for "
                   f"model {mdl}")
            print(msg)
            def wrapper(num):
                tasks.radial_profile(s, num, pids, overwrite=args.overwrite,
                                     full_radius=False, days_overwrite=0)
            with Pool(args.np) as p:
                p.map(wrapper, s.nums)

        # Find critical tes
        if args.critical_tes:
            print(f"find critical tes for cores for model {mdl}")
            for pid in pids:
                cores = s.cores[pid]
                def wrapper(num):
                    tasks.critical_tes(s, pid, num, overwrite=args.overwrite)
                with Pool(args.np) as p:
                    p.map(wrapper, cores.index)

        # Calculate Lagrangian properties
        if args.lagrangian_props:
            def wrapper(pid):
                for method in [1, 2, 3]:
                    s.select_cores(method)
                    if pid in s.cores:
                        tasks.lagrangian_props(s, pid, method=method, overwrite=args.overwrite)
            print(f"Calculate Lagrangian properties for model {mdl}")
            with Pool(args.np) as p:
                p.map(wrapper, pids)

        # Find observables
        if args.observables:
            print(f"Calculate observable core properties for model {mdl}")
            for pid in pids:
                cores = s.cores[pid]
                cores = cores.loc[:cores.attrs['numcoll']]
                def wrapper(num):
                    tasks.observables(s, pid, num, overwrite=args.overwrite)
                with Pool(args.np) as p:
                    p.map(wrapper, cores.index)


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
                    tasks.calculate_linewidth_size(s, num, seed=seed, overwrite=args.overwrite, ds=ds)
                with Pool(args.np) as p:
                    p.map(wrapper, np.arange(1000))

                def wrapper2(pid):
                    tasks.calculate_linewidth_size(s, num, pid=pid, overwrite=args.overwrite, ds=ds)
                with Pool(args.np) as p:
                    p.map(wrapper2, s.good_cores())

            def wrapper2(pid):
                ncrit = s.cores[pid].attrs['numcrit']
                tasks.calculate_linewidth_size(s, ncrit, pid=pid, overwrite=args.overwrite)
            with Pool(args.np) as p:
                p.map(wrapper2, s.good_cores())

        # make plots
        if args.plot_core_evolution:
            print(f"draw core evolution plots for model {mdl}")
            for pid in pids:
                for method in [1, 2]:
                    s.select_cores(method)
                    if pid not in s.good_cores():
                        continue
                    cores = s.cores[pid]
                    def wrapper(num):
                        tasks.plot_core_evolution(s, pid, num, method=method,
                                                  overwrite=args.overwrite)
                    with Pool(args.np) as p:
                        p.map(wrapper, cores.index)

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
            for pid in s.good_cores():
                tasks.plot_diagnostics(s, pid, overwrite=args.overwrite)

        # make movie
        if args.make_movie:
            print(f"create movies for model {mdl}")
            srcdir = Path(s.savdir, "figures")
            plot_prefix = [
#                    config.PLOT_PREFIX_PDF_PSPEC,
                    config.PLOT_PREFIX_SINK_HISTORY,
                          ]
            for prefix in plot_prefix:
                subprocess.run(["make_movie", "-p", prefix, "-s", srcdir, "-d",
                                srcdir])
            prefix = config.PLOT_PREFIX_CORE_EVOLUTION
            for pid in pids:
                for method in [1, 2]:
                    s.select_cores(method)
                    if pid not in s.good_cores():
                        continue
                    prf = f"{prefix}.par{pid}.ver{method}"
                    subprocess.run(["make_movie", "-p", prf, "-s", srcdir,
                                    "-d", srcdir])
