from pathlib import Path
import argparse
import subprocess
from multiprocessing import Pool
import pyathena as pa
from pyathena.core_formation.tasks import *
from pyathena.core_formation.config import *
from grid_dendro import energy

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
                  M5J2P4N512='/scratch/gpfs/sm69/cores/M5.J2.P4.N512',
                  M5J2P5N512='/scratch/gpfs/sm69/cores/M5.J2.P5.N512',
                  M5J2P6N512='/scratch/gpfs/sm69/cores/M5.J2.P6.N512',
                  M5J2P7N512='/scratch/gpfs/sm69/cores/M5.J2.P7.N512')
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
    parser.add_argument("--use-phitot", default=False, action="store_true",
                        help="Use total gravitational potential for analysis")
    parser.add_argument("--correct-tidal-radius", action="store_true",
                        help="Find envelop tidal radius")

    args = parser.parse_args()

    # Select models
    for mdl in args.models:
        s = sa.set_model(mdl)

        s.use_phitot = True if args.use_phitot else False

        # Combine output files.
        if args.join_partab:
            print(f"Combine partab files for model {mdl}", flush=True)
            combine_partab(s, remove=True, include_last=False)

        if args.join_partab_full:
            print(f"Combine all partab files for model {mdl}", flush=True)
            combine_partab(s, remove=True, include_last=True)

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
                p.map(wrapper, s.pids)
            s._load_cores()

        if args.correct_tidal_radius:
            for pid in s.pids:
                # Check if file exists
                ofname = Path(s.basedir, 'cores',
                              'rtidal_correction.par{}.p'.format(pid))
                ofname.parent.mkdir(exist_ok=True)
                if ofname.exists() and not args.overwrite:
                    print("[correct_tidal_radius] file already exists."
                          " Skipping...")
                    continue

                # Do not use s.cores, which might have already been
                # preimage corrected. Read from raw data.
                fname = Path(s.basedir, 'cores', 'cores.par{}.p'.format(pid))
                cores = pd.read_pickle(fname)
                nid, rtidal = tools.find_rtidal_envelop(s, cores, tol=1.1)
                def wrapper(num):
                    msg = '[correct_tidal_radius] processing model {} pid {} num {}'
                    print(msg.format(s.basename, pid, num))
                    ds = s.load_hdf5(num)
                    gd = s.load_dendrogram(num)
                    rho = gd.filter_data(ds.dens, nid.loc[num], drop=True)
                    mtidal = (rho*s.dV).sum()
                    return (num, mtidal)
                with Pool(args.np) as p:
                    mtidal = p.map(wrapper, cores.index)
                mtidal = pd.Series(data = map(lambda x: x[1], mtidal),
                                   index = map(lambda x: x[0], mtidal),
                                   name='envelop_tidal_mass')
                mtidal = mtidal.sort_index()
                res = pd.DataFrame({nid.name:nid,
                                    rtidal.name:rtidal,
                                    mtidal.name:mtidal})
                res.to_pickle(ofname)

        # Calculate radial profiles of t_coll cores and pickle them.
        if args.radial_profile:
            # TODO(SMOON) radial profile calculation is too expensive.
            # Better to parallelize over `num`, or output individual radial profiles
            # as a seperate files.
            msg = "calculate and save radial profiles of t_coll cores for model {}"
            print(msg.format(mdl), flush=True)
            for pid in s.pids:
                def wrapper(num):
                    save_radial_profiles(s, pid, num, overwrite=args.overwrite)
                with Pool(args.np) as p:
                    p.map(wrapper, s.cores[pid].index)

                # Remove combined rprofs which will be outdated.
                fname = Path(s.basedir, 'radial_profile',
                             'radial_profile.par{}.nc'.format(pid))
                if fname.exists():
                    fname.unlink()
            s._load_radial_profiles()

        # Find critical tes
        if args.critical_tes:
            print(f"find critical tes for t_coll cores for model {mdl}", flush=True)
            for pid in s.pids:
                def wrapper(num):
                    save_critical_tes(s, pid, num, overwrite=args.overwrite)
                with Pool(args.np) as p:
                    p.map(wrapper, s.cores[pid].index)
            s._load_cores()

        # Resample AMR data into uniform grid
#        print(f"resample AMR to uniform for model {mdl}", flush=True)
#        resample_hdf5(s)

        # make plots
        if args.plot_core_evolution:
            print(f"draw t_coll cores plots for model {mdl}", flush=True)
            for pid in s.pids:
                # Read snapshot at t=t_coll and set plot limits
                num = s.tcoll_cores.loc[pid].num
                ds = s.load_hdf5(num, load_method='pyathena')
                gd = s.load_dendrogram(num)
                core = s.cores[pid].loc[num]
                data = dict(rho=ds.dens.to_numpy(),
                            vel1=(ds.mom1/ds.dens).to_numpy(),
                            vel2=(ds.mom2/ds.dens).to_numpy(),
                            vel3=(ds.mom3/ds.dens).to_numpy(),
                            prs=s.cs**2*ds.dens.to_numpy(),
                            phi=ds.phigas.to_numpy(),
                            dvol=s.dV)
                reff, engs = energy.calculate_cumulative_energies(gd, data, core.nid)
                emax = tools.roundup(max(engs['ekin'].max(), engs['ethm'].max()), 1)
                emin = tools.rounddown(engs['egrv'].min(), 1)
                rmax = tools.roundup(reff.max(), 2)
                def wrapper(num):
                    make_plots_core_evolution(s, pid, num,
                                              overwrite=args.overwrite,
                                              emin=emin, emax=emax, rmax=rmax)
                with Pool(args.np) as p:
                    p.map(wrapper, s.cores[pid].index)

        if args.plot_sink_history:
            print(f"draw sink history plots for model {mdl}", flush=True)
            make_plots_sinkhistory(s, overwrite=args.overwrite)

        if args.plot_pdfs:
            print(f"draw PDF-power spectrum plots for model {mdl}", flush=True)
            make_plots_PDF_Pspec(s, overwrite=args.overwrite)

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
