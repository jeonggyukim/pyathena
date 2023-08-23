from pathlib import Path
import argparse
from multiprocessing import Pool
import pyathena as pa
from pyathena.tigress_gc import tasks, config

if __name__ == "__main__":
    # load all models
    models = dict(Binf='/tigress/sm69/public_html/M1Binf_512',
                  B100='/tigress/sm69/public_html/M1B100_512',
                  B30='/tigress/sm69/M1B30_512',
                  B10='/tigress/sm69/M1B10_512',
                  L0='/projects/EOSTRIKE/TIGRESS-GC/L0_512',
                  L1='/projects/EOSTRIKE/TIGRESS-GC/L1_512',
                  L2='/projects/EOSTRIKE/TIGRESS-GC/L2_512',
                  L3='/projects/EOSTRIKE/TIGRESS-GC/L3_512',
                  S0='/projects/EOSTRIKE/TIGRESS-GC/S0_256',
                  S1='/projects/EOSTRIKE/TIGRESS-GC/S1_256',
                  S2='/projects/EOSTRIKE/TIGRESS-GC/S2_256',
                  S3='/projects/EOSTRIKE/TIGRESS-GC/S3_256')
    sa = pa.LoadSimTIGRESSGCAll(models)

    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs='+', type=str,
                        help="List of models to process")
    parser.add_argument("--np", type=int, default=1,
                        help="Number of processors")
    parser.add_argument("--prfm", action="store_true",
                        help="Write prfm quantities")
    parser.add_argument("--azimuthal-average", action="store_true",
                        help="Calculate azimuthally averaged quantities")
    parser.add_argument("--ring-average", action="store_true",
                        help="Calculate ring masked averages")
    parser.add_argument("--time-average", action="store_true",
                        help="Produce time averaged snapshots")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="Overwrite everything")

    args = parser.parse_args()

    ts = {mdl:200 for mdl in models}
    te = {mdl:300 for mdl in models}
    ts['L0'], te['L0'] = 300, 400
    ts['L1'], te['L1'] = 300, 400


    # Select models
    for mdl in args.models:
        s = sa.set_model(mdl)

        # Calculate PRFM quantities
        if args.prfm:
            def wrapper(num):
                tasks.prfm_quantities(s, num, overwrite=args.overwrite)
            print(f"Calculate PRFM quantities for model {mdl}")
            with Pool(args.np) as p:
                p.map(wrapper, s.nums[config.NUM_START:], 1)

        # Calculate azimuthal averages
        if args.azimuthal_average:
            print(f"Calculate azimuthal averages for model {mdl}")
            tasks.save_azimuthal_averages(s, overwrite=args.overwrite)

        # Calculate ring averages
        if args.ring_average:
            fname = Path(s.basedir, "time_averages", "prims.nc")
            if fname.exists() and mdl in config.Rmax:
                print(f"Calculate ring masked averages for model {mdl}")
                tasks.save_ring_averages(s, config.Rmax[mdl], mf_crit=0.9, overwrite=args.overwrite)
        # Calculate time averages
        if args.time_average:
            if mdl not in ['B100', 'B30', 'B10']:
                print(f"Calculate time averages for model {mdl}")
                tasks.save_time_averaged_snapshot(s, ts[mdl], te[mdl],
                                                  overwrite=args.overwrite)
