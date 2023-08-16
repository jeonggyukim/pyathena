import argparse
import pyathena as pa
from pyathena.tigress_gc.tasks import *

if __name__ == "__main__":
    # load all models
    models = dict(Binf='/tigress/sm69/M1Binf_512',
                  B100='/tigress/sm69/M1B100_512',
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
    parser.add_argument("-oa", "--overwrite_azimuthal_averages", action="store_true",
                        help="Overwrite azimuthal averages")
    parser.add_argument("-or", "--overwrite_ring_averages", action="store_true",
                        help="Overwrite ring averages")
    parser.add_argument("-ot", "--overwrite_time_averages", action="store_true",
                        help="Overwrite time averages")

    args = parser.parse_args()

    ts = {mdl:200 for mdl in models}
    te = {mdl:300 for mdl in models}
    ts['L0'], te['L0'] = 300, 400
    ts['L1'], te['L1'] = 300, 400

    Rmax = dict(L0=710,
                L1=710,
                L2=760,
                L3=870,
                S0=185,
                S1=205,
                S2=245,
                S3=245)

    # Select models
    for mdl in args.models:
        s = sa.set_model(mdl)

        # Calculate azimuthal averages
        print(f"Calculate azimuthal averages for model {mdl}")
        save_azimuthal_averages(s, overwrite=args.overwrite_azimuthal_averages)

        # Calculate ring averages
        fname = Path(s.basedir, "time_averages", "prims.nc")
        if fname.exists() and mdl in Rmax:
            print(f"Calculate ring masked averages for model {mdl}")
            save_ring_averages(s, Rmax[mdl], mf_crit=0.9, overwrite=args.overwrite_ring_averages)

        # Calculate time averages
        if mdl not in ['B100', 'B30', 'B10']:
            print(f"Calculate time averages for model {mdl}")
            save_time_averaged_snapshot(s, ts[mdl], te[mdl],
                                        overwrite=args.overwrite_time_averages)
