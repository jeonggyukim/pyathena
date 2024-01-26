from pathlib import Path
import pandas as pd

from .load_sim_tigress_ncr import LoadSimTIGRESSNCRAll

num_range = dict()
num_range['R8_8pc'] = [255,459]
num_range['R8_4pc'] = [255,459]
num_range['LGR4_2pc'] = [511,714]

def get_summary(s, model):

    df = dict()
    df['num_range'] = num_range[model]

    return df

def load_sim_ncr_rad_all(savdir_base='/tigress/jk11/NCR-RAD',
                         verbose=False):

    models = dict(
        R8_8pc='/tigress/changgoo/TIGRESS-NCR/R8_8pc_NCR.full.xy2048.eps0.0',
        R8_4pc='/tigress/changgoo/TIGRESS-NCR/R8_4pc_NCR.full.xy2048.eps0.np768.has',
        LGR4_2pc='/tigress/changgoo/TIGRESS-NCR/LGR4_2pc_NCR.full',
    )

    sa = LoadSimTIGRESSNCRAll(models)

    df_list = []
    for mdl in sa.models:
        savdir = str(Path(savdir_base,
                          Path(sa.basedirs[mdl]).name))
        s = sa.set_model(mdl, savdir=savdir, verbose=verbose)
        df = get_summary(s, mdl)
        df_list.append(pd.DataFrame(pd.Series(df, name=mdl)).T)

    df = pd.concat(df_list, sort=True).sort_index(ascending=False)

    return sa, df

if __name__ == '__main__':
    sa, df = load_sim_ncr_rad_all(verbose=True)

    for mdl in sa.models[1:]:
        print(mdl)
        s = sa.set_model(mdl, verbose='INFO')
        print(s.savdir)
        nums = [n for n in range(*df.loc[mdl]['num_range'])]
        for k in s.phs.keys():
            print(k)
            zp = s.read_zprof_from_vtk_all(nums, phase_set_name=k, force_override=False)
