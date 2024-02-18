from pathlib import Path
import pandas as pd

from .load_sim_tigress_ncr import LoadSimTIGRESSNCRAll

# Radiation paper (Linzer+)
models1 = dict(
    R8_8pc='/tigress/changgoo/TIGRESS-NCR/R8_8pc_NCR.full.xy2048.eps0.0',
    R8_4pc='/tigress/changgoo/TIGRESS-NCR/R8_4pc_NCR.full.xy2048.eps0.np768.has',
    LGR4_2pc='/tigress/changgoo/TIGRESS-NCR/LGR4_2pc_NCR.full',
)

# Metallicity suite (Kim+)
models2 = dict(
    R8_Z1='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b1.v3.iCR4.Zg1.Zd1.xy2048.eps0.0',
    R8_Z03='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b1.v3.iCR4.Zg0.3.Zd0.3.xy4096.eps0.0',
    R8_Z01='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b1.v3.iCR4.Zg0.1.Zd0.1.xy4096.eps0.0',
    R8_Zg01_Zd0025='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b1.v3.iCR4.Zg0.1.Zd0.025.xy4096.eps0.0',
    LGR4_Z1='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR.full.b1.v3.iCR4.Zg1.Zd1.xy1024.eps1.e-8',
    LGR4_Z03='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR.full.b1.v3.iCR4.Zg0.3.Zd0.3.xy2048.eps1.e-8',
    LGR4_Z01='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR.full.b1.v3.iCR4.Zg0.1.Zd0.1.xy2048.eps1.e-8',
    LGR4_Zg01_Zd0025='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR.full.b1.v3.iCR4.Zg0.1.Zd0.025.xy2048.eps1.e-8'
)

# Time range analyzed
tMyr_range = dict()

# Radiation paper (Linzer+)
tMyr_range['R8_8pc'] = [250,450]
tMyr_range['R8_4pc'] = [250,450]
tMyr_range['LGR4_2pc'] = [250,350]

# Metallicity suite
tMyr_range['R8_Z1'] = [438,977]
tMyr_range['R8_Z03'] = [438,977]
tMyr_range['R8_Z01'] = [438,977]
tMyr_range['R8_Zg01_Zd0025'] = [438,977]

tMyr_range['LGR4_Z1'] = [204,488]
tMyr_range['LGR4_Z03'] = [204,488]
tMyr_range['LGR4_Z01'] = [204,488]
tMyr_range['LGR4_Zg01_Zd0025'] = [204,488]

def get_summary(s, model):

    df = dict()
    df['tMyr_range'] = tMyr_range[model]
    df['nums'] = s.get_output_nums(df['tMyr_range'], out_fmt='vtk')
    df['nums_range'] = [df['nums'][0], df['nums'][-1]]
    df['nums_starpar'] = s.get_output_nums(df['tMyr_range'], out_fmt='starpar_vtk')
    df['nums_starpar_range'] = [df['nums_starpar'][0], df['nums_starpar'][-1]]

    return df

def load_sim_ncr_rad_all(model_set='radiation_paper',
                         savdir_base='/tigress/jk11/NCR-RAD',
                         verbose=False):
    """
    Load all simulations

    Parameters
    ----------
    model_set : str
        'radiation_paper' or 'lowz'
    savdir_base : str
        Base directory for saving results
    verbose : bool
        Produce verbose messages.

    Returns
    -------
    sa, df : LoadSimTIGRESSNCRAll, pandas DataFrame for summary
    """

    if model_set == 'radiation_paper':
        models = models1
    elif model_set == 'lowz':
        models = models2

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
