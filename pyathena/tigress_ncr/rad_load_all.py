from pathlib import Path
import pandas as pd
import cmasher as cmr
import matplotlib as mpl

from .load_sim_tigress_ncr import LoadSimTIGRESSNCRAll

# Radiation paper (Linzer+)
models1 = dict(
    R8_8pc='/tigress/changgoo/TIGRESS-NCR/R8_8pc_NCR.full.xy2048.eps0.0',
    R8_4pc='/tigress/changgoo/TIGRESS-NCR/R8_4pc_NCR.full.xy2048.eps0.np768.has',
    LGR4_2pc='/tigress/changgoo/TIGRESS-NCR/LGR4_2pc_NCR.full',
)

# Metallicity suite (KimCG+)
# https://github.com/PrincetonUniversity/Athena-TIGRESS/wiki/TIGRESS-NCR-Metallicity-Suite
models2 = dict(
    S05_Z1='/projects/EOSTRIKE/TIGRESS-NCR/LGR8_8pc_NCR_S05.full.b10.v3.iCR5.Zg1.Zd1.xy4096.eps0.0',
    S05_Z01='/projects/EOSTRIKE/TIGRESS-NCR/LGR8_8pc_NCR_S05.full.b10.v3.iCR5.Zg0.1.Zd0.1.xy8192.eps0.0',

    R8_Z1='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b1.v3.iCR4.Zg1.Zd1.xy2048.eps0.0',
    R8_Z03='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b1.v3.iCR4.Zg0.3.Zd0.3.xy4096.eps0.0',
    R8_Z01='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b1.v3.iCR4.Zg0.1.Zd0.1.xy4096.eps0.0',
    R8_Zg01_Zd0025='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b1.v3.iCR4.Zg0.1.Zd0.025.xy4096.eps0.0',

    R8_b10_Z1='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b10.v3.iCR4.Zg1.Zd1.xy2048.eps0.0',
    R8_b10_Z03='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b10.v3.iCR4.Zg0.3.Zd0.3.xy4096.eps0.0',
    R8_b10_Z01='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b10.v3.iCR4.Zg0.1.Zd0.1.xy4096.eps0.0',
    R8_b10_Zg01_Zd0025='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b10.v3.iCR4.Zg0.1.Zd0.025.xy4096.eps0.0',

    S30_Z1='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR_S30.full.b1.v3.iCR4.Zg1.Zd1.xy1024.eps0.0',
    S30_Z01='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR_S30.full.b1.v3.iCR4.Zg0.1.Zd0.1.xy2048.eps0.0',

    LGR4_Z1='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR.full.b1.v3.iCR4.Zg1.Zd1.xy1024.eps1.e-8',
    LGR4_Z03='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR.full.b1.v3.iCR4.Zg0.3.Zd0.3.xy2048.eps1.e-8',
    LGR4_Z01='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR.full.b1.v3.iCR4.Zg0.1.Zd0.1.xy2048.eps1.e-8',
    LGR4_Zg01_Zd0025='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR.full.b1.v3.iCR4.Zg0.1.Zd0.025.xy2048.eps1.e-8',

    LGR4_b10_Z1='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR.full.b10.v3.iCR4.Zg1.Zd1.xy1024.eps1.e-8',
    LGR4_b10_Z01='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR.full.b10.v3.iCR4.Zg0.1.Zd0.1.xy2048.eps1.e-8',

    S100_Z1r='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR_S100.full.b1.v3.iCR5.Zg1.Zd1.xy1024.eps1.e-8.rstZ01',
    S100_Z1='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR_S100.full.b1.v3.iCR4.Zg1.Zd1.xy1024.eps1.e-8',
    S100_Z01='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR_S100.full.b1.v3.iCR4.Zg0.1.Zd0.1.xy2048.eps1.e-8',

    S150_Om100q0_Z1='/projects/EOSTRIKE/TIGRESS-NCR/LGR2_4pc_NCR_S150.full.b2.Om01.q0.v3.iCR5.Zg1.Zd1.xy1024.eps1.e-8',
    S150_Om100q0_Z01='/projects/EOSTRIKE/TIGRESS-NCR/LGR2_4pc_NCR_S150.full.b2.Om01.q0.v3.iCR5.Zg0.1.Zd0.1.xy2048.eps1.e-8',
    S150_Om200_Z1r='/projects/EOSTRIKE/TIGRESS-NCR/LGR2_4pc_NCR_S150.full.b2.Om02.v3.iCR5.Zg1.Zd1.xy1024.eps1.e-8.rstZ01',
    S150_Om200_Z1='/projects/EOSTRIKE/TIGRESS-NCR/LGR2_4pc_NCR_S150.full.b2.Om02.v3.iCR5.Zg1.Zd1.xy1024.eps1.e-8',
    S150_Om200_Z01='/projects/EOSTRIKE/TIGRESS-NCR/LGR2_4pc_NCR_S150.full.b2.Om02.v3.iCR5.Zg0.1.Zd0.1.xy2048.eps1.e-8',

)

# Time range analyzed
tMyr_range = dict()

# Radiation paper (Linzer+)
tMyr_range['R8_8pc'] = [250,450]
tMyr_range['R8_4pc'] = [250,450]
tMyr_range['LGR4_2pc'] = [250,350]

# Metallicity suite (KimCG+)
tMyr_range['S05_Z1'] = [614,1703]
tMyr_range['S05_Z01'] = [614,1830]

tMyr_range['R8_Z1'] = [438,977]
tMyr_range['R8_Z03'] = [438,977]
tMyr_range['R8_Z01'] = [438,977]
tMyr_range['R8_Zg01_Zd0025'] = [438,977]

tMyr_range['R8_b10_Z1'] = [438,977]
tMyr_range['R8_b10_Z03'] = [438,977]
tMyr_range['R8_b10_Z01'] = [438,977]
tMyr_range['R8_b10_Zg01_Zd0025'] = [438,977]

tMyr_range['S30_Z1'] = [438,977]
tMyr_range['S30_Z01'] = [438,977]

tMyr_range['LGR4_Z1'] = [204,488]
tMyr_range['LGR4_Z03'] = [204,488]
tMyr_range['LGR4_Z01'] = [204,488]
tMyr_range['LGR4_Zg01_Zd0025'] = [204,488]

tMyr_range['LGR4_b10_Z1'] = [204,488]
tMyr_range['LGR4_b10_Z01'] = [204,488]

tMyr_range['S100_Z1r'] = [204,487]
tMyr_range['S100_Z1'] = [204,425]
tMyr_range['S100_Z01'] = [204,488]

tMyr_range['S150_Om100q0_Z1'] = [122,378]
tMyr_range['S150_Om100q0_Z01'] = [122,353]
tMyr_range['S150_Om200_Z1r'] = [153,336]
tMyr_range['S150_Om200_Z1'] = [153,397]
tMyr_range['S150_Om200_Z01'] = [153,412]

def get_summary(s, model, force_override=False):

    df = dict()
    df['tMyr_range'] = tMyr_range[model]
    df['nums'] = s.get_output_nums(df['tMyr_range'], out_fmt='vtk')
    df['nums_range'] = [df['nums'][0], df['nums'][-1]]
    df['nums_starpar'] = s.get_output_nums(df['tMyr_range'], out_fmt='starpar_vtk')
    df['nums_starpar_range'] = [df['nums_starpar'][0], df['nums_starpar'][-1]]
    df['beta'] = s.par['problem']['beta']
    df['Zd'] = float(s.par['problem']['Z_dust'])
    df['Zg'] = float(s.par['problem']['Z_gas'])

    # Summary from history
    h = s.read_hst_rad(force_override=force_override)
    df['fesc_LyC_cumul'] = h['fesc_LyC_cumul'].iloc[-1]

    # Set plot kwargs
    norm_Zg = mpl.colors.LogNorm(10.0**-1.3, 10.0**0.3)
    df['markeredgecolor'] = mpl.colors.rgb2hex(cmr.guppy(norm_Zg(df['Zd'])))

    return df

def load_sim_ncr_rad_all(model_set='radiation_paper',
                         savdir_base='/tigress/jk11/NCR-RAD',
                         force_override=False,
                         verbose=False):
    """
    Load all simulations for radiation analysis

    Parameters
    ----------
    model_set : str
        'radiation_paper' or 'lowZ'
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
    elif model_set == 'lowZ':
        models = models2
    else:
        raise ValueError

    sa = LoadSimTIGRESSNCRAll(models)

    df_list = []
    print('[load_sim_ncr_rad_all]:', end=' ')
    for mdl in sa.models:
        print(mdl, end=' ')
        savdir = str(Path(savdir_base, Path(sa.basedirs[mdl]).name))
        s = sa.set_model(mdl, savdir=savdir, verbose=verbose)
        df = get_summary(s, mdl, force_override=force_override)
        df_list.append(pd.DataFrame(pd.Series(df, name=mdl)).T)

    df = pd.concat(df_list, sort=True).sort_index(ascending=False)

    return sa, df

def load_zprof_new(s):
    """Load zprof that has new phase definitions. LGR4_2pc has old definitions, see
    https://github.com/PrincetonUniversity/Athena-TIGRESS/wiki/Phase-definition#ncr-new-phase-sep-history-branch
    """
    if s.config_time < pd.to_datetime('2022-03-15 00:00:00 -04:00'):
        raise RuntimeError('config_time {0:s} indicates that this simulation has '+\
                           'old phase definitions. Do not use this funtion.'.\
                           format(str(s.config_time)))

    zp = dict()
    phases = ['c','u','w1','w2','h1','h2', # based on temperature only
              # Temperature and abundances molecular (xH2 > 0.25)
              'CUMM',
              # Ionized (xHII > 0.5)
              'CUIM','WPIM','WCIM',
              # Neither molecular nor ionized (say neutral)
              'CNM','UNM','WNM']
    for j in range(13):
        ph = 'phase{}'.format(j + 1)
        zp_ =  s._read_zprof(phase=ph)
        zp[phases[j]] = zp_

    zp['cu'] = zp['c'] + zp['u']
    zp['w'] = zp['w1'] + zp['w2']
    zp['h'] = zp['h1'] + zp['h2']
    zp['WIM'] = zp['WPIM'] + zp['WCIM']
    zp['2p'] = zp['cu'] + zp['w']
    zp['pi'] = zp['WPIM'] + zp['WCIM']
    zp['whole'] = zp['cu'] + zp['w'] + zp['h']

    return zp

def read_zpa_and_hst(sa, df, mdl, phase_set_name=None, force_override=False):
    s = sa.simdict[mdl]
    nums = df.loc[mdl]['nums']
    if phase_set_name is None:
        phase_set_name = list(s.phase_set.keys())[-1]

    zpa = s.read_zprof_from_vtk_all(nums,
                                    phase_set_name=phase_set_name,
                                    force_override=force_override)

    zpa, h = s.merge_zprof_with_hst(zpa)

    return s, zpa, h


if __name__ == '__main__':
    sa, df = load_sim_ncr_rad_all(verbose=True)

    for mdl in sa.models[1:]:
        print(mdl)
        s = sa.set_model(mdl, verbose='INFO')
        print(s.savdir)
        nums = [n for n in range(*df.loc[mdl]['num_range'])]
        for k in s.phase_set.keys():
            print(k)
            zp = s.read_zprof_from_vtk_all(nums, phase_set_name=k, force_override=False)
