import numpy as np

def get_nums_starpar(s, tMyr_range=np.array([250, 451])):
    """Function to get starpar vtk snapshot numbers

    Time range hardcoded
    """

    tMyr_range_def = {'R8-4pc': np.array([250, 451])/s.u.Myr/dt_starpar,
                      'LGR4-2pc': np.array([250, 351])/s.u.Myr/dt_starpar,
                      }

    if s.par['output3']['out_fmt'] == 'starpar_vtk':
        dt_starpar = s.par['output3']['dt']
    else:
        raise ValueError('Cannot find starpar_vtk output dt')

    if s.basename.startswith('R8_4pc_NCR.full.xy2048.eps0.np768.has'):
        tMyr_range = tMyr_range_def['R8-4pc']
    elif s.basename.startswith('LGR4_2pc_NCR.full'):
        tMyr_range = tMyr_range_def['LGR4-2pc']
    else:
        raise ValueError('Cannot find matching model name')

    nums_sp = [num for num in range(*tuple([int(t) for t in tMyr_range]))]

    return nums_sp

def get_Qi_L_FUV_distribution_all(s, force_override=False):
    """Function to calculate radiation source statistics
    """
    nums = get_nums_starpar(s)
    spa = s.read_starpar_all(nums=nums, savdir=s.savdir,
                             force_override=force_override)
    # Instantaneous Qi,sp and L_FUV,sp in all snapshots
    Qi = []
    L_FUV = []
    # Total number of radiation sources
    ntot_Qi = []
    ntot_FUV = []
    # Qi,sp and L_FUV,sp that account for >90% of the total
    Qi_90 = []
    L_FUV_90 = []
    # Number of such sources
    n90_Qi = []
    n90_FUV = []
    for i, sp in spa['sp'].items():
        # print(i, end=' ')
        Qi.append(list(sp['Qi'].values))
        L_FUV.append(sp['L_FUV'].values)

        Qi_srt = sp['Qi'].sort_values(ascending=False)
        L_FUV_srt = sp['L_FUV'].sort_values(ascending=False)

        idx_Qi = Qi_srt.cumsum() < 0.9*Qi_srt.sum()
        idx_FUV = L_FUV_srt.cumsum() < 0.9*L_FUV_srt.sum()

        n90_Qi.append(idx_Qi.sum())
        n90_FUV.append(idx_FUV.sum())
        Qi_90.append(Qi_srt[idx_Qi])
        L_FUV_90.append(L_FUV_srt[idx_FUV])

        ntot_Qi.append(len(sp['Qi'].values))
        ntot_FUV.append(len(sp['L_FUV'].values))

    # Convert list of list to 1d array
    import itertools
    Qi = np.array(list(itertools.chain.from_iterable(Qi)))
    L_FUV = np.array(list(itertools.chain.from_iterable(L_FUV)))
    Qi_90 = np.array(list(itertools.chain.from_iterable(Qi_90)))
    L_FUV_90 = np.array(list(itertools.chain.from_iterable(L_FUV_90)))

    time_code = spa['time'].values
    time_Myr = spa['time'].values*s.u.Myr
    r = dict(spa=spa, time_code=time_code, time_Myr=time_Myr,
             Qi=Qi, L_FUV=L_FUV, ntot_Qi=ntot_Qi, ntot_FUV=ntot_FUV,
             n90_Qi=n90_Qi, n90_FUV=n90_FUV, Qi_90=Qi_90, L_FUV_90=L_FUV_90)

    return r
