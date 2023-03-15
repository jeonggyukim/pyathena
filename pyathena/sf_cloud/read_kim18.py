import glob
import os.path as osp
import pandas as pd
import numpy as np

from ..load_sim import LoadSim, LoadSimAll
from ..io.read_hst import read_hst
from ..util.units import Units

def read_kim18():
    import glob
    from pyathena.util.cloud import Cloud

    dirs = glob.glob('/projects/EOSTRIKE/SF-CLOUD/Kim-etal-2018-all/M*')
    dirs += glob.glob('/projects/EOSTRIKE/SF-CLOUD/Kim-etal-2018-all/PHRP/radps_sp_phrp/M*')
    dirs = sorted(dirs)
    
    models = dict()
    for i,d in enumerate(dirs):
        models[osp.basename(d)] = d

    sa = LoadSimAll(models)
    dat = dict()
    dat['model'] = []
    dat['hst'] = []
    dat['SFE'] = []
    dat['M0'] = []
    dat['R0'] = []
    dat['Sigma0'] = []
    dat['tff0'] = []
    dat['feedback'] = []
    
    
    for mdl in sa.models:
        print(mdl, end=' ')
        s = sa.set_model(mdl, verbose=50)
        #print(s.files['hst'])
        h  = read_hst(s.files['hst'])
        dat['model'].append(mdl)
        dat['M0'].append(s.par['problem']['M_GMC'])
        dat['R0'].append(s.par['problem']['rcloud'])
        cl = Cloud(s.par['problem']['M_GMC'], s.par['problem']['rcloud'])
        dat['Sigma0'].append(cl.Sigma.value)
        dat['tff0'].append(cl.tff.value)
        u = Units(kind='LT',muH=1.4)
        Mconv = u.Msun*s.domain['Lx'].prod()
        h['mass_sp'] = (h['mass_stars'] + h['mass_stars_esc'])*Mconv
        dat['hst'].append(h)
        # Final SFE
        dat['SFE'].append(h['mass_sp'].iloc[-1]/s.par['problem']['M_GMC'])

        if 'radp' in mdl:
            dat['feedback'].append('RP')
        elif 'phot' in mdl:
            dat['feedback'].append('PH')
        else:
            dat['feedback'].append('PHRP')
            #print("Feedback not recognized Error!")

    # Add missing data
    models_miss = ['M1E5R05_phot', 'M1E5R10_phot', 'M1E6R25_phot']
    SFE_miss = [0.6033810143042913, 0.35370611183355016, 0.5604681404421328]
    M0_miss = [1e5, 1e5, 1e6]
    R0_miss = [5, 10, 25]
    Sigma0_miss = [Cloud(M0_,R0_).Sigma.value for M0_,R0_ in zip(M0_miss, R0_miss)]
    tff0_miss = [Cloud(M0_,R0_).tff.value for M0_,R0_ in zip(M0_miss, R0_miss)]

    feedback_miss = ['PH', 'PH', 'PH']
    hst_miss = [None, None, None]

    dat['model'] += models_miss
    dat['SFE'] += SFE_miss
    dat['M0'] += M0_miss
    dat['R0'] += R0_miss
    dat['Sigma0'] += Sigma0_miss
    dat['tff0'] += tff0_miss
    dat['feedback'] += feedback_miss
    dat['hst'] += hst_miss
    df = pd.DataFrame(dat, index=dat['model'])
    
    def get_markersize(M):
        log10M = [4., 5., 6.]
        ms = [40.0, 120.0, 360.0]
        return np.interp(np.log10(M), log10M, ms)
    
    df['markersize'] = get_markersize(df['M0'])
    
    return df
