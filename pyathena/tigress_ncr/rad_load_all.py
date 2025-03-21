import os
import shutil
import os.path as osp
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import cmasher as cmr
import matplotlib as mpl
import astropy.constants as ac
import astropy.units as au
from scipy.integrate import cumtrapz

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
    #R8_Z3='/scratch/gpfs/changgoo/TIGRESS-NCR/R8_8pc_NCR.full.b1.v3.iCR4.Zg3.Zd3.xy1024.eps1.e-8',

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
    #LGR4_Z3='/scratch/gpfs/changgoo/TIGRESS-NCR/LGR4_4pc_NCR.full.b1.v3.iCR4.Zg3.Zd3.xy1024.eps1.e-8',
    #LGR4_Z3_xy512='/scratch/gpfs/changgoo/TIGRESS-NCR/LGR4_4pc_NCR.full.b1.v3.iCR4.Zg3.Zd3.xy512.eps1.e-7',

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

# Subset of metallicity suite (R8 and LGR4) and low-resolution
models3 = dict(
    R8_4pc='/projects/EOSTRIKE/TIGRESS-NCR/R8_4pc_NCR.full.xy2048.eps0.np768.has',
    LGR4_2pc='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_2pc_NCR.full',

    R8_Z1='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b1.v3.iCR4.Zg1.Zd1.xy2048.eps0.0',
    R8_Z03='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b1.v3.iCR4.Zg0.3.Zd0.3.xy4096.eps0.0',
    R8_Z01='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b1.v3.iCR4.Zg0.1.Zd0.1.xy4096.eps0.0',
    R8_Zg01_Zd0025='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b1.v3.iCR4.Zg0.1.Zd0.025.xy4096.eps0.0',
    R8_Z3='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b1.v3.iCR4.Zg3.Zd3.xy1024.eps1.e-8',

    R8_b10_Z1='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b10.v3.iCR4.Zg1.Zd1.xy2048.eps0.0',
    R8_b10_Z03='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b10.v3.iCR4.Zg0.3.Zd0.3.xy4096.eps0.0',
    R8_b10_Z01='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b10.v3.iCR4.Zg0.1.Zd0.1.xy4096.eps0.0',
    R8_b10_Zg01_Zd0025='/projects/EOSTRIKE/TIGRESS-NCR/R8_8pc_NCR.full.b10.v3.iCR4.Zg0.1.Zd0.025.xy4096.eps0.0',

    LGR4_Z1='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR.full.b1.v3.iCR4.Zg1.Zd1.xy1024.eps1.e-8',
    LGR4_Z03='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR.full.b1.v3.iCR4.Zg0.3.Zd0.3.xy2048.eps1.e-8',
    LGR4_Z01='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR.full.b1.v3.iCR4.Zg0.1.Zd0.1.xy2048.eps1.e-8',
    LGR4_Zg01_Zd0025='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR.full.b1.v3.iCR4.Zg0.1.Zd0.025.xy2048.eps1.e-8',
    LGR4_Z3='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR.full.b1.v3.iCR4.Zg3.Zd3.xy1024.eps1.e-8',

    LGR4_b10_Z1='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR.full.b10.v3.iCR4.Zg1.Zd1.xy1024.eps1.e-8',
    LGR4_b10_Z01='/projects/EOSTRIKE/TIGRESS-NCR/LGR4_4pc_NCR.full.b10.v3.iCR4.Zg0.1.Zd0.1.xy2048.eps1.e-8',

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

tMyr_range['R8_Z3'] = [438,977]
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

tMyr_range['LGR4_Z3'] = [204,444]
# tMyr_range['LGR4_Z3_xy512'] = [204,450] # maybe running
tMyr_range['LGR4_Z1'] = [204,488]
tMyr_range['LGR4_Z03'] = [204,488]
tMyr_range['LGR4_Z01'] = [204,488]
tMyr_range['LGR4_Zg01_Zd0025'] = [204,488]

tMyr_range['LGR4_b10_Z1'] = [204,488]
tMyr_range['LGR4_b10_Z01'] = [204,488]

tMyr_range['S100_Z1r'] = [204,487]
tMyr_range['S100_Z1'] = [204,425]
tMyr_range['S100_Z01'] = [204,488]

tMyr_range['S150_Om100q0_Z1'] = [153,378]
tMyr_range['S150_Om100q0_Z01'] = [153,353]
tMyr_range['S150_Om200_Z1r'] = [153,336]
tMyr_range['S150_Om200_Z1'] = [153,397]
tMyr_range['S150_Om200_Z01'] = [153,412]

def float_to_nsf(x: float, precision: int=3) -> str:
    """Convert a float into a string with a chosen number of significant figures.
    """
    s = []
    for x_ in np.atleast_1d(x):
        s.append(np.format_float_positional(x_, precision=precision,
                                            unique=False, fractional=False, trim='k'))

    if len(s) == 1:
        return s[0]
    else:
        return ', '.join(np.array(s))

def print_summary(sa, df, model, q=np.array([0.16, 0.5, 0.84])):
    print('Model :', model)
    print('---------------')
    s = sa.set_model(model)
    zpa = s.read_zprof_from_vtk_all(nums=df.loc[model]['nums'], force_override=False)
    dx = s.domain['dx'][2]
    zmid = slice(-dx, dx)
    z2p = zpa.sel(phase=['CpU', 'WIM', 'WNM']).sum(dim='phase').sel(z=zmid)
    zpw = zpa.sel(phase=['WIM','WNM'])
    # zwh = zpa.sel(phase='whole').sel(z=zmid)
    zwh = zpa.sel(phase=['CpU','WIM','WNM','hot']).sum(dim='phase')
    print(r'Sigma_gas [Msun/pc^2] :',
          float_to_nsf(zpa['hst_Sigma_gas'].quantile(q, dim='time')))
    print(r'Sigma_SFR,10Myr [Msun/yr/kpc^2]:',
          float_to_nsf(zpa['hst_sfr10'].quantile(q, dim='time')))
    print(r'Sigma_FUV [Lsun/pc^2]: ',
          float_to_nsf(zpa['hst_Sigma_FUV'].quantile(q, dim='time')))
    print(r'Sigma_FUV/(4pi) [erg s-1 cm^-2 sr-1]: ',
          float_to_nsf(zpa['hst_Sigma_FUV'].quantile(q, dim='time')/(4.0*np.pi)*(1.0*au.L_sun/au.pc**2).cgs.value))
    print(r'Phi_LyC [10^7 cm^-2 s-1] :',
          float_to_nsf(zpa['hst_Phi_LyC'].quantile(q, dim='time')*(1/au.kpc**2).to('cm-2').value*1e-7))

    # Scale heights and distance between sources
    print(r'H_w [pc] :',
          float_to_nsf(zpw.weighted(zpw['nH'].sum(dim='z')).mean(dim='phase')['H_nH'].quantile(q, dim='time')))
    print(r'H_CpU [pc] :',
          float_to_nsf(zpa.sel(phase='CpU')['H_nH'].quantile(q, dim='time')))
    print(r'l_star [pc] :',
          float_to_nsf(zpa['src_lsrc'].quantile(q, dim='time')))
    print(r'l_star_90_Qi [pc] :',
          float_to_nsf(zpa['src_lsrc_90_Qi'].quantile(q, dim='time')))
    print(r'l_star_90_LFUV [pc] :',
          float_to_nsf(zpa['src_lsrc_90_LFUV'].quantile(q, dim='time')))
    print(r'H_star_Qi [pc]: ',
          float_to_nsf(zpa['src_H_Qi'].quantile(q, dim='time')))
    print(r'H_star_LFUV [pc]: ',
          float_to_nsf(zpa['src_H_LFUV'].quantile(q, dim='time')))
    print(r'(z_max - z_min)/2 [pc]: ',
          float_to_nsf(zpa['src_z_max_min_over_two'].quantile(q, dim='time')))

    print(r'16,50,84th quantiles (over snapshots) of volume averaged J_FUV_mid,2p :',
          float_to_nsf((z2p['J_FUV'].mean(dim='z')/\
                        z2p['frac'].mean(dim='z')).quantile(q, dim='time')))
    print(r'Volume- and time-averaged J_FUV,mid,2p = \int Theta_2p J_FUV dVdt / \int Theta_2p dV :',
          float_to_nsf(z2p['J_FUV'].sum()/z2p['frac'].sum()))
    print(r'Mass-weighted, time-averaged J_FUV,mid,2p = \int Theta_2p J_FUV nH dVdt / \int Theta_2p nH dVdt',
          float_to_nsf(z2p['J_FUV_w_nH'].weighted(z2p['nH']).mean(dim='z').mean(dim='time')))
    print(r'Volume- and time-averaged calJ,mid,2p = \int Theta_2p calJ dVdt / \int Theta_2p dV :',
          float_to_nsf(z2p['calJ_FUV'].sum()/z2p['frac'].sum()))
    print(r'Mass-weighted, time-averaged calJ,mid,2p = \int Theta_2p calJ nH dVdt / \int Theta_2p nH dVdt',
          float_to_nsf(z2p['calJ_FUV_w_nH'].weighted(z2p['nH']).mean(dim='z').mean(dim='time')))

    def get_cdf_and_quantiles(x, q, w=None):
        """Calculated weighted quantiles
        """
        if w is None:
            return None, np.quantile(x, q)
        else:
            sidx = np.argsort(x)
            xs = x[sidx]
            ws = w[sidx]
            ws_cumsum = np.cumsum(ws)
            cdf = ws_cumsum/ws_cumsum[-1]
            return cdf, xs[np.searchsorted(cdf, q)]

    d = s.read_slice_all(df.loc[model]['nums'], force_override=False)
    dd = d.where(d['T'] < 3.5e4)
    idx = np.isnan(dd['nH'].data.flatten())
    x = dd['J_FUV'].data.flatten()[~idx]
    print(r'Volume-weighted quantiles : J_FUV,2p,mid', float_to_nsf(get_cdf_and_quantiles(x, q)[1]))
    w = dd['nH'].data.flatten()[~idx] # mass weighted
    print(r'Mass-weighted quantiles : J_FUV,2p,mid', float_to_nsf(get_cdf_and_quantiles(x, q, w)[1]))
    w = (dd['J_FUV']*dd['nH']).data.flatten()[~idx] # absorption/emission weighted
    print(r'Absorption-weighted quantiles : J_FUV,2p,mid', float_to_nsf(get_cdf_and_quantiles(x, q, w)[1]))

    x = dd['calJ_FUV'].data.flatten()[~idx]
    print(r'Volume-weighted quantiles : calJ,2p,mid', float_to_nsf(get_cdf_and_quantiles(x, q)[1]))
    w = dd['nH'].data.flatten()[~idx] # mass weighted
    print(r'Mass-weighted quantiles : calJ,2p,mid', float_to_nsf(get_cdf_and_quantiles(x, q, w)[1]))
    w = (dd['J_FUV']*dd['nH']).data.flatten()[~idx] # absorption/emission weighted
    print(r'Absorption-weighted quantiles : calJ,2p,mid', float_to_nsf(get_cdf_and_quantiles(x, q, w)[1]))
    print('')

    return zpa

def get_zprof_summary(zp, zslice, suffix):
    find_weight = lambda v: v.split('_w_')[1] if '_w_' in v else None

    def find_vars_and_weights(zp):
        vv = [v for v in list(zp.variables) if v not in list(zp.coords)]
        weights = []
        for v in vv:
            w = find_weight(v)
            if w is not None:
                weights.append(w)

        return vv, list(set(weights))

    vv, weights = find_vars_and_weights(zp)
    try:
        vv.remove('nHI')
        vv.remove('nHI_w_nH')
        vv.remove('nHI_w_ne')
        vv.remove('nHI_w_nesq')
    except ValueError:
        pass

    zp_slc = zp.sel(z=zslice)
    zpw_mean = dict()
    for w in weights:
        zpw_mean[w] = zp_slc.weighted(zp_slc[w]).mean(dim=['time', 'z'])

    r = dict()
    for v in vv:
        w = find_weight(v)
        if w:
            r[v+'_'+suffix] = float(zpw_mean[w][v])
        else:
            if v != 'frac':
                dd = zp_slc[v]/zp_slc['frac']
            else:
                dd = zp_slc[v]
            r[v+'_'+suffix] = float((dd).mean(dim=['time','z']))

    return r

def get_summary(s, model, zprof_summary=True, force_override=False):

    df = dict()
    try:
        df['tMyr_range'] = tMyr_range[model]
        df['nums'] = s.get_output_nums(df['tMyr_range'], out_fmt='vtk')
        df['nums_range'] = [df['nums'][0], df['nums'][-1]]
        df['nums_starpar'] = s.get_output_nums(df['tMyr_range'], out_fmt='starpar_vtk')
        df['nums_starpar_range'] = [df['nums_starpar'][0], df['nums_starpar'][-1]]
    except KeyError:
        print('tMyr_range not found {0:s}'.format(model))
        ds1 = s.load_vtk(s.nums[0])
        ds2 = s.load_vtk(s.nums[-1])
        print('Time of the first/last vtk snapshot [Myr]: ',
              ds1.domain['time']*s.u.Myr, ds2.domain['time']*s.u.Myr)

    df['Zd'] = float(s.par['problem']['Z_dust'])
    df['Zg'] = float(s.par['problem']['Z_gas'])
    df['beta'] = float(s.par['problem']['beta'])
    df['Omega'] = float(s.par['problem']['Omega'])
    df['qshear'] = float(s.par['problem']['qshear'])

    # Summary from hst_rad
    h = s.read_hst_rad(tMyr_range=df['tMyr_range'], force_override=force_override)
    # Cumulative (time-averaged) escape fraction should be re-calculated for a given
    # tMyr_range
    h['fesc_LyC_cumul'] = cumtrapz(h['Lesc_LyC'].fillna(0), h['tMyr'], initial=0.0)/\
            cumtrapz(h['Ltot_LyC'], h['tMyr'], initial=0.0)
    h['fesc_FUV_cumul'] = cumtrapz(h['Lesc_FUV'].fillna(0), h['tMyr'], initial=0.0)/\
            cumtrapz(h['Ltot_FUV'], h['tMyr'], initial=0.0)
    df['fesc_LyC_avg'] = h['fesc_LyC_cumul'].iloc[-1]
    df['fesc_FUV_avg'] = h['fesc_FUV_cumul'].iloc[-1]

    # Mean
    df.update({k +'_mean': v for k, v in h.mean().to_dict().items()})
    # Quantiles
    hq = h.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    for idx in hq.index:
        for c in hq.columns:
            df['_'.join([c, str(int(100.0*idx)) + 'th'])] = hq.loc[idx, c]

    if not zprof_summary:
        return df

    # Summary from zprof
    zp = s.load_zprof_for_sfr()
    vv = [v for v in list(zp.variables) if v not in list(zp.coords)]
    dx = s.domain['dx'][2]
    for ph in zp.phase.data:
        zp_mid = zp.sel(phase=ph).sel(z=slice(-dx, dx))
        for v in vv:
            df[v + '_' + ph] = float(zp_mid[v].mean(dim=['time','z']))

    def add_zeta_pi_and_chi(r, s):
        # Convert Erad_LyC to zeta_pi
        sigma_pi_H = s.par['opacity']['sigma_HI_PH']
        hnu = (s.par['radps']['hnu_PH']*au.eV).cgs.value
        conv = sigma_pi_H/hnu*ac.c.cgs.value
        kk = [k for k in r.keys() if k.startswith('Erad_LyC') and\
              not k.startswith('Erad_LyC_mask')]
        for k in kk:
            r[k.replace('Erad_LyC', 'zeta_pi')]= r[k]*conv

        # Convert Erad_FUV to chi_FUV
        Erad_FUV0 = s.par['cooling']['Erad_PE0'] + s.par['cooling']['Erad_LW0']
        kk = [k for k in r.keys() if k.startswith('Erad_LW')]
        for k in kk:
            r[k.replace('Erad_LW', 'chi_FUV')] = \
                (r[k] + r[k.replace('LW', 'PE')])/Erad_FUV0

        return r

    zp = s.read_zprof_from_vtk_all(nums=df['nums'], phase_set_name='warm_eq_LyC_ma',
                                   force_override=False)
    zp_pi = zp.sel(phase=['w1_eq_LyC', 'w1_eq_LyC_pi']).sum(dim='phase')
    zp_pi2 = zp.sel(phase=['w1_eq_LyC', 'w1_eq_LyC_pi',
                           'w2_eq_LyC']).sum(dim='phase')
    zp_CpU = zp.sel(phase=['CpU_noLyC', 'CpU_LyC']).sum(dim='phase')
    zslice = slice(-100,100)
    for zp_ph, suffix in zip([zp_pi, zp_pi2, zp_CpU], ['pi', 'pi2', 'CpU']):
        r = get_zprof_summary(zp_ph, zslice=zslice, suffix=suffix)
        # add_zeta_pi_and_chi(r, s)
        df.update(r)

    # Set plot kwargs
    df['s'] = np.sqrt(df['W_2p'])
    norm_Zg = mpl.colors.LogNorm(0.03, 3.0)
    # norm_Zg = mpl.colors.LogNorm(10.0**-1.3, 10.0**0.3)
    df['markeredgecolor'] = mpl.colors.rgb2hex(mpl.cm.plasma_r(norm_Zg(df['Zg'])))
    df['marker'] = 'o'
    if df['Zd'] != df['Zg']:
        df['marker'] = 's'
    if df['beta'] > 5.0:
        df['marker'] = '*'
    if df['Omega'] == 0.2:
        df['marker'] = '^'
    elif df['qshear'] < 0.1:
        df['marker'] = 'v'

    return df

def load_sim_ncr_rad_all(model_set, savdir_base='/tigress/jk11/NCR-RAD',
                         zprof_summary=True, force_override=False, verbose=False):
    """
    Load all simulations for radiation analysis

    Parameters
    ----------
    model_set : str
        'radiation_paper' or 'lowZ'
    savdir_base : str
        Base directory for saving results.
    zprof_summary : bool
        Get summary from zprof.
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
    elif model_set == 'lowZ-subset':
        models = models3
    else:
        raise ValueError

    sa = LoadSimTIGRESSNCRAll(models)
    # Check if pickle exists
    fpkl = osp.join(savdir_base, 'pickles/{0:s}.p'.format(model_set))
    if not force_override and osp.isfile(fpkl):
        read_from_pkl = True
    else:
        read_from_pkl = False
        df_list = []

    print('[load_sim_ncr_rad_all]:', end=' ')
    for mdl in sa.models:
        print(mdl, end=' ')
        savdir = str(Path(savdir_base, Path(sa.basedirs[mdl]).name))
        s = sa.set_model(mdl, savdir=savdir, verbose=verbose)
        if not read_from_pkl:
            df_ = get_summary(s, mdl, zprof_summary, force_override=force_override)
            df_list.append(pd.DataFrame(pd.Series(df_, name=mdl)).T)

    if not read_from_pkl:
        if not osp.exists(osp.dirname(fpkl)):
            os.makedirs(osp.dirname(fpkl))

        df = pd.concat(df_list, sort=True).sort_index(ascending=False)
        df = df.apply(pd.to_numeric, errors='ignore')
        df.to_pickle(fpkl)
    else:
        df = pd.read_pickle(fpkl)

    return sa, df

def copy_zprof(model_set):
    """Copy all post-processed zprof from source directories
    """
    sa, df = load_sim_ncr_rad_all(model_set=model_set, verbose=False)

    for mdl in sa.models:
        print('Model: ', mdl, end=' ')
        copy = False
        s = sa.set_model(mdl)
        fpattern = osp.join(s.basedir, 'zprof/{0:s}.*.{1:s}.mod.nc'.format(s.problem_id, s.basename))
        fnames = glob(fpattern)
        zprof_dir = osp.join(s.savdir, 'zprof')
        if not osp.isdir(zprof_dir):
            os.makedirs(zprof_dir)

        for fsrc in fnames:
            fdst = osp.join(zprof_dir, osp.basename(fsrc))
            # More stringent test would be to use filecmp.cmp, but it is too slow
            if not os.path.isfile(fdst) or not osp.getsize(fdst) == osp.getsize(fsrc):
                copy = True
                print('Copying ', fsrc)
                shutil.copyfile(fsrc, fdst)

        if not copy:
            print('All zprof exist. No need to copy.')


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

    zpa = s.merge_zprof_with_hst(zpa, force_override=force_override)
    h = s.read_hst_rad(force_override=force_override)

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
