import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from ..sixray_test import sixray_test
from .cool import heatCR, heatPE
from .cool_rosen95 import CoolRosen95
from ..util.rad_uvb import f_sshld_R13

def get_equil(f_Lambda, z=0.0, manual=True, Gamma_pe0=None, 
              heating_pi_UVB=True, heating_pe=True, heating_cr=False,
              sshld=False, chi_PE=1.0, Zd=1.0, xi_cr=0.0, coll_ion=True):

    from pyathena.microphysics.cool import get_xn_eq
    from pyathena.util import rad_uvb
    
    # Read FG UV Background
    r = rad_uvb.read_FG20()

    if not manual:
        zeta_pi_H = float(r['ds_int']['zeta_pi_H'].sel(z=z, method='nearest'))# /au.s
        q_pi_H = float(r['ds_int']['q_pi_H'].sel(z=z, method='nearest'))# *au.erg
        q_zeta_pi_H = float(r['ds_int']['q_zeta_pi_H'].sel(z=z, method='nearest'))# *au.erg/au.s
    else:
        if z != 0:
            raise ValueError("z!=0 not supported")

        # Values from Table D1 in FG21
        zeta_pi_H = 3.62e-14 # /au.s
        q_zeta_pi_H = 2.03e-25 # *au.erg/au.s

    # Table does not provide this information
    sigma_pi_H = float(r['ds_int']['sigma_mean_pi_H'].sel(z=z,method='nearest'))
    print('redshift, zeta_pi, q_pi*zeta_pi, sigma_pi:',z,zeta_pi_H,q_zeta_pi_H,sigma_pi_H)
    
    def f_net_cool(T, nH):
        # WD PE heating with xe_eq from UV background
        
        if sshld:
            if sshld == 'RAMSES':
                f_sshld = np.exp(-nH/1e-2)
            else:
                f_sshld = f_sshld_R13(nH, sigma_pi_H, T, zeta_pi_H)
        else:
            f_sshld = 1.0

        xn = get_xn_eq(T, nH, f_sshld*zeta_pi_H, 1.5*xi_cr, coll_ion=coll_ion)
        
        Gamma_tot = 0.0
        if heating_pi_UVB:
            Gamma_tot += f_sshld*q_zeta_pi_H*xn
        if heating_pe:
            if Gamma_pe0 is not None:
                Gamma_tot += Gamma_pe0
            else:
                Gamma_tot += heatPE(nH, T, 1.0 - xn, Zd, chi_PE)
        if heating_cr:
            Gamma_tot += heatCR(nH, 1.0-xn, xn, 0.0, xi_cr)
        
        return Gamma_tot - nH*f_Lambda(T)
    
    nH = np.logspace(-2,3,500)
    T = np.zeros_like(nH)
    root_result = []

    for i,nH_ in enumerate(nH):
        T[i], root_result_ = brentq(f_net_cool, 1e0, 1e5, args=(nH_,), full_output=True)
        root_result.append(root_result_)
        if not root_result_.converged:
            print('Not converged nH Teq',nH_,T[i],root_result_)
        
    if sshld:
        if sshld == 'RAMSES':
            f_sshld = np.exp(-nH/1e-2)
        else:
            f_sshld = f_sshld_R13(nH, sigma_pi_H, T, zeta_pi_H)
    else:
        f_sshld = 1.0
    
    if Gamma_pe0 is None:
        xn = get_xn_eq(T, nH, f_sshld*zeta_pi_H, 1.5*xi_cr, coll_ion=coll_ion)
    else:
        xn = np.repeat(1.0,len(T))

    if heating_pe:
        if Gamma_pe0 is None:
            Gamma_pe = heatPE(nH, T, 1.0 - xn, Zd, chi_PE)
        else:
            Gamma_pe = np.repeat(Gamma_pe0,len(T))
    else:
        Gamma_pe = np.repeat(0.0,len(T))
        
    if heating_pi_UVB:
        Gamma_pi_UVB = f_sshld*q_zeta_pi_H*xn
    else:
        Gamma_pi_UVB = np.repeat(0.0,len(T))
    
    if heating_cr:
        Gamma_cr = heatCR(nH, 1.0-xn, xn, 0.0, xi_cr)
    else:
        Gamma_cr = np.repeat(0.0,len(T))
    
    res = dict(nH=nH, T=T, xn=xn, f_sshld=f_sshld, 
               heating_pe=heating_pe, heating_pi_UVB=heating_pi_UVB, 
               heating_cr=heating_cr, Gamma_pe=Gamma_pe,
               zeta_pi_H=zeta_pi_H,
               Gamma_pi_UVB=Gamma_pi_UVB, Gamma_cr=Gamma_cr,
               root_result=root_result)
    
    return res

def get_f_Lambda_newcool_grackle_rosen(newcool=True, grackle=True, Rosen95=True):

    r = dict()
    
    if newcool:
        models = dict(
            Unshld_CRvar_Z1='/tigress/jk11/NEWCOOL-TESTS/SIXRAY-TEST-UNSHIELDED/Unshld.CRvar.Zg1.0.Zd1.0/'
        )
        sa, da = sixray_test.load_sixray_test_all(models=models, cool=True)
        d = da['Unshld_CRvar_Z1'].sel(log_chi_PE=0.0, method='nearest')
        idx = np.diff(d['T'],prepend=0) > 0.0
        f_newcool = interp1d(d['T'][~idx][:-2:], d['cool_rate'][~idx][:-2:]/d['nH'][~idx][:-2:]**2, 
                            bounds_error=False, fill_value='extrapolate')

        r['f_newcool'] = f_newcool
        
    if Rosen95:
        cr = CoolRosen95()
        f_rosen95 = interp1d(cr.T_extrapolate, cr.LambdaRosen95(cr.T_extrapolate),
                             bounds_error=False, fill_value='extrapolate')
        r['f_rosen95'] = f_rosen95

    if grackle:
        fname = '/tigress/changgoo/cooling-curves/grackle_3.2_cooling_curve_Z1.txt'
        nH,dens,temp,Ne,LambdaG = np.loadtxt(fname)
        T = np.logspace(0.3,4.2,3000)
        # Cutoff weird part
        f_grackle = interp1d(temp[75:], LambdaG[75:], bounds_error=False, fill_value='extrapolate')
        r['f_grackle'] = f_grackle
    
    return r


def plt_equil(axes_, rr, plt_kwargs):

    l0,=axes_[0].loglog(rr['nH'], (2.1 - rr['xn'])*rr['nH']*rr['T'], **plt_kwargs)
    l1,=axes_[1].loglog(rr['nH'], rr['T'], **plt_kwargs)
    l2,=axes_[2].loglog(rr['nH'], rr['Gamma_cr']+rr['Gamma_pi_UVB']+rr['Gamma_pe'], **plt_kwargs)
    l3,=axes_[3].loglog(rr['nH'], 1.0 - rr['xn'], **plt_kwargs)

#     if rr['heating_pe']:
#         axes_[2].loglog(rr['nH'],rr['Gamma_pe'], ls=':', **plt_kwargs)
#     if rr['heating_pe']:
#         axes_[2].loglog(rr['nH'],rr['Gamma_pi_UVB'], ls='--', **plt_kwargs)
#     if rr['heating_cr']:
#         axes_[2].loglog(rr['nH'],rr['Gamma_cr'], ls='-.', **plt_kwargs)        

    return (l0,l1,l2,l3)
