import numpy as np
import pandas as pd

from .sixray_test import get_cool_data

def NHtrans_NHItot_alphaG_B16(nH, chi, R=3.0e-17, sigmad=1.9e-21, D0_ISRF=5.7e-11, col_factor=2.0):
    """Function to compute
    1. NHtrans: H column up to a point where HI-H2 transition occurs
    2. NHItot: Total HI column
    3. alpha*G: dimensionless parameter
    (see Sternberg+14 and Bialy & Sternberg 2016)

    Parameters
    ----------
    nH : float
        Density of the slab
    chi : float
        Incident FUV radiation field
    R : float
        H2 formation rate. Default value is 3.0e-17 [cm^3 s^-1]
    sigmad : float
        Dust cross section [cm^2]
    D0_ISRF : float
        Free-space H2 dissociation rate for chi=1 [s^-1]

    Returns
    -------
    (NHtrans, NHItot, alphaG) : tuple
    """


    beta = 0.7
    sg = sigmad/1.9e-21
    sd = 2.36e-3
    alpha = chi*D0_ISRF/(R*nH)
    Wgtot = 8.8e13/(1.0 + 8.9*sg)**0.37
    G = Wgtot*(sg*1.9e-21)/sd

    res = (beta*np.log((alpha*G/col_factor)**(1.0/beta) + 1.0)/sigmad,
           np.log((alpha*G/col_factor) + 1.0)/sigmad,
           alpha*G,alpha,G)

    return res

def analyze_HIH2_all(sa, models=None):

    if models is None:
        models = sa.models

    dd = dict()
    rr = dict()
    for mdl in models:
        print('Model:', mdl, end=' ')
        s = sa.set_model(mdl)
        dd[mdl] = get_cool_data(s, s.nums[-1], cool=False)
        r = analyze_HIH2(s, mdl, dd)
        rr[mdl] = pd.DataFrame(r)

    df = pd.concat([v for k,v in rr.items()])

    return df, dd

def analyze_HIH2(s, mdl, dd):

    from scipy import interpolate

    NHtrans = []
    NHItot = []
    NHItot22 = [] # HI column density upto NH<10^22/cm^2
    NHItot23 = [] # HI column density upto NH<10^23/cm^2
    NHItot24 = [] # HI column density upto NH<10^24/cm^2
    nH = []
    chi = []
    dx = []
    dx_cgs = []
    alphaG = []
    alpha = []
    G = []
    NHtrans_B16 = []
    NHItot_B16 = []
    xi_CR0 = []
    nx = s.par['domain1']['Nx3']
    xHI_final = []
    xicr_over_RnH_trans = []
    zeta_pd_over_RnH_trans = []
    fshld_H2_trans = []

    NH0 = 1.87e21
    for log_chi_PE in dd[mdl].log_chi_PE.data:
        for log_nH in dd[mdl].log_nH.data:
            Zd = s.par['problem']['Z_dust']
            D0_ISRF = s.par['cooling']['xi_diss_H2_ISRF']
            R = s.par['cooling']['kgr_H2']*Zd
            sigmad = s.par['opacity']['sigma_dust_LW0']*Zd
            d = dd[mdl].sel(log_nH=log_nH, log_chi_PE=log_chi_PE, method='nearest')
            dx_cgs_ = s.domain['dx'][2]*s.u.length.cgs.value
            d['NH'] = d['nH'].cumsum()*dx_cgs_
            d['NHI'] = d['nHI'].cumsum()*dx_cgs_
            d['Av'] = Zd*d['NH']/NH0
            d['2xH2'] = 2.0*d['xH2']

            dx.append(s.domain['dx'][2])
            dx_cgs.append(dx_cgs_)
            nH.append(10.0**log_nH)
            chi.append(10.0**log_chi_PE)
            xi_CR0.append(s.par['problem']['xi_CR0'])
            xHI_final.append((d['xHI'].data)[-1])
            NHtrans_B16_, NHItot_B16_, alphaG_, alpha_, G_ = NHtrans_NHItot_alphaG_B16(
                10.0**log_nH, 10.0**log_chi_PE, R, sigmad, D0_ISRF)

            NHtrans_B16.append(NHtrans_B16_)
            NHItot_B16.append(NHItot_B16_)
            alphaG.append(alphaG_)
            alpha.append(alpha_)
            G.append(G_)

            try:
                x = np.log10(d['NH']).data
                y = d['2xH2'].data
                # ratio between CRIR and formation at Ntrans
                y2 = d['xi_CR']/(R*d['nH'])
                # self-shielding factor fshld_H2 at Ntrans
                y3 = d['chi_H2_ext']/10.0**log_chi_PE
                # ratio btw PD and formation at Ntrans
                y4 = d['chi_H2_ext']*s.par['cooling']['xi_diss_H2_ISRF']/(R*d['nH'])
                x_ = np.linspace(x.min(),x.max(),500000)
                f = interpolate.interp1d(x, y, kind='linear')
                f2 = interpolate.interp1d(x, y2, kind='linear')
                f3 = interpolate.interp1d(x, y3, kind='linear')
                f4 = interpolate.interp1d(x, y4, kind='linear')
                y_ = f(x_)
                #NHtrans.append(10.0**x_[np.where(y_ >= 0.5)[0][0]])
                NHtrans.append(10.0**x_[np.where(y_ < 0.5)[-1][-1]])
                NHItot.append(d['NHI'][-1])
                xicr_over_RnH_trans.append(f2(x_[np.where(y_ < 0.5)[-1][-1]]))
                fshld_H2_trans.append(f3(x_[np.where(y_ < 0.5)[-1][-1]]))
                zeta_pd_over_RnH_trans.append(f4(x_[np.where(y_ < 0.5)[-1][-1]]))
            except IndexError:
                NHtrans.append(np.nan)
                NHItot.append(np.nan)
                xicr_over_RnH_trans.append(np.nan)
                fshld_H2_trans.append(np.nan)
                zeta_pd_over_RnH_trans.append(np.nan)

            x = np.log10(d['NH']).data
            y = np.log10(d['NHI']).data
            x_ = np.linspace(x.min(),x.max(),100000)
            f = interpolate.interp1d(x, y, kind='linear')
            y_ = f(x_)
            try:
                NHItot22.append(10.0**y_[np.where(x_ < 22.0)[-1][-1]])
            except IndexError:
                NHItot22.append(np.nan)
            try:
                NHItot23.append(10.0**y_[np.where(x_ < 23.0)[-1][-1]])
            except IndexError:
                NHItot23.append(np.nan)
            try:
                NHItot24.append(10.0**y_[np.where(x_ < 24.0)[-1][-1]])
            except IndexError:
                NHItot24.append(np.nan)

    r = dict()
    r['nH'] = np.array(nH)
    r['dx'] = np.array(dx)
    r['dx_cgs'] = np.array(dx_cgs)
    r['NHtrans'] = np.array(NHtrans)
    r['NHItot'] = np.array(NHItot)
    r['NHItot22'] = np.array(NHItot22)
    r['NHItot23'] = np.array(NHItot23)
    r['NHItot24'] = np.array(NHItot24)
    r['alphaG'] = np.array(alphaG)
    r['alpha'] = np.array(alpha)
    r['G'] = np.array(G)
    r['NHtrans_B16'] = np.array(NHtrans_B16)
    r['NHItot_B16'] = np.array(NHItot_B16)
    r['chi'] = np.array(chi)
    r['NHmin'] = r['nH']*r['dx_cgs']
    r['NHmax'] = r['nH']*r['dx_cgs']*nx
    r['xi_CR0'] = np.array(xi_CR0)
    r['xHI_final'] = xHI_final
    r['xicr_over_RnH_trans'] = np.array(xicr_over_RnH_trans)
    r['fshld_H2_trans'] = np.array(fshld_H2_trans)
    r['zeta_pd_over_RnH_trans'] = np.array(zeta_pd_over_RnH_trans)

    return r
