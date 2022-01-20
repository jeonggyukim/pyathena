from pyathena.microphysics.cool_gnat12 import CoolGnat12
from pyathena.microphysics.cool_wiersma09 import CoolWiersma09
from pyathena.microphysics.rec_rate import RecRate
from pyathena.microphysics.cool \
import get_xCII, coeff_kcoll_H, coeff_alpha_rr_H, coeff_alpha_gr_H, coolHI, \
    coolRec, coolCII, coolOI, coolHIion
from pyathena.microphysics.cool import coolCII,coolOII,coolLya,coolrecH,coolffH,coolHI

from scipy.optimize import brentq
import numpy as np
import astropy.units as au
import astropy.constants as ac

def get_xHII(nH, xe, xH2, xeM, T, xi_CR, G_PE, Z_d, zeta_pi, gr_rec=True):

    kcoll = coeff_kcoll_H(T)
    kcr = (1.5 + 2.3*xH2)*xi_CR  # assume xH2=0
    alpha_rr = coeff_alpha_rr_H(T)
    if gr_rec:
        alpha_gr = coeff_alpha_gr_H(T, G_PE, nH*xe, Z_d)
    else:
        alpha_gr = 0.0

    a = 1.0 + kcoll/alpha_rr
    b = (kcr + zeta_pi - kcoll*nH*(1.0 - xeM))/(alpha_rr*nH) + xeM + alpha_gr/alpha_rr
    c = -(kcr + zeta_pi + xeM*nH*kcoll)/(alpha_rr*nH)

    xHII_eq = (-b + np.sqrt(b**2 - 4*a*c))/(2.0*a)

    return xHII_eq

def f_xe(xe, xH2, nH, T, xi_CR, G_CI, G_PE, Z_d, Z_g, zeta_pi, xOstd, xCstd, gr_rec=True):
    xCII_eq = get_xCII(nH, xe, xH2, T, Z_d, Z_g, xi_CR, G_PE, G_CI, xCstd, gr_rec)
    xHII_eq = get_xHII(nH, xe, xH2, xCII_eq, T, xi_CR, G_PE, Z_d, zeta_pi, gr_rec)
    xOII_eq = xHII_eq*xOstd

    return xe - xHII_eq - xCII_eq - xOII_eq

def get_xe_arr(nH, T, xH2, xeM, xi_CR, G_PE, G_CI, zeta_pi, Z_d, Z_g,
               xCstd, xOstd, gr_rec=True):
    xHII = []
    xCII = []
    xOII = []
    for nH_, T_ in zip(nH,T):
        if T_ > 3.5e4:
            xHII_ = get_xHII(nH_, xe, xH2, xeM, T_, xi_CR, G_PE, Z_d, zeta_pi, gr_rec)
            xHII.append(xHII_)
            xCII.append(xCstd*Z_g)
            xOII.append(xHII_*Z_g*xOstd)
        else:
            eps = 1e-11
            xe = brentq(f_xe, -1, 2, args=(xH2, nH_, T_, xi_CR, G_CI, G_PE, Z_d, Z_g,
                                           zeta_pi, xOstd, xCstd, gr_rec))
            xHII_ = get_xHII(nH_, xe, xH2, xeM, T_, xi_CR, G_PE, Z_d, zeta_pi, gr_rec)
            xCII_ = get_xCII(nH_, xe, xH2, T_, Z_d, Z_g, xi_CR, G_PE, G_CI, xCstd, gr_rec)

            xHII.append(xHII_)
            xCII.append(xCII_)
            xOII.append(xHII_*Z_g*xOstd)

    xHII = np.array(xHII)
    xCII = np.array(xCII)
    xOII = np.array(xOII)
    xOI = xOstd*Z_g - xOII
    xe = xHII + xCII + xOII

    return xe, xHII, xCII, xOII, xOI

# Transition function
def f1(T, T0=2e4, T1=3.5e4):
    return np.where(T > T1, 1.0,
                    np.where(T <= T0, 0.0, 1.0/(1.0 + np.exp(-10.0*(T - 0.5*(T0+T1))/(T1-T0)))))

def get_CIE():
    cg = CoolGnat12(abundance='Asplund09')
    cw = CoolWiersma09()
    rec = RecRate()
    elem_no_ion_frac = []
    xe = dict()
    xe_tot = np.zeros_like(cg.temp)
    cool = dict()
    cool_tot = np.zeros_like(cg.temp)
    # Elements for which CIE ion_frac is available
    elements = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Fe']

    for e in elements:
        xe[e] = np.zeros_like(cg.temp)
        cool[e] = np.zeros_like(cg.temp)

# Note that Gnat & Ferland provided Lambda_GF = cool_rate/(n_elem*ne)
# Need to get the total electron abundance first to obtain cool_rate/nH^2 = Lambda_GF*Abundance*x_e

    for e in elements:
        nstate = cg.info.loc[e]['number'] + 1
        A = cg.info.loc[e]['abd']
        #print(e, A, nstate)
        for i in range(nstate):
            xe[e] += A*i*cg.ion_frac[e + str(i)].values
            #cool[e] += A*cg.ion_frac[e + str(i)].values*cg.cool_cie_per_ion[e][:,i]

    for e in elements:
        xe_tot += xe[e]
        #cool_tot += cool[e]

    for e in elements:
        nstate = cg.info.loc[e]['number'] + 1
        A = cg.info.loc[e]['abd']
        for i in range(nstate):
            cool[e] += xe_tot*A*cg.ion_frac[e + str(i)].values*cg.cool_cie_per_ion[e][:,i]

    for e in elements:
        cool_tot += cool[e]

# Interpolation
    from scipy.interpolate import interp1d

    cgi_metal = interp1d(cg.temp, cool_tot - cool['He'] - cool['H'], bounds_error=False, fill_value=0.0)
    cgi_He = interp1d(cg.temp, cool['He'], bounds_error=False, fill_value=0.0)
    cgi_xe_mH = interp1d(cg.temp, xe_tot - xe['H'], bounds_error=False, fill_value=0.0)
    cgi_xe_mHHe = interp1d(cg.temp, xe_tot -xe['H'] - xe['He'], bounds_error=False, fill_value=0.0)
    cgi_xe_He = interp1d(cg.temp, xe['He'], bounds_error=False, fill_value=0.0)
# Apply the minimum value floor for x_e,He
    xe['He'][xe['He'] == 0.0] = 1e-8

    return cgi_metal, cgi_He, cgi_xe_mH, cgi_xe_mHHe, cgi_xe_He


def get_cool(nH, xCstd=1.6e-4, xOstd=3.2e-4,
             Z_g=1.0, Z_d=1.0, G_CI=1.0, G_PE=1.0, xH2=0.0, xi_CR=2e-16, gr_rec=True,
             Qion = 0.0,zeta_pi = None):

    cgi_metal, cgi_He, cgi_xe_mH, cgi_xe_mHHe, cgi_xe_He = get_CIE()
    if zeta_pi is None:
        Q = Qion
        r = (5.0*au.pc).cgs.value
        Fion = Q/(4.0*np.pi*r**2)
        sigma_pi = 3e-18
        zeta_pi = Fion*sigma_pi
    #print(zeta_pi)

    # print('zeta_pi: {0:.3e}'.format(zeta_pi))

    xeM0 = xCstd*Z_g

    T = np.logspace(3.5, 8, 1001)
    nH = np.full_like(T, nH)
    xe_eq, xHII_eq, xCII_eq, xOII_eq, xOI_eq = get_xe_arr(nH, T, xH2, xeM0, xi_CR, G_PE,  G_CI,
                                                          zeta_pi, Z_d, Z_g, xCstd, xOstd, gr_rec=gr_rec)

    xHI_eq = 1.0 - xHII_eq
    xOII_eq = np.array(xOII_eq)
    xOI_eq = xOstd*Z_g - xOII_eq

    # print(cgi_xe_mHHe(T))
    # Add CIE electron abundance resulting from He and metals
    xe_eq += f1(T)*(cgi_xe_He(T) + Z_g*cgi_xe_mHHe(T))


    cool_H = coolLya(nH,T,xHI_eq,xe_eq)/nH +  coolffH(nH,T,xe_eq,xHII_eq)/nH + \
             coolrecH(nH,T,xe_eq,xHII_eq)/nH + coolHIion(nH,T,xe_eq,xHI_eq)/nH
    cool_Hffrec = coolffH(nH,T,xe_eq,xHII_eq)/nH + coolrecH(nH,T,xe_eq,xHII_eq)/nH
    cool_HLya = coolLya(nH,T,xHI_eq,xe_eq)/nH
    cool_CII_ = coolCII(nH,T,xe_eq,xHI_eq,0.0,xCII_eq)/nH
    cool_OI_ = coolOI(nH,T,xe_eq,xHI_eq,0.0,xOI_eq)/nH
    cool_OII_ = 4.0*coolOII(nH,T,xe_eq,xOII_eq)/nH
    cool_other =  cool_CII_ + cool_OI_ + cool_OII_
    if gr_rec:
        cool_grRec = coolRec(nH,T,xe_eq,Z_d,G_PE)/nH
        cool_other += cool_grRec
    else:
        cool_grRec = 0.0

    res = dict(T=T, xe_eq=xe_eq, xHI_eq=xHI_eq, xHII_eq=xHII_eq,
               xCII_eq=xCII_eq, xOI_eq=xOI_eq, xOII_eq=xOII_eq,
               cool_H=cool_H, cool_other=cool_other,
               cool_CII=cool_CII_, cool_OI=cool_OI_,
               cool_OII=cool_OII_, cool_grRec=cool_grRec,
               cool_Hffrec=cool_Hffrec, cool_HLya=cool_HLya)

    return res

def get_equilibrium_cooling(nH = 1.0, Z_g=1.0, Z_d=1.0, G_PE=1.0,
                            xi_CR=2.e-16, zeta_pi=1.e-8, gr_rec=True):

    cgi_metal, cgi_He, cgi_xe_mH, cgi_xe_mHHe, cgi_xe_He = get_CIE()

    r0 = get_cool(nH, xCstd=1.6e-4, xOstd=3.2e-4,
                  Z_g=Z_g, Z_d=Z_d, G_CI=G_PE*nH, G_PE=G_PE*nH, xH2=0.0,
                  xi_CR=xi_CR*nH, gr_rec=gr_rec,
                  zeta_pi=0.0)

    T = r0['T']
    xe_eq1 = r0['xe_eq']
    cool_tot1 = r0['cool_H'] + (1.0 - f1(T))*r0['cool_other'] + \
               f1(T)*(Z_g*cgi_metal(T)+cgi_He(T))

# Ionized
    r1 = get_cool(nH, xCstd=1.6e-4, xOstd=3.2e-4,
                  Z_g=Z_g, Z_d=Z_d, G_CI=G_PE*nH, G_PE=G_PE*nH, xH2=0.0,
                  xi_CR=xi_CR*nH, gr_rec=gr_rec,
                  zeta_pi=zeta_pi)
    T = r1['T']
    xe_eq2 = r1['xe_eq']
    cool_tot2 = r1['cool_H'] + (1.0 - f1(T))*r1['cool_other'] + \
                f1(T)*(Z_g*cgi_metal(T)+cgi_He(T))

    return T, cool_tot1, cool_tot2
