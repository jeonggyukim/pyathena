import numpy as np
import astropy.constants as ac
import astropy.units as au

from ..util.spline import GlobalSpline2D

# Original C version implemented in Athena-TIGRESS
# See also Gong, Ostriker, & Wolfire (2017) and https://github.com/munan/tigress_cooling

def get_xe_mol(nH, xH2, xe, T=20.0, xi_cr=1e-16, Z_g=1.0, Z_d=1.0):
    xe_max = 1.2006199779862501
    k1620 = 1e-14*Z_d
    k1622 = 1e-14*Z_d
    k1621 = 1e-9
    k1619 = 1.0e-7*(T*1e-2)**(-0.5)
    phi_s = (1.0 - xe/xe_max)*0.67/(1.0 + xe/0.05)
    xS = 5.3e-6*Z_g # From Draine's Table 9.5 (Diffuse H2)
    A = k1619*(1.0 + k1621/k1622*xS)
    B = k1620 + k1621*xS
    return 2.0*xH2*((B**2 + 4.0*A*xi_cr*(1.0 + phi_s)/nH)**0.5 - B)/(2.0*k1619)

def get_xCII(nH, xe, xH2, T, Z_d, Z_g, xi_CR, G_PE, G_CI,
             xCstd=1.6e-4, gr_rec=True, CRPhotC=True, iCII_rec_rate=0):

    xCtot = xCstd*Z_g
    small_ = 1e-50
    k_C_cr = 3.85*xi_CR
    k_C_photo = 3.5e-10*G_CI
    if CRPhotC:
        k_C_photo += 520.0*2.0*xH2*xi_CR

    if (iCII_rec_rate == 0):
        lnT = np.log(T)
        k_Cplus_e = np.where(T < 10.0,
                             9.982641225129824e-11,
                             np.exp(-0.7529152*lnT - 21.293937))
    else:
        k_Cplus_e = CII_rec_rate(T)

    if gr_rec:
        psi_gr = 1.7*G_PE*np.sqrt(T)/(nH*xe + small_) + small_
        cCp_ = np.array([45.58, 6.089e-3, 1.128, 4.331e2, 4.845e-2,0.8120, 1.333e-4])
        k_Cplus_gr = 1.0e-14*cCp_[0]/(1.0 + cCp_[1]*np.power(psi_gr, cCp_[2]) *
                                      (1.0 + cCp_[3] * np.power(T, cCp_[4])
                                       * np.power( psi_gr, -cCp_[5]-cCp_[6]*lnT ))) * Z_d
    else:
        k_Cplus_gr = 0.0

    k_Cplus_H2 = 3.3e-13 * np.power(T, -1.3) * np.exp(-23./T)

    c = (k_C_cr + k_C_photo) / nH
    al = k_Cplus_e*xe + k_Cplus_gr + k_Cplus_H2*xH2 + c
    ar = xCtot * c

    return ar / al


def get_xCO(nH, xH2, xCII, xOII, Z_d, Z_g, xi_CR, chi_CO,
            xCstd=1.6e-4, xOstd=3.2e-4):

    xCtot = xCstd*Z_g
    xOtot = xOstd*Z_g
    kcr16 = xi_CR*1e16
    term1 = np.maximum(4e3*Z_d/kcr16**2,1.0)
    ncrit = np.power(term1, chi_CO**(1.0/3.0))*(50*kcr16/np.power(Z_d,1.4))
    xCO = nH**2/(nH**2 + ncrit**2)
    xCO = xCO*(2.0*xH2)
    xCO = xCO*np.minimum(xCtot - xCII,xOtot-xOII)

    return xCO,ncrit

def get_charge_param(nH, T, xe, chi_PE, phi=1.0):
    # Charging parameter
    # (WD01 does not recommend using their eqaution for x < 100)
    # return np.maximum(1.7*chi_PE*np.sqrt(T)/(xe*nH*phi), 100.0)
    return 1.7*chi_PE*np.sqrt(T)/(xe*nH*phi) + 50.0

def heatPE(nH, T, xe, Z_d, chi_PE):
    # Weingartner & Draine (2001) Table 2
    # Rv = 3.1, bC=4.0, distribution A, ISRF
    CPE_ = np.array([5.22, 2.25, 0.04996, 0.00430, 0.147, 0.431, 0.692])
    x = get_charge_param(nH, T, xe, chi_PE)
    eps = (CPE_[0] + CPE_[1]*np.power(T, CPE_[4]))/ \
        (1. + CPE_[2]*np.power(x, CPE_[5])*(1. + CPE_[3]*np.power(x, CPE_[6])))

    return 1.7e-26*chi_PE*Z_d*eps

def heatPE_BT94(nH, T, xe, Z_d, chi_PE):
    x = get_charge_param(nH, T, xe, chi_PE)
    eps_BT94 =  4.87e-2/(1.0 + 4e-3*x**0.73) + 3.65e-2*(T*1e-4)**0.7/(1.0 + 2e-4*x)
    return 1.7e-24*chi_PE*Z_d*eps_BT94

def heatPE_W03(nH, T, xe, Z_d, chi_PE, phi=0.5):
    x = get_charge_param(nH, T, xe, chi_PE, phi=phi)
    eps_BT94 =  4.87e-2/(1.0 + 4e-3*x**0.73) + 3.65e-2*(T*1e-4)**0.7/(1.0 + 2e-4*x)
    # Multiply by 1.3 (due to increased PAH abundance)
    return 1.3*1.7e-24*chi_PE*Z_d*eps_BT94

def heatCR(nH, xe, xHI, xH2, xi_CR):

    eV_cgs = (1.0*au.eV).cgs.value
    qHI = (6.5 + 26.4*np.sqrt(xe / (xe + 0.07)))*eV_cgs

    # Heating rate per ionization in molecular region
    # See Appendix B in Krumholz 2014 (Despotic)
    log_nH = np.log10(nH)
    qH2 = np.zeros_like(nH)
    qH2 = np.where(log_nH < 2.0, 10.0*eV_cgs, 0.0) + \
          np.where(np.logical_and(log_nH >= 2.0, log_nH < 4.0),
                   (10 + 3*(log_nH - 2.0)*0.5)*eV_cgs, 0.0) + \
          np.where(np.logical_and(log_nH >= 4.0, log_nH < 7.0),
                   (13 + 4*(log_nH - 4.0)/3)*eV_cgs, 0.0) + \
          np.where(np.logical_and(log_nH >= 7.0, log_nH < 10.0),
                   (17 + (log_nH - 7.0)/3)*eV_cgs, 0.0) + \
          np.where(log_nH >= 10.0, 18.0*eV_cgs, 0.0)

    #return xi_CR*(xHI*qHI + 2.0*xH2*qH2 + 4.6e-10*xe) # last term included in Bialy+19
    return xi_CR*(xHI*qHI + 2.0*xH2*qH2)

def heatCR_old(nH, xe, xHI, xH2, xi_CR):

    eV_cgs = (1.0*au.eV).cgs.value
    xHetot = 0.1
    ktot = xi_CR*((2.3*xH2 + 1.5*xHI)*(xHI + 2.0*xH2) + 1.1*xHetot)
    qHI = (6.5 + 26.4*np.sqrt(xe / (xe + 0.07)))*eV_cgs

    # Heating rate per ionization in molecular region
    # See Appendix B in Krumholz 2014 (Despotic)
    log_nH = np.log10(nH)
    qH2 = np.zeros_like(nH)
    qH2 = np.where(log_nH < 2.0, 10.0*eV_cgs, 0.0) + \
          np.where(np.logical_and(log_nH >= 2.0, log_nH < 4.0),
                   (10 + 3*(log_nH - 2.0)*0.5)*eV_cgs, 0.0) + \
          np.where(np.logical_and(log_nH >= 4.0, log_nH < 7.0),
                   (13 + 4*(log_nH - 4.0)/3)*eV_cgs, 0.0) + \
          np.where(np.logical_and(log_nH >= 7.0, log_nH < 10.0),
                   (17 + (log_nH - 7.0)/3)*eV_cgs, 0.0) + \
          np.where(log_nH >= 10.0, 18.0*eV_cgs, 0.0)

    return ktot*(xHI*qHI + 2.0*xH2*qH2)

def heatH2(nH, T, xHI, xH2, Z_d, kgr_H2, xi_diss_H2, ikgr_H2, iH2heating):

    eV_cgs = (1.0*au.eV).cgs.value
    f_pump = 8.0

    if ikgr_H2 == 0: # Constant coeff
        kgr = kgr_H2*Z_d
    else:
        T2 = T*1e-2
        kgr = kgr_H2*Z_d*sqrt(T2)*2.0/(1+0.4*np.sqrt(T2)+0.2*T2+0.08*T2*T2)

    if iH2heating == 1:
        A = 2.0e-7
        D = xi_diss_H2
        t = 1.0 + T*1e-3
        geff_H  = np.power(10.0, -11.06 + 0.0555/t - 2.390/(t*t))
        geff_H2 = np.power(10.0, -11.08 - 3.671/t - 2.023/(t*t))
        ncrit = (A + D) / (geff_H*xHI + geff_H2*xH2)
        f = 1.0/(1.0 + ncrit/nH)
        heatH2form = kgr*nH*xHI*(0.2 + 4.2*f)*eV_cgs
        heatH2diss = xi_diss_H2*xH2*0.4*eV_cgs
        heatH2pump = xi_diss_H2*xH2*f_pump*2.0*f*eV_cgs
    else:
        # Hollenbach & McKee (1978) Eq (6.43), (6.45)
        de = 1.6*xHI*np.exp(-(400.0/T)**2) + 1.4*xH2*np.exp(-12000.0/(1200.0 + T))
        ncrit = 1e6/np.sqrt(T)/de
        f = nH/(nH + ncrit)
        heatH2form = kgr*nH*xHI*(0.2 + 4.2*f)*eV_cgs
        heatH2diss = xi_diss_H2*xH2*0.4*eV_cgs
        heatH2pump = xi_diss_H2*xH2*f_pump*2.2*f*eV_cgs

    return heatH2form,heatH2diss,heatH2pump

def heatH2pump(nH, T, xHI, xH2, xi_diss_H2):
    # Hollenbach & McKee (1978)
    f_pump = 9.0 # Use Draine & Bertoldi (1996) value (9.0 in HM79)
    eV_cgs = (1.0*au.eV).cgs.value
    de = 1.6*xHI*np.exp(-(400.0/T)**2) + 1.4*xH2*np.exp(-12000.0/(1200.0 + T))
    ncrit = 1e6/np.sqrt(T)/de
    f = nH/(nH + ncrit)

    return f_pump*2.2*xi_diss_H2*xH2*f*eV_cgs

def heatH2diss(xH2, xi_diss_H2):
    eV_cgs = (1.0*au.eV).cgs.value

    return 0.4*xi_diss_H2*xH2*eV_cgs

def heatH2pump_Burton90(nH, T, xHI, xH2, xi_diss_H2):
    # Burton, Hollenbach, & Tielens (1990) (Eq. A1)
    kpump = 9.0*xi_diss_H2
    Cdex = 1e-12*(1.4*np.exp(-18100.0/(T + 1200.0))*xH2 + \
                  np.exp(-1000.0/T)*xHI)*np.sqrt(T)*nH
    Crad = 2e-7
    Epump = 2.0*1.602e-12*Cdex/(Cdex + Crad)
    return kpump*Epump*xH2


def q10CII_(nH, T, xe, xHI, xH2):
    """Compute collisional de-excitation rate [s^-1]
    """

    # Ortho-to-para ratio of H2
    fp_ = 0.25
    fo_ = 0.75

    # Eqs (17.16) and (17.17) in Draine (2011)
    T2 = T*1e-2;
    k10e = 4.53e-8*np.sqrt(1.0e4/T)
    # Omega10e = (1.55+1.25*T*1e-4)/(1 + 0.35*(T*1e-4)**1.25)
    # k10e = 8.629e-8/np.sqrt(T*1e-4)*Omega10e

    k10HI = 7.58e-10*np.power(T2, 0.1281+0.0087*np.log(T2))

    k10oH2 = np.zeros_like(T)
    k10pH2 = np.zeros_like(T)

    # For T< 500K, use fit in Wiesenfeld & Goldsmith (2014)
    # For high temperature, use Glover+Jappsen 2007; scales similar to HI
    tmp = np.power(T, 0.07)
    k10oH2 = np.where(T < 500.0,
                      (5.33 + 0.11*T2)*1.0e-10,
                      3.74757785025e-10*tmp)
    k10pH2 = np.where(T < 500.0,
                      (4.43 + 0.33*T2)*1.0e-10,
                      3.88997286356e-10*tmp)

    k10H2 = k10oH2*fo_ + k10pH2*fp_

    return nH*(k10e*xe + k10HI*xHI + k10H2*xH2)


def coolCII(nH, T, xe, xHI, xH2, xCII):

    g0CII_ = 2.
    g1CII_ = 4.

    A10CII_ = 2.3e-6
    E10CII_ = 1.26e-14
    kB_cgs = ac.k_B.cgs.value

    q10 = q10CII_(nH, T, xe, xHI, xH2)
    q01 = (g1CII_/g0CII_)*q10*np.exp(-E10CII_/(kB_cgs*T))

    return q01/(q01 + q10 + A10CII_)*A10CII_*E10CII_*xCII

def coolHIion(nH, T, xe, xHI):
    eV_cgs = (1.0*au.eV).cgs.value
    return 13.6*eV_cgs*coeff_kcoll_H(T)*nH*xe*xHI

def coolCI(nH, T, xe, xHI, xH2, xCI):

    kB_cgs = ac.k_B.cgs.value
    fp_ = 0.25
    fo_ = 0.75

    # CI, 3 level system
    g0CI_ = 1
    g1CI_ = 3
    g2CI_ = 5
    A10CI_ = 7.880e-08
    A20CI_ = 1.810e-14
    A21CI_ = 2.650e-07
    E10CI_ = 3.261e-15
    E20CI_ = 8.624e-15
    E21CI_ = 5.363e-15

    # e-collisional coefficents (Johnson, Burke, & Kingston 1987; JPhysB, 20, 2553)
    T2 = T*1e-2
    lnT2 = np.log(T2)
    lnT = np.log(T)
    # ke(u,l) = fac*gamma(u,l)/g(u)

    fac = 8.629e-8*np.sqrt(1.0e4/T)

    # Collisional strength (valid for T < 10^4 K)
    lngamma10e = np.zeros_like(T)
    lngamma20e = np.zeros_like(T)
    lngamma21e = np.zeros_like(T)
    lngamma10e = np.where(T < 1.0e3,
                          (((-6.56325e-4*lnT -1.50892e-2)*lnT + 3.61184e-1)*\
                           lnT -7.73782e-1)*lnT - 9.25141,
                          (((1.0508e-1*lnT - 3.47620)*lnT + 4.2595e1)*\
                           lnT- 2.27913e2)*lnT + 4.446e2)
    lngamma20e = np.where(T < 1.0e3,
                          (((0.705277e-2*lnT - 0.111338)*lnT + 0.697638)*
                           lnT - 1.30743)*lnT -7.69735,
                          (((9.38138e-2*lnT - 3.03283)*lnT +3.61803e1)*\
                           lnT - 1.87474e2)*lnT +3.50609e2)
    lngamma21e = np.where(T < 1.0e3,
                          (((2.35272e-3*lnT - 4.18166e-2)*lnT + 0.358264)*\
                           lnT - 0.57443)*lnT -7.4387,
                          (((9.78573e-2*lnT - 3.19268)*lnT +3.85049e1)*\
                           lnT - 2.02193e2)*lnT +3.86186e2)

    k10e = fac * np.exp(lngamma10e)/g1CI_
    k20e = fac * np.exp(lngamma20e)/g2CI_
    k21e = fac * np.exp(lngamma21e)/g2CI_
    # Draine's HI/H2 collisional rates (Appendix F Table F.6)
    # NOTE: this is more updated than the LAMBDA database.
    k10HI = 1.26e-10 * np.power(T2, 0.115+0.057*lnT2)
    k20HI = 0.89e-10 * np.power(T2, 0.228+0.046*lnT2)
    k21HI = 2.64e-10 * np.power(T2, 0.231+0.046*lnT2)

    k10H2p = 0.67e-10 * np.power(T2, -0.085+0.102*lnT2)
    k10H2o = 0.71e-10 * np.power(T2, -0.004+0.049*lnT2)
    k20H2p = 0.86e-10 * np.power(T2, -0.010+0.048*lnT2)
    k20H2o = 0.69e-10 * np.power(T2, 0.169+0.038*lnT2)
    k21H2p = 1.75e-10 * np.power(T2, 0.072+0.064*lnT2)
    k21H2o = 1.48e-10 * np.power(T2, 0.263+0.031*lnT2)

    k10H2 = k10H2p*fp_ + k10H2o*fo_
    k20H2 = k20H2p*fp_ + k20H2o*fo_
    k21H2 = k21H2p*fp_ + k21H2o*fo_

    # The totol collisonal rates
    q10 = nH*(k10HI*xHI + k10H2*xH2 + k10e*xe)
    q20 = nH*(k20HI*xHI + k20H2*xH2 + k20e*xe)
    q21 = nH*(k21HI*xHI + k21H2*xH2 + k21e*xe)
    q01 = (g1CI_/g0CI_) * q10 * np.exp(-E10CI_/(kB_cgs*T))
    q02 = (g2CI_/g0CI_) * q20 * np.exp(-E20CI_/(kB_cgs*T))
    q12 = (g2CI_/g1CI_) * q21 * np.exp(-E21CI_/(kB_cgs*T))

    return cool3Level_(q01,q10,q02,q20,q12,q21,A10CI_,A20CI_,
                       A21CI_,E10CI_,E20CI_,E21CI_,xCI)

def coolneb(nH, T, xe, xHII, Z_g):
    aNEB_ = np.array([-0.0050817, 0.00765822, 0.11832144, -0.50515842,
                      0.81569592,-0.58648172,0.69170381])
    T4 = T*1e-4
    lnT4 = np.log(T4)
    lnT4_2 = lnT4*lnT4
    lnT4_3 = lnT4_2*lnT4
    lnT4_4 = lnT4_3*lnT4
    lnT4_5 = lnT4_4*lnT4
    lnT4_6 = lnT4_5*lnT4
    poly_fit = np.power(10.0,
                        aNEB_[0]*lnT4_6 +
                        aNEB_[1]*lnT4_5 +
                        aNEB_[2]*lnT4_4 +
                        aNEB_[3]*lnT4_3 +
                        aNEB_[4]*lnT4_2 +
                        aNEB_[5]*lnT4 + aNEB_[6])
    f_red = 1/(1.0 + 0.12*np.power(xe*nH*1e-2, 0.38 - 0.12*lnT4))

    return 3.677602203699553e-21*\
        Z_g*xHII*xe*nH/np.sqrt(T)*np.exp(-38585.52/T)*poly_fit*f_red

def coolOII(nH, T, xe, xOII):

    T4 = T*1e-4
    kB_cgs = ac.k_B.cgs.value
    # OII, 3 level system
    g0OII_ = 4  # 4S_3/2
    g1OII_ = 6  # 2D_5/2
    g2OII_ = 4  # 2D_3/2
    A10OII_ = 3.6e-5
    A20OII_ = 1.6e-4
    A21OII_ = 1.3e-7
    E10OII_ = (ac.h*ac.c/(3728.8*au.angstrom)).to('erg').value
    E20OII_ = (ac.h*ac.c/(3726.0*au.angstrom)).to('erg').value
    E21OII_ = (ac.h*ac.c/(497.1*au.micron)).to('erg').value

    # Draine (2011)
    Omega10e = 0.803*T4**(0.023-0.008*np.log(T4))
    Omega20e = 0.550*T4**(0.054-0.004*np.log(T4))
    Omega21e = 1.434*T4**(-0.176+0.004*np.log(T4))

    prefactor = 8.629e-8/np.sqrt(T4)
    k10e = prefactor*Omega10e/g1OII_
    k20e = prefactor*Omega20e/g2OII_
    k21e = prefactor*Omega21e/g2OII_

    # Total collisional rates
    q10 = nH*k10e*xe
    q20 = nH*k20e*xe
    q21 = nH*k21e*xe
    q01 = (g1OII_/g0OII_) * q10 * np.exp(-E10OII_/(kB_cgs*T))
    q02 = (g2OII_/g0OII_) * q20 * np.exp(-E20OII_/(kB_cgs*T))
    q12 = (g2OII_/g1OII_) * q21 * np.exp(-E21OII_/(kB_cgs*T))

    return cool3Level_(q01, q10, q02, q20, q12, q21, A10OII_, A20OII_,
                       A21OII_, E10OII_, E20OII_, E21OII_, xOII)

def coolOI(nH, T, xe, xHI, xH2, xOI):

    kB_cgs = ac.k_B.cgs.value

    # Ortho-to-para ratio of H2
    fp_ = 0.25
    fo_ = 0.75

    # OI, 3 level system
    g0OI_ = 5
    g1OI_ = 3
    g2OI_ = 1
    A10OI_ = 8.910e-05
    A20OI_ = 1.340e-10
    A21OI_ = 1.750e-05
    E10OI_ = 3.144e-14
    E20OI_ = 4.509e-14
    E21OI_ = 1.365e-14

    T2 = T*1e-2
    lnT2 = np.log(T2)
    # Collisional rates from  Draine (2011) (Appendix F Table F.6)
    # HI
    k10HI = 3.57e-10*np.power(T2, 0.419-0.003*lnT2)
    k20HI = 3.19e-10*np.power(T2, 0.369-0.006*lnT2)
    k21HI = 4.34e-10*np.power(T2, 0.755-0.160*lnT2)
    # H2
    k10H2p = 1.49e-10 * np.power(T2, 0.264+0.025*lnT2)
    k10H2o = 1.37e-10 * np.power(T2, 0.296+0.043*lnT2)
    k20H2p = 1.90e-10 * np.power(T2, 0.203+0.041*lnT2)
    k20H2o = 2.23e-10 * np.power(T2, 0.237+0.058*lnT2)
    k21H2p = 2.10e-12 * np.power(T2, 0.889+0.043*lnT2)
    k21H2o = 3.00e-12 * np.power(T2, 1.198+0.525*lnT2)
    k10H2 = k10H2p*fp_ + k10H2o*fo_
    k20H2 = k20H2p*fp_ + k20H2o*fo_
    k21H2 = k21H2p*fp_ + k21H2o*fo_

    # Electrons; fit from Bell+1998
    k10e = 5.12e-10 * np.power(T, -0.075)
    k20e = 4.86e-10 * np.power(T, -0.026)
    k21e = 1.08e-14 * np.power(T, 0.926)
    # Total collisional rates
    q10 = nH*(k10HI*xHI + k10H2*xH2 + k10e*xe)
    q20 = nH*(k20HI*xHI + k20H2*xH2 + k20e*xe)
    q21 = nH*(k21HI*xHI + k21H2*xH2 + k21e*xe)
    q01 = (g1OI_/g0OI_) * q10 * np.exp(-E10OI_/(kB_cgs*T))
    q02 = (g2OI_/g0OI_) * q20 * np.exp(-E20OI_/(kB_cgs*T))
    q12 = (g2OI_/g1OI_) * q21 * np.exp(-E21OI_/(kB_cgs*T))

    return cool3Level_(q01, q10, q02, q20, q12, q21, A10OI_, A20OI_,
                       A21OI_, E10OI_, E20OI_, E21OI_, xOI)

def coolHISmith21(nH, T, xe, xHI):

    g1 = 2.0
    prefactor = 8.62913e-06
    Tinv = 1.0/T
    T6 = T*1e-6
    T6_SQR = T6*T6
    T6_CUB = T6_SQR*T6

    Upsilon_12_cool = np.where(T6 > 0.3, 3.7354906,
                               0.616414 + 16.8152*T6 - 32.0571*T6_SQR + 35.5428*T6_CUB)
    Upsilon_13_cool = np.where(T6 > 0.3, 0.8098996999999998,
                               0.217382 + 3.92604*T6 - 10.6349*T6_SQR + 13.7721*T6_CUB)
    Upsilon_14_cool = np.where(T6 > 0.3, 0.3261425,
                               0.0959324 + 1.89951*T6 - 6.96467*T6_SQR + 10.6362*T6_CUB)
    Upsilon_15_cool = np.where(T6 > 0.3, 0.16427759999999997,
                               0.0747075 + 0.670939*T6 - 2.28512*T6_SQR + 3.4796*T6_CUB)

    # Total = sum_n E1n*exp(-T1n/T)*Upsilon_1n_Cool
    total = 1.63490e-11*Upsilon_12_cool*np.exp(-118415.6*Tinv) + \
        1.93766e-11*Upsilon_13_cool*np.exp(-140344.4*Tinv) + \
        2.04363e-11*Upsilon_14_cool*np.exp(-148019.5*Tinv) + \
        2.09267e-11*Upsilon_15_cool*np.exp(-151572.0*Tinv)

    return xHI*nH*xe*prefactor/(g1*np.sqrt(T))*total


def coolLya(nH, T, xe, xHI):

    # HI, 2 level system
    A10HI_ = 6.265e8
    E10HI_ = 1.634e-11
    g0HI_ = 1
    g1HI_ = 3

    ne = xe*nH
    T4 = T*1.0e-4
    # fac = 6.3803e-9*np.power(T4, 1.17)
    fac = 5.30856e-08*np.power(T4,1.4897e-01)/(1.0 + np.power(0.2*T4, 0.64897))
    k01e = fac*np.exp(-11.84/T4)
    q01 = k01e*ne
    q10 = (g0HI_/g1HI_)*fac*ne

    return q01/(q01 + q10 + A10HI_)*A10HI_*E10HI_*xHI

def coolHI(nH, T, xHI, xe):

    # Neutral Hydrogen cooling (Lya + Lyb + two photon) taken from DESPOTIC

    #TLyA = (3.0/4.0*(ac.h*ac.c*ac.Ryd).to('eV')/ac.k_B).to('K').value
    #TLyB = (8.0/9.0*(ac.h*ac.c*ac.Ryd).to('eV')/ac.k_B).to('K').value

    TLyA = 118415.63430152694
    TLyB = 140344.45546847637

    kB = ac.k_B.cgs.value
    upsilon2s = 0.35
    upsilon2p = 0.69
    upsilon3s = 0.077
    upsilon3p = 0.14
    upsilon3d = 0.073
    fac = 8.629e-6/(2*np.sqrt(T))
    exfacLyA = np.exp(-TLyA/T)
    exfacLyB = np.exp(-TLyB/T)
    Lambda2p = fac * exfacLyA * upsilon2s * xHI * xe * nH * kB * TLyA
    LambdaLyA = fac * exfacLyA * upsilon2p * xHI * xe * nH * kB * TLyA
    LambdaLyB = fac * exfacLyB * (upsilon3s + upsilon3p + upsilon3d) * xHI * xe * nH * kB * TLyB

    return Lambda2p + LambdaLyA + LambdaLyB

def coolH2G17(nH, T, xHI, xH2, xHII, xe, xHe=0.1):
    """
    H2 Cooling from Gong et al. (2017)
    """

    Tmax_H2 = 6000.  # maximum temperature above which use Tmax
    Tmin_H2 = 10.    # minimum temperature below which cut off cooling

    # Note: limit extended to T< 10K and T>6000K
    T = np.where(T > Tmax_H2, Tmax_H2, T)

    logT3 = np.log10(T*1.0e-3)
    logT3_2 = logT3 * logT3
    logT3_3 = logT3_2 * logT3
    logT3_4 = logT3_3 * logT3
    logT3_5 = logT3_4 * logT3

    # HI
    LHI = np.where(T < 100.0,
                   np.power(10, -16.818342e0 +3.7383713e1*logT3 \
                            + 5.8145166e1*logT3_2 + 4.8656103e1*logT3_3 \
                            + 2.0159831e1*logT3_4 + 3.8479610e0*logT3_5), 0.0)
    LHI += np.where(np.logical_and(T >= 100.0, T < 1000.0),
                    np.power(10, -2.4311209e1 +3.5692468e0*logT3 \
                             - 1.1332860e1*logT3_2 - 2.7850082e1*logT3_3 \
                             - 2.1328264e1*logT3_4 - 4.2519023e0*logT3_5), 0.0)
    LHI += np.where(T >= 1000.0,
                    np.power(10, -2.4311209e1 +4.6450521e0*logT3 + \
                             - 3.7209846e0*logT3_2 + 5.9369081e0*logT3_3
                             - 5.5108049e0*logT3_4 + 1.5538288e0*logT3_5), 0.0)

    # H2
    LH2 = np.power(10, -2.3962112e1 +2.09433740e0*logT3 \
                   -0.77151436e0*logT3_2 +0.43693353e0*logT3_3 \
                   -0.14913216e0*logT3_4 -0.033638326e0*logT3_5)
    # He
    LHe = np.power(10, -2.3689237e1 +2.1892372e0*logT3 \
                   -0.81520438e0*logT3_2 +0.29036281e0*logT3_3 \
                   -0.16596184e0*logT3_4 +0.19191375e0*logT3_5)
    # H+
    LHplus = np.power(10, -2.1716699e1 +1.3865783e0*logT3 \
                      -0.37915285e0*logT3_2 +0.11453688e0*logT3_3 \
                      -0.23214154e0*logT3_4 +0.058538864e0*logT3_5)
    # e
    Le = np.where(T < 200.0,
                  np.power(10, -3.4286155e1 -4.8537163e1*logT3 \
                           -7.7121176e1*logT3_2 -5.1352459e1*logT3_3 \
                           -1.5169150e1*logT3_4 -0.98120322e0*logT3_5),
                  np.power(10, -2.2190316e1 +1.5728955e0*logT3 \
                           -0.213351e0*logT3_2 +0.96149759e0*logT3_3 \
                           -0.91023195e0*logT3_4 +0.13749749e0*logT3_5)
                  )

    # total cooling in low density limit
    Gamma_n0 = LHI*xHI*nH + LH2*xH2*nH + LHe*xHe*nH + LHplus*xHII*nH + Le*xe*nH
    # cooling rate at LTE, from Hollenbach + McKee 1979
    T3 = T*1.0e-3
    Gamma_LTE_HR = (9.5e-22*np.power(T3, 3.76))/(1.+0.12*np.power(T3, 2.1))* \
        np.exp(-np.power(0.13/T3, 3))+ 3.e-24*np.exp(-0.51/T3)
    Gamma_LTE_HV = 6.7e-19*np.exp(-5.86/T3) + 1.6e-18*np.exp(-11.7/T3)
    Gamma_LTE = Gamma_LTE_HR +  Gamma_LTE_HV
    # Total cooling rate
    Gamma_tot = np.where(Gamma_n0 > 1e-100,
                         Gamma_LTE / (1.0 + Gamma_LTE/Gamma_n0),
                         0.0)
    Gamma_tot = np.where(T >= Tmin_H2, Gamma_tot, 0.0)

    return Gamma_tot * xH2;

def coolH2rovib(nH, T, xHI, xH2):
    """
    Cooling by rotation-vibration lines of H2
    from Moseley et al. (2021)
    """

    n1 = 50.0
    n2 = 450.0
    n3 = 25.0
    n4 = 900
    T3 = T*1e-3
    T3inv = 1.0/T3
    nH2 = xH2*nH
    nHI = xHI*nH
    x1 = nHI + 5.0*nH2
    x2 = nHI + 4.5*nH2
    x3 = nHI + 0.75*nH2
    x4 = nHI + 0.05*nH2
    sqrtT3 = np.power(T3,0.5)
    f1 = 1.1e-25*sqrtT3*np.exp(-0.51*T3inv)* \
        (0.7*x1/(1.0 + x1/n1) + 0.3*x1/(1.0 + x1/(10.0*n1)))
    f2 = 2.0e-25*T3*np.exp(-T3inv)* \
        (0.35*x2/(1.0 + x2/n2) + 0.65*x2/(1.0 + x2/(10.0*n2)))
    f3 = 2.4e-24*sqrtT3*T3*np.exp(-2.0*T3inv)* \
        (x3/(1.0 + x3/n3))
    f4 = 1.7e-23*sqrtT3*T3*np.exp(-4.0*T3inv)* \
        (0.45*x4/(1.0 + x4/n4) + 0.55*x4/(1.0 + x4/(10.0*n4)))

    return xH2*(f1 + f2 + f3 + f4)

def coolH2colldiss(nH, T, xHI, xH2):
    eV_cgs=(1.*au.eV).cgs.value
    xi_coll_H2=coeff_coll_H2(nH, T, xHI, xH2)
    return 4.48*eV_cgs*xH2*xi_coll_H2


def coolffH(nH, T, xe, xHII):
    """free-free power for hydrogen (Z=1)
    """
    # Frequency-averaged Gaunt factor (Eq.10.11 in Draine 2011)
    gff_T = 1.0 + 0.44/(1.0 + 0.058* np.log(T/10**5.4)**2)
    return 1.422e-25*gff_T*(T*1e-4)**0.5*nH*xe*xHII

def coolrecH(nH, T, xe, xHII):
    from .rec_rate import RecRate
    rec = RecRate()
    Err_B = (0.684 - 0.0416*np.log(T*1e-4))*ac.k_B.cgs.value*T
    return Err_B*rec.get_rec_rate_H_caseB(T)*nH*xe*xHII

def coolRec(nH, T, xe, Z_d, chi_PE):
    # Weingartner & Draine (2001) Table 3
    # Rv = 3.1, bC=4.0, distribution A, ISRF
    DPE_ = np.array([0.4535, 2.234, -6.266, 1.442, 0.05089])
    ne = nH*xe
    x = get_charge_param(nH, T, xe, chi_PE)
    lnx = np.log(x)

    return 1.0e-28*Z_d*ne*np.power(T, DPE_[0] + DPE_[1]/lnx)*\
        np.exp((DPE_[2]+(DPE_[3]-DPE_[4]*lnx)*lnx))

def coolRec_BT94(nH, T, xe, Z_d, chi_PE):
    # Eq 44 in Bakes & Tielens (1994)
    ne = nH*xe
    x = get_charge_param(nH, T, xe, chi_PE)
    beta = 0.735*np.power(T,-0.068)

    return 3.49e-30*np.power(T, 0.944)*np.power(x, beta)*ne*Z_d

def coolRec_W03(nH, T, xe, Z_d, chi_PE, phi=0.5):
    ne = nH*xe
    x = get_charge_param(nH, T, xe, chi_PE, phi=phi)
    beta = 0.735*np.power(T,-0.068)
    return 4.65e-30*np.power(T, 0.944)*np.power(x, beta)*ne*Z_d*phi

def cooldust(nH, T, Td, Z_d):
    # From Krumholz (2014) or Goldsmith (2001)
    # Strictly speaking the coupling constant alpha_gd depends on chemical composition
    alpha_gd = 3.2e-34
    return Z_d*alpha_gd*nH*np.sqrt(T)*(T - Td)

def cool3Level_(q01, q10, q02, q20, q12, q21,
                A10, A20, A21, E10, E20, E21, xs):
    # Equilibrium level population including
    # collisional excitation/de-excitation, spontatneous emission
    # but ignoring absorption and stimulated emission.
    R10 = q10 + A10
    R20 = q20 + A20
    R21 = q21 + A21
    a0 = R10*R20 + R10*R21 + q12*R20
    a1 = q01*R20 + q01*R21 + R21*q02
    a2 = q02*R10 + q02*q12 + q12*q01
    de = a0 + a1 + a2
    f1 = a1 / de
    f2 = a2 / de

    return (f1*A10*E10 + f2*(A20*E20 + A21*E21))*xs


def cool2Level_(q01, q10, A10, E10, xs):
    f1 = q01 / (q01 + q10 + A10)
    return f1*A10*E10*xs


def coolCO(nH, T, xe, xHI, xH2, xCO, dvdr):

    # CO cooling table data from Omukai+2010
    TCO_ = np.array([10,20,30,50,80,100,300,600,1000,1500,2000])

    NeffCO_ = np.array([14.0, 14.5, 15.0, 15.5, 16.0, 16.5,
                        17.0, 17.5, 18.0, 18.5, 19.0])

    # JKIM: Values are slightly different from Omukai+2010 for T > 100K, any reason?
    L0CO_ = np.array([24.77, 24.38, 24.21, 24.03, 23.89, 23.82,
                      23.34238089, 22.99832519, 22.75384686, 22.56640625, 22.43740866])

    LLTECO_ = np.array([[21.08, 20.35, 19.94, 19.45, 19.01, 18.80, 17.81, 17.23, 16.86, 16.66, 16.55],
                        [21.09, 20.35, 19.95, 19.45, 19.01, 18.80, 17.81, 17.23, 16.86, 16.66, 16.55],
                        [21.11, 20.37, 19.96, 19.46, 19.01, 18.80, 17.81, 17.23, 16.86, 16.66, 16.55],
                        [21.18, 20.40, 19.98, 19.47, 19.02, 18.81, 17.82, 17.23, 16.87, 16.66, 16.55],
                        [21.37, 20.51, 20.05, 19.52, 19.05, 18.83, 17.82, 17.23, 16.87, 16.66, 16.55],
                        [21.67, 20.73, 20.23, 19.64, 19.13, 18.90, 17.85, 17.25, 16.88, 16.67, 16.56],
                        [22.04, 21.05, 20.52, 19.87, 19.32, 19.06, 17.92, 17.28, 16.90, 16.69, 16.58],
                        [22.44, 21.42, 20.86, 20.19, 19.60, 19.33, 18.08, 17.38, 16.97, 16.75, 16.63],
                        [22.87, 21.82, 21.24, 20.55, 19.95, 19.66, 18.34, 17.59, 17.15, 16.91, 16.78],
                        [23.30, 22.23, 21.65, 20.94, 20.32, 20.03, 18.67, 17.89, 17.48, 17.26, 17.12],
                        [23.76, 22.66, 22.06, 21.35, 20.71, 20.42, 19.03, 18.26, 17.93, 17.74, 17.61]])

    nhalfCO_ = np.array([[3.29, 3.49 ,3.67 ,3.97, 4.30, 4.46, 5.17, 5.47, 5.53, 5.30, 4.70],
                         [3.27, 3.48 ,3.66 ,3.96, 4.30, 4.45, 5.16, 5.47, 5.53, 5.30, 4.70],
                         [3.22, 3.45 ,3.64 ,3.94, 4.29, 4.45, 5.16, 5.47, 5.53, 5.30, 4.70],
                         [3.07, 3.34 ,3.56 ,3.89, 4.26, 4.42, 5.15, 5.46, 5.52, 5.30, 4.70],
                         [2.72, 3.09 ,3.35 ,3.74, 4.16, 4.34, 5.13, 5.45, 5.51, 5.29, 4.68],
                         [2.24, 2.65 ,2.95 ,3.42, 3.92, 4.14, 5.06, 5.41, 5.48, 5.26, 4.64],
                         [1.74, 2.15 ,2.47 ,2.95, 3.49, 3.74, 4.86, 5.30, 5.39, 5.17, 4.53],
                         [1.24, 1.65 ,1.97 ,2.45, 3.00, 3.25, 4.47, 5.02, 5.16, 4.94, 4.27],
                         [0.742, 1.15 ,1.47 ,1.95, 2.50, 2.75, 3.98, 4.57, 4.73, 4.52, 3.84],
                         [0.242, 0.652,0.966 ,1.45, 2.00, 2.25, 3.48, 4.07, 4.24, 4.03, 3.35],
                         [-0.258, 0.152,0.466 ,0.95, 1.50, 1.75, 2.98, 3.57, 3.74, 3.53, 2.85]])

    alphaCO_ = np.array([[0.439, 0.409, 0.392, 0.370, 0.361, 0.357, 0.385, 0.437, 0.428, 0.354, 0.322],
                         [0.436, 0.407, 0.391, 0.368, 0.359, 0.356, 0.385, 0.437, 0.427, 0.354, 0.322],
                         [0.428, 0.401, 0.385, 0.364, 0.356, 0.352, 0.383, 0.436, 0.427, 0.352, 0.320],
                         [0.416, 0.388, 0.373, 0.353, 0.347, 0.345, 0.380, 0.434, 0.425, 0.349, 0.316],
                         [0.416, 0.378, 0.360, 0.338, 0.332, 0.330, 0.371, 0.429, 0.421, 0.341, 0.307],
                         [0.450, 0.396, 0.367, 0.334, 0.322, 0.317, 0.355, 0.419, 0.414, 0.329, 0.292],
                         [0.492, 0.435, 0.403, 0.362, 0.339, 0.329, 0.343, 0.406, 0.401, 0.317, 0.276],
                         [0.529, 0.473, 0.441, 0.404, 0.381, 0.370, 0.362, 0.410, 0.392, 0.316, 0.272],
                         [0.555, 0.503, 0.473, 0.440, 0.423, 0.414, 0.418, 0.446, 0.404, 0.335, 0.289],
                         [0.582, 0.528, 0.499, 0.469, 0.457, 0.451, 0.470, 0.487, 0.432, 0.364, 0.310],
                         [0.596, 0.546, 0.519, 0.492, 0.483, 0.479, 0.510, 0.516, 0.448, 0.372, 0.313]])

    from scipy import interpolate
    x = TCO_
    y = NeffCO_

    # ISSUE: interp2d doesn't extrapolate as per documentation,
    # instead uses nearest-neighbor
    # See https://github.com/scipy/scipy/issues/8099
    # Using this code
    # https://github.com/pig2015/mathpy/blob/master/polation/globalspline.py
    L0CO = interpolate.interp1d(TCO_, L0CO_, kind='linear',
                                bounds_error=False, fill_value='extrapolate')
    LLTECO = GlobalSpline2D(x, y, LLTECO_, kind='linear')
    nhalfCO = GlobalSpline2D(x, y, nhalfCO_, kind='linear')
    alphaCO = GlobalSpline2D(x, y, alphaCO_, kind='linear')

    # LLTECO = interpolate.interp2d(x, y, LLTECO_, kind='linear')
    # nhalfCO = interpolate.interp2d(x, y, nhalfCO_, kind='linear')
    # alphaCO = interpolate.interp2d(x, y, alphaCO_, kind='linear')

    def enumerate2D(arr1, arr2):
        assert arr1.shape == arr2.shape, "Error - dimensions."
        for indexes, data in np.ndenumerate(arr1):
            yield indexes, data, arr2[indexes]

    kB_cgs = ac.k_B.cgs.value
    TmaxCO = 2000.0;

    # Calculate effective column of CO
    # maximum escape probability length, in cgs unites
    Leff_CO_max = 3.086e20 # 100 pc
    # vth/Leff_CO_max
    grad_small = np.sqrt(2.0*kB_cgs*T/4.68e-23)/Leff_CO_max
    gradeff = np.maximum(dvdr, grad_small)
    NCOeff = (nH*xCO)/gradeff

    # Maximum temperature (2000 K) above which we use TmaxCO
    # for cooling rate interpolation
    T1 = np.where(T < 2000.0, T, 2000.0)
    facT = np.power(1. - np.exp(-T1), 1.0e3) # Shuts off cooling if T < ~10 K

    # Small number for a very small NCOeff
    eps = 1.0e13
    log_NCOeff = np.log10(NCOeff*1.0e5 + eps) # unit of effective column: cm^-2/km/s
    Troot4 = np.power(T1, 0.25)
    neff = nH*(xH2 + 1.75*Troot4*xHI + 680.1/Troot4*xe)
    L0 = 10.0**(-L0CO(T1))

    alpha = np.zeros_like(T1)
    LLTE = np.zeros_like(T1)
    nhalf = np.zeros_like(T1)
    for idx, T1_, log_NCOeff_ in enumerate2D(T1,log_NCOeff):
        alpha[idx] = alphaCO(T1_,float(log_NCOeff_))
        LLTE[idx] = 10.0**(-LLTECO(T1_,float(log_NCOeff_)))
        nhalf[idx] = 10.0**(nhalfCO(T1_,float(log_NCOeff_)))


    inv_LCO = 1./L0 + neff/LLTE + 1./L0*np.power(neff/nhalf, alpha)*(1. - nhalf*L0/LLTE)

    return (1./inv_LCO)*neff*xCO*facT

def fshld_H2(NH2, b5=3.0):
    x = 2.0e-15*NH2
    sqrt_onepx = np.sqrt(1.0 + x)
    return 0.965/(1.0 + x/b5)**2 + 0.034970262640168365/sqrt_onepx*np.exp(-8.5e-4*sqrt_onepx)

def fshld_CO(logNH2,logNCO):

    # # CO column density for DB table
    # logNCOvDB_ = np.array([0, 13, 14, 15, 16, 17, 18, 19])
    # # H2 column densities for DB table
    # logNH2vDB_ = np.array([0, 19, 20, 21, 22, 23])
    # # Tabulated shielding factors
    # ThetavDB_ = np.array([[1.0, 9.681e-1, 7.764e-1, 3.631e-1, 7.013e-2, 1.295e-2, 1.738e-3, 9.985e-5],
    #                       [8.215e-1, 7.916e-1, 6.160e-1, 2.749e-1, 5.351e-2, 1.065e-2, 1.519e-3, 8.818e-5],
    #                       [7.160e-1, 6.900e-1, 5.360e-1, 2.359e-1, 4.416e-2, 8.769e-3, 1.254e-3, 7.558e-5],
    #                       [3.500e-1, 3.415e-1, 2.863e-1, 1.360e-1, 2.500e-2, 4.983e-3, 7.151e-4, 3.796e-5],
    #                       [4.973e-2, 4.877e-2, 4.296e-2, 2.110e-2, 4.958e-3, 9.245e-4, 1.745e-4, 8.377e-6],
    #                       [1.310e-4, 1.293e-4, 1.160e-4, 6.346e-5, 1.822e-5, 6.842e-6, 3.622e-6, 3.572e-7]])

    # Visser 2009 Table 5
    logNCOV09_ = np.array([0.000,10.000,10.200,10.400,10.600,10.800,11.000,11.200,11.400,11.600,11.800,
                           12.000,12.200,12.400,12.600,12.800,13.000,13.200,13.400,13.600,13.800,14.000,
                           14.200,14.400,14.600,14.800,15.000,15.200,15.400,15.600,15.800,16.000,16.200,
                           16.400,16.600,16.800,17.000,17.200,17.400,17.600,17.800,18.000,18.200,18.400,
                           18.600,18.800,19.000])

    logNH2V09_ = np.array([0.000,15.000,15.200,15.400,15.600,15.800,16.000,16.200,16.400,16.600,16.800,
                           17.000,17.200,17.400,17.600,17.800,18.000,18.200,18.400,18.600,18.800,19.000,
                           19.200,19.400,19.600,19.800,20.000,20.200,20.400,20.600,20.800,21.000,21.200,
                           21.400,21.600,21.800,22.000,22.200,22.400,22.600,22.800,23.000])

    ThetaV09_ = np.array(
      [[1.000e+00,9.997e-01,9.995e-01,9.992e-01,9.988e-01,9.981e-01,9.970e-01,9.953e-01,9.926e-01,9.883e-01,
        9.817e-01,9.716e-01,9.563e-01,9.338e-01,9.021e-01,8.599e-01,8.080e-01,7.498e-01,6.900e-01,6.323e-01,
        5.777e-01,5.250e-01,4.720e-01,4.177e-01,3.614e-01,3.028e-01,2.434e-01,1.871e-01,1.387e-01,1.012e-01,
        7.401e-02,5.467e-02,4.075e-02,3.063e-02,2.323e-02,1.775e-02,1.362e-02,1.044e-02,7.963e-03,6.037e-03,
        4.541e-03,3.378e-03,2.470e-03,1.759e-03,1.210e-03,8.046e-04,5.240e-04],
       [8.985e-01,8.983e-01,8.981e-01,8.978e-01,8.974e-01,8.967e-01,8.956e-01,8.939e-01,8.913e-01,8.871e-01,
        8.807e-01,8.707e-01,8.558e-01,8.338e-01,8.030e-01,7.621e-01,7.122e-01,6.569e-01,6.014e-01,5.494e-01,
        5.021e-01,4.578e-01,4.137e-01,3.681e-01,3.201e-01,2.694e-01,2.177e-01,1.683e-01,1.256e-01,9.224e-02,
        6.786e-02,5.034e-02,3.767e-02,2.844e-02,2.167e-02,1.664e-02,1.282e-02,9.868e-03,7.556e-03,5.743e-03,
        4.325e-03,3.223e-03,2.363e-03,1.690e-03,1.169e-03,7.815e-04,5.112e-04],
       [8.966e-01,8.963e-01,8.962e-01,8.959e-01,8.955e-01,8.948e-01,8.937e-01,8.920e-01,8.894e-01,8.852e-01,
        8.788e-01,8.688e-01,8.539e-01,8.319e-01,8.011e-01,7.602e-01,7.103e-01,6.551e-01,5.996e-01,5.476e-01,
        5.004e-01,4.562e-01,4.122e-01,3.667e-01,3.190e-01,2.685e-01,2.171e-01,1.679e-01,1.254e-01,9.214e-02,
        6.781e-02,5.031e-02,3.765e-02,2.842e-02,2.166e-02,1.663e-02,1.282e-02,9.865e-03,7.554e-03,5.741e-03,
        4.323e-03,3.222e-03,2.362e-03,1.689e-03,1.169e-03,7.811e-04,5.110e-04],
       [8.949e-01,8.946e-01,8.944e-01,8.941e-01,8.937e-01,8.930e-01,8.920e-01,8.903e-01,8.876e-01,8.834e-01,
        8.770e-01,8.671e-01,8.521e-01,8.302e-01,7.993e-01,7.585e-01,7.086e-01,6.533e-01,5.979e-01,5.460e-01,
        4.988e-01,4.546e-01,4.107e-01,3.655e-01,3.179e-01,2.677e-01,2.165e-01,1.676e-01,1.252e-01,9.204e-02,
        6.776e-02,5.028e-02,3.763e-02,2.841e-02,2.165e-02,1.662e-02,1.281e-02,9.861e-03,7.551e-03,5.739e-03,
        4.322e-03,3.220e-03,2.361e-03,1.689e-03,1.168e-03,7.808e-04,5.108e-04],
       [8.932e-01,8.929e-01,8.927e-01,8.924e-01,8.920e-01,8.913e-01,8.903e-01,8.886e-01,8.859e-01,8.818e-01,
        8.753e-01,8.654e-01,8.504e-01,8.285e-01,7.976e-01,7.568e-01,7.069e-01,6.517e-01,5.962e-01,5.444e-01,
        4.972e-01,4.531e-01,4.094e-01,3.642e-01,3.169e-01,2.669e-01,2.159e-01,1.672e-01,1.250e-01,9.193e-02,
        6.770e-02,5.025e-02,3.761e-02,2.839e-02,2.164e-02,1.661e-02,1.281e-02,9.858e-03,7.549e-03,5.737e-03,
        4.320e-03,3.219e-03,2.360e-03,1.688e-03,1.168e-03,7.805e-04,5.106e-04],
       [8.915e-01,8.912e-01,8.911e-01,8.908e-01,8.904e-01,8.897e-01,8.886e-01,8.869e-01,8.843e-01,8.801e-01,
        8.737e-01,8.637e-01,8.488e-01,8.269e-01,7.960e-01,7.551e-01,7.053e-01,6.501e-01,5.947e-01,5.428e-01,
        4.957e-01,4.517e-01,4.080e-01,3.630e-01,3.159e-01,2.661e-01,2.154e-01,1.669e-01,1.248e-01,9.182e-02,
        6.764e-02,5.022e-02,3.759e-02,2.838e-02,2.162e-02,1.661e-02,1.280e-02,9.854e-03,7.546e-03,5.735e-03,
        4.319e-03,3.218e-03,2.359e-03,1.687e-03,1.167e-03,7.802e-04,5.104e-04],
       [8.899e-01,8.896e-01,8.895e-01,8.892e-01,8.888e-01,8.881e-01,8.870e-01,8.853e-01,8.827e-01,8.785e-01,
        8.721e-01,8.621e-01,8.472e-01,8.253e-01,7.944e-01,7.536e-01,7.037e-01,6.485e-01,5.931e-01,5.413e-01,
        4.942e-01,4.503e-01,4.067e-01,3.618e-01,3.148e-01,2.653e-01,2.148e-01,1.665e-01,1.246e-01,9.170e-02,
        6.758e-02,5.018e-02,3.757e-02,2.837e-02,2.161e-02,1.660e-02,1.280e-02,9.851e-03,7.544e-03,5.733e-03,
        4.317e-03,3.216e-03,2.358e-03,1.686e-03,1.167e-03,7.799e-04,5.103e-04],
       [8.855e-01,8.852e-01,8.850e-01,8.848e-01,8.843e-01,8.837e-01,8.826e-01,8.809e-01,8.782e-01,8.741e-01,
        8.676e-01,8.577e-01,8.428e-01,8.209e-01,7.900e-01,7.492e-01,6.993e-01,6.442e-01,5.888e-01,5.371e-01,
        4.901e-01,4.463e-01,4.028e-01,3.582e-01,3.114e-01,2.622e-01,2.120e-01,1.642e-01,1.227e-01,9.024e-02,
        6.640e-02,4.917e-02,3.667e-02,2.759e-02,2.096e-02,1.608e-02,1.239e-02,9.538e-03,7.308e-03,5.558e-03,
        4.189e-03,3.123e-03,2.290e-03,1.638e-03,1.133e-03,7.572e-04,4.958e-04],
       [8.834e-01,8.831e-01,8.829e-01,8.826e-01,8.822e-01,8.815e-01,8.805e-01,8.788e-01,8.761e-01,8.720e-01,
        8.655e-01,8.556e-01,8.406e-01,8.187e-01,7.879e-01,7.471e-01,6.972e-01,6.421e-01,5.867e-01,5.350e-01,
        4.881e-01,4.443e-01,4.010e-01,3.565e-01,3.099e-01,2.609e-01,2.111e-01,1.635e-01,1.223e-01,9.003e-02,
        6.629e-02,4.911e-02,3.664e-02,2.757e-02,2.095e-02,1.607e-02,1.238e-02,9.533e-03,7.305e-03,5.555e-03,
        4.187e-03,3.121e-03,2.289e-03,1.637e-03,1.132e-03,7.567e-04,4.955e-04],
       [8.814e-01,8.811e-01,8.809e-01,8.807e-01,8.802e-01,8.796e-01,8.785e-01,8.768e-01,8.741e-01,8.700e-01,
        8.635e-01,8.536e-01,8.387e-01,8.168e-01,7.859e-01,7.451e-01,6.953e-01,6.401e-01,5.848e-01,5.331e-01,
        4.862e-01,4.425e-01,3.993e-01,3.549e-01,3.085e-01,2.597e-01,2.101e-01,1.629e-01,1.219e-01,8.978e-02,
        6.616e-02,4.905e-02,3.661e-02,2.755e-02,2.094e-02,1.606e-02,1.237e-02,9.529e-03,7.302e-03,5.553e-03,
        4.185e-03,3.120e-03,2.288e-03,1.636e-03,1.131e-03,7.562e-04,4.952e-04],
       [8.792e-01,8.789e-01,8.788e-01,8.785e-01,8.781e-01,8.774e-01,8.763e-01,8.746e-01,8.720e-01,8.678e-01,
        8.614e-01,8.515e-01,8.365e-01,8.146e-01,7.838e-01,7.429e-01,6.931e-01,6.380e-01,5.827e-01,5.310e-01,
        4.842e-01,4.405e-01,3.974e-01,3.531e-01,3.068e-01,2.583e-01,2.090e-01,1.620e-01,1.214e-01,8.948e-02,
        6.601e-02,4.897e-02,3.657e-02,2.753e-02,2.092e-02,1.605e-02,1.237e-02,9.523e-03,7.297e-03,5.549e-03,
        4.182e-03,3.117e-03,2.286e-03,1.634e-03,1.130e-03,7.557e-04,4.948e-04],
       [8.766e-01,8.764e-01,8.762e-01,8.759e-01,8.755e-01,8.748e-01,8.737e-01,8.720e-01,8.694e-01,8.652e-01,
        8.588e-01,8.489e-01,8.339e-01,8.120e-01,7.812e-01,7.404e-01,6.906e-01,6.355e-01,5.802e-01,5.285e-01,
        4.817e-01,4.381e-01,3.951e-01,3.509e-01,3.049e-01,2.566e-01,2.076e-01,1.611e-01,1.207e-01,8.911e-02,
        6.581e-02,4.887e-02,3.652e-02,2.749e-02,2.090e-02,1.603e-02,1.235e-02,9.515e-03,7.291e-03,5.545e-03,
        4.178e-03,3.114e-03,2.284e-03,1.633e-03,1.129e-03,7.549e-04,4.943e-04],
       [8.735e-01,8.732e-01,8.730e-01,8.728e-01,8.723e-01,8.716e-01,8.706e-01,8.689e-01,8.662e-01,8.621e-01,
        8.556e-01,8.457e-01,8.308e-01,8.089e-01,7.781e-01,7.372e-01,6.874e-01,6.324e-01,5.771e-01,5.255e-01,
        4.787e-01,4.352e-01,3.923e-01,3.483e-01,3.025e-01,2.546e-01,2.060e-01,1.598e-01,1.199e-01,8.861e-02,
        6.554e-02,4.874e-02,3.644e-02,2.745e-02,2.087e-02,1.601e-02,1.234e-02,9.504e-03,7.284e-03,5.539e-03,
        4.173e-03,3.111e-03,2.281e-03,1.630e-03,1.128e-03,7.540e-04,4.937e-04],
       [8.697e-01,8.694e-01,8.692e-01,8.689e-01,8.685e-01,8.678e-01,8.668e-01,8.651e-01,8.624e-01,8.583e-01,
        8.518e-01,8.419e-01,8.270e-01,8.051e-01,7.743e-01,7.335e-01,6.837e-01,6.286e-01,5.734e-01,5.218e-01,
        4.751e-01,4.317e-01,3.889e-01,3.451e-01,2.996e-01,2.520e-01,2.039e-01,1.582e-01,1.188e-01,8.795e-02,
        6.517e-02,4.854e-02,3.634e-02,2.738e-02,2.083e-02,1.598e-02,1.232e-02,9.490e-03,7.273e-03,5.530e-03,
        4.167e-03,3.105e-03,2.277e-03,1.628e-03,1.126e-03,7.527e-04,4.929e-04],
       [8.652e-01,8.649e-01,8.647e-01,8.644e-01,8.640e-01,8.633e-01,8.623e-01,8.606e-01,8.579e-01,8.538e-01,
        8.473e-01,8.374e-01,8.225e-01,8.006e-01,7.698e-01,7.290e-01,6.793e-01,6.242e-01,5.690e-01,5.175e-01,
        4.709e-01,4.276e-01,3.849e-01,3.414e-01,2.962e-01,2.490e-01,2.014e-01,1.563e-01,1.175e-01,8.712e-02,
        6.469e-02,4.828e-02,3.619e-02,2.729e-02,2.077e-02,1.594e-02,1.229e-02,9.471e-03,7.259e-03,5.519e-03,
        4.158e-03,3.098e-03,2.271e-03,1.624e-03,1.123e-03,7.509e-04,4.919e-04],
       [8.600e-01,8.597e-01,8.595e-01,8.593e-01,8.588e-01,8.582e-01,8.571e-01,8.554e-01,8.528e-01,8.486e-01,
        8.422e-01,8.323e-01,8.173e-01,7.955e-01,7.647e-01,7.239e-01,6.742e-01,6.192e-01,5.640e-01,5.126e-01,
        4.660e-01,4.229e-01,3.804e-01,3.371e-01,2.923e-01,2.456e-01,1.985e-01,1.541e-01,1.159e-01,8.608e-02,
        6.406e-02,4.791e-02,3.598e-02,2.717e-02,2.069e-02,1.589e-02,1.226e-02,9.445e-03,7.239e-03,5.504e-03,
        4.146e-03,3.089e-03,2.264e-03,1.618e-03,1.119e-03,7.486e-04,4.904e-04],
       [8.543e-01,8.540e-01,8.539e-01,8.536e-01,8.532e-01,8.525e-01,8.514e-01,8.497e-01,8.471e-01,8.429e-01,
        8.365e-01,8.266e-01,8.117e-01,7.898e-01,7.591e-01,7.183e-01,6.686e-01,6.137e-01,5.586e-01,5.072e-01,
        4.608e-01,4.178e-01,3.755e-01,3.325e-01,2.880e-01,2.418e-01,1.953e-01,1.516e-01,1.140e-01,8.481e-02,
        6.326e-02,4.743e-02,3.569e-02,2.699e-02,2.058e-02,1.582e-02,1.221e-02,9.411e-03,7.213e-03,5.483e-03,
        4.130e-03,3.076e-03,2.254e-03,1.611e-03,1.115e-03,7.456e-04,4.885e-04],
       [8.482e-01,8.479e-01,8.477e-01,8.475e-01,8.470e-01,8.464e-01,8.453e-01,8.436e-01,8.410e-01,8.368e-01,
        8.304e-01,8.205e-01,8.056e-01,7.838e-01,7.531e-01,7.124e-01,6.627e-01,6.079e-01,5.529e-01,5.016e-01,
        4.553e-01,4.125e-01,3.704e-01,3.277e-01,2.836e-01,2.379e-01,1.920e-01,1.489e-01,1.120e-01,8.334e-02,
        6.227e-02,4.680e-02,3.530e-02,2.675e-02,2.043e-02,1.572e-02,1.214e-02,9.363e-03,7.177e-03,5.456e-03,
        4.108e-03,3.059e-03,2.241e-03,1.602e-03,1.108e-03,7.415e-04,4.859e-04],
       [8.416e-01,8.414e-01,8.412e-01,8.409e-01,8.405e-01,8.398e-01,8.387e-01,8.371e-01,8.344e-01,8.303e-01,
        8.238e-01,8.140e-01,7.991e-01,7.773e-01,7.466e-01,7.060e-01,6.565e-01,6.017e-01,5.468e-01,4.957e-01,
        4.496e-01,4.069e-01,3.651e-01,3.228e-01,2.792e-01,2.340e-01,1.886e-01,1.461e-01,1.098e-01,8.169e-02,
        6.110e-02,4.602e-02,3.479e-02,2.643e-02,2.023e-02,1.559e-02,1.205e-02,9.298e-03,7.129e-03,5.419e-03,
        4.079e-03,3.037e-03,2.225e-03,1.590e-03,1.100e-03,7.364e-04,4.827e-04],
       [8.345e-01,8.342e-01,8.340e-01,8.337e-01,8.333e-01,8.326e-01,8.316e-01,8.299e-01,8.273e-01,8.231e-01,
        8.167e-01,8.069e-01,7.920e-01,7.703e-01,7.397e-01,6.992e-01,6.498e-01,5.952e-01,5.405e-01,4.895e-01,
        4.436e-01,4.012e-01,3.597e-01,3.178e-01,2.747e-01,2.300e-01,1.853e-01,1.433e-01,1.075e-01,7.995e-02,
        5.981e-02,4.510e-02,3.417e-02,2.602e-02,1.995e-02,1.541e-02,1.193e-02,9.209e-03,7.063e-03,5.369e-03,
        4.041e-03,3.008e-03,2.204e-03,1.575e-03,1.090e-03,7.299e-04,4.785e-04],
       [8.265e-01,8.262e-01,8.260e-01,8.258e-01,8.254e-01,8.247e-01,8.236e-01,8.220e-01,8.193e-01,8.152e-01,
        8.088e-01,7.990e-01,7.842e-01,7.626e-01,7.321e-01,6.918e-01,6.425e-01,5.881e-01,5.337e-01,4.830e-01,
        4.373e-01,3.952e-01,3.542e-01,3.127e-01,2.701e-01,2.261e-01,1.820e-01,1.406e-01,1.053e-01,7.817e-02,
        5.843e-02,4.407e-02,3.343e-02,2.550e-02,1.960e-02,1.516e-02,1.176e-02,9.086e-03,6.973e-03,5.302e-03,
        3.991e-03,2.971e-03,2.176e-03,1.556e-03,1.077e-03,7.215e-04,4.731e-04],
       [8.176e-01,8.173e-01,8.171e-01,8.169e-01,8.164e-01,8.158e-01,8.147e-01,8.130e-01,8.104e-01,8.063e-01,
        8.000e-01,7.903e-01,7.756e-01,7.540e-01,7.237e-01,6.836e-01,6.347e-01,5.806e-01,5.265e-01,4.761e-01,
        4.308e-01,3.891e-01,3.485e-01,3.076e-01,2.656e-01,2.222e-01,1.787e-01,1.378e-01,1.031e-01,7.637e-02,
        5.701e-02,4.297e-02,3.260e-02,2.488e-02,1.915e-02,1.484e-02,1.152e-02,8.919e-03,6.851e-03,5.212e-03,
        3.925e-03,2.922e-03,2.141e-03,1.531e-03,1.061e-03,7.108e-04,4.662e-04],
       [8.073e-01,8.070e-01,8.069e-01,8.066e-01,8.062e-01,8.055e-01,8.045e-01,8.028e-01,8.002e-01,7.962e-01,
        7.899e-01,7.802e-01,7.657e-01,7.443e-01,7.143e-01,6.746e-01,6.261e-01,5.725e-01,5.189e-01,4.689e-01,
        4.240e-01,3.828e-01,3.427e-01,3.023e-01,2.609e-01,2.182e-01,1.753e-01,1.350e-01,1.008e-01,7.452e-02,
        5.552e-02,4.179e-02,3.167e-02,2.417e-02,1.861e-02,1.443e-02,1.122e-02,8.697e-03,6.689e-03,5.093e-03,
        3.838e-03,2.859e-03,2.096e-03,1.500e-03,1.040e-03,6.969e-04,4.573e-04],
       [7.949e-01,7.946e-01,7.944e-01,7.942e-01,7.938e-01,7.931e-01,7.921e-01,7.904e-01,7.879e-01,7.839e-01,
        7.777e-01,7.682e-01,7.538e-01,7.328e-01,7.032e-01,6.640e-01,6.162e-01,5.633e-01,5.104e-01,4.611e-01,
        4.168e-01,3.760e-01,3.364e-01,2.966e-01,2.559e-01,2.138e-01,1.716e-01,1.320e-01,9.827e-02,7.248e-02,
        5.388e-02,4.048e-02,3.064e-02,2.335e-02,1.796e-02,1.393e-02,1.084e-02,8.411e-03,6.477e-03,4.938e-03,
        3.725e-03,2.778e-03,2.038e-03,1.460e-03,1.013e-03,6.792e-04,4.459e-04],
       [7.784e-01,7.782e-01,7.780e-01,7.778e-01,7.774e-01,7.767e-01,7.757e-01,7.741e-01,7.716e-01,7.677e-01,
        7.617e-01,7.524e-01,7.383e-01,7.177e-01,6.888e-01,6.504e-01,6.037e-01,5.519e-01,5.000e-01,4.516e-01,
        4.081e-01,3.680e-01,3.290e-01,2.898e-01,2.498e-01,2.085e-01,1.671e-01,1.283e-01,9.528e-02,7.009e-02,
        5.199e-02,3.899e-02,2.946e-02,2.242e-02,1.722e-02,1.334e-02,1.038e-02,8.056e-03,6.209e-03,4.740e-03,
        3.581e-03,2.675e-03,1.965e-03,1.409e-03,9.785e-04,6.566e-04,4.313e-04],
       [7.553e-01,7.550e-01,7.549e-01,7.546e-01,7.543e-01,7.536e-01,7.527e-01,7.511e-01,7.487e-01,7.450e-01,
        7.391e-01,7.301e-01,7.166e-01,6.967e-01,6.688e-01,6.317e-01,5.865e-01,5.364e-01,4.861e-01,4.391e-01,
        3.966e-01,3.575e-01,3.194e-01,2.811e-01,2.420e-01,2.017e-01,1.614e-01,1.236e-01,9.158e-02,6.719e-02,
        4.972e-02,3.723e-02,2.808e-02,2.134e-02,1.637e-02,1.267e-02,9.843e-03,7.635e-03,5.885e-03,4.496e-03,
        3.402e-03,2.546e-03,1.874e-03,1.346e-03,9.352e-04,6.280e-04,4.129e-04],
       [7.223e-01,7.220e-01,7.219e-01,7.216e-01,7.213e-01,7.207e-01,7.198e-01,7.183e-01,7.160e-01,7.125e-01,
        7.069e-01,6.984e-01,6.856e-01,6.668e-01,6.404e-01,6.053e-01,5.624e-01,5.149e-01,4.670e-01,4.221e-01,
        3.812e-01,3.434e-01,3.066e-01,2.695e-01,2.317e-01,1.929e-01,1.540e-01,1.177e-01,8.696e-02,6.364e-02,
        4.701e-02,3.515e-02,2.649e-02,2.010e-02,1.540e-02,1.190e-02,9.231e-03,7.150e-03,5.506e-03,4.207e-03,
        3.186e-03,2.388e-03,1.761e-03,1.266e-03,8.811e-04,5.923e-04,3.899e-04],
       [6.758e-01,6.756e-01,6.754e-01,6.752e-01,6.749e-01,6.743e-01,6.735e-01,6.722e-01,6.701e-01,6.668e-01,
        6.618e-01,6.540e-01,6.423e-01,6.250e-01,6.008e-01,5.686e-01,5.292e-01,4.854e-01,4.410e-01,3.991e-01,
        3.607e-01,3.249e-01,2.898e-01,2.546e-01,2.186e-01,1.816e-01,1.448e-01,1.104e-01,8.136e-02,5.941e-02,
        4.382e-02,3.274e-02,2.465e-02,1.869e-02,1.430e-02,1.103e-02,8.540e-03,6.601e-03,5.074e-03,3.871e-03,
        2.931e-03,2.198e-03,1.623e-03,1.169e-03,8.145e-04,5.484e-04,3.619e-04],
       [6.127e-01,6.125e-01,6.124e-01,6.122e-01,6.119e-01,6.114e-01,6.107e-01,6.095e-01,6.077e-01,6.049e-01,
        6.005e-01,5.937e-01,5.835e-01,5.685e-01,5.473e-01,5.192e-01,4.847e-01,4.461e-01,4.067e-01,3.691e-01,
        3.342e-01,3.012e-01,2.686e-01,2.358e-01,2.022e-01,1.678e-01,1.335e-01,1.016e-01,7.474e-02,5.447e-02,
        4.013e-02,2.996e-02,2.255e-02,1.708e-02,1.304e-02,1.004e-02,7.759e-03,5.983e-03,4.587e-03,3.490e-03,
        2.637e-03,1.975e-03,1.459e-03,1.052e-03,7.343e-04,4.956e-04,3.281e-04],
       [5.310e-01,5.309e-01,5.308e-01,5.306e-01,5.304e-01,5.300e-01,5.295e-01,5.285e-01,5.271e-01,5.248e-01,
        5.212e-01,5.158e-01,5.076e-01,4.955e-01,4.784e-01,4.557e-01,4.276e-01,3.959e-01,3.632e-01,3.313e-01,
        3.010e-01,2.718e-01,2.426e-01,2.129e-01,1.825e-01,1.513e-01,1.202e-01,9.136e-02,6.707e-02,4.880e-02,
        3.591e-02,2.678e-02,2.014e-02,1.523e-02,1.161e-02,8.915e-03,6.873e-03,5.286e-03,4.039e-03,3.062e-03,
        2.305e-03,1.721e-03,1.270e-03,9.161e-04,6.405e-04,4.336e-04,2.884e-04],
       [4.328e-01,4.327e-01,4.326e-01,4.325e-01,4.323e-01,4.321e-01,4.317e-01,4.310e-01,4.300e-01,4.283e-01,
        4.258e-01,4.220e-01,4.161e-01,4.075e-01,3.952e-01,3.788e-01,3.584e-01,3.350e-01,3.103e-01,2.854e-01,
        2.609e-01,2.364e-01,2.115e-01,1.857e-01,1.592e-01,1.320e-01,1.048e-01,7.956e-02,5.830e-02,4.233e-02,
        3.109e-02,2.315e-02,1.737e-02,1.310e-02,9.965e-03,7.636e-03,5.872e-03,4.504e-03,3.430e-03,2.589e-03,
        1.940e-03,1.443e-03,1.061e-03,7.650e-04,5.354e-04,3.635e-04,2.432e-04],
       [3.260e-01,3.260e-01,3.259e-01,3.258e-01,3.258e-01,3.256e-01,3.253e-01,3.250e-01,3.243e-01,3.234e-01,
        3.219e-01,3.196e-01,3.161e-01,3.110e-01,3.036e-01,2.937e-01,2.810e-01,2.661e-01,2.497e-01,2.324e-01,
        2.143e-01,1.953e-01,1.753e-01,1.543e-01,1.325e-01,1.098e-01,8.726e-02,6.618e-02,4.841e-02,3.506e-02,
        2.569e-02,1.907e-02,1.426e-02,1.073e-02,8.134e-03,6.216e-03,4.768e-03,3.647e-03,2.768e-03,2.081e-03,
        1.552e-03,1.150e-03,8.429e-04,6.062e-04,4.238e-04,2.882e-04,1.941e-04],
       [2.241e-01,2.241e-01,2.241e-01,2.241e-01,2.240e-01,2.240e-01,2.238e-01,2.237e-01,2.234e-01,2.229e-01,
        2.223e-01,2.212e-01,2.196e-01,2.172e-01,2.137e-01,2.089e-01,2.025e-01,1.947e-01,1.854e-01,1.748e-01,
        1.629e-01,1.496e-01,1.349e-01,1.192e-01,1.025e-01,8.509e-02,6.767e-02,5.133e-02,3.752e-02,2.712e-02,
        1.982e-02,1.466e-02,1.092e-02,8.184e-03,6.184e-03,4.713e-03,3.606e-03,2.751e-03,2.083e-03,1.561e-03,
        1.161e-03,8.580e-04,6.273e-04,4.495e-04,3.130e-04,2.126e-04,1.440e-04],
       [1.394e-01,1.394e-01,1.394e-01,1.393e-01,1.393e-01,1.393e-01,1.393e-01,1.392e-01,1.391e-01,1.390e-01,
        1.387e-01,1.384e-01,1.378e-01,1.369e-01,1.356e-01,1.338e-01,1.312e-01,1.277e-01,1.232e-01,1.176e-01,
        1.106e-01,1.024e-01,9.286e-02,8.230e-02,7.094e-02,5.903e-02,4.704e-02,3.576e-02,2.616e-02,1.892e-02,
        1.381e-02,1.020e-02,7.568e-03,5.648e-03,4.253e-03,3.233e-03,2.470e-03,1.882e-03,1.423e-03,1.067e-03,
        7.942e-04,5.872e-04,4.290e-04,3.062e-04,2.119e-04,1.432e-04,9.745e-05],
       [7.604e-02,7.604e-02,7.603e-02,7.603e-02,7.603e-02,7.602e-02,7.601e-02,7.599e-02,7.596e-02,7.591e-02,
        7.584e-02,7.573e-02,7.555e-02,7.528e-02,7.486e-02,7.422e-02,7.327e-02,7.189e-02,6.995e-02,6.729e-02,
        6.380e-02,5.941e-02,5.415e-02,4.814e-02,4.160e-02,3.472e-02,2.779e-02,2.123e-02,1.561e-02,1.134e-02,
        8.295e-03,6.124e-03,4.536e-03,3.374e-03,2.534e-03,1.924e-03,1.470e-03,1.122e-03,8.510e-04,6.409e-04,
        4.796e-04,3.565e-04,2.612e-04,1.862e-04,1.281e-04,8.610e-05,5.885e-05],
       [3.382e-02,3.382e-02,3.382e-02,3.381e-02,3.381e-02,3.381e-02,3.381e-02,3.380e-02,3.379e-02,3.378e-02,
        3.376e-02,3.372e-02,3.366e-02,3.357e-02,3.343e-02,3.322e-02,3.289e-02,3.239e-02,3.165e-02,3.060e-02,
        2.915e-02,2.725e-02,2.490e-02,2.219e-02,1.922e-02,1.612e-02,1.299e-02,1.003e-02,7.466e-03,5.480e-03,
        4.038e-03,2.990e-03,2.213e-03,1.643e-03,1.232e-03,9.352e-04,7.164e-04,5.497e-04,4.205e-04,3.201e-04,
        2.425e-04,1.823e-04,1.348e-04,9.649e-05,6.645e-05,4.483e-05,3.108e-05],
       [1.108e-02,1.108e-02,1.108e-02,1.108e-02,1.108e-02,1.108e-02,1.107e-02,1.107e-02,1.107e-02,1.107e-02,
        1.106e-02,1.105e-02,1.103e-02,1.101e-02,1.097e-02,1.090e-02,1.081e-02,1.066e-02,1.043e-02,1.011e-02,
        9.646e-03,9.033e-03,8.269e-03,7.385e-03,6.427e-03,5.434e-03,4.441e-03,3.492e-03,2.653e-03,1.982e-03,
        1.479e-03,1.102e-03,8.166e-04,6.052e-04,4.528e-04,3.439e-04,2.644e-04,2.048e-04,1.589e-04,1.233e-04,
        9.539e-05,7.329e-05,5.529e-05,4.038e-05,2.845e-05,1.982e-05,1.437e-05],
       [2.364e-03,2.364e-03,2.364e-03,2.364e-03,2.364e-03,2.364e-03,2.363e-03,2.363e-03,2.363e-03,2.362e-03,
        2.360e-03,2.358e-03,2.355e-03,2.350e-03,2.342e-03,2.330e-03,2.310e-03,2.281e-03,2.236e-03,2.171e-03,
        2.078e-03,1.955e-03,1.802e-03,1.625e-03,1.434e-03,1.236e-03,1.034e-03,8.355e-04,6.521e-04,4.978e-04,
        3.767e-04,2.830e-04,2.105e-04,1.558e-04,1.162e-04,8.813e-05,6.808e-05,5.344e-05,4.250e-05,3.411e-05,
        2.751e-05,2.211e-05,1.752e-05,1.356e-05,1.029e-05,7.862e-06,6.317e-06],
       [2.983e-04,2.982e-04,2.982e-04,2.982e-04,2.982e-04,2.982e-04,2.982e-04,2.982e-04,2.981e-04,2.980e-04,
        2.979e-04,2.977e-04,2.974e-04,2.970e-04,2.962e-04,2.951e-04,2.933e-04,2.905e-04,2.863e-04,2.801e-04,
        2.713e-04,2.595e-04,2.446e-04,2.268e-04,2.066e-04,1.841e-04,1.592e-04,1.325e-04,1.059e-04,8.236e-05,
        6.326e-05,4.825e-05,3.638e-05,2.723e-05,2.054e-05,1.583e-05,1.253e-05,1.023e-05,8.603e-06,7.410e-06,
        6.479e-06,5.689e-06,4.977e-06,4.328e-06,3.767e-06,3.331e-06,3.040e-06],
       [2.297e-05,2.297e-05,2.297e-05,2.296e-05,2.296e-05,2.296e-05,2.296e-05,2.296e-05,2.296e-05,2.296e-05,
        2.295e-05,2.295e-05,2.294e-05,2.292e-05,2.290e-05,2.286e-05,2.280e-05,2.270e-05,2.255e-05,2.233e-05,
        2.200e-05,2.154e-05,2.089e-05,2.002e-05,1.886e-05,1.736e-05,1.548e-05,1.326e-05,1.093e-05,8.801e-06,
        7.086e-06,5.751e-06,4.678e-06,3.829e-06,3.200e-06,2.755e-06,2.446e-06,2.236e-06,2.094e-06,1.996e-06,
        1.922e-06,1.857e-06,1.796e-06,1.737e-06,1.684e-06,1.640e-06,1.608e-06],
       [1.546e-06,1.546e-06,1.546e-06,1.546e-06,1.546e-06,1.546e-06,1.545e-06,1.545e-06,1.545e-06,1.545e-06,
        1.545e-06,1.545e-06,1.545e-06,1.545e-06,1.544e-06,1.544e-06,1.542e-06,1.541e-06,1.538e-06,1.534e-06,
        1.527e-06,1.516e-06,1.500e-06,1.477e-06,1.442e-06,1.393e-06,1.329e-06,1.251e-06,1.168e-06,1.092e-06,
        1.032e-06,9.867e-07,9.497e-07,9.197e-07,8.973e-07,8.813e-07,8.701e-07,8.624e-07,8.574e-07,8.539e-07,
        8.513e-07,8.490e-07,8.467e-07,8.444e-07,8.421e-07,8.397e-07,8.374e-07],
       [3.938e-07,3.938e-07,3.938e-07,3.938e-07,3.938e-07,3.938e-07,3.938e-07,3.938e-07,3.938e-07,3.938e-07,
        3.938e-07,3.938e-07,3.938e-07,3.938e-07,3.938e-07,3.938e-07,3.938e-07,3.937e-07,3.937e-07,3.937e-07,
        3.937e-07,3.936e-07,3.935e-07,3.933e-07,3.931e-07,3.928e-07,3.923e-07,3.918e-07,3.913e-07,3.908e-07,
        3.904e-07,3.901e-07,3.898e-07,3.896e-07,3.894e-07,3.893e-07,3.893e-07,3.892e-07,3.891e-07,3.891e-07,
        3.890e-07,3.890e-07,3.889e-07,3.887e-07,3.885e-07,3.881e-07,3.875e-07]]
      )

    from scipy import interpolate
    x, y = np.meshgrid(logNCOV09_,logNH2V09_)
    z = ThetaV09_
    Theta = interpolate.interp2d(x, y, z, kind='linear')

    return Theta(logNCO,logNH2)


def get_CI_lev(nH, T, xe, xHI, xH2):

    kB_cgs = ac.k_B.cgs.value
    fp_ = 0.25
    fo_ = 0.75

    # CI, 3 level system
    g0CI_ = 1
    g1CI_ = 3
    g2CI_ = 5
    A10CI_ = 7.880e-08
    A20CI_ = 1.810e-14
    A21CI_ = 2.650e-07
    E10CI_ = 3.261e-15
    E20CI_ = 8.624e-15
    E21CI_ = 5.363e-15

    # e-collisional coefficents (Johnson, Burke, & Kingston 1987; JPhysB, 20, 2553)
    T2 = T*1e-2
    lnT2 = np.log(T2)
    lnT = np.log(T)
    # ke(u,l) = fac*gamma(u,l)/g(u)

    fac = 8.629e-8*np.sqrt(1.0e4/T)

    # Collisional strength (valid for T < 10^4 K)
    lngamma10e = np.zeros_like(T)
    lngamma20e = np.zeros_like(T)
    lngamma21e = np.zeros_like(T)
    lngamma10e = np.where(T < 1.0e3,
                          (((-6.56325e-4*lnT -1.50892e-2)*lnT + 3.61184e-1)*\
                           lnT -7.73782e-1)*lnT - 9.25141,
                          (((1.0508e-1*lnT - 3.47620)*lnT + 4.2595e1)*\
                           lnT- 2.27913e2)*lnT + 4.446e2)
    lngamma20e = np.where(T < 1.0e3,
                          (((0.705277e-2*lnT - 0.111338)*lnT + 0.697638)*
                           lnT - 1.30743)*lnT -7.69735,
                          (((9.38138e-2*lnT - 3.03283)*lnT +3.61803e1)*\
                           lnT - 1.87474e2)*lnT +3.50609e2)
    lngamma21e = np.where(T < 1.0e3,
                          (((2.35272e-3*lnT - 4.18166e-2)*lnT + 0.358264)*\
                           lnT - 0.57443)*lnT -7.4387,
                          (((9.78573e-2*lnT - 3.19268)*lnT +3.85049e1)*\
                           lnT - 2.02193e2)*lnT +3.86186e2)

    k10e = fac * np.exp(lngamma10e)/g1CI_
    k20e = fac * np.exp(lngamma20e)/g2CI_
    k21e = fac * np.exp(lngamma21e)/g2CI_
    # Draine's HI/H2 collisional rates (Appendix F Table F.6)
    # NOTE: this is more updated than the LAMBDA database.
    k10HI = 1.26e-10 * np.power(T2, 0.115+0.057*lnT2)
    k20HI = 0.89e-10 * np.power(T2, 0.228+0.046*lnT2)
    k21HI = 2.64e-10 * np.power(T2, 0.231+0.046*lnT2)

    k10H2p = 0.67e-10 * np.power(T2, -0.085+0.102*lnT2)
    k10H2o = 0.71e-10 * np.power(T2, -0.004+0.049*lnT2)
    k20H2p = 0.86e-10 * np.power(T2, -0.010+0.048*lnT2)
    k20H2o = 0.69e-10 * np.power(T2, 0.169+0.038*lnT2)
    k21H2p = 1.75e-10 * np.power(T2, 0.072+0.064*lnT2)
    k21H2o = 1.48e-10 * np.power(T2, 0.263+0.031*lnT2)

    k10H2 = k10H2p*fp_ + k10H2o*fo_
    k20H2 = k20H2p*fp_ + k20H2o*fo_
    k21H2 = k21H2p*fp_ + k21H2o*fo_

    # The totol collisonal rates
    q10 = nH*(k10HI*xHI + k10H2*xH2 + k10e*xe)
    q20 = nH*(k20HI*xHI + k20H2*xH2 + k20e*xe)
    q21 = nH*(k21HI*xHI + k21H2*xH2 + k21e*xe)
    q01 = (g1CI_/g0CI_) * q10 * np.exp(-E10CI_/(kB_cgs*T))
    q02 = (g2CI_/g0CI_) * q20 * np.exp(-E20CI_/(kB_cgs*T))
    q12 = (g2CI_/g1CI_) * q21 * np.exp(-E21CI_/(kB_cgs*T))

    R10 = q10 + A10CI_
    R20 = q20 + A20CI_
    R21 = q21 + A21CI_
    a0 = R10*R20 + R10*R21 + q12*R20
    a1 = q01*R20 + q01*R21 + R21*q02
    a2 = q02*R10 + q02*q12 + q12*q01
    de = a0 + a1 + a2

    f0 = a0 / de
    f1 = a1 / de
    f2 = a2 / de

    return (f0, f1, f2)


def get_OI_lev(nH, T, xe, xHI, xH2):

    kB_cgs = ac.k_B.cgs.value

    # Ortho-to-para ratio of H2
    fp_ = 0.25
    fo_ = 0.75

    # OI, 3 level system
    g0OI_ = 5
    g1OI_ = 3
    g2OI_ = 1
    A10OI_ = 8.910e-05
    A20OI_ = 1.340e-10
    A21OI_ = 1.750e-05
    E10OI_ = 3.144e-14
    E20OI_ = 4.509e-14
    E21OI_ = 1.365e-14

    T2 = T*1e-2
    lnT2 = np.log(T2)
    # Collisional rates from  Draine (2011) (Appendix F Table F.6)
    # HI
    k10HI = 3.57e-10*np.power(T2, 0.419-0.003*lnT2)
    k20HI = 3.19e-10*np.power(T2, 0.369-0.006*lnT2)
    k21HI = 4.34e-10*np.power(T2, 0.755-0.160*lnT2)
    # H2
    k10H2p = 1.49e-10 * np.power(T2, 0.264+0.025*lnT2)
    k10H2o = 1.37e-10 * np.power(T2, 0.296+0.043*lnT2)
    k20H2p = 1.90e-10 * np.power(T2, 0.203+0.041*lnT2)
    k20H2o = 2.23e-10 * np.power(T2, 0.237+0.058*lnT2)
    k21H2p = 2.10e-12 * np.power(T2, 0.889+0.043*lnT2)
    k21H2o = 3.00e-12 * np.power(T2, 1.198+0.525*lnT2)
    k10H2 = k10H2p*fp_ + k10H2o*fo_
    k20H2 = k20H2p*fp_ + k20H2o*fo_
    k21H2 = k21H2p*fp_ + k21H2o*fo_

    # Electrons; fit from Bell+1998
    k10e = 5.12e-10 * np.power(T, -0.075)
    k20e = 4.86e-10 * np.power(T, -0.026)
    k21e = 1.08e-14 * np.power(T, 0.926)
    # Total collisional rates
    q10 = nH*(k10HI*xHI + k10H2*xH2 + k10e*xe)
    q20 = nH*(k20HI*xHI + k20H2*xH2 + k20e*xe)
    q21 = nH*(k21HI*xHI + k21H2*xH2 + k21e*xe)
    q01 = (g1OI_/g0OI_) * q10 * np.exp(-E10OI_/(kB_cgs*T))
    q02 = (g2OI_/g0OI_) * q20 * np.exp(-E20OI_/(kB_cgs*T))
    q12 = (g2OI_/g1OI_) * q21 * np.exp(-E21OI_/(kB_cgs*T))

    R10 = q10 + A10OI_
    R20 = q20 + A20OI_
    R21 = q21 + A21OI_
    a0 = R10*R20 + R10*R21 + q12*R20
    a1 = q01*R20 + q01*R21 + R21*q02
    a2 = q02*R10 + q02*q12 + q12*q01
    de = a0 + a1 + a2

    f0 = a0 / de
    f1 = a1 / de
    f2 = a2 / de

    return (f0, f1, f2)

def coeff_kcoll_H(T):
    """Collisional ionization
    """
    lnT = np.log(T)
    lnTe = np.log(T*8.6173e-5)
    k_coll = np.where(T > 3.0e3,
                np.exp((-3.271396786e1+ (1.35365560e1 + (- 5.73932875 + (1.56315498
                      + (- 2.877056e-1 + (3.48255977e-2 + (-2.63197617e-3
                      + (1.11954395e-4 + (-2.03914985e-6)
                         *lnTe)*lnTe)*lnTe)*lnTe)*lnTe)*lnTe)*lnTe)*lnTe)),
                      0.0)
    return k_coll

def coeff_alpha_rr_H(T):
    """Gong+17's fit to Ferland+92 radiative recombination
    """
    Tinv = 1/T
    bb = 315614.0*Tinv
    cc = 115188.0*Tinv
    dd = 1.0 + np.power(cc, 0.407)
    alpha_rr = 2.753e-14*np.power(bb, 1.5)*np.power(dd, -2.242)

    return alpha_rr

def coeff_alpha_gr_H(T, G_PE, ne,  Z_d):
    """Grain-assisted recombination
    """
    lnT = np.log(T)
    small_ = 1e-50
    cHp_ = np.array([12.25, 8.074e-6,1.378,5.087e2,1.586e-2,0.4723,1.102e-5])
    psi_gr = 1.7*G_PE*np.sqrt(T)/(ne + small_) + small_
    alpha_gr = 1.0e-14*cHp_[0] / \
        (1.0 + cHp_[1]*np.power(psi_gr, cHp_[2]) *
        (1.0 + cHp_[3] * np.power(T, cHp_[4])*np.power(psi_gr, -cHp_[5]-cHp_[6]*lnT)))*Z_d

    return alpha_gr

def coeff_coll_H2(nH,T,xHI,xH2):
    """destruction by collision
    Glover & Mac Low (2007)
    (15) H2 + *H -> 3 *H
    (16) H2 + H2 -> H2 + 2 *H
    --(9) Density dependent. See Glover+MacLow2007
    """
    temp_coll_ = 7.0e2
    Tinv = 1/T
    logT4 = np.log10(T*1e-4);
    k9l_ = 6.67e-12 * np.sqrt(T) * np.exp(-(1. + 63590.*Tinv))
    k9h_ = 3.52e-9 * np.exp(-43900.0*Tinv)
    k10l_ = 5.996e-30 * np.power(T, 4.1881) / np.power((1.0 + 6.761e-6*T), 5.6881) * \
            np.exp(-54657.4*Tinv)
    k10h_ = 1.3e-9 * np.exp(-53300.0*Tinv)
    ncrH2_ = np.power(10, (4.845 - 1.3*logT4 + 1.62*logT4*logT4))
    ncrHI_ = np.power(10, (3.0 - 0.416*logT4 - 0.327*logT4*logT4))
    ncrinv = np.clip(xHI/ncrHI_ + 2.0*xH2/ncrH2_,0.0,None)
    n2ncr = nH * ncrinv;
    k_H2_HI = np.power(10, np.log10(k9h_) * n2ncr/(1. + n2ncr)
                        + np.log10(k9l_) / (1. + n2ncr))
    k_H2_H2 = np.power(10, np.log10(k10h_) * n2ncr/(1. + n2ncr)
              + np.log10(k10l_) / (1. + n2ncr))
    xi_coll_H2 = k_H2_H2*nH*xH2 + k_H2_HI*nH*xHI
    return np.where(T>temp_coll_,xi_coll_H2,0.0)


def CII_rec_rate(T):
    A = 2.995e-9
    B = 0.7849
    T0 =  6.670e-3
    T1 = 1.943e6
    C = 0.1597
    T2 = 4.955e4
    BN = B + C * np.exp(-T2/T)
    term1 = np.sqrt(T/T0)
    term2 = np.sqrt(T/T1)
    alpha_rr = A/(term1*np.power(1.0+term1, 1.0-BN)*np.power(1.0+term2, 1.0+BN) )
    alpha_dr = np.power( T, -3.0/2.0 ) * ( 6.346e-9 * np.exp(-1.217e1/T) +
                                           9.793e-09 * np.exp(-7.38e1/T) +
                                           1.634e-06 * np.exp(-1.523e+04/T) )
    return alpha_rr + alpha_dr


def get_xn_eq(T, nH, zeta_pi=0.0, zeta_cr=0.0, coll_ion=True):
    """Function to compute equilibrium neutral fraction
    """
    T = np.atleast_1d(T)
    nH = np.atleast_1d(nH)
    if coll_ion:
        zeta_ci = nH*coeff_kcoll_H(T)
    else:
        zeta_ci = 0.0

    zeta_rec = nH*coeff_alpha_rr_H(T)

    aa = 1.0 + zeta_ci/zeta_rec
    bb = -(2.0 + (zeta_pi + zeta_cr + zeta_ci)/zeta_rec)
    x = -bb/(2.0*aa)*(1 - (np.lib.scimath.sqrt(1 - 4.0*aa/bb**2)).real)

    return x
