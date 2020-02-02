import numpy as np
import astropy.constants as ac
import astropy.units as au

# Original C version implemented in Athena-TIGRESS
# See also Gong, Ostriker, & Wolfire (2017) and https://github.com/munan/tigress_cooling

def get_xCO(nH, xH2, xCII, Z_d, Z_g, xi_CR, chi_CO, xCstd=1.6e-4):

    xCtot = xCstd*Z_g
    kcr16 = xi_CR*1e16
    term1 = np.maximum(4e3*Z_d/kcr16**2,1.0)
    ncrit2 = 2.0*np.power(term1,chi_CO**(1.0/3.0))*(50*kcr16/np.power(Z_d,1.4))
    
    xCO = np.where(nH > ncrit2, xCtot, nH/ncrit2)
    xCO = np.minimum(xCO, 2.0*xH2*xCtot)
    xCO = np.minimum(xCO, xCtot - xCII)
    
    return xCO

def heatPE(nH, T, xe, Z_d, chi_PE):

    # Weingartner & Draine (2001) Table 2
    # Rv = 3.1, bC=4.0, distribution A, ISRF
    CPE_ = np.array([5.22, 2.25, 0.04996, 0.00430, 0.147, 0.431, 0.692])
    # Charging parameter
    # JKIM: adding 50 for safety..?
    # (WD01 does not recommend using their eqaution for x <100)
    x = 1.7*chi_PE*np.sqrt(T)/(xe*nH) + 50.0
    eps = (CPE_[0] + CPE_[1]*np.power(T, CPE_[4]) )/ \
        (1. + CPE_[2]*np.power(x, CPE_[5])*(1. + CPE_[3]*np.power(x, CPE_[6])))
    
    return 1.7e-26*chi_PE*Z_d*eps


def heatCR(nH, xe, xHI, xH2, xi_CR):

    # Heating rate per ionization in atomic region
    # See Eq.30.1 in Draine (2011)
    eV_cgs = (1.0*au.eV).cgs.value
    xHetot = 0.1
    # JKIM: Isn't the last term 1.5*xHetot?
    ktot = xi_CR*((2.3*xH2 + 1.5*xHI)*(xHI + 2.0*xH2) + 1.1*xHetot)
    qHI = (6.5 + 26.4*np.sqrt(xe / (xe + 0.07)))*eV_cgs
    
    # Heating rate per ionization in molecular region
    # See Appendix B in Krumholz 2014 (Despotic)
    log_nH = np.log10(nH)
    qH2 = np.zeros_like(nH)
    qH2[np.where(log_nH < 2.0)] = 10.0*eV_cgs
    idx = np.where(np.logical_and(log_nH >= 2.0, log_nH < 4.0))
    qH2[idx] = (10 + 3*(log_nH[idx] - 2.0)*0.5)*eV_cgs
    idx = np.where(np.logical_and(log_nH >= 4.0, log_nH < 7.0))
    qH2[idx] = (13 + 4*(log_nH[idx] - 4.0)/3)*eV_cgs
    
    # (We're not likely to have this high density cells)
    idx = np.where(np.logical_and(log_nH >= 7.0, log_nH < 10.0))
    qH2[idx] = (17 + (log_nH[idx] - 7.0)/3)*eV_cgs
    idx = np.where(log_nH >= 10.0)
    qH2[idx] = 18.0*eV_cgs
    
    return ktot*(xHI*qHI + 2.0*xH2*qH2)

def heatH2pump(nH, T, xHI, xH2, xi_diss_H2):
    # Hollenbach & McKee (1978)
    eV_cgs = (1.0*au.eV).cgs.value
    de = 1.6*xHI*np.exp(-(400.0/T)**2) + 1.4*xH2*np.exp(-12000.0/(1200.0 + T))
    ncrit = 1e6/np.sqrt(T)/de
    f = nH/(nH + ncrit)

    return 18.4*xi_diss_H2*xH2*f*eV_cgs

def heatH2pump_Burton90(nH, T, xHI, xH2, xi_diss_H2):
    # Burton, Hollenbach, & Tielens (1990)
    kpump = 6.94*xi_diss_H2
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
    k10HI = 7.58e-10*np.power(T2, 0.1281+0.0087*np.log(T2))

    k10oH2 = np.zeros_like(T)
    k10pH2 = np.zeros_like(T)

    # fit in Wiesenfeld & Goldsmith (2014)
    idx = np.where(T < 500.0)
    k10oH2[idx] = (5.33 + 0.11*T2[idx])*1.0e-10
    k10pH2[idx] = (4.43 + 0.33*T2[idx])*1.0e-10
    
    # Glover+Jappsen 2007, for high temperature scales similar to HI
    idx = np.where(T >= 500.0)
    tmp = np.power(T[idx], 0.07)
    k10oH2[idx] = 3.74757785025e-10*tmp
    k10pH2[idx] = 3.88997286356e-10*tmp

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
    idx = np.where(T < 1.0e3)
    lnT_ = lnT[idx]
    lngamma10e[idx] = (((-6.56325e-4*lnT_ -1.50892e-2)*lnT_ + 3.61184e-1)*\
                       lnT_ -7.73782e-1)*lnT_ - 9.25141

    lngamma20e[idx] = (((0.705277e-2*lnT_ - 0.111338)*lnT_ + 0.697638)*
                       lnT_ - 1.30743)*lnT_ -7.69735
    lngamma21e[idx] = (((2.35272e-3*lnT_ - 4.18166e-2)*lnT_ + 0.358264)*\
                       lnT_ - 0.57443)*lnT_ -7.4387
    idx = np.where(T >= 1.0e3)
    lnT_ = lnT[idx]
    lngamma10e[idx] = (((1.0508e-1*lnT_ - 3.47620)*lnT_ + 4.2595e1)*\
                       lnT_- 2.27913e2)*lnT_ + 4.446e2
    lngamma20e[idx] = (((9.38138e-2*lnT - 3.03283)*lnT_ +3.61803e1)*\
                       lnT_ - 1.87474e2)*lnT_ +3.50609e2
    lngamma21e[idx] = (((9.78573e-2*lnT_ - 3.19268)*lnT_ +3.85049e1)*\
                       lnT_ - 2.02193e2)*lnT_ +3.86186e2
    
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


def coolLya(nH, T, xe, xHI):
    
    # HI, 2 level system
    A10HI_ = 6.265e8
    E10HI_ = 1.634e-11
    g0HI_ = 1
    g1HI_ = 3

    ne = xe*nH
    T4 = T*1.0e-4
    fac = 6.3803e-9*np.power(T4, 1.17)
    k01e = fac*np.exp(-11.84/T4)
    q01 = k01e*ne
    q10 = (g0HI_/g1HI_)*fac*ne

    return q01/(q01 + q10 + A10HI_)*A10HI_*E10HI_*xHI


def coolRec(nH, T, xe, Z_d, chi_PE):
    # Weingartner & Draine (2001) Table 3
    # Rv = 3.1, bC=4.0, distribution A, ISRF    
    DPE_ = np.array([0.4535, 2.234, -6.266, 1.442, 0.05089])
    ne = nH*xe
    x = 1.7*chi_PE*np.sqrt(T)/ne + 50.
    lnx = np.log(x)
    
    return 1.0e-28*Z_d*ne*np.power(T, DPE_[0] + DPE_[1]/lnx)*\
        np.exp((DPE_[2]+(DPE_[3]-DPE_[4]*lnx)*lnx))


def cool3Level_(q01, q10, q02, q20, q12, q21,
                A10, A20, A21, E10, E20, E21, xs):
    
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
