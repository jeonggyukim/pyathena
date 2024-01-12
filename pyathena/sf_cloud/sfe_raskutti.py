import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import astropy.units as au
import astropy.constants as ac

def Sigma_E(SFE, Psi):
    """Eddington surface density"""
    return SFE/(1.0 + SFE)*Psi/(2.0*np.pi*ac.c.cgs.value*ac.G.cgs.value)

def mu_M(Sigma_cl, SFE, x, sigma):
    return np.log(Sigma_cl*(1.0 - SFE)/(4.0*x**2)) + 0.5*sigma**2

def y_E(Sigma_cl, SFE, x, sigma, Psi):
    """
    Returns the random variable with a standard normal distribution
    for mass-weighted surface density distribution corresponding to Sigma_E
    y_E = (ln(Sigma_E) - mu_M)/(sqrt(2)*sigma_lnSigma)
    """
    Sigma_E_ = Sigma_E(SFE,Psi)
    muM_ = mu_M(Sigma_cl, SFE, x, sigma)

    return (np.log(Sigma_E_) - muM_)/(np.sqrt(2.0)*sigma)

def argmax_eps_of(Sigmacl, Psi=2000.0, sigma=1.0, x=1.0):
    SFE=np.linspace(0.0001, 0.9999, num=1000)
    yE = y_E(Sigmacl, SFE, x, sigma, Psi)
    eps_of = 0.5*(1.0 - SFE)*(1.0 + sp.erf(yE))
    return SFE, yE, eps_of, SFE[np.argmax(eps_of)]

def eps_min_max(Sigmacl, Psi=2000.0, sigma=1.0, x=1.0):
    """
    Compute final SF efficiency given Sigmacl, Psi, sigma_lnSigma, x
    """
    if not isinstance(Sigmacl, (np.ndarray, np.generic)):
        if isinstance(Sigmacl, float):
            Sigmacl = np.asarray([Sigmacl])
        else:
            Sigmacl = np.asarray(Sigmacl)

    eps_min = np.zeros_like(Sigmacl)
    eps_max = np.zeros_like(Sigmacl)
    for i, Sigmacl_ in enumerate(Sigmacl):
        SFE, yE, eps_of, SFE_min = argmax_eps_of(Sigmacl_, Psi=Psi, sigma=sigma, x=x)
        eps_min[i] = SFE_min
        eps_max[i] = 1.0 - max(eps_of)

    return eps_min, eps_max
