from scipy.optimize import brentq
from .cool import get_xCII, coeff_kcoll_H, coeff_alpha_rr_H, coeff_alpha_gr_H, coolLya, coolRec, coolCII, coolOI
import numpy as np


def get_xHII(nH, xe, xH2, xeM, T, xi_CR, G_PE, Z_d, zeta_pi, gr_rec):
    """Equilibrium HII fraction assuming xH2=0.0

    xe : guess for xe
    xeM : electron abundance contributed by heavy elements
    xi_CR : primary CR ionization rate
    gr_rec : include grain-assisted recombination
    """

    kcoll = coeff_kcoll_H(T)
    kcr = (1.5 + 2.3*xH2)*xi_CR
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

def get_xe(xe, xH2, nH, T, xi_CR, G_PE, G_CI, Z_d, Z_g, zeta_pi, gr_rec):

    xCII_eq = get_xCII(nH, xe, xH2, T, Z_d, Z_g, xi_CR, G_PE, G_CI, gr_rec)
    xHII_eq = get_xHII(nH, xe, xH2, xCII_eq, T, xi_CR, G_PE, Z_d, zeta_pi, gr_rec)

    return xe - xHII_eq - xCII_eq

def get_xe_arr(nH, T, xH2, xeM, xi_CR, G_PE, G_CI, Z_d, Z_g, zeta_pi, gr_rec):
    """Compute xe_eq for an array of nH and T

    Assume xH2=0
    """
    xHII = []
    xCII = []
    for nH_, T_ in zip(nH,T):
        print(T_)
        xe = brentq(get_xe, 1e-10, 1.0, args=(xH2, nH_, T_, xi_CR, G_PE, G_CI,
                                                Z_d, Z_g, zeta_pi, gr_rec))
        xHII_ = get_xHII(nH_, xe, xH2, xeM, T_, xi_CR, G_PE, Z_d, zeta_pi, gr_rec)
        xCII_ = get_xCII(nH_, xe, xH2, T_, Z_d, Z_g, xi_CR, G_PE, G_CI, 1.6e-4, gr_rec)
        xHII.append(xHII_)
        xCII.append(xCII_)

    xHII = np.array(xHII)
    xCII = np.array(xCII)
    xe = xHII + xCII
    return xe, xHII, xCII
