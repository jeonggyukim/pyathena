"""Equilibrium hydrogen-ionisation and carbon-ionisation helpers.

Ports `get_xHII`, `get_xCII`, `coeff_kcoll_H`, `coeff_alpha_rr_H`,
`coeff_alpha_gr_H` from `pyathena.microphysics.cool` /
`pyathena.microphysics.get_xe_eq` into the chemistry-rewrite namespace.
Used to build physically-consistent test states: instead of hand-
tuning (xHI, xHII, xe) triplets, callers pass (T, nH, xH2, xi_CR,
G_PE, ...) and the helper returns the equilibrium (xHII, xCII, xe,
xHI) by solving the H + C ionisation balance jointly.

The H2 abundance is NOT solved here -- callers pass `xH2` as an
input. Equilibrium xH2 depends on the full chemistry network
(formation on grains + photodissociation + CR dissociation) which
this helper does not model. Pass `xH2 = 0` for atomic regimes and
`xH2 ~ 0.45` for molecular cores.

Conservation:

    xHI = 1 - 2 * xH2 - xHII

is enforced after solving for xHII. Caller does not have to set xHI.

Solver: fixed-point iteration on xe = xHII(xe) + xCII(xe). Damped
update with relaxation factor 0.5 for stability when starting far
from the equilibrium. Converges in ~ 20-30 iterations for any
reasonable (T, nH, xi_CR) combination; per-cell vectorised over
numpy arrays.
"""
from __future__ import annotations

import numpy as np

_SMALL = 1.0e-50


def coeff_kcoll_H(T):
    """Collisional ionisation of HI by electrons (Gnat + Sternberg
    2007 fit; T_e = T * k_B in eV)."""
    lnTe = np.log(T * 8.6173e-5)
    k_coll = np.where(
        T > 3.0e3,
        np.exp(
            -3.271396786e1 + (1.35365560e1 + (-5.73932875 + (1.56315498
            + (-2.877056e-1 + (3.48255977e-2 + (-2.63197617e-3
            + (1.11954395e-4 + (-2.03914985e-6)
            * lnTe) * lnTe) * lnTe) * lnTe) * lnTe) * lnTe) * lnTe) * lnTe),
        0.0,
    )
    return k_coll


def coeff_alpha_rr_H(T):
    """HII radiative case-B recombination (Gong+17 fit to Ferland+92)."""
    Tinv = 1.0 / T
    bb = 315614.0 * Tinv
    cc = 115188.0 * Tinv
    dd = 1.0 + np.power(cc, 0.407)
    return 2.753e-14 * np.power(bb, 1.5) * np.power(dd, -2.242)


def coeff_alpha_gr_H(T, G_PE, ne, Z_d):
    """Grain-assisted HII recombination (WD01 Table 3)."""
    lnT = np.log(T)
    cHp = np.array(
        [12.25, 8.074e-6, 1.378, 5.087e2, 1.586e-2, 0.4723, 1.102e-5])
    psi_gr = 1.7 * G_PE * np.sqrt(T) / (ne + _SMALL) + _SMALL
    alpha_gr = 1.0e-14 * cHp[0] / (
        1.0 + cHp[1] * np.power(psi_gr, cHp[2]) * (
            1.0 + cHp[3] * np.power(T, cHp[4])
            * np.power(psi_gr, -cHp[5] - cHp[6] * lnT))) * Z_d
    return alpha_gr


def get_xCII(nH, xe, xH2, T, Z_d, Z_g, xi_CR, G_PE, G_CI,
             xCstd=1.6e-4, gr_rec=True, CRPhotC=True):
    """Equilibrium CII fraction. Port of
    `pyathena.microphysics.cool.get_xCII` (iCII_rec_rate = 0 branch).
    """
    xCtot = xCstd * Z_g
    k_C_cr = 3.85 * xi_CR
    k_C_photo = 3.5e-10 * G_CI
    if CRPhotC:
        k_C_photo = k_C_photo + 520.0 * 2.0 * xH2 * xi_CR

    lnT = np.log(T)
    k_Cplus_e = np.where(
        T < 10.0,
        9.982641225129824e-11,
        np.exp(-0.7529152 * lnT - 21.293937),
    )

    if gr_rec:
        psi_gr = 1.7 * G_PE * np.sqrt(T) / (nH * xe + _SMALL) + _SMALL
        cCp = np.array(
            [45.58, 6.089e-3, 1.128, 4.331e2, 4.845e-2, 0.8120, 1.333e-4])
        k_Cplus_gr = 1.0e-14 * cCp[0] / (
            1.0 + cCp[1] * np.power(psi_gr, cCp[2]) * (
                1.0 + cCp[3] * np.power(T, cCp[4])
                * np.power(psi_gr, -cCp[5] - cCp[6] * lnT))) * Z_d
    else:
        k_Cplus_gr = 0.0

    k_Cplus_H2 = 3.3e-13 * np.power(T, -1.3) * np.exp(-23.0 / T)

    c = (k_C_cr + k_C_photo) / nH
    al = k_Cplus_e * xe + k_Cplus_gr + k_Cplus_H2 * xH2 + c
    ar = xCtot * c
    return ar / al


def get_xHII(nH, xe, xH2, xeM, T, xi_CR, G_PE, Z_d, zeta_pi=0.0,
             gr_rec=True):
    """Equilibrium HII fraction at fixed xe. Port of
    `pyathena.microphysics.get_xe_eq.get_xHII`.

    xeM : electron abundance contributed by heavy elements (xCII).
    """
    kcoll = coeff_kcoll_H(T)
    kcr = (1.5 + 2.3 * xH2) * xi_CR
    alpha_rr = coeff_alpha_rr_H(T)
    if gr_rec:
        alpha_gr = coeff_alpha_gr_H(T, G_PE, nH * xe, Z_d)
    else:
        alpha_gr = 0.0

    a = 1.0 + kcoll / alpha_rr
    b = ((kcr + zeta_pi - kcoll * nH * (1.0 - xeM)) / (alpha_rr * nH)
         + xeM + alpha_gr / alpha_rr)
    c = -(kcr + zeta_pi + xeM * nH * kcoll) / (alpha_rr * nH)

    return (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


def eq_xHII_xe(T, nH, xH2=0.0, xi_CR=1.0e-16, G_PE=1.0, G_CI=1.0,
               Z_d=1.0, Z_g=1.0, zeta_pi=0.0, gr_rec=True,
               xCstd=1.6e-4, max_iter=50, tol=1.0e-8,
               xe_init=1.6e-4):
    """Solve jointly for (xHII, xCII, xe) given (T, nH, xH2) and the
    ionising-rate inputs (xi_CR, G_PE, G_CI, zeta_pi).

    Conservation: xHI = 1 - 2*xH2 - xHII is enforced on return.

    Returns
    -------
    xHII, xCII, xe, xHI : ndarray
        All shaped like broadcast(T, nH, xH2).
    """
    T = np.asarray(T, dtype=np.float64)
    nH = np.asarray(nH, dtype=np.float64)
    shape = np.broadcast_shapes(T.shape, nH.shape)
    if np.isscalar(xH2):
        xH2 = np.full(shape, float(xH2))
    else:
        xH2 = np.broadcast_to(np.asarray(xH2, dtype=np.float64),
                              shape).copy()
    T = np.broadcast_to(T, shape).copy()
    nH = np.broadcast_to(nH, shape).copy()

    xe = np.full(shape, float(xe_init))
    for _ in range(max_iter):
        xCII = get_xCII(nH, xe, xH2, T, Z_d, Z_g, xi_CR, G_PE, G_CI,
                        xCstd=xCstd, gr_rec=gr_rec)
        xHII = get_xHII(nH, xe, xH2, xCII, T, xi_CR, G_PE, Z_d,
                        zeta_pi=zeta_pi, gr_rec=gr_rec)
        xe_new = xHII + xCII
        delta = np.max(np.abs(xe_new - xe) / np.maximum(xe, 1.0e-30))
        # Damped update for stability.
        xe = 0.5 * (xe + xe_new)
        if delta < tol:
            break

    # Final pass at converged xe.
    xCII = get_xCII(nH, xe, xH2, T, Z_d, Z_g, xi_CR, G_PE, G_CI,
                    xCstd=xCstd, gr_rec=gr_rec)
    xHII = get_xHII(nH, xe, xH2, xCII, T, xi_CR, G_PE, Z_d,
                    zeta_pi=zeta_pi, gr_rec=gr_rec)
    xe = xHII + xCII
    xHI = 1.0 - 2.0 * xH2 - xHII
    return xHII, xCII, xe, xHI
