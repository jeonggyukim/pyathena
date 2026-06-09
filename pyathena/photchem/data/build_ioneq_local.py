"""Build CIE ionization fractions x_q(T) per element using pyathena's
own collisional-ionization, recombination, and charge-transfer rate
code. For ions where pyathena's tabulation is missing (notably the
neutral and low-q Fe states, q = 0 .. 6), fall back to CHIANTI v11
recombination / ionization rates via ChiantiPy. The CHIANTI fallback
fills gaps but does NOT override pyathena's data where it exists, so
the comparison plot continues to reveal pyathena vs CHIANTI rate
deltas.

CLI:
    XUVTOP=$HOME/Dropbox/Projects/CHIANTI_db \\
        python -m pyathena.photchem.data.build_ioneq_local
"""

import os
import numpy as np

from pyathena.microphysics.ci_rate import (
    CollIonRate, CollIonRateCHIANTI)
from pyathena.microphysics.rec_rate import RecRate, RecRateCHIANTI
from pyathena.microphysics.ct_rate import ChargeTransferRate
from .build_ioneq_tables import ELEMENTS, read_ioneq, write_ascii


# CHIANTI element symbol lowercase used in ion names (`fe_2`, `o_3`,
# etc.). Matches CHIANTI's data directory naming.
_CH_ELEM = {
    'H':  'h',  'He': 'he', 'C':  'c',  'N':  'n',  'O':  'o',
    'Ne': 'ne', 'Mg': 'mg', 'Si': 'si', 'S':  's',  'Ar': 'ar',
    'Ca': 'ca', 'Fe': 'fe',
}


def _chianti_rates(element, Z, T_grid):
    """Return (k_ci_full, alpha_rec_full) arrays of shape (Z+1, NT)
    computed entirely from CHIANTI v11 via ChiantiPy. Each
    k_ci_full[q] is the rate for X^q -> X^(q+1) (and is zero for
    q=Z); each alpha_rec_full[q] is the rate for X^q -> X^(q-1)
    (zero for q=0).
    """
    import ChiantiPy.core as ch
    NT = len(T_grid)
    k_ci = np.zeros((Z + 1, NT))
    arec = np.zeros((Z + 1, NT))
    sym = _CH_ELEM[element]
    for q in range(Z + 1):
        ion_name = f'{sym}_{q + 1}'   # CHIANTI 1-based
        try:
            ion = ch.ion(ion_name, temperature=T_grid)
        except (FileNotFoundError, KeyError, IOError, OSError):
            continue
        # k_ci: only meaningful for q < Z (q=Z is fully stripped).
        if q < Z:
            try:
                ion.ionizRate()
                k_ci[q] = np.asarray(ion.IonizRate['rate'])
            except Exception:
                pass
        # alpha_rec: only meaningful for q > 0.
        if q > 0:
            try:
                ion.recombRate()
                arec[q] = np.asarray(ion.RecombRate['rate'])
            except Exception:
                pass
    # Sanitize: replace NaN/negative with 0.
    k_ci = np.where(np.isfinite(k_ci) & (k_ci > 0), k_ci, 0.0)
    arec = np.where(np.isfinite(arec) & (arec > 0), arec, 0.0)
    return k_ci, arec


def _safe(call):
    """Return 0 instead of raising / returning NaN/negative."""
    try:
        v = float(call())
        if not np.isfinite(v) or v < 0:
            return 0.0
        return v
    except (IndexError, KeyError, ValueError):
        return 0.0


def cie_xq_for_element(Z, T_grid, x_HI_arr, x_HII_arr, use_ct=True,
                       element=None):
    """Compute CIE x_q(T) for one element using CHIANTI v11-backed
    CI + recombination rates (via RecRateCHIANTI /
    CollIonRateCHIANTI) and optional charge transfer with H.

    Parameters
    ----------
    Z         : int
        Atomic number.
    T_grid    : (NT,)
        Temperature grid [K].
    x_HI_arr  : (NT,)
        Neutral-H CIE fraction at each T (from CHIANTI's
        chianti.ioneq H column).
    x_HII_arr : (NT,)
    use_ct    : bool
        Include CT contributions if True.
    element   : str
        Element symbol (e.g., 'Fe'). Required by the CHIANTI
        rate classes.

    Returns
    -------
    x_q : (Z+1, NT) ndarray
    """
    if element is None:
        raise ValueError("`element` is required for CHIANTI-backed rates")
    # Use CHIANTI v11-backed CI + recombination rates so the
    # comparison plot doesn't have data-coverage gaps (notably Fe
    # I-VI rec rates that pyathena's Badnell subset lacks). Same
    # underlying Verner-Ferland + Badnell fits as pyathena's
    # RecRate / CollIonRate; CHIANTI just has more complete
    # tabulations.
    ci = CollIonRateCHIANTI(T_grid=T_grid, elements=[(element, Z)])
    rc = RecRateCHIANTI(T_grid=T_grid, elements=[(element, Z)])
    ct = ChargeTransferRate()
    NT = len(T_grid)
    x_q = np.zeros((Z + 1, NT))
    # Pre-compute CHIANTI fallback arrays once (each gives all q,
    # all T). Used cell-by-cell below when pyathena returns 0.
    if element is not None:
        ci_ch, arec_ch = _chianti_rates(element, Z, T_grid)
    else:
        ci_ch, arec_ch = None, None
    for k, T in enumerate(T_grid):
        x_HI = x_HI_arr[k]
        x_HII = x_HII_arr[k]
        log_ratio = np.zeros(Z + 1)
        for q in range(Z):
            # source of q+1 from q (per X^q per n_H)
            # k_ci: X^q + e -> X^(q+1) + 2e; rate ~ n_e ~ x_HII n_H
            kci = _safe(lambda: ci.get_ci_rate(Z, Z - q, T))
            src = kci * x_HII
            if use_ct and Z > 1:
                # k_CT_ion: X^q + H+ -> X^(q+1) + H; rate ~ n_H+
                # ~ x_HII n_H (H+ is the collider).
                kctI = _safe(lambda: ct.get_ct_ion_rate(Z, Z - q, T))
                src += kctI * x_HII
            # sink for q+1 to q (per X^(q+1) per n_H)
            # alpha_rec: X^(q+1) + e -> X^q + photon; rate ~ x_HII n_H
            arec = _safe(lambda: rc.get_rec_rate(Z, Z - q - 1, T))
            snk = arec * x_HII
            if use_ct and Z > 1:
                # k_CT_rec: X^(q+1) + H -> X^q + H+; rate ~ n_HI
                # ~ x_HI n_H (neutral H is the collider).
                kctR = _safe(lambda: ct.get_ct_rec_rate(Z, Z - q - 1, T))
                snk += kctR * x_HI
            if snk <= 0.0 and src <= 0.0:
                log_ratio[q + 1] = -np.inf
            elif snk <= 0.0:
                log_ratio[q + 1] = log_ratio[q] + 40.0
            elif src <= 0.0:
                log_ratio[q + 1] = log_ratio[q] - 40.0
            else:
                log_ratio[q + 1] = log_ratio[q] + np.log(src / snk)
        mx = np.max(log_ratio)
        weights = np.exp(log_ratio - mx)
        weights = weights / weights.sum()
        x_q[:, k] = weights
    return x_q


def main():
    """Build two CIE x_q(T) tables per element on the CHIANTI T grid:
       - `ioneq_local_<element>.txt`     : collisional only (no CT)
       - `ioneq_local_ct_<element>.txt`  : collisional + CT (CT
                                           contribution weighted by
                                           CHIANTI's CIE H state)
    The pair lets the comparison plot isolate the effect of CT.
    """
    out_dir = os.path.dirname(os.path.abspath(__file__))
    H = read_ioneq(os.path.join(out_dir, 'ioneq_H.txt'))
    log_T = H['log_T']
    T_grid = 10.0 ** log_T
    x_HI = H['x_q'][0]
    x_HII = H['x_q'][1]
    print(f"Building local CIE on T grid {len(T_grid)} pts, "
          f"{T_grid[0]:.2g} -> {T_grid[-1]:.2g} K")
    for element, Z in ELEMENTS.items():
        for use_ct, suffix in [(False, ''), (True, 'ct_')]:
            x_q = cie_xq_for_element(
                Z, T_grid, x_HI, x_HII, use_ct=use_ct,
                element=element)
            out_path = os.path.join(
                out_dir, f'ioneq_local_{suffix}{element}.txt')
            write_ascii(out_path, element, Z, log_T, x_q)
            col_sum = x_q.sum(axis=0)
            tag = 'with CT' if use_ct else 'collisional only'
            print(f"  {element:2s} (Z={Z:2d}, {tag}): wrote "
                  f"{out_path}  (col sums "
                  f"{col_sum.min():.4f} - {col_sum.max():.4f})")


if __name__ == '__main__':
    main()
