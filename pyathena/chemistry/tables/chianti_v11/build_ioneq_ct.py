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
        python -m pyathena.microphysics.chianti_v11.build_ioneq_ct
"""

import os
import numpy as np

from pyathena.microphysics.ci_rate import (
    CollIonRate, CollIonRateCHIANTI)
from pyathena.microphysics.rec_rate import RecRate, RecRateCHIANTI
from pyathena.microphysics.ct_rate import ChargeTransferRate
from .build_ioneq import ELEMENTS, read_ioneq, write_ascii




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
       - `ioneq_<element>.txt`     : collisional only (no CT)
       - `ioneq_ct_<element>.txt`  : collisional + CT (CT
                                           contribution weighted by
                                           CHIANTI's CIE H state)
    The pair lets the comparison plot isolate the effect of CT.
    """
    # Guard against silent failure: the CHIANTI rate classes need
    # XUVTOP to load .rrparams / .drparams / .diparams from the
    # CHIANTI database. Without it, every ChiantiPy lookup returns
    # an error that the rate classes catch, producing empty rate
    # tables. The CIE sequential solver then puts every element at
    # q=0 with no warning.
    if not os.environ.get('XUVTOP'):
        raise RuntimeError(
            "XUVTOP environment variable not set. ChiantiPy cannot "
            "load CHIANTI data without it; build_ioneq_ct would "
            "silently produce all-neutral x_q tables. Set XUVTOP "
            "to your CHIANTI v11 data directory before running, "
            "e.g.:\n"
            "    export XUVTOP=$HOME/Dropbox/Projects/CHIANTI_db")
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..', '..', '..', 'data', 'microphysics', 'chianti_v11'))
    os.makedirs(out_dir, exist_ok=True)
    H = read_ioneq(os.path.join(out_dir, 'ioneq_H.txt'))
    log_T = H['log_T']
    T_grid = 10.0 ** log_T
    x_HI = H['x_q'][0]
    x_HII = H['x_q'][1]
    print(f"Building our CIE on T grid {len(T_grid)} pts, "
          f"{T_grid[0]:.2g} -> {T_grid[-1]:.2g} K")
    # Below this x_HII threshold, the gas has effectively no H+
    # available, so both source and sink rates vanish and the
    # equilibrium time -> infinity. The mathematical CIE limit is
    # undefined (depends on which vanishing quantity wins). Force
    # neutral state for x_HII below this threshold (physical: real
    # cold gas at log T < 4 stays at whatever initial state since
    # nothing actually happens fast enough to equilibrate).
    _XHII_FLOOR = 1.0e-6
    # Only build the +CT variant: the no-CT (use_ct=False) result
    # is identical to `ioneq_<X>.txt` by construction (same CHIANTI
    # rates, same sequential balance, no extra physics), so it's
    # redundant. Keep the CT variant which adds H charge transfer.
    for element, Z in ELEMENTS.items():
        for use_ct, suffix in [(True, 'ct_')]:
            x_q = cie_xq_for_element(
                Z, T_grid, x_HI, x_HII, use_ct=use_ct,
                element=element)
            # Force all-neutral where H is essentially neutral.
            mask_cold = x_HII < _XHII_FLOOR
            if mask_cold.any():
                x_q[:, mask_cold] = 0.0
                x_q[0, mask_cold] = 1.0
            out_path = os.path.join(
                out_dir, f'ioneq_{suffix}{element}.txt')
            write_ascii(out_path, element, Z, log_T, x_q)
            col_sum = x_q.sum(axis=0)
            tag = 'with CT' if use_ct else 'collisional only'
            print(f"  {element:2s} (Z={Z:2d}, {tag}): wrote "
                  f"{out_path}  (col sums "
                  f"{col_sum.min():.4f} - {col_sum.max():.4f})")


if __name__ == '__main__':
    main()
