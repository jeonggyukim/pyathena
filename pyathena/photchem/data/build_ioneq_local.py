"""Build CIE ionization fractions x_q(T) per element using pyathena's
own collisional-ionization, recombination, and charge-transfer rate
code.

In CIE the sequential balance at each (T, charge q) is

    n(X^(q+1)) / n(X^q) = (k_ci * x_HII + k_CT_ion * x_HI) /
                          (alpha_rec * x_HII + k_CT_rec * x_HI)

where the H ionization fractions x_HI(T) and x_HII(T) themselves
are taken from the CHIANTI v11 CIE table (so the metals are
"trace-element" weighted by the standard CIE H state). The n_H
that multiplies both source and sink cancels in the ratio.

CHIANTI / GS07 / Mazzotta+98 CIE tables include CT in this same
spirit. Pure-collisional CIE (k_CT = 0) is a separate physical
regime that disagrees with CHIANTI at temperatures where x_HI is
non-negligible.

Writes per-element ASCII tables under this directory:
    ioneq_local_<element>.txt
Format matches `ioneq_<element>.txt` (CHIANTI v11).

CLI:
    python -m pyathena.photchem.data.build_ioneq_local
"""

import os
import numpy as np

from pyathena.microphysics.ci_rate import CollIonRate
from pyathena.microphysics.rec_rate import RecRate
from pyathena.microphysics.ct_rate import ChargeTransferRate
from .build_ioneq_tables import ELEMENTS, read_ioneq, write_ascii


def _safe(call):
    """Return 0 instead of raising / returning NaN/negative."""
    try:
        v = float(call())
        if not np.isfinite(v) or v < 0:
            return 0.0
        return v
    except (IndexError, KeyError, ValueError):
        return 0.0


def cie_xq_for_element(Z, T_grid, x_HI_arr, x_HII_arr, use_ct=True):
    """Compute CIE x_q(T) for one element using pyathena's k_ci,
    alpha_rec, and optionally charge-transfer with H.

    Parameters
    ----------
    Z         : int
    T_grid    : (NT,)
    x_HI_arr  : (NT,)  CIE neutral-H fraction at each T (e.g., from
                       CHIANTI's chianti.ioneq H column)
    x_HII_arr : (NT,)
    use_ct    : bool   include CT contributions if True

    Returns
    -------
    x_q : (Z+1, NT) ndarray
    """
    ci = CollIonRate()
    rc = RecRate(caseB=False)
    ct = ChargeTransferRate()
    NT = len(T_grid)
    x_q = np.zeros((Z + 1, NT))
    for k, T in enumerate(T_grid):
        x_HI = x_HI_arr[k]
        x_HII = x_HII_arr[k]
        log_ratio = np.zeros(Z + 1)
        for q in range(Z):
            # source of q+1 from q (per X^q per n_H)
            kci = _safe(lambda: ci.get_ci_rate(Z, Z - q, T))
            src = kci * x_HII
            if use_ct and Z > 1:
                kctI = _safe(lambda: ct.get_ct_ion_rate(Z, Z - q, T))
                src += kctI * x_HI
            # sink for q+1 to q (per X^(q+1) per n_H)
            arec = _safe(lambda: rc.get_rec_rate(Z, Z - q - 1, T))
            snk = arec * x_HII
            if use_ct and Z > 1:
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
                Z, T_grid, x_HI, x_HII, use_ct=use_ct)
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
