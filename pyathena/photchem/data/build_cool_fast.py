"""Experimental fast cooling-table builder.

!!! PARTIAL VALIDATION -- use with care !!!
The batched matrix-solve path produces BB cooling values that
disagree with the canonical `build_cool_tables.py` (ChiantiPy-based)
output by ~2 dex (factor ~100) for BB-dominated mid-q ions. Pinning
the discrepancy is a TODO.

However, for ions where the BB channel is negligible compared to
FF + FB (typically: fully-stripped q=Z, near-fully-stripped H-like
and He-like q=Z-1, q=Z-2), this builder gives results matching the
canonical builder to machine precision. Such ions can use the fast
builder's output safely. The list of "safe" ions (to be built up by
direct comparison) belongs in a constant at the top of this file
when the validation is exhaustive.

Likely culprits to investigate:
   - 4*pi factor in ChiantiPy's intensity formula (~12x; necessary
     but not sufficient).
   - Missing proton-impact (.psplups) coupling channel; only matters
     for some Fe/heavy-Z ions but could be ~5-10x.
   - hc/lambda factor in 'photon' default flux mode of ChiantiPy.
   - More transitions in ChiantiPy's internal populate than my
     filtered .scups subset.
   - Bug in matrix indexing or dE sign.

Right debug path: compare ChiantiPy's `ion.Population['population']`
against our batched f-array for a single test ion (e.g., OIII at
T=1e5 K, n_e=1). If populations match -> bug is in emissivity sum
or 4*pi. If populations differ -> bug is in matrix construction.

Speedup target: ~30 sec for full followed-element set vs ~15 min
canonical (15-30x). Currently achieves ~6.5x at ~140 sec total but
with broken BB values, so unusable.

CURRENT STATUS: NOT used in `build_all.py`. Stays as scratch /
sandbox until BB mismatch is fixed.

Implementation strategy:
   1. `ion = ch.ion(name, temperature=T_grid)` -- ChiantiPy reads
      atomic data files (.elvlc, .wgfa, .scups, .psplups) once.
   2. Extract:
        - level energies E_i      from ion.Elvlc
        - radiative rates A_ij    from ion.Wgfa
        - electron Upsilons Y_ij(T) from ion.Scups via batch spline
          eval on the full T_grid
   3. Build rate tensor R(T) shape (NT, N, N):
        R[t, i, j] = -sum_{k != i} k_{i->k}(t)            if i == j
        R[t, i, j] =  k_{j->i}(t)                          if i != j
      with closure: replace last row with [1, 1, ..., 1] = 1.
   4. Batch solve: f = np.linalg.solve(R, b) -> (NT, N).
   5. Sum emissivity over transitions:
      Lambda_q[t] = sum_{ij} f[t, upper_ij] * A_ij * dE_ij
   6. Continuum (FF, FB, 2gamma) still via ChiantiPy (small cost
      compared to BB).

CLI:
    XUVTOP=$HOME/Dropbox/Projects/CHIANTI_db \\
        python -m pyathena.photchem.data.build_cool_fast
"""

import os
import warnings
import numpy as np

from .build_ioneq_tables import ELEMENTS
from .build_cool_tables import (
    _ELEM_SYM, _NE_REF, _safe_loss, write_ascii)


def _compute_BB_batched(ion, T_grid):
    """Vectorized bound-bound cooling efficiency [erg cm^3 / s,
    per ion per electron] at all T_grid points in one matrix solve.

    Returns array of shape (NT,).
    """
    NT = len(T_grid)
    # Extract atomic data from already-constructed ion. Some ions
    # lack Wgfa / Scups data and ChiantiPy raises AttributeError.
    if not (hasattr(ion, 'Elvlc') and hasattr(ion, 'Wgfa')):
        return None
    elvlc = ion.Elvlc
    wgfa = ion.Wgfa
    # Energy levels in cm^-1; convert to erg.
    E_cm = np.asarray(elvlc['ecm'])
    E_cm_th = np.asarray(elvlc['ecmth'])
    E_eff = np.where(E_cm > 0.0, E_cm, E_cm_th)
    E_erg = E_eff * 1.986e-16   # cm^-1 -> erg
    N = len(E_erg)
    # Degeneracy g = 2J + 1.
    mult = np.asarray(elvlc['mult'])
    # Radiative transition data.
    # Wgfa lists (lvl1=lower, lvl2=upper, wvl, gf, avalue).
    lvl1 = np.asarray(wgfa['lvl1'], dtype=int) - 1   # 0-based
    lvl2 = np.asarray(wgfa['lvl2'], dtype=int) - 1
    avalue = np.asarray(wgfa['avalue'])
    # Filter out non-decaying / unphysical rows.
    keep = (avalue > 0) & (lvl2 > lvl1)
    lvl1 = lvl1[keep]
    lvl2 = lvl2[keep]
    avalue = avalue[keep]
    dE = E_erg[lvl2] - E_erg[lvl1]
    # Sanity: drop transitions with non-positive energy.
    keep = dE > 0
    lvl1, lvl2, avalue, dE = lvl1[keep], lvl2[keep], avalue[keep], dE[keep]
    # Collisional Upsilon: skip if no scups; fall back to Burgess-Tully
    # constant approximation (Y=1).
    # ChiantiPy's batch upsilon evaluation API:
    try:
        ion.upsilonDescale()   # evaluates Upsilons at ion's T_grid
        ups_dict = ion.Upsilon
        # ups_dict['upsilon'] has shape (NWgfa_ish, NT). Index by
        # the .scups 'lvl1', 'lvl2' arrays.
        scups = ion.Scups
        s_lvl1 = np.asarray(scups['lvl1'], dtype=int) - 1
        s_lvl2 = np.asarray(scups['lvl2'], dtype=int) - 1
        ups_arr = np.asarray(ups_dict['upsilon'])
    except Exception:
        return None
    # Boltzmann factor for upward rates.
    kB = 1.380649e-16
    # beta = 8.629e-8: collision rate prefactor.
    beta = 8.629e-8
    # Excitation/de-excitation collisional rates:
    # q_down(j -> i) = beta * Y_ij / (g_j * sqrt(T))
    # q_up(i -> j) = q_down * (g_j/g_i) * exp(-(E_j - E_i)/kT)
    sqrtT = np.sqrt(T_grid)
    s_dE = E_erg[s_lvl2] - E_erg[s_lvl1]
    s_dE = np.maximum(s_dE, 1e-40)
    g_up = mult[s_lvl2]
    g_lo = mult[s_lvl1]
    # Build rate matrix R[t, i, j] (NT, N, N): rate INTO i FROM j.
    R = np.zeros((NT, N, N))
    # Loop over scups transitions: each contributes 4 entries.
    for k in range(len(s_lvl1)):
        i_lo = s_lvl1[k]
        i_up = s_lvl2[k]
        # Down: from upper to lower.
        q_down = beta * ups_arr[k] / (g_up[k] * sqrtT)
        boltz = np.exp(-s_dE[k] / (kB * T_grid))
        q_up_rate = q_down * (g_up[k] / g_lo[k]) * boltz
        # Multiply by n_e (we use _NE_REF for now).
        q_down *= _NE_REF
        q_up_rate *= _NE_REF
        # OUT of upper, INTO lower:
        R[:, i_lo, i_up] += q_down
        R[:, i_up, i_up] -= q_down
        R[:, i_up, i_lo] += q_up_rate
        R[:, i_lo, i_lo] -= q_up_rate
    # Add radiative decay rates to matrix.
    for k in range(len(lvl1)):
        i_lo = lvl1[k]
        i_up = lvl2[k]
        R[:, i_lo, i_up] += avalue[k]
        R[:, i_up, i_up] -= avalue[k]
    # Closure: replace last row with [1, 1, ..., 1] = 1.
    R[:, -1, :] = 1.0
    b = np.zeros((NT, N, 1))
    b[:, -1, 0] = 1.0
    # Batch solve. np.linalg.solve sig for batched A: (..., M, M)
    # and B: (..., M, K) -> (..., M, K). We have R: (NT, N, N) and
    # b: (NT, N, 1), result (NT, N, 1), squeeze to (NT, N).
    try:
        f = np.linalg.solve(R, b).squeeze(-1)   # (NT, N)
    except np.linalg.LinAlgError:
        return None
    # Sum emissivity over transitions.
    if len(lvl1) == 0:
        return np.zeros(NT)
    f_upper = f[:, lvl2]   # shape (NT, N_trans)
    Lambda_q = np.sum(f_upper * avalue * dE, axis=1)  # (NT,)
    # ChiantiPy multiplies emissivity by 4*pi in boundBoundLoss
    # (intensity = 4*pi * em). We match that convention so the
    # output normalization is identical.
    Lambda_q = 4.0 * np.pi * Lambda_q / _NE_REF
    return Lambda_q


def cooling_for_element_fast(element, Z, T_grid):
    """Drop-in replacement for `cooling_for_element` using batched
    BB solve. FF/FB/2gamma still via ChiantiPy (small cost)."""
    import ChiantiPy.core as ch
    NT = len(T_grid)
    Lambda_q = np.zeros((Z + 1, NT))
    sym = _ELEM_SYM[element]
    for q in range(Z + 1):
        ion_name = f'{sym}_{q + 1}'
        total = np.zeros(NT)
        # BB + 2gamma on ion class.
        ion = None
        if q < Z:
            try:
                ion = ch.ion(ion_name, temperature=T_grid,
                             eDensity=_NE_REF)
                ion.Abundance = 1.0
                ion.IoneqOne = np.ones(NT)
            except Exception:
                ion = None
            if ion is not None:
                BB = _compute_BB_batched(ion, T_grid)
                if BB is not None:
                    BB = np.where(np.isfinite(BB) & (BB > 0), BB, 0.0)
                    total += BB
                N_electrons = Z - q
                if N_electrons in (1, 2):
                    total += _safe_loss(ion, 'twoPhotonLoss',
                                        'TwoPhotonLoss', NT)
        # FF on continuum class.
        try:
            cont = ch.continuum(ion_name, temperature=T_grid)
            cont.Abundance = 1.0
            cont.IoneqOne = np.ones(NT)
        except Exception:
            cont = None
        if cont is not None and q >= 1:
            total += _safe_loss(cont, 'freeFreeLoss',
                                'FreeFreeLoss', NT)
        # FB on continuum of X^(q+1), attributed to row q
        # (GF12 / Cloudy convention).
        if q < Z:
            next_ion_name = f'{sym}_{q + 2}'
            try:
                cont_next = ch.continuum(
                    next_ion_name, temperature=T_grid)
                cont_next.Abundance = 1.0
                cont_next.IoneqOne = np.ones(NT)
            except Exception:
                cont_next = None
            if cont_next is not None:
                total += _safe_loss(cont_next, 'freeBoundLoss',
                                    'FreeBoundLoss', NT)
        Lambda_q[q] = total
    return Lambda_q


def main():
    if not os.environ.get('XUVTOP'):
        raise RuntimeError(
            "XUVTOP environment variable not set.")
    warnings.filterwarnings('ignore')
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..', '..', '..', 'data', 'microphysics', 'chianti_v11'))
    os.makedirs(out_dir, exist_ok=True)
    log_T = np.linspace(3.0, 9.0, 121)
    T_grid = 10.0 ** log_T
    import time
    print(f"FAST builder on T grid {len(T_grid)} pts "
          f"({T_grid[0]:.2g} -> {T_grid[-1]:.2g} K)")
    total_start = time.time()
    for element, Z in ELEMENTS.items():
        t0 = time.time()
        Lambda_q = cooling_for_element_fast(element, Z, T_grid)
        dt = time.time() - t0
        out_path = os.path.join(out_dir, f"cool_fast_{element}.txt")
        write_ascii(out_path, element, Z, log_T, Lambda_q)
        nonzero_q = int(np.sum(Lambda_q.max(axis=1) > 0))
        max_q = float(Lambda_q.max())
        print(f"  {element:2s} (Z={Z:2d}): {dt:5.1f}s, "
              f"{nonzero_q}/{Z+1} ions, max={max_q:.2e}")
    total_dt = time.time() - total_start
    print(f"Total: {total_dt:.1f}s "
          f"(vs ~900s for ChiantiPy-based build).")


if __name__ == '__main__':
    main()
