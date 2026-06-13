"""Build CIE high-charge-pool tables for `NCRNetwork3PlusIons16`.

The Phase 6 multi-ion network tracks the LOW charge states of each
metal element explicitly (CI / CII for C; OI / OII / OIII for O; ...)
and absorbs the HIGH charge states into a single CIE-equilibrium
"pool" ghost row. This builder writes the three tables the pool
ghost needs, on the CHIANTI v11 log-T grid that
`build_ioneq` and `build_cool` already use:

    x_high_frac(T)  = sum_{q >= q_max_tracked} ioneq_q(T)
    q_high_mean(T)  = sum_{q >= q_max_tracked} q * ioneq_q(T)
                       / x_high_frac(T)
    Lambda_high(T) = sum_{q >= q_max_tracked} ioneq_q(T) * L_q(T)

The `q_max_tracked` cutoff matches the `element_groups` declaration
on `NCRNetwork3PlusIons16`: tracked = (CI, CII) -> q_max = 2; the
pool covers q >= 2 (i.e. C III - C VII). Per-ion radiative loss
curves L_q(T) come from `cool_<element>.txt` (CHIANTI bound-bound +
two-photon + free-free + free-bound summed); per-charge ionisation
fractions come from `ioneq_<element>.txt`.

Inputs (read from `data/microphysics/chianti_v11/`):

- ioneq_<element>.txt -- (NT, Z+1) CIE ionisation fractions
- cool_<element>.txt  -- (NT, Z+1) per-ion radiative loss [erg cm^3/s]

Outputs (written to `data/chemistry/cie_high_pool_<element>.txt`):

- log_T_K          [log10 K]
- x_high_frac       [-]
- q_high_mean       [-]   (mean charge averaged over the pool)
- Lambda_high       [erg cm^3 / s per H nucleus, before x_std * Z_g]

The runtime cooling rate for the pool of element X follows

    Lambda_X^pool [erg/s/cm^3] = Lambda_high(T) * x_std_X * Z_g
                                  * n_e * n_H

with `x_std_X` the standard solar abundance of element X relative to
hydrogen (`x_std_C = 1.6e-4`, `x_std_O = 3.2e-4`, ...). The pool ghost
row `x_high_X` is set to `x_high_frac(T) * x_std_X * Z_g`, the
population-weighted mean charge is needed for the electron-fraction
sum (`x_high_X * q_high_mean(T)` contributes to `x_e`), and the per-
electron, per-ion sum `Lambda_high(T)` is what the cooling channel
reads.

CLI:
    XUVTOP=$HOME/Dropbox/Projects/CHIANTI_db \\
        python -m pyathena.chemistry.tables.chianti_v11.build_cie_high_pool

The builder is data-table-only: it does not need ChiantiPy at run
time; it just sums the pre-built CHIANTI tables.
"""
from __future__ import annotations

import os
from typing import Iterable, Tuple

import numpy as np

from .build_ioneq import read_ioneq


# Element groups must match the corresponding declaration on
# `NCRNetwork3PlusIons16.element_groups`. The third tuple element is
# the evolved ion-name tuple; only the count enters the cutoff, so
# we store the cutoff explicitly here.
ELEMENT_GROUPS: Tuple[Tuple[str, int], ...] = (
    ('C', 2),   # tracked: CI, CII;          pool: q = 2 .. 6
    ('N', 2),   # tracked: NI, NII;          pool: q = 2 .. 7
    ('O', 3),   # tracked: OI, OII, OIII;    pool: q = 3 .. 8
    ('S', 3),   # tracked: SI, SII, SIII;    pool: q = 3 .. 16
)


def _read_cool(path: str) -> dict:
    """Parse a `cool_<element>.txt` file into a `(NT, Z+1)` array
    plus the log-T column.

    Returns a dict with keys `log_T`, `element`, `Z`, `L_q` (shape
    `(Z+1, NT)`). Format matches `build_cool.py`: lines starting `#`
    are header; the header line containing 'cooling efficiency for'
    + 'Z=' carries the metadata; subsequent non-empty lines are
    `log_T_K  L_0  L_1  ... L_Z`.
    """
    log_T = []
    rows = []
    Z = None
    element = None
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if s.startswith('#'):
                if 'efficiency for' in s and 'Z=' in s:
                    parts = s.split('efficiency for')[-1].split('(Z=')
                    element = parts[0].strip()
                    Z = int(parts[1].rstrip(')').strip())
                continue
            if not s:
                continue
            parts = s.split()
            log_T.append(float(parts[0]))
            rows.append([float(p) for p in parts[1:]])
    log_T_arr = np.asarray(log_T)
    L_q = np.asarray(rows).T   # shape (Z+1, NT)
    return dict(log_T=log_T_arr, element=element, Z=Z, L_q=L_q)


def compute_pool(
    ioneq: np.ndarray,
    L_q: np.ndarray,
    q_max_tracked: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum the high-charge pool contributions on the existing log-T
    grid.

    Parameters
    ----------
    ioneq : (Z+1, NT) ndarray
        CIE charge fractions; column q = 0 is neutral, q = Z is
        fully stripped.
    L_q : (Z+1, NT) ndarray
        Per-ion radiative loss [erg cm^3 / s]. Column q = 0 is the
        neutral atom (which often has zero bound-bound emission and
        is included for completeness).
    q_max_tracked : int
        Number of low charge states the multi-ion network resolves
        explicitly. Pool covers `q >= q_max_tracked` (so
        `q_max_tracked = 2` means CI + CII tracked; pool starts at
        CIII).

    Returns
    -------
    x_high_frac : (NT,) ndarray
    q_high_mean : (NT,) ndarray
        Mean charge over the pool. Returns 0 where the pool fraction
        is below `1e-30` to avoid divide-by-zero noise in the
        deep-cold tail.
    Lambda_high : (NT,) ndarray
    """
    q = np.arange(ioneq.shape[0], dtype=np.float64)
    sel = q >= q_max_tracked
    pool_frac = ioneq[sel, :]      # (n_pool, NT)
    pool_q = q[sel]                 # (n_pool,)
    pool_L = L_q[sel, :]            # (n_pool, NT)

    x_high = pool_frac.sum(axis=0)
    sum_q_frac = (pool_q[:, None] * pool_frac).sum(axis=0)
    # q_mean = sum_q q * ioneq_q / x_high; avoid 0/0 in cold cells.
    safe = x_high > 1.0e-30
    q_mean = np.where(safe, sum_q_frac / np.where(safe, x_high, 1.0),
                      0.0)
    Lambda_high = (pool_frac * pool_L).sum(axis=0)
    return x_high, q_mean, Lambda_high


def write_ascii(path: str, element: str, Z: int, q_max_tracked: int,
                log_T: np.ndarray, x_high: np.ndarray,
                q_mean: np.ndarray, Lambda_high: np.ndarray) -> None:
    """Write `cie_high_pool_<element>.txt` in the project ASCII
    convention.
    """
    NT = log_T.size
    pool_range = f'q = {q_max_tracked} .. {Z}'
    with open(path, 'w') as f:
        f.write(f'# CIE high-charge-pool tables for {element} '
                f'(Z={Z}, pool {pool_range})\n')
        f.write(f'# Source: ioneq_{element}.txt + cool_{element}.txt\n')
        f.write(f'# x_high_frac  = sum_{{q >= {q_max_tracked}}}'
                f' ioneq_q(T)\n')
        f.write(f'# q_high_mean  = sum_{{q >= {q_max_tracked}}}'
                f' q * ioneq_q / x_high_frac  (0 where pool is empty)\n')
        f.write(f'# Lambda_high  = sum_{{q >= {q_max_tracked}}}'
                f' ioneq_q * L_q     [erg cm^3 / s]\n')
        f.write(f'# T grid: {NT} log-spaced points, '
                f'10^{log_T[0]:.2f} K -> 10^{log_T[-1]:.2f} K\n')
        f.write(f'# Generated by '
                f'pyathena/chemistry/tables/chianti_v11/'
                f'build_cie_high_pool.py\n#\n')
        cols = ['log_T_K', 'x_high_frac', 'q_high_mean', 'Lambda_high']
        f.write('# ' + '  '.join(f'{c:<13s}' for c in cols) + '\n')
        for it in range(NT):
            row = (f'{log_T[it]:13.4f}  {x_high[it]:13.6e}  '
                   f'{q_mean[it]:13.6e}  {Lambda_high[it]:13.6e}')
            f.write('  ' + row + '\n')


def read_cie_high_pool(path: str) -> dict:
    """Read a previously-built pool table back into a dict.

    Returns keys `log_T`, `element`, `Z`, `q_max_tracked`,
    `x_high_frac`, `q_high_mean`, `Lambda_high`.
    """
    log_T = []
    x_high = []
    q_mean = []
    Lambda_high = []
    element = None
    Z = None
    q_max = None
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if s.startswith('#'):
                if 'tables for' in s and 'Z=' in s:
                    parts = s.split('tables for')[-1]
                    element = parts.split('(Z=')[0].strip()
                    Z_str = parts.split('Z=')[1].split(',')[0]
                    Z = int(Z_str.strip())
                    pool_part = parts.split('pool q =')
                    if len(pool_part) > 1:
                        q_max = int(pool_part[1].split('..')[0].strip())
                continue
            if not s:
                continue
            cols = s.split()
            log_T.append(float(cols[0]))
            x_high.append(float(cols[1]))
            q_mean.append(float(cols[2]))
            Lambda_high.append(float(cols[3]))
    return dict(
        log_T=np.asarray(log_T),
        element=element,
        Z=Z,
        q_max_tracked=q_max,
        x_high_frac=np.asarray(x_high),
        q_high_mean=np.asarray(q_mean),
        Lambda_high=np.asarray(Lambda_high),
    )


def _resolve_input_path(name: str) -> str:
    """Resolve `name` against the chianti_v11 data dir."""
    here = os.path.dirname(__file__)
    return os.path.abspath(
        os.path.join(here, '..', '..', '..', '..',
                     'data', 'microphysics', 'chianti_v11', name))


def _resolve_output_dir() -> str:
    here = os.path.dirname(__file__)
    return os.path.abspath(
        os.path.join(here, '..', '..', '..', '..',
                     'data', 'chemistry'))


def main(elements: Iterable[Tuple[str, int]] = ELEMENT_GROUPS) -> None:
    """Build the pool tables for every entry in `ELEMENT_GROUPS`."""
    out_dir = _resolve_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    print(f'Writing CIE high-pool tables to {out_dir}')
    for element, q_max in elements:
        ioneq_path = _resolve_input_path(f'ioneq_{element}.txt')
        cool_path = _resolve_input_path(f'cool_{element}.txt')
        if not (os.path.isfile(ioneq_path) and os.path.isfile(cool_path)):
            print(f'  {element}: SKIP (missing {ioneq_path} or '
                  f'{cool_path})')
            continue
        ioneq = read_ioneq(ioneq_path)
        cool = _read_cool(cool_path)
        assert ioneq['Z'] == cool['Z'], (
            f'Z mismatch for {element}: ioneq={ioneq["Z"]} vs '
            f'cool={cool["Z"]}')
        assert ioneq['log_T'].shape == cool['log_T'].shape, (
            f'log_T length mismatch for {element}')
        x_high, q_mean, Lambda_high = compute_pool(
            ioneq['x_q'], cool['L_q'], q_max)
        out_path = os.path.join(out_dir,
                                f'cie_high_pool_{element}.txt')
        write_ascii(out_path, element, ioneq['Z'], q_max,
                    ioneq['log_T'], x_high, q_mean, Lambda_high)
        # Brief summary at the peak of the pool.
        peak_i = int(np.argmax(x_high))
        print(f'  {element:2s} (Z={ioneq["Z"]:2d}, q>={q_max}): '
              f'wrote {os.path.basename(out_path)}  '
              f'(peak x_high={x_high[peak_i]:.3f} at '
              f'log T = {ioneq["log_T"][peak_i]:.2f}, '
              f'Lambda_peak = {Lambda_high.max():.3e})')


if __name__ == '__main__':
    main()
