"""Per-line cooling decomposition for OIII, with three solver paths.

Benchmark plot: Lambda_line / (n_OIII * n_e) vs T for each of the
five dominant OIII collisionally excited lines, compared between:

  1. pyathena.photchem 5-level: collisional + radiative within
     2p^2 3P + 1D + 1S, electron + proton impact from CHIANTI v11
     Upsilon tables we extract offline.
  2. ChiantiPy.populate(popCorrect=False): many-level (~177)
     statistical-equilibrium populations, no recombination
     correction.
  3. ChiantiPy.populate(popCorrect=True): same as above + cascade
     from O IV recombination (CIE-assumed x_OIV).

Lines tabulated:
   - [OIII] 5008  (1D_2 -> 3P_2; classic nebular green)
   - [OIII] 4960  (1D_2 -> 3P_1)
   - [OIII] 4364  (1S_0 -> 1D_2; auroral, T_e diagnostic w/ 5008)
   - [OIII] 52 um (3P_2 -> 3P_1; FIR fine structure)
   - [OIII] 88 um (3P_1 -> 3P_0; FIR fine structure)

This is a benchmark for the per-line accuracy of each solver path.
More ions to be added in follow-up plots.

Run:
    XUVTOP=$HOME/Dropbox/Projects/CHIANTI_db pytest \\
        tests/microphysics/test_plot_nebular_cooling.py
"""

import os
import warnings
import pytest


@pytest.fixture(scope='module', autouse=True)
def _xuvtop():
    os.environ.setdefault(
        'XUVTOP', os.path.expanduser('~/Dropbox/Projects/CHIANTI_db'))
    warnings.filterwarnings('ignore')


# OIII lines (CHIANTI 1-based level indexing):
#   1: 3P_0  2: 3P_1  3: 3P_2  4: 1D_2  5: 1S_0
# Group [OIII] 4960+5008 (both from 1D_2) into one "doublet" entry
# since they're physically tied (same upper level, just different
# lower J of the 3P term).
OIII_LINES = [
    # label, list of (upper, lower) transitions to sum
    ('[OIII] 4960 + 5008',  [(4, 3), (4, 2)]),  # 1D_2 -> 3P_2, 3P_1
    ('[OIII] 4364 (auroral)', [(5, 4)]),         # 1S_0 -> 1D_2
    ('[OIII] 52 um',  [(3, 2)]),                 # 3P_2 -> 3P_1
    ('[OIII] 88 um',  [(2, 1)]),                 # 3P_1 -> 3P_0
]

# OII lines (CHIANTI 1-based level indexing):
#   1: 4S_3/2  2: 2D_5/2  3: 2D_3/2  4: 2P_1/2  5: 2P_3/2
# [OII] 3726 + 3729 is the famous n_e diagnostic doublet (both from
# 2D levels to 4S ground).
OII_LINES = [
    ('[OII] 3726 + 3729', [(2, 1), (3, 1)]),     # 2D_5/2, 3/2 -> 4S
    ('[OII] 7320 + 7330', [(4, 2), (5, 3)]),     # 2P -> 2D auroral
]


def _chianti_pops_and_A(ion_name, T_grid, n_e, popCorrect):
    """Compute populations + per-transition emissivities from
    ChiantiPy.populate. Returns dict
    {(upper_1based, lower_1based): array(T_grid)} of per-ion line
    emissivity Lambda_line / n_ion = f_up * A * E_ij [erg/s].
    """
    import ChiantiPy.core as ch
    import numpy as np
    out = {}
    n_T = len(T_grid)
    for k, T in enumerate(T_grid):
        ion = ch.ion(ion_name,
                     temperature=np.array([T, T*1.001]),
                     eDensity=n_e)
        ion.populate(popCorrect=popCorrect)
        f = ion.Population['population'][0]
        E_cm = np.asarray(ion.Elvlc['ecm'])
        E_cm_th = np.asarray(ion.Elvlc['ecmth'])
        E = np.where(E_cm > 0.0, E_cm, E_cm_th) * 1.986e-16
        wg = ion.Wgfa
        for w in range(len(wg['lvl1'])):
            l1 = int(wg['lvl1'][w])
            l2 = int(wg['lvl2'][w])
            av = float(wg['avalue'][w])
            if av <= 0 or E[l2-1] <= E[l1-1]:
                continue
            key = (l2, l1)
            if key not in out:
                out[key] = np.zeros(n_T)
            out[key][k] = f[l2-1] * av * (E[l2-1] - E[l1-1])
    return out


def _pyathena_pops_and_A(module_name, T_grid, n_e):
    """Same as _chianti_pops_and_A but from pyathena's 5-level
    solver. Module-name is e.g. 'o_3', 'o_2'.
    """
    import numpy as np
    from importlib import import_module
    mod = import_module(f'pyathena.photchem.coolants.{module_name}')
    tab = mod._C.table()
    A = tab['A']
    E = tab['E_erg']
    out = {}
    n_T = len(T_grid)
    for k, T in enumerate(T_grid):
        f = mod.populations(T, n_e)
        for i in range(5):
            for j in range(i):
                if A[i, j] <= 0:
                    continue
                key = (i + 1, j + 1)   # 0-based -> CHIANTI 1-based
                if key not in out:
                    out[key] = np.zeros(n_T)
                out[key][k] = float(
                    f[i] * A[i, j] * (E[i] - E[j]))
    return out


def _sum_line_group(per_transition_dict, transitions):
    """Sum per-transition emissivity arrays for a list of (up, lo)
    transitions into a single curve."""
    import numpy as np
    total = None
    for (up, lo) in transitions:
        arr = per_transition_dict.get((up, lo))
        if arr is None:
            continue
        total = arr if total is None else total + arr
    if total is None:
        # Return zeros of right length using the first available
        # transition's shape, or None.
        for arr in per_transition_dict.values():
            return np.zeros_like(arr)
        return None
    return total


def test_plot_O_nebular_per_line(figures_dir, save_figures):
    """Benchmark: per-line cooling vs T for OII + OIII at HII
    conditions, normalized by n_H * n_e (Draine 2011 Fig 27.1(b)
    style). Two panels (one per ion), each showing:
      - pyathena 5-level (solid),
      - ChiantiPy popCorrect=False (dashed),
      - ChiantiPy popCorrect=True (dotted).
    Lines are grouped by physical pairing (4960+5008 doublet,
    3726+3729 doublet, etc.).

    Assumed abundance + ionization fractions (typical inner HII):
      n_O / n_H = 3.2e-4
      x_OII  = n(O+) / n_O = 0.3
      x_OIII = n(O++) / n_O = 0.7
    """
    if not save_figures:
        pytest.skip('plot generation disabled (--no-figures)')
    import matplotlib.pyplot as plt
    import numpy as np

    n_e = 1e3
    T_grid = np.logspace(3.5, 5.0, 25)
    # Abundance + ionization fractions (inner HII fiducial)
    AO_H = 3.2e-4    # n_O / n_H
    X_ION = {'OII': 0.3, 'OIII': 0.7}
    # Per-ion line group definitions: (display_label, pyathena_mod,
    # chianti_name, line_defs)
    ion_panels = [
        ('OIII', 'o_3', 'o_3', OIII_LINES),
        ('OII',  'o_2', 'o_2', OII_LINES),
    ]
    method_specs = [
        ('pyathena 5-lvl', '-',  None),
        ('ChiantiPy -rec', '--', False),
        ('ChiantiPy +rec', ':',  True),
    ]
    color_per_line = {
        '[OIII] 4960 + 5008': 'C0',
        '[OIII] 4364 (auroral)': 'C1',
        '[OIII] 52 um': 'C2',
        '[OIII] 88 um': 'C3',
        '[OII] 3726 + 3729': 'C0',
        '[OII] 7320 + 7330': 'C1',
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, (ion_label, pymod, ch_name, line_defs) in enumerate(
            ion_panels):
        ax = axes[ax_idx]
        # Pre-compute per-transition tables for each method.
        tables = {}
        for method_name, ls, popCorrect in method_specs:
            if method_name == 'pyathena 5-lvl':
                tables[method_name] = _pyathena_pops_and_A(
                    pymod, T_grid, n_e)
            else:
                tables[method_name] = _chianti_pops_and_A(
                    ch_name, T_grid, n_e, popCorrect=popCorrect)
        # Conversion factor from Lambda / n_ion to
        # Lambda / (n_H * n_e):
        #   Lambda / (n_H * n_e)
        #     = (Lambda / n_ion) * (n_ion / n_H) / n_e
        #     = (Lambda / n_ion) * (A_O * x_ion) / n_e
        x_ion = X_ION[ion_label]
        factor = AO_H * x_ion / n_e
        # Plot per line group, all methods.
        for label, transitions in line_defs:
            col = color_per_line[label]
            for method_name, ls, _ in method_specs:
                vals = _sum_line_group(
                    tables[method_name], transitions)
                if vals is None or np.all(vals <= 0):
                    continue
                ax.loglog(T_grid, vals * factor, ls, color=col,
                          label=f'{label} ({method_name})')
        ax.set_xlabel(r'$T\,[{\rm K}]$')
        ax.set_ylabel(
            r'$\Lambda_{\rm line} / (n_{\rm H}\,n_e)\,'
            r'[\rm erg\,cm^3\,s^{-1}]$')
        ax.set_title(f'{ion_label} per-line cooling, '
                     rf'$x_{{{ion_label}}}={x_ion}$, '
                     rf'$n_e = {n_e:g}$ cm$^{{-3}}$')
        ax.set_xlim(3e3, 1e5)
        ax.set_ylim(1e-28, 1e-22)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize='xx-small', loc='lower right', ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / 'O_nebular_per_line_cooling.png',
                dpi=150)
    plt.close(fig)
