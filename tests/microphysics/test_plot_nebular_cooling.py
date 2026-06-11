"""Per-line cooling decomposition for OIII, with three solver paths.

Benchmark plot: Lambda_line / (n_OIII * n_e) vs T for each of the
five dominant OIII collisionally excited lines, compared between:

  1. pyathena.microphysics 5-level: collisional + radiative within
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

# OII lines: 2p^3 ground configuration 4S + 2D + 2P.
#   1: 4S_3/2  2: 2D_5/2  3: 2D_3/2  4: 2P_1/2  5: 2P_3/2
OII_LINES = [
    ('[OII] 3726 + 3729', [(2, 1), (3, 1)]),     # 2D -> 4S
    ('[OII] 7320 + 7330', [(4, 2), (5, 3)]),     # 2P -> 2D auroral
]
# CI lines: 2p^2 ground configuration 3P + 1D + 1S.
#   1: 3P_0  2: 3P_1  3: 3P_2  4: 1D_2  5: 1S_0
CI_LINES = [
    ('[CI] 9824 + 9849', [(4, 3), (4, 2)]),      # 1D -> 3P
    ('[CI] 8727 (auroral)', [(5, 4)]),
    ('[CI] 370 um',  [(3, 2)]),                  # 3P_2 -> 3P_1
    ('[CI] 609 um',  [(2, 1)]),                  # 3P_1 -> 3P_0
]
# CII lines: 2p ground configuration 2P_{1/2,3/2}.
#   1: 2P_1/2  2: 2P_3/2
CII_LINES = [
    ('[CII] 158 um', [(2, 1)]),                  # 2P_3/2 -> 2P_1/2
]
# NI lines: 2p^3 ground configuration 4S + 2D + 2P.
NI_LINES = [
    ('[NI] 5198 + 5200', [(2, 1), (3, 1)]),      # 2D -> 4S
    ('[NI] 3466 (auroral)', [(4, 2), (5, 3)]),   # 2P -> 2D
]
# NII lines: 2p^2 ground configuration 3P + 1D + 1S (same as OIII).
NII_LINES = [
    ('[NII] 6548 + 6583', [(4, 3), (4, 2)]),
    ('[NII] 5755 (auroral)', [(5, 4)]),
    ('[NII] 122 um',  [(3, 2)]),
    ('[NII] 205 um',  [(2, 1)]),
]
# NeII lines: 2p^5 ground configuration 2P_{1/2,3/2}.
NEII_LINES = [
    ('[NeII] 12.8 um', [(2, 1)]),
]
# NIII lines: 2s^2 2p ground configuration 2P_{1/2,3/2}.
NIII_LINES = [
    ('[NIII] 57.3 um', [(2, 1)]),
]
# NeIII lines: 2p^4 ground configuration 3P + 1D + 1S (np^4 inverted).
#   1: 3P_2  2: 3P_1  3: 3P_0  4: 1D_2  5: 1S_0
NEIII_LINES = [
    ('[NeIII] 3869 + 3968', [(4, 1), (4, 2)]),   # 1D -> 3P
    ('[NeIII] 3343 (auroral)', [(5, 4)]),
    ('[NeIII] 15.6 um', [(2, 1)]),
    ('[NeIII] 36.0 um', [(3, 2)]),
]
# SII lines: 3p^3 ground configuration 4S + 2D + 2P (same as OII).
SII_LINES = [
    ('[SII] 6716 + 6731', [(2, 1), (3, 1)]),
    ('[SII] 4068 + 4076 (auroral)', [(4, 1), (5, 1)]),
]
# SIII lines: 3p^2 ground configuration 3P + 1D + 1S (same as OIII).
SIII_LINES = [
    ('[SIII] 9069 + 9532', [(4, 3), (4, 2)]),
    ('[SIII] 6312 (auroral)', [(5, 4)]),
    ('[SIII] 18.7 um', [(3, 2)]),
    ('[SIII] 33.5 um', [(2, 1)]),
]


def _chianti_pops_and_A(ion_name, T_grid, n_e, popCorrect):
    """Compute populations + per-transition emissivities from
    ChiantiPy.populate. Returns dict
    {(upper_1based, lower_1based): array(T_grid)} of per-ion line
    emissivity Lambda_line / n_ion = f_up * A * E_ij [erg/s].
    """
    import ChiantiPy.core as ch
    import numpy as np
    n_T = len(T_grid)
    # ChiantiPy ion accepts T and eDensity as arrays of equal length.
    T_arr = np.asarray(T_grid, dtype=float)
    ne_arr = np.full(n_T, float(n_e))
    ion = ch.ion(ion_name, temperature=T_arr, eDensity=ne_arr)
    ion.populate(popCorrect=popCorrect)
    pop = ion.Population['population']   # (n_T, n_lvls)
    E_cm = np.asarray(ion.Elvlc['ecm'])
    E_cm_th = np.asarray(ion.Elvlc['ecmth'])
    E = np.where(E_cm > 0.0, E_cm, E_cm_th) * 1.986e-16
    wg = ion.Wgfa
    out = {}
    for w in range(len(wg['lvl1'])):
        l1 = int(wg['lvl1'][w])
        l2 = int(wg['lvl2'][w])
        av = float(wg['avalue'][w])
        if av <= 0 or E[l2-1] <= E[l1-1]:
            continue
        key = (l2, l1)
        # pop[:, l2-1] is the upper-level population across T_grid.
        out[key] = pop[:, l2-1] * av * (E[l2-1] - E[l1-1])
    return out


def _pyathena_pops_and_A(module_name, T_grid, n_e):
    """Same as _chianti_pops_and_A but from pyathena's 5-level
    solver. Module-name is e.g. 'o_3', 'o_2'.
    """
    import numpy as np
    from importlib import import_module
    mod = import_module(f'pyathena.microphysics.coolants.{module_name}')
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


def _make_element_panel_figure(figures_dir, element_letter, A_X,
                               ion_rows, color_per_line, X_ION,
                               output_name):
    """Render per-line cooling 2x4 grid (rows=ions, cols=n_e) for
    one element. Reusable across O / C / N / Ne / S / Ar.

    Parameters
    ----------
    element_letter : str
        Suptitle prefix, e.g. 'O' / 'C' / 'N' / 'S' / 'Ne' / 'Ar'.
    A_X : float
        Asplund09 elemental abundance n_X / n_H.
    ion_rows : list of (ion_label, pymod_name, ch_name, line_defs)
        e.g. [('CI', 'c_1', 'c_1', CI_LINES),
              ('CII', 'c_2', 'c_2', CII_LINES)]
    color_per_line : dict
        Mapping from line label to matplotlib color spec.
    X_ION : dict
        Fractional abundance of each ion stage relative to the
        total element population (e.g. {'CI': 0.05, 'CII': 0.9}).
    output_name : str
        Saved file basename (e.g. 'cool_nebular_C.png').
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patheffects as path_effects
    from pyathena.plt_tools.line_annotation import line_annotate

    n_e_cols = [1.0, 1e2, 1e4, 1e6]
    n_e_ref = 1.0
    # T grid spans WIM to lower transition-region: 3e3 to 1e5 K.
    T_grid = np.logspace(np.log10(3e3), 5.0, 50)
    stroke = [path_effects.withStroke(
        linewidth=2.0, foreground='white')]

    n_rows = len(ion_rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4.5 * n_rows),
                             sharex=True, sharey=True,
                             squeeze=False)
    # Pre-compute tables.
    tables_py = {ion[0]: {} for ion in ion_rows}
    tables_ch_neg = {ion[0]: {} for ion in ion_rows}
    for ion_label, pymod, ch_name, _ in ion_rows:
        for ne in [n_e_ref] + n_e_cols:
            if ne not in tables_py[ion_label]:
                tables_py[ion_label][ne] = _pyathena_pops_and_A(
                    pymod, T_grid, ne)
        for ne in n_e_cols:
            tables_ch_neg[ion_label][ne] = _chianti_pops_and_A(
                ch_name, T_grid, ne, popCorrect=False)

    for row_idx, (ion_label, pymod, ch_name, line_defs) in enumerate(
            ion_rows):
        x_ion = X_ION[ion_label]
        for col_idx, n_e_col in enumerate(n_e_cols):
            ax = axes[row_idx, col_idx]
            ours_lines = {}
            total_ref = np.zeros_like(T_grid)
            total_py = np.zeros_like(T_grid)
            total_ch = np.zeros_like(T_grid)
            f_ref = A_X * x_ion / n_e_ref
            f_py  = A_X * x_ion / n_e_col
            f_ch  = A_X * x_ion / n_e_col
            for label, transitions in line_defs:
                col = color_per_line[label]
                # Reference: pyathena at n_e_ref = 1 cm^-3 (faint).
                vals_ref = _sum_line_group(
                    tables_py[ion_label][n_e_ref], transitions)
                if vals_ref is not None and not np.all(vals_ref <= 0):
                    ax.loglog(T_grid, vals_ref * f_ref, '-',
                              color=col, lw=1.0, alpha=0.35)
                    total_ref += vals_ref
                # Pyathena at this column's n_e (thick solid).
                vals_py = _sum_line_group(
                    tables_py[ion_label][n_e_col], transitions)
                if vals_py is not None and not np.all(vals_py <= 0):
                    ln, = ax.loglog(T_grid, vals_py * f_py, '-',
                                    color=col, lw=2.0)
                    ours_lines[label] = ln
                    total_py += vals_py
                # ChiantiPy populate(-rec) at this n_e: thick dashed.
                vals_ch = _sum_line_group(
                    tables_ch_neg[ion_label][n_e_col], transitions)
                if vals_ch is not None and not np.all(vals_ch <= 0):
                    ax.loglog(T_grid, vals_ch * f_ch, '--',
                              color=col, lw=3.5, alpha=0.45)
                    total_ch += vals_ch
            # Black total-cooling curves.
            if np.any(total_ref > 0):
                ax.loglog(T_grid, total_ref * f_ref, '-',
                          color='black', lw=1.0, alpha=0.35)
            if np.any(total_py > 0):
                ax.loglog(T_grid, total_py * f_py, '-',
                          color='black', lw=2.0)
            if np.any(total_ch > 0):
                ax.loglog(T_grid, total_ch * f_ch, '--',
                          color='black', lw=3.5, alpha=0.45)
            # Inline per-line labels, only in left-most column.
            if col_idx == 0 and ours_lines:
                sorted_labels = sorted(
                    ours_lines.items(),
                    key=lambda kv:
                        int(np.argmax(kv[1].get_ydata())))
                for li, (label, ln) in enumerate(sorted_labels):
                    ydata = ln.get_ydata()
                    if not np.any(ydata > 0):
                        continue
                    ymax = float(np.max(ydata))
                    mask = ydata >= 0.5 * ymax
                    idx = np.where(mask)[0]
                    if len(idx) == 0:
                        continue
                    i_mid = (idx[0] + idx[-1]) // 2
                    T_annot = float(T_grid[i_mid])
                    T_annot = float(np.clip(T_annot, 4e3, 5e4))
                    dy = +8 if li % 2 == 0 else -10
                    line_annotate(label, ln, x=T_annot,
                                  xytext=(0, dy),
                                  fontsize='medium',
                                  color=ln.get_color(),
                                  ha='center',
                                  path_effects=stroke)
            # Title shows n_e for this column.
            exp = int(np.log10(n_e_col))
            ax.set_title(
                rf'{ion_label}, $n_e=10^{{{exp}}}$ cm$^{{-3}}$')
            ax.set_xlim(4e3, 2e4)
            ax.set_xscale('linear')
            ax.set_ylim(1e-28, 1e-22)
            ax.grid(True, which='both', alpha=0.3)
        # Legend only in the top-left panel.
        if row_idx == 0:
            ax0 = axes[0, 0]
            ax0.plot([], [], 'k-', lw=1.0, alpha=0.35,
                     label=rf'pyathena ref ($n_e=1$ cm$^{{-3}}$)')
            ax0.plot([], [], 'k-', lw=2.0,
                     label=r'pyathena (this column $n_e$)')
            ax0.plot([], [], 'k--', lw=3.5, alpha=0.45,
                     label=r'ChiantiPy populate ($-$rec)')
            ax0.legend(fontsize='small', loc='lower right',
                       framealpha=0.85, handlelength=3.5)

    # Outer labels.
    for col_idx in range(4):
        axes[n_rows - 1, col_idx].set_xlabel(r'$T\,[{\rm K}]$')
    for row_idx in range(n_rows):
        axes[row_idx, 0].set_ylabel(
            r'$\Lambda_{\rm line} / (n_{\rm H}\,n_e)\,'
            r'[\rm erg\,cm^3\,s^{-1}]$')
    # Assumption summary for the suptitle.
    ion_strs = ', '.join(
        rf'$x_{{\rm {label}}}={x:g}$'
        for label, x in X_ION.items())
    # Format abundance as a x 10^b instead of '3.20e-04'.
    mantissa, exp = f'{A_X:.1e}'.split('e')
    A_X_tex = rf'{float(mantissa):g}\times10^{{{int(exp)}}}'
    fig.suptitle(
        f'{element_letter} per-line cooling vs density: '
        f'pyathena 5-level (solid) vs ChiantiPy populate '
        f'(thick dashed)' '\n'
        rf'$n_{{\rm {element_letter},\,tot}}/n_{{\rm H}}={A_X_tex}$ '
        f'(Asplund 2009); {ion_strs}')
    fig.tight_layout()
    fig.savefig(figures_dir / output_name, dpi=150)
    plt.close(fig)


def test_plot_O_nebular_per_line(figures_dir, save_figures):
    """Per-line cooling for OII + OIII vs T at multiple n_e."""
    if not save_figures:
        pytest.skip('plot generation disabled (--no-figures)')
    _make_element_panel_figure(
        figures_dir,
        element_letter='O',
        A_X=3.2e-4,    # Asplund09 / Draine 11 Table 1.4
        ion_rows=[
            ('OII',  'o_2', 'o_2', OII_LINES),
            ('OIII', 'o_3', 'o_3', OIII_LINES),
        ],
        color_per_line={
            '[OIII] 4960 + 5008': 'C0',
            '[OIII] 4364 (auroral)': 'C1',
            '[OIII] 52 um': 'C2',
            '[OIII] 88 um': 'C3',
            '[OII] 3726 + 3729': 'C0',
            '[OII] 7320 + 7330': 'C1',
        },
        X_ION={'OII': 0.3, 'OIII': 0.7},
        output_name='cool_nebular_O.png',
    )


def test_plot_C_nebular_per_line(figures_dir, save_figures):
    """Per-line cooling for CI + CII vs T at multiple n_e."""
    if not save_figures:
        pytest.skip('plot generation disabled (--no-figures)')
    _make_element_panel_figure(
        figures_dir,
        element_letter='C',
        A_X=2.95e-4,
        ion_rows=[
            ('CI',  'c_1', 'c_1', CI_LINES),
            ('CII', 'c_2', 'c_2', CII_LINES),
        ],
        color_per_line={
            '[CI] 9824 + 9849': 'C0',
            '[CI] 8727 (auroral)': 'C1',
            '[CI] 370 um': 'C2',
            '[CI] 609 um': 'C3',
            '[CII] 158 um': 'C0',
        },
        X_ION={'CI': 0.05, 'CII': 0.95},
        output_name='cool_nebular_C.png',
    )


def test_plot_N_nebular_per_line(figures_dir, save_figures):
    """Per-line cooling for NI + NII vs T at multiple n_e."""
    if not save_figures:
        pytest.skip('plot generation disabled (--no-figures)')
    _make_element_panel_figure(
        figures_dir,
        element_letter='N',
        A_X=7.41e-5,
        ion_rows=[
            ('NII',  'n_2', 'n_2', NII_LINES),
            ('NIII', 'n_3', 'n_3', NIII_LINES),
        ],
        color_per_line={
            '[NII] 6548 + 6583': 'C0',
            '[NII] 5755 (auroral)': 'C1',
            '[NII] 122 um': 'C2',
            '[NII] 205 um': 'C3',
            '[NIII] 57.3 um': 'C0',
        },
        X_ION={'NII': 0.5, 'NIII': 0.5},
        output_name='cool_nebular_N.png',
    )


def test_plot_Ne_nebular_per_line(figures_dir, save_figures):
    """Per-line cooling for NeII + NeIII vs T at multiple n_e."""
    if not save_figures:
        pytest.skip('plot generation disabled (--no-figures)')
    _make_element_panel_figure(
        figures_dir,
        element_letter='Ne',
        A_X=9.33e-5,
        ion_rows=[
            ('NeII',  'ne_2', 'ne_2', NEII_LINES),
            ('NeIII', 'ne_3', 'ne_3', NEIII_LINES),
        ],
        color_per_line={
            '[NeII] 12.8 um': 'C0',
            '[NeIII] 3869 + 3968': 'C0',
            '[NeIII] 3343 (auroral)': 'C1',
            '[NeIII] 15.6 um': 'C2',
            '[NeIII] 36.0 um': 'C3',
        },
        X_ION={'NeII': 0.1, 'NeIII': 0.9},
        output_name='cool_nebular_Ne.png',
    )


def test_plot_S_nebular_per_line(figures_dir, save_figures):
    """Per-line cooling for SII + SIII vs T at multiple n_e."""
    if not save_figures:
        pytest.skip('plot generation disabled (--no-figures)')
    _make_element_panel_figure(
        figures_dir,
        element_letter='S',
        A_X=1.45e-5,
        ion_rows=[
            ('SII',  's_2', 's_2', SII_LINES),
            ('SIII', 's_3', 's_3', SIII_LINES),
        ],
        color_per_line={
            '[SII] 6716 + 6731': 'C0',
            '[SII] 4068 + 4076 (auroral)': 'C1',
            '[SIII] 9069 + 9532': 'C0',
            '[SIII] 6312 (auroral)': 'C1',
            '[SIII] 18.7 um': 'C2',
            '[SIII] 33.5 um': 'C3',
        },
        X_ION={'SII': 0.3, 'SIII': 0.6},
        output_name='cool_nebular_S.png',
    )
