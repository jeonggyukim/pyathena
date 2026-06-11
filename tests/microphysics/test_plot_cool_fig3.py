"""Benchmark: total CIE cooling efficiency Lambda_e(T) per H per
electron broken down by element, in the style of Gnat & Ferland
2012 Fig 3.

Two curves per element:
  solid  -- GF12 reference: their `{Element}{CIE}` column (which is
             Sum_q x_q^GS07 * Lambda_q^GF12 per element-atom per
             electron) times Asplund09 solar abundance.
  dashed -- ours: Sum_q x_q^our_CIE * Lambda_q^our_per_ion per
             element-atom per electron times the same Asplund09
             solar abundance. x_q from `ioneq_<element>.txt`
             (ChiantiPy.core.ioneq.calculate, current CHIANTI v11
             rates), Lambda_q from `cool_<element>.txt`
             (bound-bound + free-bound + free-free + two-photon
             via ChiantiPy).

Per-element curves use the same color for both methods (line style
distinguishes source). Black thick curves show the summed total:
sum over all followed elements.

For a fair comparison we use the *same* Asplund09 elemental
abundance ratios from `pyathena.microphysics.abundance_solar
.AbundanceSolar` (=Draine 2011 Table 1.4) on both methods. The
remaining differences reflect atomic-data updates 2012 -> 2025
(Cloudy vintage vs CHIANTI v11) plus ioneq solver differences
(GS07 ioneq vs `ChiantiPy.core.ioneq.calculate`).
"""

from pathlib import Path
import pytest
import numpy as np


GF12_DIR = Path(__file__).parent.parent.parent / 'data' / \
    'microphysics' / 'Gnat_Ferland12_tables'
_GF12_NAME = {
    'H':  'Hydrogen',  'He': 'Helium',   'C':  'Carbon',
    'N':  'Nitrogen',  'O':  'Oxygen',   'Ne': 'Neon',
    'Mg': 'Magnesium', 'Si': 'Silicon',  'S':  'Sulfur',
    'Ar': 'Argon',     'Ca': 'Calcium',  'Fe': 'Iron',
}
ELEMENTS = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Fe']


def _read_gf12_cie_per_atom(element):
    """Return (T, Lambda_X_CIE_per_atom) from GF12 table. Their
    `{Element}{CIE}` column is Sum_q x_q^CIE * Lambda_q per
    element-atom per electron (no abundance baked in)."""
    fname = GF12_DIR / f'{_GF12_NAME[element]}.txt'
    with open(fname) as f:
        lines = f.read().splitlines()
    data_lines = []
    for ln in lines:
        s = ln.lstrip()
        if not s:
            continue
        try:
            float(s.split()[0])
        except ValueError:
            continue
        data_lines.append(ln)
    arr = np.array([[float(v) for v in ln.split()]
                    for ln in data_lines])
    return arr[:, 0], arr[:, -1]


def _ours_cie_per_atom(element):
    """Compute Sum_q x_q^our_CIE * Lambda_q^per_ion from our local
    tables. Returns (log_T, Lambda_X_CIE_per_atom)."""
    from pyathena.photchem.data.build_cool_tables import read_cool
    from pyathena.photchem.data.build_ioneq_tables import read_ioneq
    base = Path(__file__).parent.parent.parent / 'data' / \
        'microphysics' / 'chianti_v11'
    cool = read_cool(str(base / f'cool_{element}.txt'))
    ioneq = read_ioneq(str(base / f'ioneq_{element}.txt'))
    # Both on same T grid (101 log-spaced pts from 1e4 to 1e9 K).
    L = (ioneq['x_q'] * cool['Lambda_q']).sum(axis=0)
    return cool['log_T'], L


def _abundance(element):
    from pyathena.microphysics.abundance_solar import AbundanceSolar
    a = AbundanceSolar(Zprime=1.0)
    df = a.df
    return float(df[df['X'] == element]['NX_NH'].iloc[0])


def test_plot_cool_fig3(figures_dir, save_figures):
    """GF12 Fig 3 reproduction: Lambda_e per H per electron, per
    element + total, CIE solar."""
    if not save_figures:
        pytest.skip('plot generation disabled (--no-figures)')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from pyathena.plt_tools.line_annotation import line_annotate

    stroke = [path_effects.withStroke(
        linewidth=2.0, foreground='white')]

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap('tab10')
    # Common abundance + same color per element across both
    # methods. Track total in parallel for both.
    gf_total = None
    ours_total = None
    gf_T = None
    ours_log_T = None
    gf_lines = []
    for ei, element in enumerate(ELEMENTS):
        A_X = _abundance(element)
        color = cmap(ei % 10)
        # GF12 (reference): dashed, semi-transparent.
        T_gf, L_gf = _read_gf12_cie_per_atom(element)
        L_gf_per_H = L_gf * A_X
        ax.loglog(T_gf, L_gf_per_H, '--', color=color, lw=3.5,
                  alpha=0.45)
        # Ours (primary): solid.
        log_T_o, L_o = _ours_cie_per_atom(element)
        L_o_per_H = L_o * A_X
        ln_ours, = ax.loglog(10 ** log_T_o, L_o_per_H, '-',
                             color=color, lw=1.4)
        gf_lines.append((element, ln_ours, color))
        # Accumulate totals
        if gf_total is None:
            gf_total = L_gf_per_H.copy()
            gf_T = T_gf
        else:
            # T grids should match; just add
            gf_total = gf_total + L_gf_per_H
        if ours_total is None:
            ours_total = L_o_per_H.copy()
            ours_log_T = log_T_o
        else:
            ours_total = ours_total + L_o_per_H

    # Total curves on top (black thick); ours solid, GF12 dashed.
    ax.loglog(gf_T, gf_total, 'k--', lw=3.5, alpha=0.55)
    ax.loglog(10 ** ours_log_T, ours_total, 'k-', lw=2.4)

    # Inline element labels at the T where each curve peaks.
    for element, ln, color in gf_lines:
        xdata = ln.get_xdata()
        ydata = ln.get_ydata()
        if not np.any(ydata > 0):
            continue
        i_peak = int(np.argmax(ydata))
        T_annot = float(xdata[i_peak])
        if T_annot < 1.5e4 or T_annot > 5e8:
            continue
        line_annotate(rf'${{\rm {element}}}$', ln, x=T_annot,
                      xytext=(0, 4), fontsize='small',
                      color=color, ha='center',
                      path_effects=stroke)

    # Single legend distinguishing line styles only (thick black =
    # totals, solid = GF12, dashed = ours).
    ax.plot([], [], 'k--', lw=3.5, alpha=0.55,
            label='GF12 (Cloudy 2012)')
    ax.plot([], [], 'k-',  lw=2.4,
            label='ours (CHIANTI v11)')
    ax.legend(fontsize='small', loc='lower right', framealpha=0.7,
              handlelength=4.0)
    ax.set_xlabel(r'$T\,[{\rm K}]$')
    ax.set_ylabel(
        r'$\Lambda_e\,[\rm erg\,cm^3\,s^{-1}]$ (per H per $e$)')
    ax.set_title(
        r'CIE cooling efficiency $\Lambda_e(T)$ per element + total'
        '\n'
        r'cooling rate per unit volume = $n_{\rm H}\,n_e\,\Lambda_e$'
        r' (Asplund 2009 solar abundances)')
    ax.set_xlim(1e4, 1e9)
    ax.set_ylim(1e-25, 1e-21)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / 'cool_cie.png', dpi=300)
    plt.close(fig)
