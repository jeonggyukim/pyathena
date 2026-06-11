"""Benchmark: ion-by-ion CIE cooling efficiency Lambda_q(T) from our
CHIANTI v11 build vs the Gnat & Ferland 2012 (ApJS 199 20)
reference tables shipped with pyathena.

GF12's published per-ion cooling tables are at
`pyathena/data/microphysics/Gnat_Ferland12_tables/<Element>.txt`
(file name is the full element name, e.g., `Oxygen.txt`). Columns
are per ion per electron in the same convention as ours: no
abundance or ionization fraction baked in. Their final
`{Element}{CIE}` column is the GS07-CIE-weighted sum
`sum_q x_q^CIE * Lambda_q`.

Two figures (light + heavy split), per-element 2x3 grid (one panel
per element). For each element, per-ion Lambda_q(T) curves
overlaid: GF12 solid, ours (CHIANTI v11) dashed. Color-coded by
charge state q via viridis.
"""

from pathlib import Path
import pytest
import numpy as np


GF12_DIR = Path(__file__).parent.parent.parent / 'data' / \
    'microphysics' / 'Gnat_Ferland12_tables'

# Full element-name mapping used in GF12 file naming.
_GF12_NAME = {
    'H':  'Hydrogen',  'He': 'Helium',   'C':  'Carbon',
    'N':  'Nitrogen',  'O':  'Oxygen',   'Ne': 'Neon',
    'Mg': 'Magnesium', 'Si': 'Silicon',  'S':  'Sulfur',
    'Ar': 'Argon',     'Ca': 'Calcium',  'Fe': 'Iron',
}
COMPARE_FIG1 = ['H',  'He', 'C', 'N', 'O']
COMPARE_FIG2 = ['Ne', 'Mg', 'Si', 'S', 'Fe']


def _read_gf12(element):
    """Read GF12's per-element table.

    Returns dict with keys:
      'T'        : (NT,) linear T in K
      'Lambda_q' : (Z+1, NT) per ion per electron
      'Lambda_CIE' : (NT,) GS07-CIE-weighted total per element atom
    """
    fname = GF12_DIR / f'{_GF12_NAME[element]}.txt'
    with open(fname) as f:
        lines = f.read().splitlines()
    # Data lines start after two blocks of dashes. Simpler: find
    # lines that start with a digit/'.' that parse as float and
    # have at least 3 columns of floats.
    data_lines = []
    for ln in lines:
        s = ln.lstrip()
        if not s:
            continue
        first = s.split()[0]
        try:
            float(first)
        except ValueError:
            continue
        data_lines.append(ln)
    arr = np.array([[float(v) for v in ln.split()]
                    for ln in data_lines])
    T = arr[:, 0]
    # Columns 1..(Z+1) are per-ion Lambda_q; last column is CIE total.
    Lambda_q = arr[:, 1:-1].T   # (Z+1, NT)
    Lambda_CIE = arr[:, -1]
    return dict(T=T, Lambda_q=Lambda_q, Lambda_CIE=Lambda_CIE)


def _read_ours(element):
    """Read our cool_<element>.txt."""
    from pyathena.photchem.data.build_cool_tables import read_cool
    base = Path(__file__).parent.parent.parent / 'data' / \
        'microphysics' / 'chianti_v11'
    return read_cool(str(base / f'cool_{element}.txt'))


def _ion_label(element, q):
    if q == 0:
        return rf'${{\rm {element}}}$'
    if q == 1:
        return rf'${{\rm {element}}}^+$'
    return rf'${{\rm {element}}}^{{{q}+}}$'


def _make_panel(figures_dir, elements_subset, fig_name):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from pyathena.plt_tools.line_annotation import line_annotate

    stroke = [path_effects.withStroke(
        linewidth=2.0, foreground='white')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9),
                             sharex=True, sharey=True)
    axes = axes.flatten()
    for ax in axes[len(elements_subset):]:
        ax.set_visible(False)
    for ax, element in zip(axes, elements_subset):
        gf = _read_gf12(element)
        ours = _read_ours(element)
        Z = gf['Lambda_q'].shape[0] - 1
        log_T_gf = np.log10(gf['T'])
        log_T_o  = ours['log_T']
        import cmasher as cmr
        cmap = cmr.combine_cmaps(
            cmr.get_sub_cmap(cmr.ocean, 0.15, 0.85),
            cmr.get_sub_cmap(cmr.amber, 0.15, 0.85))
        gf_lines = []
        for q in range(Z + 1):
            color = cmap(q / max(Z, 1))
            ln_gf, = ax.loglog(10 ** log_T_gf, gf['Lambda_q'][q],
                               '-', color=color, lw=1.4)
            ax.loglog(10 ** log_T_o, ours['Lambda_q'][q], '--',
                      color=color, lw=3.5, alpha=0.45)
            gf_lines.append(ln_gf)
        # Inline labels at the T where each ion's Lambda_q peaks.
        for q in range(Z + 1):
            arr = gf['Lambda_q'][q]
            if not np.any(arr > 0):
                continue
            i_peak = int(np.argmax(arr))
            T_annot = float(10 ** log_T_gf[i_peak])
            # For fully-stripped ions whose cooling rises monotonic
            # in T (FF dominant), the argmax falls at the top of
            # the table -- pin the label to the right edge of the
            # plotted range so it stays visible.
            if T_annot > 0.9 * 1e8:
                T_annot = 5e7
            if T_annot < 1.5e4:
                continue
            line_annotate(_ion_label(element, q), gf_lines[q],
                          x=T_annot, xytext=(0, 4),
                          fontsize='small',
                          color=cmap(q / max(Z, 1)),
                          ha='center', path_effects=stroke)
        ax.plot([], [], 'k-',  lw=1.4, label='GF12 (Cloudy)')
        ax.plot([], [], 'k--', lw=3.5, alpha=0.45,
                label='ours (CHIANTI v11)')
        ax.legend(fontsize='small', loc='lower left',
                  framealpha=0.7)
        ax.set_title(f'{element} (Z={Z})')
        ax.set_xlim(1e4, 1e8)
        ax.set_ylim(1e-24, 1e-16)
        ax.grid(True, which='both', alpha=0.3)
    for ax in axes[:len(elements_subset)]:
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_xlabel(r'$T\,[{\rm K}]$')
    for ax in axes[::3]:
        ax.set_ylabel(
            r'$\Lambda_q\,[\rm erg\,cm^3\,s^{-1}]$ '
            r'(per ion per $e$)')
    fig.suptitle(
        r'Per-ion cooling efficiency $\Lambda_q(T)$ '
        r'(low-density limit; no abundance or $x_q$ baked in)'
        '\n'
        r'cooling rate per unit volume from ion $X^q$ = '
        r'$n(X^q)\,n_e\,\Lambda_q$;  $\Lambda_q$ = sum of '
        r'bound-bound + two-photon + bremsstrahlung + '
        r'recombination radiation'
        '\n'
        r'CHIANTI v11 (ours, dashed) vs '
        r'Gnat & Ferland 2012 (Cloudy, solid)')
    fig.tight_layout()
    fig.savefig(figures_dir / fig_name, dpi=300)
    plt.close(fig)


def test_plot_cool_vs_gf12(figures_dir, save_figures):
    if not save_figures:
        pytest.skip('plot generation disabled (--no-figures)')
    _make_panel(figures_dir, COMPARE_FIG1,
                'cool_chianti_v11_vs_GF12_light.png')
    _make_panel(figures_dir, COMPARE_FIG2,
                'cool_chianti_v11_vs_GF12_heavy.png')
