"""Benchmark: CIE x_q(T) from our CHIANTI v11.0.2 ioneq tables vs
the Gnat & Sternberg 2007 (ApJS 168, 213) reference shipped with
pyathena. CHIANTI as the gold standard reference; deltas
relative to GS07 quantify the atomic-data updates 2007 -> 2025.

Two figures (5 elements each, 2x3 grid with one empty panel):
  - Light: H, He, C, N, O
  - Heavy: Ne, Mg, Si, S, Fe

Inline charge-state labels via `pyathena.plt_tools.line_annotation`
with white-stroke patheffects.
"""

from pathlib import Path
import pytest
import numpy as np


GS07_PATH = Path(__file__).parent.parent.parent / 'data' / \
    'microphysics' / 'Gnat_Sternberg07_cie_ion_frac.txt'

# GS07 column ordering: H, He, C, N, O, Ne, Mg, Si, S, Fe.
GS07_ELEMENT_ORDER = [
    ('H', 1), ('He', 2), ('C', 6), ('N', 7), ('O', 8),
    ('Ne', 10), ('Mg', 12), ('Si', 14), ('S', 16), ('Fe', 26),
]
COMPARE_FIG1 = ['H',  'He', 'C', 'N', 'O']     # light
COMPARE_FIG2 = ['Ne', 'Mg', 'Si', 'S', 'Fe']   # heavy


def _read_gs07():
    """Read GS07 ASCII table; return {element: (T_K, x_q_array)}."""
    with open(GS07_PATH) as f:
        lines = f.read().splitlines()
    data_lines = []
    for ln in lines:
        s = ln.lstrip()
        if not s:
            continue
        first = s.split()[0]
        # Data lines start with a temperature in scientific
        # notation like 1.00e+04.
        if 'e' in first.lower() and any(c.isdigit() for c in first):
            try:
                float(first)
                data_lines.append(ln)
            except ValueError:
                continue
    arr = np.array([[float(v) for v in ln.split()]
                    for ln in data_lines])
    T = arr[:, 0]
    out = {}
    col = 1
    for element, Z in GS07_ELEMENT_ORDER:
        ncol = Z + 1
        out[element] = (T, arr[:, col:col + ncol].T)   # (Z+1, NT)
        col += ncol
    return out


def _read_chianti_local(element):
    """Return (log_T, x_q) from our CHIANTI v11 ioneq file."""
    from pyathena.photchem.data.build_ioneq_tables import read_ioneq
    base = Path(__file__).parent.parent.parent / 'pyathena' / \
        'photchem' / 'data'
    d = read_ioneq(str(base / f'ioneq_{element}.txt'))
    return d['log_T'], d['x_q']


def _ion_label(element, q):
    """LaTeX label for charge state q of element."""
    if q == 0:
        return rf'${{\rm {element}}}$'
    if q == 1:
        return rf'${{\rm {element}}}^+$'
    return rf'${{\rm {element}}}^{{{q}+}}$'


def _make_panel(figures_dir, elements_subset, fig_name, gs07):
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
        log_T_ch, x_ch = _read_chianti_local(element)
        T_gs, x_gs = gs07[element]
        log_T_gs = np.log10(T_gs)
        Z = x_ch.shape[0] - 1
        cmap = plt.get_cmap('viridis')

        ch_lines = []
        for q in range(Z + 1):
            color = cmap(q / Z)
            ln_ch, = ax.semilogy(log_T_ch, x_ch[q], '-',
                                 color=color, lw=1.4)
            ax.semilogy(log_T_gs, x_gs[q], '--', color=color, lw=1.0)
            ch_lines.append(ln_ch)
        # Inline charge-state labels on the CHIANTI curve.
        for q in range(Z + 1):
            if np.max(x_ch[q]) < 1e-3:
                continue
            i_peak = int(np.argmax(x_ch[q]))
            xa = float(log_T_ch[i_peak])
            if xa < 4.05 or xa > 7.95:
                continue
            line_annotate(_ion_label(element, q), ch_lines[q],
                          x=xa, xytext=(0, 4),
                          fontsize='x-small',
                          color=cmap(q / Z), ha='center',
                          path_effects=stroke)
        ax.plot([], [], 'k-',  lw=1.4, label='CHIANTI v11')
        ax.plot([], [], 'k--', lw=1.0, label='Gnat & Sternberg 2007')
        ax.legend(fontsize='xx-small', loc='lower left',
                  framealpha=0.7)
        ax.set_title(f'{element} (Z={Z})')
        ax.set_xlim(4.0, 8.0)
        ax.set_ylim(1e-5, 2)
        ax.grid(True, which='both', alpha=0.3)
    for ax in axes[-3:]:
        ax.set_xlabel(r'$\log_{10}(T/{\rm K})$')
    for ax in axes[::3]:
        ax.set_ylabel(r'$x_q = n(X^{q+})/n(X)$')
    fig.suptitle('CIE x_q(T): CHIANTI v11 (solid) '
                 'vs Gnat & Sternberg 2007 (dashed)')
    fig.tight_layout()
    fig.savefig(figures_dir / fig_name, dpi=150)
    plt.close(fig)


def test_plot_ioneq_chianti_vs_gs07(figures_dir, save_figures):
    if not save_figures:
        pytest.skip('plot generation disabled (--no-figures)')
    gs07 = _read_gs07()
    _make_panel(figures_dir, COMPARE_FIG1,
                'ioneq_chianti_v11_vs_GS07_light.png', gs07)
    _make_panel(figures_dir, COMPARE_FIG2,
                'ioneq_chianti_v11_vs_GS07_heavy.png', gs07)
