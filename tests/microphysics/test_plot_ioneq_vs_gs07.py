"""Benchmark: CIE ionization fractions x_q(T) per element from our
CHIANTI v11 ioneq tables vs the Gnat & Sternberg 2007 (ApJS 168 213)
reference tables shipped with pyathena.

GS07 used Cloudy + atomic data from that vintage; the discrepancies
vs our CHIANTI v11 tables quantify the atomic-data updates since
2007.
"""

import os
import warnings
from pathlib import Path
import pytest
import numpy as np


GS07_PATH = Path(__file__).parent.parent.parent / 'data' / \
    'microphysics' / 'Gnat_Sternberg07_cie_ion_frac.txt'

# Element configuration in GS07: ordering of charge-state columns
# (Z+1 columns per element, q=0..Z), in declaration order.
GS07_ELEMENT_ORDER = [
    ('H', 1), ('He', 2), ('C', 6), ('N', 7), ('O', 8),
    ('Ne', 10), ('Mg', 12), ('Si', 14), ('S', 16), ('Ar', 18),
    ('Ca', 20), ('Fe', 26),
]
# Followed elements to compare. Subset of the above + that we have
# in our CHIANTI v11 build.
COMPARE = ['C', 'N', 'O', 'S', 'Ne', 'Ar']


def _read_gs07():
    """Read GS07 ASCII table; return dict {element: (T, x_q array)}."""
    # First find the line where data starts (after two long
    # dashed lines following the byte-by-byte description block).
    out = {}
    raw_rows = []
    with open(GS07_PATH) as f:
        lines = f.read().splitlines()
    # The data lines start with a temperature value (digit followed
    # by 'e' or '.'); skip everything before the first such line.
    data_lines = []
    for ln in lines:
        s = ln.lstrip()
        if not s:
            continue
        first = s.split()[0]
        if first.replace('.', '').replace('e', '').replace('+', '')\
                .replace('-', '').isdigit() and 'e' in first.lower():
            data_lines.append(ln)
    arr = np.array([[float(v) for v in ln.split()]
                    for ln in data_lines])
    T = arr[:, 0]
    col = 1
    for element, Z in GS07_ELEMENT_ORDER:
        ncol = Z + 1
        x_q = arr[:, col:col + ncol].T   # (Z+1, NT)
        out[element] = (T, x_q)
        col += ncol
    return out


def _read_our_ioneq(element):
    """Read our CHIANTI v11 ioneq table."""
    from pyathena.photchem.data.build_ioneq_tables import read_ioneq
    path = Path(__file__).parent.parent.parent / 'pyathena' / \
        'photchem' / 'data' / f'ioneq_{element}.txt'
    d = read_ioneq(str(path))
    return d['log_T'], d['x_q']


def test_plot_ioneq_comparison(figures_dir, save_figures):
    """Per-element plot of x_q(T) for our CHIANTI v11 vs GS07."""
    if not save_figures:
        pytest.skip('plot generation disabled (--no-figures)')
    import matplotlib.pyplot as plt
    gs07 = _read_gs07()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9),
                             sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, element in zip(axes, COMPARE):
        Z = dict(GS07_ELEMENT_ORDER)[element]
        # GS07 values
        T_gs, x_q_gs = gs07[element]
        log_T_gs = np.log10(T_gs)
        # Ours
        log_T_o, x_q_o = _read_our_ioneq(element)
        # Plot each charge state q = 0..Z
        cmap = plt.get_cmap('viridis')
        for q in range(Z + 1):
            color = cmap(q / Z)
            ax.semilogy(log_T_gs, x_q_gs[q], '-',
                        color=color, lw=1.0,
                        label=f'q={q}' if element == COMPARE[0] else None)
            ax.semilogy(log_T_o, x_q_o[q], '--',
                        color=color, lw=1.0)
        ax.set_title(f'{element} (Z={Z})')
        ax.set_ylim(1e-5, 2)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim(4.0, 8.0)
    for ax in axes[-3:]:
        ax.set_xlabel(r'$\log_{10}(T/{\rm K})$')
    for ax in axes[::3]:
        ax.set_ylabel(r'$x_q = n(X^{q+})/n(X)$')
    # Single combined legend in the first subplot.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, fontsize='xx-small',
                       loc='lower left', ncol=2)
    fig.suptitle('CIE ionization fractions: '
                 'CHIANTI v11.0.2 (dashed) vs Gnat & Sternberg 2007 '
                 '(solid)')
    fig.tight_layout()
    fig.savefig(figures_dir / 'ioneq_chianti_v11_vs_GS07.png',
                dpi=150)
    plt.close(fig)
