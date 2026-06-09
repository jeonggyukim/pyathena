"""Benchmark: CIE x_q(T) from CHIANTI v11.0.2 (gold standard) vs our
own CIE computed from `pyathena.microphysics` k_ci + alpha_rec.

The local computation uses only the pyathena rate functions
(`CollIonRate.get_ci_rate`, `RecRate.get_rec_rate`). No charge
transfer is included. Discrepancies vs CHIANTI quantify the
internal-rate-data quality of pyathena vs the modern reference.
"""

from pathlib import Path
import pytest
import numpy as np


# GS07 element set is exactly these 10 (no Ar, no Ca):
COMPARE_FIG1 = ['H',  'He', 'C', 'N', 'O']
COMPARE_FIG2 = ['Ne', 'Mg', 'Si', 'S', 'Fe']
COMPARE = COMPARE_FIG1 + COMPARE_FIG2


def _load(element):
    """Return (log_T, x_q_chianti, x_q_local, x_q_local_ct) for
    one element."""
    from pyathena.photchem.data.build_ioneq_tables import read_ioneq
    base = Path(__file__).parent.parent.parent / 'pyathena' / \
        'photchem' / 'data'
    d_ch = read_ioneq(str(base / f'ioneq_{element}.txt'))
    d_lo = read_ioneq(str(base / f'ioneq_local_{element}.txt'))
    d_lo_ct = read_ioneq(
        str(base / f'ioneq_local_ct_{element}.txt'))
    return (d_ch['log_T'], d_ch['x_q'],
            d_lo['x_q'], d_lo_ct['x_q'])


def _ion_label(element, q):
    """LaTeX label for charge state q of element."""
    if q == 0:
        return rf'${{\rm {element}}}$'
    elif q == 1:
        return rf'${{\rm {element}}}^+$'
    else:
        return rf'${{\rm {element}}}^{{{q}+}}$'


def _make_panel(figures_dir, elements_subset, fig_name):
    """Build one 2x3 panel grid for the given element subset. Inline
    labels via `line_annotate` with white-stroke patheffects; one
    label per charge state placed at the T where x_q is near its
    peak (and above a visibility floor)."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    import numpy as np
    from pyathena.plt_tools.line_annotation import line_annotate

    stroke = [path_effects.withStroke(linewidth=2.0, foreground='white')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9),
                             sharex=True, sharey=True)
    axes = axes.flatten()
    for ax in axes[len(elements_subset):]:
        ax.set_visible(False)
    for ax, element in zip(axes, elements_subset):
        log_T, x_ch, x_lo, x_lo_ct = _load(element)
        Z = x_ch.shape[0] - 1
        cmap = plt.get_cmap('viridis')
        ch_lines = []  # store CHIANTI Line2D for annotation
        for q in range(Z + 1):
            color = cmap(q / Z)
            ln_ch, = ax.semilogy(log_T, x_ch[q],    '-',
                                 color=color, lw=1.4)
            ax.semilogy(log_T, x_lo[q],    '--', color=color, lw=1.0)
            ax.semilogy(log_T, x_lo_ct[q], ':',  color=color, lw=1.0)
            ch_lines.append(ln_ch)
        # Inline labels per charge state on the CHIANTI curve.
        # Place at peak log-T (within the plot range).
        for q in range(Z + 1):
            ymax = np.max(x_ch[q])
            if ymax < 1e-3:
                # Skip annotating barely-present stages.
                continue
            i_peak = int(np.argmax(x_ch[q]))
            x_annot = log_T[i_peak]
            # Skip if peak sits at the edge of the plotted T range.
            if x_annot < 4.05 or x_annot > 7.95:
                continue
            color = cmap(q / Z)
            label = _ion_label(element, q)
            line_annotate(label, ch_lines[q], x=x_annot,
                          xytext=(0, 4), fontsize='small',
                          color=color, ha='center',
                          path_effects=stroke)
        # Method legend (line styles) on every panel, lower-left.
        ax.plot([], [], 'k-',  lw=1.4, label='CHIANTI v11')
        ax.plot([], [], 'k--', lw=1.0, label='ours, no CT')
        ax.plot([], [], 'k:',  lw=1.0, label='ours, + CT')
        ax.legend(fontsize='xx-small', loc='lower left',
                  framealpha=0.7)
        ax.set_title(f'{element} (Z={Z})')
        ax.set_xlim(4.0, 8.0)
        ax.set_ylim(1e-5, 2)
        ax.grid(True, which='both', alpha=0.3)
    # Force every visible axis to show xticklabels + xlabel (sharex
    # hides them by default in interior subplots; override).
    for ax in axes[:len(elements_subset)]:
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_xlabel(r'$\log_{10}\,T\,[{\rm K}]$')
    for ax in axes[::3]:
        ax.set_ylabel(r'$x_q = n(X^{q+})/n(X)$')
    fig.suptitle('CIE x_q(T): CHIANTI v11 vs pyathena own rates '
                 '(without / with CT). H ionization for CT weighting '
                 'taken from CHIANTI v11.')
    fig.tight_layout()
    fig.savefig(figures_dir / fig_name, dpi=300)
    plt.close(fig)


def test_plot_chianti_vs_local(figures_dir, save_figures):
    """Two figures (light + heavy elements), 6 panels each."""
    if not save_figures:
        pytest.skip('plot generation disabled (--no-figures)')
    _make_panel(
        figures_dir, COMPARE_FIG1,
        'ioneq_chianti_v11_vs_local_light.png')
    _make_panel(
        figures_dir, COMPARE_FIG2,
        'ioneq_chianti_v11_vs_local_heavy.png')
