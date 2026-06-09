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


def _make_panel(figures_dir, elements_subset, fig_name):
    """Build one 2x3 panel grid for the given element subset; if
    fewer than 6 elements, hide the trailing axes."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 9),
                             sharex=True, sharey=True)
    axes = axes.flatten()
    # Hide any axes beyond len(elements_subset)
    for ax in axes[len(elements_subset):]:
        ax.set_visible(False)
    for ax, element in zip(axes, elements_subset):
        log_T, x_ch, x_lo, x_lo_ct = _load(element)
        Z = x_ch.shape[0] - 1
        cmap = plt.get_cmap('viridis')
        for q in range(Z + 1):
            color = cmap(q / Z)
            ax.semilogy(log_T, x_ch[q],    '-',  color=color, lw=1.4)
            ax.semilogy(log_T, x_lo[q],    '--', color=color, lw=1.0)
            ax.semilogy(log_T, x_lo_ct[q], ':',  color=color, lw=1.0)
        # Per-charge-state legend on the first panel of each figure.
        if element == elements_subset[0]:
            for q in range(Z + 1):
                ax.plot([], [], color=cmap(q / Z), lw=1.4,
                        label=f'q={q}')
            ax.legend(fontsize='xx-small', loc='lower left',
                      ncol=2 if Z + 1 > 6 else 1)
        # Method legend on a different panel.
        if element == elements_subset[2]:
            ax.plot([], [], 'k-',  lw=1.4, label='CHIANTI v11')
            ax.plot([], [], 'k--', lw=1.0, label='ours (no CT)')
            ax.plot([], [], 'k:',  lw=1.0, label='ours (+ CT)')
            ax.legend(fontsize='xx-small', loc='lower left')
        ax.set_title(f'{element} (Z={Z})')
        ax.set_xlim(4.0, 8.0)
        ax.set_ylim(1e-5, 2)
        ax.grid(True, which='both', alpha=0.3)
    for ax in axes[-3:]:
        ax.set_xlabel(r'$\log_{10}(T/{\rm K})$')
    for ax in axes[::3]:
        ax.set_ylabel(r'$x_q = n(X^{q+})/n(X)$')
    fig.suptitle('CIE x_q(T): CHIANTI v11 vs pyathena own rates '
                 '(without / with CT). H ionization for CT weighting '
                 'taken from CHIANTI v11.')
    fig.tight_layout()
    fig.savefig(figures_dir / fig_name, dpi=150)
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
