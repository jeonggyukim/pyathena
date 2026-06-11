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
COMPARE_FIG1 = ['H',  'He', 'C', 'N', 'O', 'Ne']
COMPARE_FIG2 = ['Mg', 'Si', 'S', 'Ar', 'Ca', 'Fe']
COMPARE = COMPARE_FIG1 + COMPARE_FIG2


GS07_PATH = Path(__file__).parent.parent.parent / 'data' / \
    'microphysics' / 'Gnat_Sternberg07_cie_ion_frac.txt'
_GS07_ELEMENT_ORDER = [
    ('H', 1), ('He', 2), ('C', 6), ('N', 7), ('O', 8),
    ('Ne', 10), ('Mg', 12), ('Si', 14), ('S', 16), ('Fe', 26),
]


def _read_gs07_all():
    """Parse GS07 ASCII table into {element: (T_K, x_q_array)}."""
    with open(GS07_PATH) as f:
        lines = f.read().splitlines()
    data_lines = []
    for ln in lines:
        s = ln.lstrip()
        if not s:
            continue
        first = s.split()[0]
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
    for element, Z in _GS07_ELEMENT_ORDER:
        ncol = Z + 1
        out[element] = (T, arr[:, col:col + ncol].T)
        col += ncol
    return out


def _load(element, gs07_cache):
    """Return (log_T, x_q_chianti, log_T_gs, x_q_gs, x_q_local_ct).
    GS07 is None when the element isn't in GS07's published set
    (Ar, Ca)."""
    from pyathena.photchem.data.build_ioneq_tables import read_ioneq
    base = Path(__file__).parent.parent.parent / 'data' / \
        'microphysics' / 'chianti_v11'
    d_ch = read_ioneq(str(base / f'ioneq_{element}.txt'))
    d_lo_ct = read_ioneq(
        str(base / f'ioneq_ct_{element}.txt'))
    if element in gs07_cache:
        T_gs, x_gs = gs07_cache[element]
        log_T_gs = np.log10(T_gs)
    else:
        log_T_gs, x_gs = None, None
    return (d_ch['log_T'], d_ch['x_q'],
            log_T_gs, x_gs, d_lo_ct['x_q'])


def _ion_label(element, q):
    """LaTeX label for charge state q: '0', '+', '2+', ..., 'Z+'.
    Element name dropped because plot title already names it."""
    if q == 0:
        return r'$0$'
    elif q == 1:
        return r'$+$'
    else:
        return rf'${q}+$'


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
    gs07_cache = _read_gs07_all()
    for ax in axes[len(elements_subset):]:
        ax.set_visible(False)
    for ax_idx, (ax, element) in enumerate(
            zip(axes, elements_subset)):
        log_T, x_ch, log_T_gs, x_gs, x_lo_ct = _load(
            element, gs07_cache)
        Z = x_ch.shape[0] - 1
        import cmasher as cmr
        cmap = cmr.combine_cmaps(
            cmr.get_sub_cmap(cmr.ocean, 0.15, 0.85),
            cmr.get_sub_cmap(cmr.amber, 0.15, 0.85))
        ch_lines = []  # store CHIANTI Line2D for annotation
        for q in range(Z + 1):
            color = cmap(q / Z)
            ln_ch, = ax.semilogy(log_T, x_ch[q], '-',
                                 color=color, lw=1.4)
            # GS07 (Gnat & Sternberg 2007) ion fractions: thick
            # dashed with reduced alpha. Skipped for Ar/Ca since
            # GS07 doesn't include them.
            if x_gs is not None and q < x_gs.shape[0]:
                ax.semilogy(log_T_gs, x_gs[q], '--', color=color,
                            lw=3.5, alpha=0.45)
            ax.semilogy(log_T, x_lo_ct[q], ':',  color=color, lw=1.0)
            ch_lines.append(ln_ch)
        # Inline labels per charge state on the CHIANTI curve.
        # Place at peak log-T (within the plot range).
        for q in range(Z + 1):
            ymax = np.max(x_ch[q])
            if ymax < 1e-3:
                # Skip stages that never reach a visible fraction.
                continue
            # Use FWHM midpoint as label position. Handles all curve
            # shapes uniformly:
            #   - bell curve (mid-q): midpoint = argmax
            #   - saturated rising (fully stripped): midpoint inside
            #     the high-T plateau
            #   - saturated falling (neutral): midpoint inside the
            #     low-T plateau
            half = 0.5 * ymax
            mask = x_ch[q] >= half
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            i_mid = (idx[0] + idx[-1]) // 2
            # Clamp to the visible panel range so labels don't clip.
            x_annot = float(np.clip(log_T[i_mid], 3.1, 7.9))
            color = cmap(q / Z)
            label = _ion_label(element, q)
            line_annotate(label, ch_lines[q], x=x_annot,
                          xytext=(0, 4), fontsize='small',
                          color=color, ha='center',
                          path_effects=stroke)
        # Method legend only in the first (upper-left) panel.
        if ax_idx == 0:
            ax.plot([], [], 'k-',  lw=1.4,
                    label='ours (CHIANTI v11)')
            ax.plot([], [], 'k:',  lw=1.0, label='ours, + CT')
            ax.plot([], [], 'k--', lw=3.5, alpha=0.45,
                    label='Gnat & Sternberg 2007')
            ax.legend(fontsize='large', loc='lower right',
                      framealpha=0.7, handlelength=3.5)
        ax.set_title(f'{element} (Z={Z})')
        ax.set_xlim(3.0, 8.0)
        ax.set_ylim(1e-5, 2)
        ax.grid(True, which='both', alpha=0.3)
    # Force every visible axis to show xticklabels + xlabel (sharex
    # hides them by default in interior subplots; override).
    for ax in axes[:len(elements_subset)]:
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_xlabel(r'$\log_{10}\,T\,[{\rm K}]$')
    for ax in axes[::3]:
        ax.set_ylabel(r'$x_q = n(X^{q+})/n(X)$')
    fig.suptitle(
        r'CIE ionization fractions $x_q(T)$: '
        r'CHIANTI v11 (ours, solid) vs Gnat & Sternberg 2007 '
        r'(thick dashed)' '\n'
        r'Dotted: our $x_q$ including charge transfer with H')
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
