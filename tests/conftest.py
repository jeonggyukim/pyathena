"""Pytest configuration + shared fixtures for the pyathena test suite.

Adds two pieces of infrastructure used across the suite:

1. `figures_dir` fixture -> `tests/figures/` (created on demand,
   git-ignored via `tests/figures/.gitignore`).
2. `save_figures` fixture -> bool that defaults to True; flipped to
   False by passing `--no-figures` on the pytest command line.

Tests that produce sanity-visualization plots typically combine both:

    def test_something(figures_dir, save_figures):
        ...
        if save_figures:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ...
            fig.savefig(figures_dir / "my_plot.png", dpi=120)
            plt.close(fig)

This keeps plot generation on-by-default (helpful for quickly seeing
what the test sees) while still allowing a fast "tests only" run via
`pytest tests/ --no-figures`.
"""

from pathlib import Path
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--no-figures", action="store_true", default=False,
        help="Skip producing diagnostic plots from tests "
             "(plots are on by default)."
    )


@pytest.fixture(scope="session")
def figures_dir():
    """Path to `tests/figures/` (created if absent). Output not
    tracked by git -- see `tests/figures/.gitignore`.
    """
    here = Path(__file__).parent.resolve()
    fdir = here / "figures"
    fdir.mkdir(exist_ok=True)
    return fdir


@pytest.fixture(scope="session")
def save_figures(request):
    """True unless `pytest --no-figures` was passed."""
    return not request.config.getoption("--no-figures")


# -----------------------------------------------------------------------
# Per-ion color convention, mirroring `PhotChem._set_colors` in
# `pyathena/microphysics/photchem.py`. Same ion gets the same color
# across every plot in the suite, so cross-figure visual comparison
# stays consistent.
#
# Per-element colormap from photchem.py:
#   H  -> Greys, He -> Purples, C -> Blues,
#   N  -> Oranges, O -> Greens, S -> Reds
# Within an element the intensity scales with `num_ions - q` (neutral
# darkest within its colormap). Each colormap is sampled at the same
# fractional positions used by photchem.py, so an ion plotted in a
# test figure visually matches the same ion plotted in a notebook
# that uses `PhotChem.plt_rate_coeffs` or `plt_sed_sigma_pi`.
# -----------------------------------------------------------------------

# (element, num_ions_to_plot) -- the denominator of the color norm.
# Choose num_ions to match the photchem.py default species sets so the
# color levels line up exactly when both are used.
_NUM_IONS_DEFAULT = {
    "H": 2, "He": 3, "C": 4, "N": 4, "O": 4, "S": 5,
}

_CMAP_NAME = {
    "H": "Greys", "He": "Purples", "C": "Blues",
    "N": "Oranges", "O": "Greens", "S": "Reds",
}

_Z_TO_ELEMENT = {1: "H", 2: "He", 6: "C", 7: "N", 8: "O", 16: "S"}


def ion_color(Z, q):
    """Return matplotlib color hex string for ion (Z, q).

    Matches `pyathena/microphysics/photchem.py:_set_colors` so the
    same ion always gets the same color whether plotted from a test
    or from `PhotChem.plt_rate_coeffs` / `plt_sed_sigma_pi`.
    """
    import matplotlib as mpl
    elem = _Z_TO_ELEMENT[Z]
    num_ions = _NUM_IONS_DEFAULT[elem]
    cmap = mpl.colormaps[_CMAP_NAME[elem]]
    norm = mpl.colors.Normalize(0, num_ions)
    return mpl.colors.rgb2hex(cmap(norm(num_ions - q)))


def ion_label(Z, q):
    """Roman-numeral ion label, e.g., (Z=8, q=1) -> '$\\rm O$\\,II'.

    Matches the LaTeX label format in `PhotChem._set_colors`.
    """
    elem = _Z_TO_ELEMENT[Z]
    return r"$\rm " + elem + r"$\," + chr(0x215F + q + 1)


@pytest.fixture(scope="session")
def ion_colors():
    """Fixture form of `ion_color` -- returns the callable so tests
    can grab it without an explicit import.
    """
    return ion_color


@pytest.fixture(scope="session")
def ion_labels():
    """Fixture form of `ion_label`."""
    return ion_label
