# pyathena tests

Pytest suite covering pyathena's physics modules. New suites land here
under subdirectories that mirror the package layout, so adding more is
just dropping a `test_*.py` file in the right subdir.

## Layout

```
tests/
  microphysics/                    # microphysics rate / cross-section tests
    test_ct_rate_balance.py        # charge-exchange (ct_rate.py)
    test_photx_sigma.py            # photoionization cross section (photx.py)
    test_rec_rate.py               # radiative + dielectronic recomb (rec_rate.py)
    test_ci_rate.py                # collisional ionization (ci_rate.py)
  ...future suites here, e.g.:
  io/                              # data I/O
  fields/                          # diagnostic fields
  obs/                             # observability calculations
```

## Run

All tests (also produces diagnostic plots under `tests/figures/`):
```bash
conda activate pyathena
pytest tests/ -v
```

Skip plot production (faster):
```bash
pytest tests/ --no-figures -v
```

Just one module:
```bash
pytest tests/microphysics/ -v
```

Single file:
```bash
pytest tests/microphysics/test_ct_rate_balance.py -v
```

Single test:
```bash
pytest tests/microphysics/test_rec_rate.py::test_HII_caseB_at_T10000 -v
```

## Diagnostic plots

Some tests emit a PNG to `tests/figures/` as a byproduct (helpful for
quickly seeing what the test sees -- e.g., the O-H CT cross-source
comparison plot mirrors the standalone notebook). Plot generation is
ON by default; disable per-run with `pytest --no-figures`.

`tests/figures/` is git-ignored (see `tests/figures/.gitignore`) so
committed plots don't pollute the repo.

To produce a figure from a new test:
```python
def test_something(figures_dir, save_figures):
    ...
    assert ...
    if save_figures:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ...
        fig.savefig(figures_dir / "my_plot.png", dpi=200)
        plt.close(fig)
```
The `figures_dir` and `save_figures` fixtures are defined in
`tests/conftest.py`.

## What the tests are for

Each test file pins the current behavior of one module via small,
fast regression checks. Three categories appear repeatedly:

1. **Smoke** — output is finite, positive (or non-negative), no
   exceptions. Catches "method renamed and forgot to update caller"
   bugs at the rate level.
2. **Sign-convention probe** — for endothermic rates with a
   Boltzmann factor `exp(-dE/T)`, growth-vs-T direction is the
   leading sign indicator. Catches sign-flip regressions.
3. **Reference-value spot-check** — at one or two carefully chosen
   `(Z, N, T)` points, compare to published reference rates
   (Verner+96, Badnell, Draine 2011, Voronov 1997, Kingdon&Ferland
   1996, Pequignot 1996). Tolerances are deliberately loose
   (typically 5-20%) so that the test catches a 2x bug without
   triggering false failures from minor coefficient updates.

Each test file's module docstring documents:
- the API conventions it verifies (notably which `(Z, N)` indexes
  which ion stage — this differs between rec/ci/ct);
- which reference papers / equation numbers the spot-checks come
  from;
- known data-quirks pinned (e.g., `S I` charge-exchange ionization
  is a placeholder `1e-14` in the Cloudy data).

## Adding tests

Conventions:
- One test per behavior. Use `pytest.mark.parametrize` to apply the
  same behavior across many ions or temperatures.
- Use `@pytest.fixture(scope="module")` for objects that take time
  to build (the rate-class constructors read data files).
- Add a one-line docstring per test describing the physical content.
- For reference-value tests, cite the source paper + equation /
  table number in the docstring AND keep the tolerance explicit.
- If the test pins a known-buggy or placeholder value, label it
  `test_..._is_placeholder` and document the upgrade path in the
  docstring.
