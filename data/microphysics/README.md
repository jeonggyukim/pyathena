# pyathena/data/microphysics

Microphysics data files used by `pyathena.microphysics`. Tables here
are read at runtime by the rate / cooling modules and at build time
by the scripts under `pyathena/microphysics/chianti_v11/`.

## Top-level files

- `coll_ion.dat` -- Voronov 1997 (At. Data Nucl. Data Tables 65, 1)
  collisional ionization fit coefficients. Used by
  `pyathena.microphysics.ci_rate`. Source: Cloudy.

## chianti_v11/

CHIANTI v11.0.2 derived tables, built by the scripts under
`pyathena/microphysics/chianti_v11/`. Followed elements:
H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe.

### Ionization equilibrium

- `ioneq_<X>.txt` -- CIE ionization fractions x_q(T) from
  `ChiantiPy.core.ioneq.calculate`. 101 log-spaced T points from
  1e3 to 1e9 K.
- `ioneq_ct_<X>.txt` -- CIE ionization fractions with charge
  transfer included (pyathena `evolve_one_species` solver).

### Cooling tables

Per-ion cooling efficiency Lambda_q(T) in units of erg cm^3 / s, at
the low-density limit, with no abundance or ionization-fraction
weighting included.

- `cool_<X>.txt`    -- total: BB + 2g + FF + FB
- `cool_BB_<X>.txt` -- bound-bound line emission
- `cool_2g_<X>.txt` -- two-photon continuum
- `cool_FF_<X>.txt` -- free-free bremsstrahlung
- `cool_FB_<X>.txt` -- free-bound recombination radiation
  (attributed to the post-recombination ion, Gnat & Ferland 2012
  convention)

The split lets time-dependent multi-ion chemistry weight FB by
x_(q+1) (the recombining ion) and BB / 2g / FF by x_q (the emitting
ion).

### Per-ion atomic data

- `<ion>.txt` -- per-ion atomic data (energies, statistical weights,
  A coefficients, Upsilon collision strengths) for the 5-level
  coolant solver in `pyathena.microphysics.coolants`. Extracted
  from CHIANTI v11 by `build_atomic.py`. Examples: `o_3.txt`,
  `c_2.txt`, `n_3.txt`.
