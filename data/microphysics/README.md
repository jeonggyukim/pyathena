# pyathena/data/microphysics

Microphysics data files used by `pyathena.microphysics`. Tables here
are read at runtime by the rate / cooling modules and at build time
by the scripts under `pyathena/microphysics/chianti_v11/`.

## Top-level files

- `verner96_photx.dat` -- Verner et al. 1996 (ApJ 465, 487)
  photoionization cross sections. Used by
  `pyathena.microphysics.photx`.
- `badnell_rr_2023.dat` -- Badnell radiative recombination fit
  coefficients (Badnell 2006 ApJS 167, 334; updated 2023).
  `badnell_rr.dat` is the older version, kept for back-compat.
- `badnell_dr_C_2023.dat`, `badnell_dr_E_2023.dat` -- Badnell
  dielectronic recombination C and E coefficients (Badnell 2006
  A&A 447, 389; updated 2023). `badnell_dr_*.dat` are older
  versions.
- `Gnat_Ferland12_Table2.txt` -- Gnat & Ferland 2012 (ApJS 199,
  20) Table 2 (per-element CIE cooling efficiencies). Used by
  `pyathena.microphysics.cool_gnat12.CoolGnat12`.
- `Gnat_Sternberg07_cie_ion_frac.txt` -- Gnat & Sternberg 2007
  (ApJS 168, 213) CIE ionization fractions, used as the reference
  for CIE plots.
- `Grackle_equillibrium_cooling_*.dat` -- Grackle (Smith et al.
  2017 MNRAS 466, 2217) equilibrium cooling functions at Z = 0.1
  and 1.0 solar.

## cloudy/

Tables extracted from Cloudy (Ferland et al. 2017 RMxAA 53, 385).

- `coll_ion.dat` -- Voronov 1997 (At. Data Nucl. Data Tables 65,
  1) collisional ionization fit coefficients. Used by
  `pyathena.microphysics.ci_rate`.
- `ctiondata.dat` -- charge-transfer ionization (X+ + Y -> X +
  Y+) rate coefficients. Used by `pyathena.microphysics.ct_rate`.
- `ctrecombdata.dat` -- charge-transfer recombination (X + Y+ ->
  X+ + Y) rate coefficients. Used by
  `pyathena.microphysics.ct_rate`.

## ugacxdb/

UGA Charge Transfer Database (https://sites.physast.uga.edu/ugacxdb/).
Used by `pyathena.microphysics.ct_rate`.

- `ct_h2.dat`   -- H+ + H2 isotopomer charge transfer fits from
  Wang & Stancil 2002 (Physica Scripta T96, 72).
- `cti_hyd.dat` -- charge transfer ionization with H I
  (Kingdon & Ferland 1996 ApJ 442, 714 fits).
- `ctr_hyd.dat` -- charge transfer recombination with H I
  (Kingdon & Ferland 1996).
- `cti_he.dat`  -- charge transfer ionization with He I.
- `ctr_he.dat`  -- charge transfer recombination with He I.

## Gnat_Ferland12_tables/

Per-element ASCII tables from Gnat & Ferland 2012 (ApJS 199, 20).
One file per element (Hydrogen.txt, Helium.txt, ..., Zinc.txt)
containing per-ion CIE cooling efficiency Lambda_q(T) plus the
final CIE-weighted column. Format documented in the GF12 paper.

## Wiersma09/

Wiersma et al. 2009 (MNRAS 393, 99) cooling tables. Solar-scaled
metallicity-dependent cooling functions used in cosmological
simulations.

## chianti_v11/

CHIANTI v11.0.2 derived tables, built by the scripts under
`pyathena/microphysics/chianti_v11/`. Followed elements:
H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe.

### Ionization equilibrium

- `ioneq_<X>.txt` -- CIE ionization fractions x_q(T) from
  `ChiantiPy.core.ioneq.calculate`. Built by `build_ioneq.py`.
  101 log-spaced T points from 1e3 to 1e9 K.
- `ioneq_ct_<X>.txt` -- CIE ionization fractions with charge
  transfer (pyathena `evolve_one_species`). Built by
  `build_ioneq_ct.py`.

### Cooling tables

Per-ion cooling efficiency Lambda_q(T) in units of erg cm^3 / s, at
the low-density limit, with no abundance or ionization-fraction
weighting included.

- `cool_<X>.txt`    -- total: bound-bound + two-photon + free-free
  + free-bound
- `cool_BB_<X>.txt` -- bound-bound line emission
- `cool_2g_<X>.txt` -- two-photon continuum
- `cool_FF_<X>.txt` -- free-free bremsstrahlung
- `cool_FB_<X>.txt` -- free-bound recombination radiation
  (attributed to the post-recombination ion, Gnat & Ferland 2012
  convention)

The split lets time-dependent multi-ion chemistry weight free-bound
by x_(q+1) (the recombining ion) and bound-bound / two-photon /
free-free by x_q (the emitting ion).

### Per-ion atomic data

- `<ion>.txt` -- per-ion atomic data (energies, statistical weights,
  A coefficients, Upsilon collision strengths) for the 5-level
  coolant solver in `pyathena.microphysics.coolants`. Extracted
  from CHIANTI v11 by `build_atomic.py`. Examples: `o_3.txt`,
  `c_2.txt`, `n_3.txt`.
