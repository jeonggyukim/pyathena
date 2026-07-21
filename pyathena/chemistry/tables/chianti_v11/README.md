# pyathena.chemistry.tables.chianti_v11

Build scripts for the CHIANTI v11 derived atomic-data tables consumed
at runtime by `pyathena.chemistry.coolants` and the cooling-table
readers. The tables themselves live in `data/microphysics/chianti_v11/`.

## Build-time requirements

The build scripts here read the CHIANTI v11 atomic database via
[ChiantiPy](https://chiantipy.readthedocs.io/en/latest/). The runtime
readers (`IonCoolant`, the `read_*` parsers) do not. Install the
build-time extra with

    pip install pyathena[tables]

and point ChiantiPy at the CHIANTI data directory by setting the
`XUVTOP` environment variable, e.g.

    export XUVTOP=$HOME/Dropbox/Projects/CHIANTI_db

End-user runtime code that only reads pre-built tables does not need
ChiantiPy or `XUVTOP`.

## Build scripts

| Module | What it does |
|---|---|
| `build_ioneq_tables.py` | CIE ionization fractions `x_q(T)` from ChiantiPy. Output: `data/microphysics/chianti_v11/ioneq_<Element>.txt` (CHIANTI v11 reference, no CT). |
| `build_ioneq_ct.py` | Same as above + charge transfer with H. Output: `data/microphysics/chianti_v11/ioneq_ct_<Element>.txt`. Trace-element approximation. |
| `build_cool_tables.py` | Per-ion cooling efficiency `Lambda_q(T)` from ChiantiPy. Includes bound-bound + two-photon + free-free + free-bound. FB attributed to post-recombination ion (GF12 convention). Output: `data/microphysics/chianti_v11/cool_<Element>.txt`. |
| `build_cool_fast.py` | Experimental: batched matrix solve for BB cooling. ~6.5x faster than `build_cool_tables.py`. Currently mismatches canonical builder by ~2 dex for BB-dominated mid-q ions; needs debugging before becoming default. |
| `build_chianti_tables.py` | Per-ion atomic data extracted from CHIANTI (energy levels, A coefficients, Upsilon tables for electron + proton impact). Output: `<element>_<charge>.txt` files (e.g., `o_3.txt` for OIII). Used by the 5-level coolant solver. |
| `build_all.py` | Convenience: runs ioneq, ioneq_ct, and cool builders in sequence. Requires `XUVTOP` set. |

## CHIANTI atomic data file conventions

The CHIANTI database stores per-ion atomic data in a directory per
(element, charge) pair, e.g.
`$XUVTOP/o/o_3/o_3.elvlc` for OIII. File extensions:

| Extension | Contents | Format |
|---|---|---|
| `.elvlc` | **El**ectronic **lv**l **c**onfiguration | Energy levels: index, configuration, term, J, energy (cm-1), statistical weight g = 2J+1. |
| `.wgfa` | Weighted oscillator strengths + **gf** + **a**-values (radiative decay rates) | One row per radiative transition: `lvl_lower lvl_upper wvl(A) gf A_ji(s-1)`. The "wgfa" name historically refers to these four fields. |
| `.scups` | **Sc**aled **ups**ilons (collisional excitation strengths) | Burgess-Tully scaled electron-impact effective collision strengths Y_ji(T) on a transformed grid. Spline-evaluated to get Y at any T. |
| `.psplups` | **P**roton **splups**: proton-impact collision strengths | Same format as `.scups` but for proton-impact excitation. Only matters for inner-shell of low-q heavy ions. |
| `.rrparams` | RR (radiative recombination) coefficients | Verner & Ferland 1996 fit params. |
| `.drparams` | DR (dielectronic recombination) coefficients | Badnell sum-of-exponentials fit params. |
| `.diparams` | DI (direct ionization) + EA coefficients | Dere 2007 fit params. |
| `.auto` | Autoionization rates | For inner-shell Auger physics. |
| `.fblvl` | Free-bound level info | For recombination continuum (used by `freeBoundLoss`). |
| `.ip` | Ionization potential | One float per ion (in eV). |

### Notation

- **Wgfa vs wgfa**: same content, different conventions. `.wgfa` is the on-disk filename suffix; `ion.Wgfa` is the ChiantiPy attribute name (Python dict with the parsed file).
- **Scups**: ChiantiPy attribute holding the parsed `.scups` file. After `ion.upsilonDescale()`, the spline is evaluated at the requested T grid; result accessible as `ion.Upsilon['upsilon']`.
- **Statistical weight g**: `g = 2J + 1` where J is the total angular momentum. Loaded from `.elvlc` as the `mult` column. For Fe^0 ground (^3F_2, J=2): g = 5. For H^0 ground (^2S_{1/2}, J=1/2): g = 2.

### Collisional rate formula (Draine 2011 Eq 17.10)

```
q_down(j -> i)  =  beta * Y_ji(T) / (g_j * sqrt(T))
q_up(i -> j)    =  q_down * (g_j / g_i) * exp(-(E_j - E_i) / kT)
```

where beta = 8.629e-8 (collision rate prefactor in cgs). The ratio of g's enforces detailed balance.

## Cooling channel conventions

Total per-ion cooling efficiency Lambda_q(T) summed over four channels:

1. **BB** (bound-bound, line emission) -- via `ion.boundBoundLoss()`. Only for q < Z (need bound electrons).
2. **2gamma** (two-photon continuum) -- via `ion.twoPhotonLoss()`. Only for H-like (N=1) and He-like (N=2) ions.
3. **FF** (free-free, bremsstrahlung) -- via `continuum(X^q).freeFreeLoss()`. Only for q >= 1 (need positive charge for Coulomb field).
4. **FB** (free-bound, recombination radiation) -- via `continuum(X^(q+1)).freeBoundLoss()`. **Attributed to the POST-recombination ion** (X^q) per Gnat & Ferland 2012 / Cloudy convention. This row q receives the FB photon energy from `X^(q+1) + e -> X^q + photon`.

## Normalization conventions

Three common conventions for cooling rate in the literature:

| Convention | Volume cooling rate | What our tables store |
|---|---|---|
| **Lambda_q** (per ion per electron) | `n(X^q) * n_e * Lambda_q` | Our `cool_<X>.txt` files: each row q is `Lambda_q^per_ion(T)` [erg cm^3 / s]. No abundance or x_q baked in. |
| **Lambda_e** (per H per electron, GF12 / most CIE papers) | `n_H * n_e * Lambda_e` | Combine ours: `Lambda_e = sum_X A_X * sum_q x_q(T) * Lambda_q^per_ion(T)`. |
| **Lambda_NCR** (per H squared, Tigris-NCR / TIGRESS) | `n_H^2 * Lambda_NCR` | `Lambda_NCR = (n_e / n_H) * Lambda_e`. For fully-ionized solar gas, `n_e / n_H ~ 1.1-1.2`. |

## Trace-element approximation

For all CIE calculations done here, x_q is computed assuming each
element is a trace (does not affect n_e meaningfully). The result is
x_q(T) **independent of A_X**. Only T and the H state (x_HI, x_HII)
enter. CHIANTI/Mazzotta98/Dere09 all use this convention. See
`tigris-notes/notes/metal-ions.tex` Section sec:trace_approx for the
derivation.

## Tracked vs gitignored

All tables under `data/microphysics/chianti_v11/` are tracked in git
because regeneration costs ~15 minutes via the canonical ChiantiPy
path. Files at this directory:
- `ioneq_<X>.txt` -- CHIANTI ref ioneq
- `ioneq_ct_<X>.txt` -- ours + CT
- `cool_<X>.txt` -- per-ion cooling

Per-ion atomic-data files (e.g., `o_3.txt`) used by the 5-level
coolant solver also live under `data/microphysics/chianti_v11/`.
Read by `pyathena.microphysics.coolants.base.IonCoolant._load` via a
relative path. Built by `build_chianti_tables.py`.
