"""Offline builder: extract per-ion atomic data from CHIANTI v11
and save to ASCII text files in this directory.

Run once (or whenever the followed-ion list or T grid changes):

    XUVTOP=$HOME/Dropbox/Projects/CHIANTI_db \\
        python -m pyathena.photchem.data.build_chianti_tables

By default the temperature grid is 60 log-spaced points from
1e3 to 1e6 K, covering:
- PDR / CNM transition (~1e3 K),
- HII region core (~5e3-1.5e4 K),
- WIM (~6e3-1e4 K),
- HII / HI transition layer (~5e3-1e4 K),
- collisionally-ionized regime for the highest q stages we follow
  (OIII / NII / SIII CIE peaks ~3e4 K),
- non-equilibrium / shock conditions where high-q metals coexist
  with T ~ 1e5-1e6 K (supernova-driven gas).

For pure HII-region equilibrium work, the upper bound could be
relaxed to 5e4 K to halve the table size; for coronal applications
(Fe XIV etc.) extend further up. Pass --T-min, --T-max, --N-T to
override:

    python -m pyathena.photchem.data.build_chianti_tables \\
        --T-min 100 --T-max 1e6 --N-T 80

ASCII format chosen so the tables stay `cat`-/git-diffable and
readable from C++ with no external library (the tigris-NCR port
parses with std::ifstream). The format is line-based with section
markers; see `OIII.txt` for the canonical layout.

ChiantiPy is a build-time-only dependency: the runtime coolant
modules under `pyathena.photchem.coolants` parse the .txt files
directly.

The 5-level truncation captures the np^2 / np^3 / np^4 ground-
configuration coolants standard in nebular work. For C II (which
has only 2 levels in its ground 2P term) the table is padded with
dummies at indices 2-4 (E set very high so Boltzmann factor -> 0,
A and Upsilon set to 0); the runtime CII module ignores indices
>= nlev_phys.
"""

import os
import sys
import numpy as np


# Map (Z, N) reactant tuple to CHIANTI ion name (lowercase, "_"
# separator, charge q+1 by CHIANTI convention).
FOLLOWED_IONS = {
    'CI':   ('c_1',  5),
    'CII':  ('c_2',  2),   # 2P doublet only (2 levels physically)
    'NI':   ('n_1',  5),
    'NII':  ('n_2',  5),
    'OI':   ('o_1',  5),
    'OII':  ('o_2',  5),
    'OIII': ('o_3',  5),
    'SI':   ('s_1',  5),
    'SII':  ('s_2',  5),
    'SIII': ('s_3',  5),
}

# Default temperature grid for the Upsilon table: 60 log-spaced
# points from 1e3 to 1e6 K. Covers the followed-ion CIE peaks plus
# non-equilibrium / shock margin. 60 log-points gives ~1% log-linear
# interpolation; the underlying splups fits are 5-9 spline knots per
# transition. Override via CLI args (see module docstring).
DEFAULT_T_MIN = 1.0e3
DEFAULT_T_MAX = 1.0e6
DEFAULT_N_T = 60

# 1 cm^-1 = h c in erg = 1.986e-16 erg
ERG_PER_CM = 1.986e-16


def build_one(ion_name, nlev, T_grid):
    """Extract atomic data for one ion and return a dict suitable
    for `np.savez`.
    """
    import ChiantiPy.core as ch
    print(f"--- {ion_name} (taking lowest {nlev} levels) ---")
    ion = ch.ion(ion_name, temperature=T_grid)
    if not hasattr(ion, 'Elvlc'):
        raise RuntimeError(f"{ion_name}: no Elvlc data found")

    elv = ion.Elvlc
    lvl_all = np.asarray(elv['lvl'])
    ecm_raw = np.asarray(elv['ecm'])
    ecmth = np.asarray(elv['ecmth']) if 'ecmth' in elv else ecm_raw
    # CHIANTI v11 uses -1.0 (not 0) for unmeasured ecm. Prefer the
    # experimental energy where available, fall back to theoretical.
    ecm_eff = np.where(ecm_raw > 0.0, ecm_raw, ecmth)

    # Robust slicing for CII: there may be fewer than 5 lowest
    # levels of physical interest.  Pad to NLEV=5 with placeholders
    # so that arrays are a uniform shape.
    nlev_phys = min(nlev, len(lvl_all))
    keep = np.argsort(ecm_eff)[:nlev_phys]
    # Re-sort by energy ascending (so level 0 is ground).
    keep = keep[np.argsort(ecm_eff[keep])]
    lvl_keep = lvl_all[keep]

    # Pad if nlev > nlev_phys (only CII gets to here in practice).
    if nlev_phys < nlev:
        pad = nlev - nlev_phys
        # Use placeholder values: very high E (suppresses Boltzmann),
        # zero A's, zero Upsilon. The runtime two-level CII module
        # ignores indices >= nlev_phys.
        lvl_keep = np.concatenate([lvl_keep, np.full(pad, -1)])

    print(f"  level indices kept: {lvl_keep[:nlev_phys]}")

    # Build per-level arrays of size NLEV.
    # ChiantiPy v0.16 stores Elvlc fields as:
    #   'conf'  (int)  -- internal config index, always 0; not useful
    #   'term'  (str)  -- the actual configuration string ("2s2 2p2")
    #   'spin'  (int)  -- spin multiplicity 2S+1   <-- what we want for LS term
    #   'mult'  (float)-- LEVEL degeneracy g_J = 2J+1, NOT 2S+1
    #   'spd'   (str)  -- L letter (S/P/D/F/...)
    #   'j'     (float)-- total J
    # We assemble the LS term symbol "<2S+1><spd><J>" from `spin`,
    # `spd`, `j`; configuration string is taken from `term`.
    conf = np.full(nlev, '', dtype=object)
    term = np.full(nlev, '', dtype=object)
    E_erg = np.full(nlev, np.inf)
    g = np.zeros(nlev)
    j_arr = np.zeros(nlev)
    for k in range(nlev_phys):
        idx_in_elv = np.where(lvl_all == lvl_keep[k])[0][0]
        conf[k] = str(elv['term'][idx_in_elv])      # "2s2 2p2"
        spin = int(elv['spin'][idx_in_elv])         # 2S+1
        spd = str(elv['spd'][idx_in_elv])
        j_val = float(elv['j'][idx_in_elv])
        # Format J as integer if half-integer rendering not needed,
        # else as a fraction.
        if abs(j_val - round(j_val)) < 1e-6:
            j_lbl = f"{int(round(j_val))}"
        else:
            j_lbl = f"{j_val:.1f}"
        term[k] = f"{spin}{spd}{j_lbl}"             # "3P0", "1D2", etc.
        E_erg[k] = float(ecm_eff[idx_in_elv]) * ERG_PER_CM
        j_arr[k] = j_val
        g[k] = 2.0 * j_val + 1.0
    # Placeholder padding (CII case)
    for k in range(nlev_phys, nlev):
        E_erg[k] = 1e6 * ERG_PER_CM         # ~ huge so Boltzmann -> 0
        g[k] = 1.0
        j_arr[k] = 0.0

    # Einstein A: walk Wgfa and fill the 5x5 matrix.
    # CHIANTI convention for Wgfa: lvl1 = LOWER level, lvl2 = UPPER
    # level.  The avalue is the Einstein A coefficient for the
    # spontaneous emission lvl2 -> lvl1.
    A = np.zeros((nlev, nlev))
    if hasattr(ion, 'Wgfa'):
        wg = ion.Wgfa
        keep_set = set(int(x) for x in lvl_keep)
        for k, (l1, l2, av) in enumerate(zip(
                wg['lvl1'], wg['lvl2'], wg['avalue'])):
            if int(l1) in keep_set and int(l2) in keep_set and av > 0.0:
                lo = int(np.where(lvl_keep == int(l1))[0][0])
                hi = int(np.where(lvl_keep == int(l2))[0][0])
                # Sanity: upper must have higher energy than lower.
                if E_erg[hi] > E_erg[lo]:
                    A[hi, lo] = float(av)

    # Effective collision strengths Upsilon_e on T_grid.  ChiantiPy
    # has `upsilonDescale` to compute these from the scups (CHIANTI
    # v11, formerly splups in v10) spline coefficients at the
    # temperatures the ion was constructed with.  Convention
    # follows Wgfa: lvl1 = lower, lvl2 = upper.
    Upsilon_e = np.zeros((nlev, nlev, len(T_grid)))
    if hasattr(ion, 'Scups') or hasattr(ion, 'Splups'):
        ion.upsilonDescale()
        ups = ion.Upsilon
        keep_set = set(int(x) for x in lvl_keep)
        for k in range(len(ups['lvl1'])):
            l1 = int(ups['lvl1'][k])
            l2 = int(ups['lvl2'][k])
            if l1 in keep_set and l2 in keep_set:
                lo = int(np.where(lvl_keep == l1)[0][0])
                hi = int(np.where(lvl_keep == l2)[0][0])
                Y_T = np.asarray(ups['upsilon'][k])
                # Upsilon is symmetric in the transition; store
                # both [lo, hi] and [hi, lo] for convenience.
                Upsilon_e[lo, hi, :] = Y_T
                Upsilon_e[hi, lo, :] = Y_T

    return {
        'lvl': lvl_keep,
        'conf': conf,
        'term': term,
        'E_erg': E_erg,
        'g': g,
        'j': j_arr,
        'A': A,
        'T_grid': T_grid,
        'Upsilon_e': Upsilon_e,
        'nlev_phys': np.array(nlev_phys),
        'chianti_ion_name': np.array(ion_name),
    }


def parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--T-min', type=float, default=DEFAULT_T_MIN,
                   help='lower bound of temperature grid [K]')
    p.add_argument('--T-max', type=float, default=DEFAULT_T_MAX,
                   help='upper bound of temperature grid [K]')
    p.add_argument('--N-T', type=int, default=DEFAULT_N_T,
                   help='number of log-spaced T grid points')
    p.add_argument('--ions', nargs='*', default=None,
                   help='subset of ion labels to build '
                        '(default: all in FOLLOWED_IONS). '
                        f'Choices: {list(FOLLOWED_IONS)}')
    return p.parse_args(argv)


def write_ascii(out_path, label, data, T_min, T_max):
    """Write one ion's atomic data to an ASCII text file with
    section markers. Format is line-based and parseable in both
    Python (see `read_ascii` below) and C++ (`std::ifstream`).

    Sections (in order):
        LEVELS    -- 5 rows of (idx, chianti_lvl, conf, term, j, g,
                                E_erg [erg], E_K [K])
        A_COEFFS  -- N_A rows of (upper_idx, lower_idx, A [s^-1])
        T_GRID    -- N_T floats, one per line
        UPSILON_E -- N_trans rows of (upper_idx, lower_idx, then
                                       N_T Upsilon values)
    """
    nlev_phys = int(data['nlev_phys'])
    nlev = len(data['lvl'])
    A = data['A']
    Y = data['Upsilon_e']
    T = data['T_grid']
    KB_CGS = 1.380649e-16
    # Collect A and Upsilon transitions present (upper > lower).
    A_rows = []
    for i in range(nlev):
        for j in range(i):
            if A[i, j] > 0.0:
                A_rows.append((i, j, A[i, j]))
    Y_rows = []
    for i in range(nlev):
        for j in range(i):
            if np.any(Y[i, j, :] > 0.0):
                Y_rows.append((i, j, Y[i, j, :]))

    with open(out_path, 'w') as f:
        f.write(f"# pyathena.photchem atomic data : {label}\n")
        f.write(f"# CHIANTI source : ion {data['chianti_ion_name']}\n")
        f.write(f"# Levels kept    : lowest {nlev_phys} of CHIANTI "
                f"(padded to 5 if smaller)\n")
        f.write(f"# T grid         : {len(T)} log-spaced points, "
                f"{T_min:.3g} K -> {T_max:.3g} K\n")
        f.write(f"# Generated by   : "
                f"pyathena/photchem/data/build_chianti_tables.py\n")
        f.write(f"#\n")
        f.write(f"# Indices below are 0-based pyathena indices "
                f"(level 0 = ground).\n")
        f.write(f"# CHIANTI level numbers are 1-based and shown for "
                f"cross-reference.\n")
        f.write(f"#\n")
        # Section 1: levels
        f.write(f"LEVELS N={nlev_phys}\n")
        f.write(f"# idx  chianti_lvl  conf            term       "
                f"j     g    E_erg            E_K\n")
        for k in range(nlev_phys):
            conf = str(data['conf'][k])
            term = str(data['term'][k])
            E_erg = float(data['E_erg'][k])
            E_K = E_erg / KB_CGS
            f.write(f"{k:<5d} {int(data['lvl'][k]):<12d} "
                    f"{conf:<15s} {term:<10s} "
                    f"{float(data['j'][k]):<5.1f} "
                    f"{int(data['g'][k]):<4d} "
                    f"{E_erg: .6e}  {E_K: .6e}\n")
        f.write("\n")
        # Section 2: A
        f.write(f"A_COEFFS N={len(A_rows)}\n")
        f.write(f"# upper_idx  lower_idx   A [s^-1]\n")
        for (i, j, av) in A_rows:
            f.write(f"{i:<11d} {j:<11d} {av: .6e}\n")
        f.write("\n")
        # Section 3: T grid
        f.write(f"T_GRID N={len(T)}\n")
        for Tk in T:
            f.write(f"{Tk: .6e}\n")
        f.write("\n")
        # Section 4: Upsilon
        f.write(f"UPSILON_E N_trans={len(Y_rows)}\n")
        f.write(f"# upper_idx  lower_idx   "
                f"upsilon(T_GRID[0]..T_GRID[N-1])\n")
        for (i, j, Y_T) in Y_rows:
            vals = " ".join(f"{y: .6e}" for y in Y_T)
            f.write(f"{i:<11d} {j:<11d} {vals}\n")
        f.write("\n")


def read_ascii(path):
    """Companion reader for `write_ascii`.  Returns a dict with the
    same keys the runtime coolant modules expect.
    """
    import re
    with open(path) as f:
        lines = [ln for ln in f.read().splitlines()
                 if ln and not ln.lstrip().startswith('#')]
    out = {}
    i = 0
    while i < len(lines):
        head = lines[i]
        i += 1
        m = re.match(r'(\w+)\s+N(?:_trans)?=(\d+)', head)
        if not m:
            raise ValueError(f"unexpected section header: {head!r}")
        section, n = m.group(1), int(m.group(2))
        body = lines[i:i + n]
        i += n
        if section == 'LEVELS':
            idx = np.array([int(b.split()[0]) for b in body])
            lvl = np.array([int(b.split()[1]) for b in body])
            conf = [b.split()[2] for b in body]
            term = [b.split()[3] for b in body]
            j_arr = np.array([float(b.split()[4]) for b in body])
            g = np.array([int(b.split()[5]) for b in body])
            E_erg = np.array([float(b.split()[6]) for b in body])
            out.update(dict(lvl=lvl, conf=conf, term=term,
                            j=j_arr, g=g, E_erg=E_erg, nlev_phys=n))
        elif section == 'A_COEFFS':
            A = np.zeros((5, 5))
            for b in body:
                up, lo, av = b.split()
                A[int(up), int(lo)] = float(av)
            out['A'] = A
        elif section == 'T_GRID':
            out['T_grid'] = np.array([float(b) for b in body])
        elif section == 'UPSILON_E':
            Y = np.zeros((5, 5, len(out['T_grid'])))
            for b in body:
                parts = b.split()
                up, lo = int(parts[0]), int(parts[1])
                Y_T = np.array([float(p) for p in parts[2:]])
                Y[up, lo, :] = Y_T
                Y[lo, up, :] = Y_T
            out['Upsilon_e'] = Y
        else:
            raise ValueError(f"unknown section: {section!r}")
    return out


def main(argv=None):
    args = parse_args(argv)
    T_grid = np.logspace(np.log10(args.T_min),
                         np.log10(args.T_max),
                         args.N_T)
    print(f"Temperature grid: {args.N_T} points, "
          f"{args.T_min:.3g} K -> {args.T_max:.3g} K")
    ions = args.ions if args.ions else list(FOLLOWED_IONS)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for label in ions:
        ion_name, nlev = FOLLOWED_IONS[label]
        data = build_one(ion_name, nlev, T_grid)
        out_path = os.path.join(out_dir, f"{label}.txt")
        write_ascii(out_path, label, data, args.T_min, args.T_max)
        nz_A = (data['A'] > 0).sum()
        nz_Y = (data['Upsilon_e'] > 0).any(axis=2).sum() // 2
        print(f"  wrote {out_path}  "
              f"(A: {nz_A} entries, Upsilon: {nz_Y} transitions)")


if __name__ == '__main__':
    main()
