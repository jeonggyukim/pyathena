"""Single entry-point that regenerates all CIE ionization-fraction
ASCII tables in this directory from CHIANTI v11 via ChiantiPy.

Produces two sets of per-element files in
`pyathena/photchem/data/`:

  - ioneq_<element>.txt
        CHIANTI v11 reference via `ChiantiPy.core.ioneq.calculate`.
        Sequential CIE balance using CURRENT CHIANTI rate fits
        (Badnell DR + Verner-Ferland RR + Dere 2007 CI). Same data
        and same solver as the dashed "ours" curves below; this is
        the gold-standard reference.

  - ioneq_local_<element>.txt
        Our own implementation of the sequential CIE balance using
        the same CHIANTI v11 rate data (via RecRateCHIANTI /
        CollIonRateCHIANTI). Should match `ioneq_<element>.txt` to
        machine precision; differences would indicate a bug in our
        sequential solver or rate-class wrappers.

  - ioneq_local_ct_<element>.txt
        Same as `ioneq_local_<element>.txt` but with the
        charge-transfer (CT) contribution added to source / sink
        rates. CT weighted by CHIANTI's CIE H state at each T
        (trace-element approximation).

The downstream benchmark plots
`tests/microphysics/test_plot_ioneq_chianti_vs_local.py` overlay
all three; they should produce dashed-under-solid for the no-CT
case and a visibly offset dotted curve where CT shifts the
balance (cold ionized regime).

CLI:
    XUVTOP=$HOME/Dropbox/Projects/CHIANTI_db \\
        python -m pyathena.photchem.data.build_all
"""

import os

from . import build_ioneq_tables, build_ioneq_local


def main():
    """Regenerate everything in `pyathena/photchem/data/`."""
    if not os.environ.get('XUVTOP'):
        raise RuntimeError(
            "XUVTOP environment variable not set. ChiantiPy cannot "
            "load CHIANTI data without it. Set XUVTOP to your "
            "CHIANTI v11 data directory before running, e.g.:\n"
            "    export XUVTOP=$HOME/Dropbox/Projects/CHIANTI_db")
    print("=== Step 1: CHIANTI reference ioneq_<element>.txt ===")
    build_ioneq_tables.main()
    print()
    print("=== Step 2: local ioneq_local[_ct]_<element>.txt ===")
    build_ioneq_local.main()
    print()
    print("All tables regenerated.")


if __name__ == '__main__':
    main()
