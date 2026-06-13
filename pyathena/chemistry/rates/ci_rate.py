"""Voronov 1997 collisional ionization rate fits.

Provides `CollIonRate` (Voronov 1997, At. Data Nucl. Data Tables 65, 1)
and the optional `CollIonRateCHIANTI` for the same rates via
ChiantiPy. Data file at
`data/microphysics/cloudy/coll_ion.dat`.

Compatibility port of `pyathena.microphysics.ci_rate`; public API
and numerical behavior identical (rtol = 1e-12 parity test under
`tests/chemistry/parity/test_ci_rate_parity.py`).
"""
import os
import os.path as osp
from typing import Dict, Tuple
import numpy as np
import astropy.units as au
import astropy.constants as ac

from ..datapaths import _DATA_ROOT
from ..enums import InterpMode
from .photx import _ordered_ions

_CLOUDY_DIR = osp.join(_DATA_ROOT, 'microphysics', 'cloudy')


class CollIonRate(object):

    def __init__(self, interp_mode: 'InterpMode' = InterpMode.Exact):
        """
        Parameters
        ----------
        interp_mode : InterpMode, optional
            Rate-table interpolation mode. Only `InterpMode.Exact`
            (analytic Voronov fits) is supported today; the
            table-based modes land in Phase 3.5 alongside the C++
            port.
        """
        if interp_mode != InterpMode.Exact:
            raise NotImplementedError(
                f'CollIonRate interp_mode={interp_mode!r} is not '
                'implemented; table-based modes (LogLog / Nqt1 / '
                'Nqt2) land in Phase 3.5. Use InterpMode.Exact for '
                'now.')
        self.interp_mode = interp_mode
        self._read_data()

    def _read_data(self, max_rows=465):

        self.fname = os.path.join(_CLOUDY_DIR, 'coll_ion.dat')
        lines = np.loadtxt(self.fname, unpack=True, skiprows=1, max_rows=max_rows)

        # In cloudy, both eletron and atomic numbers are on C scale (starting from 0), so
        # need to add 1.
        self.N = lines[0].astype(int) + 1
        self.Z = lines[1].astype(int) + 1
        self.dE_Kel = lines[2]*(1.0*au.eV/ac.k_B).cgs.value
        self.P = lines[3] # 0 or 1
        self.A = lines[4]
        self.X = lines[5]
        self.K = lines[6]

        # Build (Z, N) -> row index dict so get_ci_rate does an O(1)
        # lookup instead of a per-call `np.where(iZ & iN)` scan.
        self._ion_idx: Dict[Tuple[int, int], int] = {
            (int(self.Z[i]), int(self.N[i])): i
            for i in range(self.Z.size)
        }

    def get_ci_rate(self, Z, N, T):
        i = self._ion_idx[(int(Z), int(N))]

        U = self.dE_Kel[i]/T
        rate = np.where(U > 80.0,
                        0.0,
                        self.A[i]*(1.0 + self.P[i]*U**0.5)/\
                        (self.X[i] + U)*U**(self.K[i])*np.exp(-U))

        return rate

    def get_ci_rate_table(self, species_set, T):
        """Strip-vectorised collisional-ionization rates.

        Returns k_CI[(Z_i, N_i), T] stacked over every ion in
        `species_set`. Output shape is `(n_species, ncell)`. Species
        that have no entry — bare electron, H2, fully-stripped ions
        outside the Voronov table — contribute a zero row.

        Parameters
        ----------
        species_set : SpeciesSet
            Iterable of ions. Picks up
            `species_set.evolved_names + species_set.ghost_names` when
            the partition is declared, else `species_set.ions`.
        T : float or array-like
            Temperature [K].

        Notes
        -----
        Today this is a thin loop wrapper over `get_ci_rate`. The C++
        port (Phase D) will replace it with a precomputed strip-shape
        table indexed by the same `(Z, N)` ordering.
        """
        T_arr = np.asarray(T, dtype=float)
        ncell = T_arr.size if T_arr.ndim > 0 else 1
        ions = _ordered_ions(species_set)

        out = np.zeros((len(ions), ncell), dtype=float)
        for i, ion in enumerate(ions):
            if ion.element in ('e', 'H2'):
                continue
            key = (int(ion.Z), int(ion.N))
            if key not in self._ion_idx:
                continue
            rate = self.get_ci_rate(ion.Z, ion.N, T_arr)
            out[i, :] = np.broadcast_to(rate, (ncell,))
        return out


class CollIonRateCHIANTI(object):
    """Collisional ionization rate coefficient using CHIANTI v11 via
    ChiantiPy. Same `get_ci_rate(Z, N, T)` API as the pyathena-native
    `CollIonRate`. Pre-tabulates rates on a fixed T grid; interpolates
    in log T at call time. See `RecRateCHIANTI` for the same pattern
    applied to recombination.
    """

    DEFAULT_ELEMENTS = [
        ('H', 1), ('He', 2), ('C', 6), ('N', 7), ('O', 8),
        ('Ne', 10), ('Mg', 12), ('Si', 14), ('S', 16), ('Ar', 18),
        ('Ca', 20), ('Fe', 26),
    ]
    _ELEM_SYM = {
        'H':  'h',  'He': 'he', 'C':  'c',  'N':  'n',  'O':  'o',
        'Ne': 'ne', 'Mg': 'mg', 'Si': 'si', 'S':  's',  'Ar': 'ar',
        'Ca': 'ca', 'Fe': 'fe',
    }

    def __init__(self, T_grid=None, elements=None):
        import ChiantiPy.core as ch
        if T_grid is None:
            T_grid = np.logspace(2.0, 9.0, 100)
        else:
            T_grid = np.asarray(T_grid, dtype=float)
        self._T_grid = T_grid
        self._lnT_grid = np.log(T_grid)
        if elements is None:
            elements = self.DEFAULT_ELEMENTS
        self._table = {}
        for element, Z in elements:
            sym = self._ELEM_SYM[element]
            for q in range(0, Z):  # q=Z is fully stripped, can't ionize further
                ion_name = f'{sym}_{q + 1}'
                try:
                    ion = ch.ion(ion_name, temperature=T_grid)
                    ion.ionizRate()
                    rate = np.asarray(ion.IonizRate['rate'])
                except Exception:
                    continue
                rate = np.where(np.isfinite(rate) & (rate > 0), rate, 0.0)
                N = Z - q                # reactant electron count
                self._table[(Z, N)] = rate

    def get_ci_rate(self, Z, N, T):
        """Same API as pyathena.microphysics.ci_rate.CollIonRate.

        Returns k_CI [cm^3 / s] for the reactant ion (Z, N). Returns
        0 if CHIANTI lacks data.
        """
        rate = self._table.get((Z, N))
        if rate is None:
            return np.zeros_like(np.asarray(T, dtype=float))
        T_arr = np.atleast_1d(np.asarray(T, dtype=float))
        out = np.interp(np.log(T_arr), self._lnT_grid, rate,
                        left=rate[0], right=rate[-1])
        return float(out[0]) if np.isscalar(T) else out
