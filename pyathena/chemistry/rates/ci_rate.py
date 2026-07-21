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
from . import _nqt
from .photx import _ordered_ions

_CLOUDY_DIR = osp.join(_DATA_ROOT, 'microphysics', 'cloudy')

# Floor applied to tabulated rates before taking log / nqt_log. Mirrors
# the C++ side's kRateFloor; sufficient for double-precision storage
# without underflowing the NQTo2 inversion (which needs 4 - 3*f > 0 in
# the fractional mantissa).
_RATE_FLOOR: float = 1.0e-100

# T-grid endpoints + size shared by the LogLog and NQT modes. Matches
# the C++ ncr_rates.hpp TabFull layout (n_grid=2000, T in [1, 1e9]).
_TAB_N: int = 2000
_TAB_T_MIN: float = 1.0
_TAB_T_MAX: float = 1.0e9


class CollIonRate(object):

    def __init__(self, interp_mode: 'InterpMode' = InterpMode.Exact):
        """
        Parameters
        ----------
        interp_mode : InterpMode, optional
            Rate-table interpolation mode. `InterpMode.Exact` (analytic
            Voronov fits) is the default; `InterpMode.LogLog`,
            `InterpMode.Nqt2`, and `InterpMode.Nqt1` precompute a
            `(n_T, n_ions)` table on construction and dispatch a
            two-point linear interpolation in the chosen log space.
        """
        if interp_mode not in (
            InterpMode.Exact, InterpMode.LogLog,
            InterpMode.Nqt2, InterpMode.Nqt1,
        ):
            raise ValueError(
                f'CollIonRate: unknown interp_mode={interp_mode!r}')
        self.interp_mode = interp_mode
        self._read_data()
        if interp_mode != InterpMode.Exact:
            self._build_table(interp_mode)

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

    def _get_ci_rate_exact(self, i_row: int, T: np.ndarray) -> np.ndarray:
        """Analytic Voronov fit for a single ion row at temperature(s) `T`.

        The dispatcher in `get_ci_rate` resolves `(Z, N)` to `i_row`
        once; the table-build path calls this directly on every row to
        populate the LogLog / Nqt2 / Nqt1 tables. `T` is treated as a
        numpy array so the result is always an ndarray (scalar T gives
        a 0-d ndarray).
        """
        U = self.dE_Kel[i_row] / T
        return np.where(
            U > 80.0,
            0.0,
            self.A[i_row] * (1.0 + self.P[i_row] * U ** 0.5)
            / (self.X[i_row] + U)
            * U ** (self.K[i_row]) * np.exp(-U),
        )

    def _build_table(self, mode: 'InterpMode') -> None:
        """Pre-tabulate rate(T_grid) for every ion row in `_ion_idx`.

        Stores the encoded rate (log10 for LogLog; nqt2_log / nqt1_log
        for the NQT modes) alongside the grid metadata so the lookup
        path can recover the rate with a single linear interpolation
        plus an inverse encoding step.
        """
        if mode == InterpMode.LogLog:
            log_T_grid = np.linspace(
                np.log10(_TAB_T_MIN), np.log10(_TAB_T_MAX), _TAB_N,
            )
            self._T_grid = 10.0 ** log_T_grid
            self._tab_x_min = float(log_T_grid[0])
            self._tab_inv_dx = float(
                (_TAB_N - 1) / (log_T_grid[-1] - log_T_grid[0])
            )
        else:  # Nqt2 / Nqt1 share the same nqt1_log T grid
            y_min = float(_nqt.nqt1_log(np.array([_TAB_T_MIN]))[0])
            y_max = float(_nqt.nqt1_log(np.array([_TAB_T_MAX]))[0])
            y_grid = np.linspace(y_min, y_max, _TAB_N)
            self._T_grid = _nqt.nqt1_exp(y_grid)
            self._tab_x_min = y_min
            self._tab_inv_dx = float((_TAB_N - 1) / (y_max - y_min))

        n_ions = int(self.Z.size)
        tab = np.empty((_TAB_N, n_ions), dtype=np.float64)
        for j in range(n_ions):
            rates = self._get_ci_rate_exact(j, self._T_grid)
            rates = np.where(rates > _RATE_FLOOR, rates, _RATE_FLOOR)
            if mode == InterpMode.LogLog:
                tab[:, j] = np.log10(rates)
            elif mode == InterpMode.Nqt2:
                tab[:, j] = _nqt.nqt2_log(rates)
            else:  # Nqt1
                tab[:, j] = _nqt.nqt1_log(rates)
        self._tab = tab

    def _table_lookup(self, i_row: int, T: np.ndarray) -> np.ndarray:
        """Two-point linear interpolation of the pre-built table.

        Returns the rate decoded out of the appropriate log space.
        Branch-free over T -- the per-mode choice is fixed at
        construction time, so the dispatch happens once in
        `get_ci_rate`.
        """
        T_arr = np.asarray(T, dtype=np.float64)
        if self.interp_mode == InterpMode.LogLog:
            x = np.log10(T_arr)
        else:
            x = _nqt.nqt1_log(T_arr)
        idx_f = (x - self._tab_x_min) * self._tab_inv_dx
        idx = np.clip(idx_f.astype(np.int64), 0, _TAB_N - 2)
        frac = idx_f - idx.astype(np.float64)
        lo = self._tab[idx, i_row]
        hi = self._tab[idx + 1, i_row]
        encoded = lo + (hi - lo) * frac
        if self.interp_mode == InterpMode.LogLog:
            return 10.0 ** encoded
        if self.interp_mode == InterpMode.Nqt2:
            return _nqt.nqt2_exp(encoded)
        return _nqt.nqt1_exp(encoded)

    def get_ci_rate(self, Z, N, T):
        i = self._ion_idx[(int(Z), int(N))]
        if self.interp_mode == InterpMode.Exact:
            return self._get_ci_rate_exact(i, T)
        return self._table_lookup(i, T)

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
