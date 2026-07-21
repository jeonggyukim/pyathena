"""Verner+96 photoionization cross sections.

Provides `PhotX` (per-ion sigma_pi(nu) lookups) and
`get_sigma_pi_H2`. The data file lives at
`data/microphysics/verner96_photx.dat`.

This is a compatibility port of `pyathena.microphysics.photx` whose
public API and numerical behavior are identical (rtol = 1e-12 parity
test under `tests/chemistry/parity/test_photx_parity.py`). The old
module stays available until cross-repo callers migrate.
"""
from typing import Dict, Tuple
import numpy as np
import os
import os.path as osp

import astropy.units as au
import astropy.constants as ac

from ..datapaths import _DATA_ROOT
from ..enums import InterpMode

__all__ = ['PhotX', 'get_sigma_pi_H2']

_MICROPHYSICS_DATA_DIR = osp.join(_DATA_ROOT, 'microphysics')


def _ordered_ions(species_set):
    """Return the `Ion` instances of `species_set` in the strip-table
    traversal order (`evolved_names + ghost_names`) when the partition
    is declared, otherwise the raw `ions` tuple.

    The strip-table accessors call this so they can honor the
    SpeciesSet partition without each one re-implementing the lookup.
    """
    evolved = getattr(species_set, 'evolved_names', None)
    ghost = getattr(species_set, 'ghost_names', None)
    idx = getattr(species_set, 'idx', None)
    ions = tuple(species_set.ions)
    if evolved is not None and ghost is not None and idx is not None:
        names = tuple(evolved) + tuple(ghost)
        return tuple(ions[idx[n]] for n in names)
    return ions


class PhotX(object):
    """
    Computes photoionization cross-sections as described in Verner 96
    http://adsabs.harvard.edu/abs/1996ApJ...465..487V
    Original python code from Rabacus implementation
    https://github.com/galtay/rabacus/tree/master/rabacus/atomic/verner/photox
    (Released under GNU GPL3 v.30;
    JKIM: okay to copy snippets of their code?)
    """

    def __init__(self, datadir=None, interp_mode: 'InterpMode' = InterpMode.Exact):
        """
        Parameters
        ----------
        datadir : str or None
            Override the default Verner+96 data directory.
        interp_mode : InterpMode, optional
            Rate-table interpolation mode. Only `InterpMode.Exact`
            (analytic evaluation) is supported at present; the
            `LogLog` / `Nqt1` / `Nqt2` table-based modes are
            scheduled for Phase 3.5 once the chemistry strip
            scaffold is in place.
        """
        # Only Exact is implemented today. Refuse anything else
        # explicitly so callers do not silently get analytic results
        # when they asked for a table mode.
        if interp_mode != InterpMode.Exact:
            raise NotImplementedError(
                f'PhotX interp_mode={interp_mode!r} is not implemented; '
                'table-based modes (LogLog / Nqt1 / Nqt2) land in '
                'Phase 3.5. Use InterpMode.Exact for now.')
        self.interp_mode = interp_mode

        # Read Verner et al. 1996 photoionization cross-section table
        if datadir is None:
            fname = os.path.join(_MICROPHYSICS_DATA_DIR, 'verner96_photx.dat')

        dat = np.loadtxt(fname, unpack=True)
        self._dat = dat

        # Organize data
        self.Z = self._dat[0]
        self.N = self._dat[1]
        self.Eth = self._dat[2]
        self.Emax = self._dat[3]
        self.E0 = self._dat[4]
        self.sigma0 = self._dat[5] * 1.0e-18
        self.ya = self._dat[6]
        self.P = self._dat[7]
        self.yw = self._dat[8]
        self.y0 = self._dat[9]
        self.y1 = self._dat[10]
        del self._dat

        # Build (Z, N) -> row index dict so get_sigma / get_Eth do an
        # O(1) lookup instead of a per-call `np.where(c1 & c2)` scan.
        self._ion_idx: Dict[Tuple[int, int], int] = {
            (int(self.Z[i]), int(self.N[i])): i
            for i in range(self.Z.size)
        }

    def get_Eth(self, Z, N, unit='eV'):
        """
        Threshold ionization energy in eV for ions defined by Z and N.

        Parameters
        ----------
        Z : int
            Atomic number (number of protons)
        N : int
            Electron number

        Returns
        -------
        Eth: float
            Threshold ionization energy in eV
        """

        indx = self._ion_idx[(int(Z), int(N))]
        Eth = self.Eth[indx]

        if unit == 'eV':
            return Eth
        elif unit == 'Angstrom':
            return ((ac.h*ac.c)/(Eth*au.eV)).to('Angstrom').value

    def get_sigma(self, Z, N, E):
        """Returns a photo-ionization cross-section for an ion defined by
        Z and N at energies E in eV.

        Parameters
        ----------
        Z : int
            Atomic number (number of protons)
        N : int
            Electron number (number of electrons)
        E : array of floats
            Calculate cross-section at these energies [eV]

        Returns
        -------
        sigma: array of floats
            Photoionization cross-sections [cm^-2]
        """

        # calculate fit
        indx = self._ion_idx[(int(Z), int(N))]
        Z = self.Z[indx]
        N = self.N[indx]
        Eth = self.Eth[indx]
        Emax = self.Emax[indx]
        E0 = self.E0[indx]
        sigma0 = self.sigma0[indx]
        ya = self.ya[indx]
        P = self.P[indx]
        yw = self.yw[indx]
        y0 = self.y0[indx]
        y1 = self.y1[indx]

        x = E / E0 - y0
        y = np.sqrt(x*x + y1*y1)

        sigma = sigma0 * ((x-1)*(x-1) + yw*yw) * y**(0.5*P - 5.5) * \
            (1 + np.sqrt(y/ya))**(-P)

        # zero cross-section below threshold
        indx = np.where(E < Eth)
        if indx[0].size > 0:
            sigma[indx] = 0.0

        return sigma

    def get_sigma_table(self, species_set, E):
        """Strip-vectorised photoionization cross sections.

        Returns sigma_pi[(Z_i, N_i), E] stacked over every ion in
        `species_set`. Output shape is `(n_ions, len(E))`.

        Parameters
        ----------
        species_set : SpeciesSet
            Iterable of ions. The traversal order follows the
            `species_set.evolved_names + species_set.ghost_names`
            convention when the partition is declared; falls back to
            `species_set.ions` (the full storage order) otherwise. Ions that lack a Verner+96
            entry — e.g. the bare electron or H2 — contribute a zero
            row.
        E : array-like of float
            Photon energies [eV].

        Returns
        -------
        sigma : np.ndarray
            Shape `(n_ions, len(E))`. Cross sections in cm^-2.

        Notes
        -----
        Today this is a thin loop wrapper over `get_sigma`. The C++
        port (Phase D) will replace it with a precomputed strip-shape
        table indexed by the same `(Z, N)` ordering.
        """
        E_arr = np.asarray(E, dtype=float)
        # Pick evolved+ghost when the SpeciesSet exposes the partition
        # via name-tuples; resolve each name back to the canonical Ion
        # through `species_set.idx`. Falls back to `species_set.ions`
        # (the full storage order) for sets that do not declare a
        # partition.
        ions = _ordered_ions(species_set)

        out = np.zeros((len(ions), E_arr.size), dtype=float)
        for i, ion in enumerate(ions):
            # Skip the bare electron and molecular species (e.g. H2)
            # outright; both lack a per-atom Verner+96 fit and the
            # `(Z, N)` lookup would collide with another ion (H2 shares
            # (2, 2) with He I).
            if ion.element in ('e', 'H2'):
                continue
            key = (int(ion.Z), int(ion.N))
            if key not in self._ion_idx:
                # No Verner fit row for this ion (atomic species
                # outside the table). Leave the strip row at zero.
                continue
            out[i, :] = self.get_sigma(ion.Z, ion.N, E_arr)
        return out

def get_sigma_pi_H2(E):
    """H2 photoionization cross-section [cm^-2]
    Table 1 in Baczynski et al. (2015)
    Piecewise constant cross-section to the analytical results
    of Liu & Shemansky (2012)

    E : array of floats
        Photon energy in eV
    """
    return 1e-18*np.piecewise(E, [E < 15.2,
                            ((E >= 15.2) & (E < 15.45)),
                            ((E >= 15.45) & (E < 15.70)),
                            ((E >= 15.7) & (E < 15.95)),
                            ((E >= 15.95) & (E < 16.20)),
                            ((E >= 16.2) & (E < 16.40)),
                            ((E >= 16.4) & (E < 16.65)),
                            ((E >= 16.65) & (E < 16.85)),
                            ((E >= 16.85) & (E < 17.0)),
                            ((E >= 17.0) & (E < 17.2)),
                            ((E >= 17.2) & (E < 17.65)),
                            ((E >= 17.65) & (E < 18.1)),
                            E >= 18.1],
                            [0.0,0.09,1.15,3.0,5.0,6.75,8.0,9.0,9.5,9.8,10.1,9.85,
                             lambda E: 9.85/(E/18.1)**3])
