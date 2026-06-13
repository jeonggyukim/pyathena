"""NCRNetwork3 — the three-species H I / H II / H2 NCR network.

Tracks three evolved hydrogen species plus six algebraic ghosts:

    evolved:  HI, HII, H2
    ghost:    electron, CI, CII, CO, OI, OII

The evolved rows are integrated as ODE state by the solver. The ghost
rows are reconstructed every substep by `fill_ghosts(state)` from the
evolved rows, T, and the configured C / O abundances (mirroring the
`CoolingOther` body in `src/photchem/ncr_rates.hpp` lines 1505-1540
of the tigris-ncr fork). This keeps the cooling layer's per-species
abundance lookups uniform without growing the ODE state vector.

Mass closure on hydrogen nuclei is `x_HI + x_HII + 2 * x_H2 = 1`.
Charge closure is `x_e = x_HII + x_CII + x_OII` (electron count from
the carbon and oxygen ghosts plus HII). Helium contributions to x_e
arrive with Phase 6.

The rate equations mirror the Tigris-NCR C++ network in
`src/photchem/ncr_rates.hpp` (functions `HIIRates`, `H2Rates`). For
each non-conserved species we write the semi-implicit decomposition

    dx_i / dt = C_i - D_i * x_i

into caller-owned (nspec, ncell) buffers. The C++ side uses the
two-species Cramer solve `(x_H2, x_HII)` with `x_HI = 1 - 2 x_H2 -
x_HII`; that solve happens inside the solver, not the network. The
network's job is just to expose C_i and D_i that match the C++
intent.

Rate contributions (matching ncr_rates.hpp:1582-1625):

  H II creation:    (k_coll(T) * n_e + xi_cr_HII + xi_ph_HI) * x_HI
                    -> C_HII = (k_coll*n_e + xi_cr_HII + xi_ph_HI)
                       times x_HI in the network output
  H II destruction: n_e * alpha_rr(T) + nH * alpha_gr(T, chi, n_e, Z_d)
                    -> D_HII = n_e * alpha_rr + nH * alpha_gr

  H2 creation:      kgr(T, Z_d) * nH * x_HI
                    -> C_H2 = kgr*nH * x_HI in the network output
  H2 destruction:   1.65 * xi_cr_H2 + xi_ph_H2 + xi_diss_H2 + xi_coll_H2
                    where xi_cr_H2 = 2 * xi_CR * (2.3 x_H2 + 1.5 x_HI)
                    -> D_H2 contains all of these

  H I creation:     D_HII * x_HII + 2 * D_H2 * x_H2  (folded back into C_HI
                    because that is the species the destruction terms
                    above are returning x_HI atoms to)
  H I destruction:  C_HII_per_xHI + 2 * C_H2_per_xHI
                    (the "per_xHI" factors are HII/H2 creation rates
                    before multiplying by x_HI)

  electron:         not integrated as a free species; the closure
                    sets x_e = x_HII (charge neutrality with only
                    H II contributing).

CR ionization rate convention: `state.xi_CR` is the primary CR
ionization rate per H I [s^-1]; the secondary-weighted form for
H II creation is `xi_cr_HII = xi_CR * (2.3 x_H2 + 1.5 x_HI)`.

Floors: species fractions are clamped to `x_floor = 1e-20` inside
`closure()` before the H mass renormalisation runs.

Above `temp_hot1 = 3.5e4 K` the grain channel turns off and H2
chemistry is zeroed entirely (matches the C++ `if (T >= temp_hot1)`
branch in `EvaluateChemistryCD`). The smooth sigmoid blend between
`temp_hot0 = 2e4 K` and `temp_hot1 = 3.5e4 K` on `alpha_gr` is also
preserved.

Divergences from `pyathena.microphysics.photchem.PhotChem`:

- PhotChem has no H2 channel at all. The H2 chemistry here is taken
  from `pyathena.microphysics.cool` (kgr, xi_coll_H2) plus the C++
  reference for the photo/CR terms. Phase 6 (`NCRNetwork3PlusIons16`)
  is where PhotChem-style ion-by-ion sweeps return.
- PhotChem updates electrons mid-sweep as `n_e = n_HII` (`L770`).
  Here we expose the same convention via `electron_fraction(state)`
  returning `x_HII`, and rely on the driver / solver to write it
  back. The network does not mutate the electron row.
- Photo-rates (`xi_ph_HI`, `xi_ph_H2`, `xi_diss_H2`) and the FUV
  intensity (`chi_FUV`) are read from optional fields on `state`:
  `state.xi_ph_HI`, `state.xi_ph_H2`, `state.xi_diss_H2`,
  `state.chi_FUV`. When absent (Phase 2 unit-test usage), they
  default to zero — i.e., the network falls back to a dark, CR-only
  regime. The integration agent will wire these up to `state.chi[]`
  in Phase 3.
"""
from __future__ import annotations

from typing import Any, ClassVar, Tuple

import numpy as np

from ..species import SpeciesSet
from .base import NetworkBase


# ---- Physical / numerical constants ------------------------------------
# Mirrors NCRRates defaults in src/photchem/ncr_rates.hpp.
TEMP_HOT0 = 2.0e4   # below this, full alpha_gr; above this, sigmoid blend
TEMP_HOT1 = 3.5e4   # above this, no alpha_gr and no H2 chemistry
X_FLOOR = 1.0e-20   # species-fraction floor; matches C++ TINY_NUMBER


# Single source of truth for the species inventory this network reads.
# Shared as a class-level attribute so callers can identity-check
# `state.species is NCRNetwork3.species` before entering `evaluate_CD`.
# The set is the 9-species ncr3_with_ghosts layout (3 evolved + 6 ghost);
# evaluate_CD / closure still operate purely on the evolved rows, while
# fill_ghosts materialises the algebraic ghost rows each substep.
_NCR3_SPECIES = SpeciesSet.ncr3_with_ghosts()

# Standard solar C and O abundances per H nucleus, mirroring the C++
# constants `NCRRates::kxCstd` / `NCRRates::kxOstd` in ncr_rates.hpp:409.
X_C_STD = 1.6e-4
X_O_STD = 3.2e-4


# ---- Rate helpers ------------------------------------------------------
def _coeff_kcoll_H(T: np.ndarray) -> np.ndarray:
    """Janev et al. fit to H I collisional ionization, taken from
    `pyathena.microphysics.cool.coeff_kcoll_H`. Zero below 3000 K.
    """
    lnTe = np.log(T * 8.6173e-5)
    k_coll = np.where(
        T > 3.0e3,
        np.exp(
            -3.271396786e1
            + (1.35365560e1
               + (-5.73932875
                  + (1.56315498
                     + (-2.877056e-1
                        + (3.48255977e-2
                           + (-2.63197617e-3
                              + (1.11954395e-4
                                 + (-2.03914985e-6)
                                 * lnTe) * lnTe) * lnTe) * lnTe) * lnTe) * lnTe) * lnTe) * lnTe
        ),
        0.0,
    )
    return k_coll


def _coeff_alpha_rr_H(T: np.ndarray) -> np.ndarray:
    """Gong+17 fit to Ferland+92 case-B H II radiative recombination.
    Matches `pyathena.microphysics.cool.coeff_alpha_rr_H`.
    """
    Tinv = 1.0 / T
    bb = 315614.0 * Tinv
    cc = 115188.0 * Tinv
    dd = 1.0 + np.power(cc, 0.407)
    return 2.753e-14 * np.power(bb, 1.5) * np.power(dd, -2.242)


def _coeff_alpha_gr_H(
    T: np.ndarray,
    chi_FUV: np.ndarray,
    ne: np.ndarray,
    Z_d: np.ndarray,
) -> np.ndarray:
    """Grain-assisted H II recombination (Draine 2011 Eq. 14.37 with
    Weingartner & Draine fit coefficients). Matches
    `pyathena.microphysics.cool.coeff_alpha_gr_H`, with the C++ sigmoid
    blend between temp_hot0 and temp_hot1 applied here too so that the
    coefficient turns off smoothly in hot gas.
    """
    small = 1.0e-50
    cHp = (12.25, 8.074e-6, 1.378, 5.087e2, 1.586e-2, 0.4723, 1.102e-5)
    lnT = np.log(T)
    psi = 1.7 * chi_FUV * np.sqrt(T) / (ne + small) + small
    alpha = (
        1.0e-14 * cHp[0]
        / (1.0 + cHp[1] * np.power(psi, cHp[2])
           * (1.0 + cHp[3] * np.power(T, cHp[4])
              * np.power(psi, -cHp[5] - cHp[6] * lnT)))
    ) * Z_d
    # Above TEMP_HOT1 the grain channel is off. Between TEMP_HOT0 and
    # TEMP_HOT1 a sigmoid blend matches the C++ reference.
    wgt = np.where(
        T >= TEMP_HOT1,
        0.0,
        np.where(
            T <= TEMP_HOT0,
            1.0,
            1.0 - 1.0 / (1.0 + np.exp(
                -10.0 * (T - 0.5 * (TEMP_HOT0 + TEMP_HOT1))
                / (TEMP_HOT1 - TEMP_HOT0)
            )),
        ),
    )
    return alpha * wgt


def _coeff_kgr_H2(T: np.ndarray, Z_d: np.ndarray) -> np.ndarray:
    """Grain-surface H2 formation rate coefficient, T-dependent form
    (`ikgr_H2 == 1` branch of `pyathena.microphysics.cool.heatH2`):

        kgr = kgr0 * Z_d * sqrt(T2) * 2 / (1 + 0.4 sqrt(T2)
              + 0.2 T2 + 0.08 T2^2)     with T2 = T / 100 K.

    kgr0 = 3e-17 cm^3 s^-1 is the standard Draine-Bertoldi reference
    rate (the C++ uses the same constant).
    """
    kgr0 = 3.0e-17
    T2 = T * 1.0e-2
    return kgr0 * Z_d * np.sqrt(T2) * 2.0 / (
        1.0 + 0.4 * np.sqrt(T2) + 0.2 * T2 + 0.08 * T2 * T2
    )


def _coeff_xi_coll_H2(
    nH: np.ndarray,
    T: np.ndarray,
    xHI: np.ndarray,
    xH2: np.ndarray,
) -> np.ndarray:
    """Glover & Mac Low (2007) H+H2 and H2+H2 collisional dissociation,
    matches `pyathena.microphysics.cool.coeff_coll_H2`. Zero below
    700 K. Returns a per-H2 destruction rate [s^-1].
    """
    temp_coll = 7.0e2
    Tinv = 1.0 / T
    logT4 = np.log10(T * 1.0e-4)
    k9l = 6.67e-12 * np.sqrt(T) * np.exp(-(1.0 + 63590.0 * Tinv))
    k9h = 3.52e-9 * np.exp(-43900.0 * Tinv)
    k10l = (5.996e-30 * np.power(T, 4.1881)
            / np.power(1.0 + 6.761e-6 * T, 5.6881)
            * np.exp(-54657.4 * Tinv))
    k10h = 1.3e-9 * np.exp(-53300.0 * Tinv)
    ncrH2 = np.power(10.0, 4.845 - 1.3 * logT4 + 1.62 * logT4 * logT4)
    ncrHI = np.power(10.0, 3.0 - 0.416 * logT4 - 0.327 * logT4 * logT4)
    ncrinv = np.clip(xHI / ncrHI + 2.0 * xH2 / ncrH2, 0.0, None)
    n2ncr = nH * ncrinv
    k_H2_HI = np.power(
        10.0,
        np.log10(k9h) * n2ncr / (1.0 + n2ncr)
        + np.log10(k9l) / (1.0 + n2ncr),
    )
    k_H2_H2 = np.power(
        10.0,
        np.log10(k10h) * n2ncr / (1.0 + n2ncr)
        + np.log10(k10l) / (1.0 + n2ncr),
    )
    xi = k_H2_H2 * nH * xH2 + k_H2_HI * nH * xHI
    return np.where(T > temp_coll, xi, 0.0)


# ---- NCRNetwork3 -------------------------------------------------------
class NCRNetwork3(NetworkBase):
    """Three-species H I / H II / H2 NCR network (Phase 2).

    Class-level metadata:

      species         -- 9-species SpeciesSet (HI, HII, H2 evolved;
                         electron, CI, CII, CO, OI, OII ghost).
      evolved         -- ('HI', 'HII', 'H2'). Solver state.
      ghost           -- ('electron', 'CI', 'CII', 'CO', 'OI', 'OII').
                         Rebuilt every substep by fill_ghosts().
      walk_order      -- single-element sequential walk used by the
                         Phase D ion sweep solver. For NCR3 the only
                         element is hydrogen and the walk visits
                         neutral -> ionised -> molecular.
      kSupportsStrips -- True; every rate helper uses np.where and
                         broadcasts cleanly over the ncell axis.
      kNeedsJacobian  -- False; the Phase 3 explicit subcycler and
                         the Phase C sync sweep are derivative-free.
    """

    species: ClassVar[SpeciesSet] = _NCR3_SPECIES
    evolved: ClassVar[Tuple[str, ...]] = ('HI', 'HII', 'H2')
    ghost:   ClassVar[Tuple[str, ...]] = (
        'electron', 'CI', 'CII', 'CO', 'OI', 'OII',
    )
    walk_order: ClassVar[Tuple[Tuple[str, ...], ...]] = (
        ('HI', 'HII', 'H2'),
    )
    kSupportsStrips: ClassVar[bool] = True
    kNeedsJacobian: ClassVar[bool] = False

    # Standard solar C and O abundances per H nucleus. Class-level
    # constants mirror the C++ `NCRRates::kxCstd` / `kxOstd` so
    # `fill_ghosts` can use them without an extra config read.
    x_C_std: ClassVar[float] = X_C_STD
    x_O_std: ClassVar[float] = X_O_STD

    # The species floor used by closure(). Instance attribute so the
    # integration agent can override it via a config object later
    # without touching the class.
    x_floor: float = X_FLOOR

    def __init__(self, x_floor: float = X_FLOOR) -> None:
        self.x_floor = float(x_floor)

    # ---- evaluate_CD ------------------------------------------------
    def evaluate_CD(
        self,
        state: Any,
        out_C: np.ndarray,
        out_D: np.ndarray,
    ) -> None:
        """Write semi-implicit (C, D) rate split for HI, HII, H2.

        Convention: `dx_i/dt = C_i - D_i * x_i`. The ghost rows
        (electron, CI, CII, CO, OI, OII) are left untouched —
        `fill_ghosts()` is the channel that rebuilds them. Indices
        are read from `state.species` so the network operates uniformly
        whether the strip carries the full 9-species set or a reduced
        layout used by legacy smoke tests.
        """
        idx = state.species.idx
        i_HI = idx['HI']
        i_HII = idx['HII']
        i_H2 = idx['H2']

        # Broadcast scalar config fields up to ncell-shape.
        nH = np.asarray(state.nH)
        T = np.asarray(state.T)
        Z_d = np.asarray(state.Z_d)
        xi_CR = np.asarray(state.xi_CR)

        xHI = state.x[i_HI]
        xHII = state.x[i_HII]
        xH2 = state.x[i_H2]
        # Electron density from the explicit electron row.
        i_e = idx['electron']
        x_e = state.x[i_e]
        ne = nH * x_e

        # Optional radiation-side inputs. Default to zero so the
        # Phase 2 dark/CR-only tests work without state extensions.
        chi_FUV = _get_optional(state, 'chi_FUV', shape=T.shape)
        xi_ph_HI = _get_optional(state, 'xi_ph_HI', shape=T.shape)
        xi_ph_H2 = _get_optional(state, 'xi_ph_H2', shape=T.shape)
        xi_diss_H2 = _get_optional(state, 'xi_diss_H2', shape=T.shape)

        # --- HII channel ---------------------------------------------
        k_coll = _coeff_kcoll_H(T)
        alpha_rr = _coeff_alpha_rr_H(T)
        alpha_gr = _coeff_alpha_gr_H(T, chi_FUV, ne, Z_d)
        xi_cr_HII = xi_CR * (2.3 * xH2 + 1.5 * xHI)

        # Source per unit x_HI: dx_HII/dt = C_HII_per_xHI * x_HI - D_HII * x_HII
        C_HII_per_xHI = k_coll * ne + xi_cr_HII + xi_ph_HI
        D_HII = ne * alpha_rr + nH * alpha_gr

        # --- H2 channel ----------------------------------------------
        # Above TEMP_HOT1, zero out H2 chemistry entirely.
        hot_mask = T >= TEMP_HOT1
        kgr = _coeff_kgr_H2(T, Z_d)
        xi_coll_H2 = _coeff_xi_coll_H2(nH, T, xHI, xH2)
        xi_cr_H2 = 2.0 * xi_CR * (2.3 * xH2 + 1.5 * xHI)
        C_H2_per_xHI = np.where(hot_mask, 0.0, kgr * nH)
        D_H2 = np.where(
            hot_mask,
            0.0,
            1.65 * xi_cr_H2 + xi_ph_H2 + xi_diss_H2 + xi_coll_H2,
        )

        # --- Fold into NetworkBase (C, D x) form ---------------------
        # HII: dx_HII/dt = C_HII_per_xHI * x_HI - D_HII * x_HII
        out_C[i_HII] = C_HII_per_xHI * xHI
        out_D[i_HII] = D_HII

        # H2: dx_H2/dt = C_H2_per_xHI * x_HI - D_H2 * x_H2
        out_C[i_H2] = C_H2_per_xHI * xHI
        out_D[i_H2] = D_H2

        # HI: conservation, dx_HI/dt = -dx_HII/dt - 2 dx_H2/dt
        # Per-species (C_HI, D_HI x_HI) split:
        out_C[i_HI] = D_HII * xHII + 2.0 * D_H2 * xH2
        out_D[i_HI] = C_HII_per_xHI + 2.0 * C_H2_per_xHI

        # Electron row: not integrated here; leave untouched and let
        # closure / driver set it from electron_fraction(state).
        out_C[i_e] = 0.0
        out_D[i_e] = 0.0

    # ---- closure ----------------------------------------------------
    def closure(self, state: Any) -> None:
        """Apply the species floor and enforce
        `x_HI + x_HII + 2 x_H2 = 1` plus rebuild the ghost rows.

        The clamp-and-renormalise pattern mirrors the C++ RAMSES-style
        clipping in `ncr_solver.hpp`:541-555. We work on `state.x`
        directly (no allocation). After enforcing hydrogen
        conservation, `fill_ghosts(state)` repopulates the metal
        ghost rows and the electron row from the new evolved values.
        """
        idx = state.species.idx
        i_HI = idx['HI']
        i_HII = idx['HII']
        i_H2 = idx['H2']

        x = state.x
        floor = self.x_floor

        # Step 1: clamp every fraction at the floor.
        np.maximum(x[i_HI], floor, out=x[i_HI])
        np.maximum(x[i_HII], floor, out=x[i_HII])
        np.maximum(x[i_H2], floor, out=x[i_H2])

        # Step 2: if x_HII + 2 x_H2 > 1, rescale both so the sum is
        # exactly 1 - floor (leaves x_HI at the floor). Otherwise
        # back out x_HI from conservation.
        ion_plus_mol = x[i_HII] + 2.0 * x[i_H2]
        overflow = ion_plus_mol > (1.0 - floor)
        if np.any(overflow):
            scale = np.where(
                overflow,
                (1.0 - floor) / np.maximum(ion_plus_mol, floor),
                1.0,
            )
            x[i_HII] *= scale
            x[i_H2] *= scale

        # Recompute HI from conservation.
        x[i_HI] = np.maximum(1.0 - x[i_HII] - 2.0 * x[i_H2], floor)

        # Step 3: rebuild ghost rows (electron + metals) from the
        # post-conservation evolved values.
        self.fill_ghosts(state)

    # ---- electron_fraction ------------------------------------------
    def electron_fraction(self, state: Any) -> np.ndarray:
        """Return x_e implied by the current ionised-H + metal
        ghosts.

        Reads the explicit electron row when the strip carries one
        (the canonical 9-species `ncr3_with_ghosts` layout), else
        falls back to `x_e = x_HII` (the minimal 4-species network
        with only H II contributing electrons). This method is
        read-only.
        """
        idx = state.species.idx
        i_e = idx.get('electron')
        if i_e is not None:
            return state.x[i_e].copy()
        return state.x[idx['HII']].copy()

    # ---- fill_ghosts -------------------------------------------------
    def fill_ghosts(self, state: Any) -> None:
        """Rebuild the ghost rows of `state.x` in place.

        Implements the algebra of `NCRRates::CoolingOther` in
        ncr_rates.hpp:1505-1540 of tigris-ncr (Chem_flag == 0 branch
        — the full radiation-field-dependent GetxCII / CO chemistry
        arrives with the GOW17 hand-off in Phase 9):

        - x_OII = x_HII * xOstd * Z_g                (Saha-like
          scaling of O+ on H+; no equivalent for OIII/OIV at NCR
          temperatures).
        - x_CII = xCstd * Z_g                        (saturated C+
          at typical FUV exposure; the smooth `GetxCII` form lands
          when chi_FUV / chi_CI / xi_CR / Z_d state inputs reach the
          Python layer).
        - x_CO  = 0                                  (no CO without
          the full GOW17 chain; the proper form arrives in Phase 9).
        - x_OI  = max(xOstd * Z_g - x_OII - x_CO, x_floor)
        - x_CI  = max(xCstd * Z_g - x_CII - x_CO, x_floor)
        - x_e   = max(x_HII + x_CII + x_OII, x_floor)
          (singly-charged ghost ions contribute +1 each; CO and
          OI are neutral.)

        TODO: replace the saturated x_CII with the full GetxCII
        prescription (chi_FUV, chi_CI, xi_CR, Z_d dependent), and
        add the GOW17 CO branch once the corresponding state inputs
        are wired up. Tracked in the chemistry-rewrite-plan §9.

        Networks that operate on a strip without the metal ghost
        rows (e.g., the legacy 4-species minimal set) only get the
        electron row populated; absent ghost names are skipped via
        `state.species.idx.get(name)`.
        """
        idx = state.species.idx
        x = state.x
        floor = self.x_floor

        i_HII = idx['HII']
        xHII = x[i_HII]

        Z_g = np.asarray(state.Z_g)
        xC_tot = self.x_C_std * Z_g
        xO_tot = self.x_O_std * Z_g

        # Resolve ghost row indices once. A `None` means the strip
        # does not carry that row (e.g., the legacy 4-species minimal
        # set has no metal ghost rows), in which case the species
        # contributes zero to charge balance — the network is not
        # meant to invent metals on a strip that does not track them.
        i_OII = idx.get('OII')
        i_OI = idx.get('OI')
        i_CII = idx.get('CII')
        i_CI = idx.get('CI')
        i_CO = idx.get('CO')
        i_e = idx.get('electron')

        # Zero per-cell baselines for charge balance when a metal
        # ghost is absent from the strip.
        zero = np.zeros_like(xHII)

        # OII: positive ion, scales with HII.
        if i_OII is not None:
            x[i_OII] = xHII * xO_tot
            x_OII_arr = x[i_OII]
        else:
            x_OII_arr = zero

        # CII: saturated at standard C abundance (Chem_flag == 0).
        if i_CII is not None:
            x[i_CII] = np.broadcast_to(xC_tot, xHII.shape).copy()
            x_CII_arr = x[i_CII]
        else:
            x_CII_arr = zero

        # CO: zero in the Chem_flag == 0 branch; the floor sits in
        # the storage row so a downstream cooling lookup never sees a
        # negative or zero abundance.
        if i_CO is not None:
            x[i_CO] = np.full_like(xHII, floor)
        x_CO_arr = zero

        # OI: remainder of the oxygen budget, floored.
        if i_OI is not None:
            x[i_OI] = np.maximum(xO_tot - x_OII_arr - x_CO_arr, floor)

        # CI: remainder of the carbon budget, floored. With the
        # Chem_flag == 0 branch x_CII = xCstd*Z_g and x_CO = 0, so
        # x_CI hits the floor — but we still write the expression so
        # the prescription matches the C++ source line by line.
        if i_CI is not None:
            x[i_CI] = np.maximum(xC_tot - x_CII_arr - x_CO_arr, floor)

        # Electron from charge balance. Singly-charged positive ions
        # contribute +1 each; CO and OI are neutral. Strips without
        # metal ghost rows fall back to the H-only count `x_e = x_HII`.
        if i_e is not None:
            x[i_e] = np.maximum(xHII + x_CII_arr + x_OII_arr, floor)


# ---- Helpers -----------------------------------------------------------
def _get_optional(state: Any, attr: str, shape: tuple) -> np.ndarray:
    """Return `getattr(state, attr)` as an array broadcastable to
    `shape`, or zeros of that shape if the attribute is missing /
    None. Used for radiation-side scalars (`chi_FUV`, `xi_ph_*`,
    `xi_diss_H2`) that the Phase 2 ChemState skeleton does not yet
    expose.
    """
    val = getattr(state, attr, None)
    if val is None:
        return np.zeros(shape, dtype=np.float64)
    return np.asarray(val, dtype=np.float64)
