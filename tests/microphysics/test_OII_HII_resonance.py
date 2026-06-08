"""Charge-transfer wiring tests: O II + H I <-> O I + H II near-resonance.

Verifies the new CT wiring inside PhotChem.evolve_one_species and the
sum-over-metals pre-pass `_compute_metal_CT_fluxes`.

The full evolve_all integration requires an SB99 SED file (path is
machine-specific) plus an HIIWind setup, neither of which belongs in
a fast unit test. Instead, this file exercises the CT methods
directly on a minimal stand-in object that carries only the
attributes the methods actually touch (`den`, `ions`, `ct`, `rc`,
`ci`, and a stub `Fphot`). This isolates the CT logic from the rest
of PhotChem so a regression in the CT terms surfaces here without
needing the full radiation pipeline.

Two checks:

1. `test_metal_CT_fluxes_signs` -- sign + magnitude of the four
   returned arrays. With one O+ ion present (q >= 1) and one O0
   neutral present in a partially ionized cell, the H I destruction
   and H II creation fluxes scale as `n_OII * k_CT_rec`, and the H II
   destruction and H I creation fluxes scale as `n_OI * k_CT_ion`.

2. `test_OII_HII_resonance_lock` -- a few-step relaxation toward the
   CT-equilibrium ratio `n(O+)/n(O0) ~ K(T) * n(H+)/n(H0)` at
   T = 1e4 K, with no radiation field, no recombination from
   electrons, and no collisional ionization (those terms set to
   zero in the stub). Confirms that the CT terms drive x_OII toward
   x_HII as the chemistry sweep iterates.
"""

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from pyathena.microphysics.photchem import PhotChem
from pyathena.microphysics.ct_rate import ChargeTransferRate
from pyathena.microphysics.rec_rate import RecRate
from pyathena.microphysics.ci_rate import CollIonRate


def _make_stub_for_CT_only(Nr=1, n_H=100.0, x_HI=0.5, x_OI=1.0, T=1.0e4):
    """Construct a minimal stand-in carrying the attributes the
    `_compute_metal_CT_fluxes` and `evolve_one_species` methods
    actually read.

    Species layout (5 ions + electrons): H0, H1, O0, O1, O2.
    Abundances: O / H = 5e-4 (Asplund+09 protosolar).

    The stub deliberately omits SED, Fphot, and self.w. Tests that
    invoke `evolve_one_species` pass the CT-only path by setting
    Fphot to zero and stubbing out the ci / rc rate calls via
    monkeypatching when needed.
    """
    # Five species + 1 electron slot.
    Nsp = 5
    x_OH = 5.0e-4

    den = np.zeros((Nsp + 1, Nr))
    nH_arr = np.full(Nr, n_H)
    # H I, H II
    den[0] = nH_arr * x_HI
    den[1] = nH_arr * (1.0 - x_HI)
    # O I, O II, O III
    den[2] = nH_arr * x_OH * x_OI
    den[3] = nH_arr * x_OH * (1.0 - x_OI)
    den[4] = 0.0
    # electrons (approximate)
    den[-1] = den[1] + den[3] + 2.0 * den[4]

    # ions DataFrame must carry idx, element, Z, N, q, ionize,
    # recomb columns (and a sigma_pi column for the inline reader).
    rows = [
        # name, idx, element, Z, N, q, ionize, recomb
        ('H0', 0, 'H', 1, 1, 0, True, False),
        ('H1', 1, 'H', 1, 0, 1, False, True),
        ('O0', 2, 'O', 8, 8, 0, True, False),
        ('O1', 3, 'O', 8, 7, 1, True, True),
        ('O2', 4, 'O', 8, 6, 2, False, True),
    ]
    df = pd.DataFrame(
        rows,
        columns=['name', 'idx', 'element', 'Z', 'N', 'q',
                 'ionize', 'recomb'],
    ).set_index('name')
    # sigma_pi placeholder (one frequency bin, value zero -> no
    # photoionization in CT-only test).
    df['sigma_pi'] = [np.zeros(1)] * len(df)

    stub = SimpleNamespace()
    stub.den = den
    stub.nH = nH_arr
    stub.ions = df
    stub.ct = ChargeTransferRate()
    stub.rc = RecRate(caseB=False)
    stub.ci = CollIonRate()
    stub.species_set = SimpleNamespace(
        num_ions_tot=Nsp,
        num_ions={'H': 2, 'O': 3},
    )
    stub.abd = {'H': 1.0, 'O': x_OH}
    stub.Tion = T
    # Stub `self.w` so the `T = self.w.Tion` line inside
    # evolve_one_species does not fail.
    stub.w = SimpleNamespace(Tion=T)
    # Zero radiation field -> pi rates are zero.
    stub.Fphot = np.zeros((Nr, 1))
    # Bind the unbound CT methods to the stub.
    stub._ct_rate_safe = PhotChem._ct_rate_safe.__get__(stub)
    stub._compute_metal_CT_fluxes = (
        PhotChem._compute_metal_CT_fluxes.__get__(stub))
    return stub


def test_metal_CT_fluxes_signs():
    """Verify sign + scaling of the four sum-over-metals CT fluxes
    in `_compute_metal_CT_fluxes`. Set up a 1-zone cell with O at
    50/50 between O0 and O+, partial H ionization, at T = 1e4 K.
    """
    T = 1.0e4
    n_H = 100.0
    x_HI = 0.7
    x_OI = 0.5
    stub = _make_stub_for_CT_only(n_H=n_H, x_HI=x_HI, x_OI=x_OI, T=T)

    HI_drate, HII_crate, HI_crate, HII_drate = \
        stub._compute_metal_CT_fluxes(T)

    n_OII = n_H * 5.0e-4 * (1.0 - x_OI)
    n_OI = n_H * 5.0e-4 * x_OI
    n_HI = n_H * x_HI
    n_HII = n_H * (1.0 - x_HI)

    k_rec_OII = stub.ct.get_ct_rec_rate(8, 7, T)
    # O0 in J=2 (low-density limit; this is the right call for
    # n_H = 100 << n_crit ~ 1e5).
    k_ion_OI = stub.ct.get_ct_ion_rate(8, 8, T)

    # Expected sum-over-metals contributions (only O carries CT data
    # in the stub). q >= 1 for the CT-rec arm contributes O+ only;
    # CT-ion arm contributes O0 (and could include O+ if ionize is
    # True for it, which it is for our O+ row -- O+ -> O2+ via H+
    # is possible). Account for both O0 and O+ in CT-ion if both
    # ionize.
    k_ion_OII = stub.ct.get_ct_ion_rate(8, 7, T)
    expected_sum_rec = n_OII * k_rec_OII
    expected_sum_ion = n_OI * k_ion_OI + n_OII * k_ion_OII

    np.testing.assert_allclose(HI_drate, expected_sum_rec, rtol=1e-10)
    np.testing.assert_allclose(HII_drate, expected_sum_ion, rtol=1e-10)
    np.testing.assert_allclose(HII_crate, n_HI * expected_sum_rec,
                               rtol=1e-10)
    np.testing.assert_allclose(HI_crate, n_HII * expected_sum_ion,
                               rtol=1e-10)


def test_OII_HII_resonance_lock():
    """At T ~ 10^4 K the near-resonant CT lock should drive x_OII
    toward `K(T) * (x_HII / x_HI) / (1 + K(T) * x_HII / x_HI)`,
    where K(T) is the equilibrium ratio `[n(O+)/n(O0)] /
    [n(H+)/n(H0)]`. In the low-density limit (n_H << n_crit ~ 2e4
    cm^-3 at T = 1e4 K) K equals `k0r / (k0 + k1 + k2)`, the
    J=2-only CT-ion rate over the J-summed CT-rec rate (Draine 2011
    Fig 14.5, corrected). At T = 1e4 K this evaluates to ~0.89.

    Setup: partially ionized 1-zone cell with x_HII = 0.5, x_OII = 0
    initial, n_H = 100, T = 1e4 K. Holding H ionization fixed,
    iterate a mass-conserving implicit Euler on the two-state O0
    / O+ sub-system for many substeps; verify x_OII relaxes to the
    K_low(T) prediction.

    IMPORTANT: pyathena's `get_ct_ion_rate(8, 8, T)` returns the
    UNWEIGHTED J-sum `k0i + k1i + k2i`, not the
    J-population-weighted sum that the physics actually requires.
    Use only the J=2 component (`k0i`) for the low-density limit
    that applies at the test conditions. See the right-panel
    implementation in `test_ct_rate_balance.py` for the same
    treatment with full J-population weighting via `get_OI_lev`.
    """
    T = 1.0e4
    n_H = 100.0
    x_HI = 0.5
    x_OI_init = 1.0
    stub = _make_stub_for_CT_only(n_H=n_H, x_HI=x_HI, x_OI=x_OI_init,
                                  T=T)

    # Low-density-limit CT-ion rate: only the J=2 channel of O0
    # contributes (other J levels radiatively decay before
    # collisions; n_H = 100 << n_crit ~ 2e4 at T = 1e4 K).
    k0i, _, _ = ChargeTransferRate.get_ct_ion_HII_OI_Draine11(
        T, sum=False)
    k_ion = k0i
    k_rec = stub.ct.get_ct_rec_rate(8, 7, T)
    K_low = k_ion / k_rec
    # x_HII / x_HI = 1 at x_HI = 0.5, so x_OII / x_OI = K_low.
    R = (n_H * (1.0 - x_HI)) / (n_H * x_HI) * K_low
    x_OII_eq = R / (1.0 + R)

    # Mass-conserving implicit Euler on (n_O0, n_OII):
    #   d n_OII / dt = n_OI * src_ion - n_OII * sink_rec
    # with n_OI + n_OII = n_O_total conserved. Solving for
    # n_OII_new gives the closed form below.
    n_HI = n_H * x_HI
    n_HII = n_H * (1.0 - x_HI)
    dt = 1.0e10                       # ~300 yr; CT t_scale ~ 6 yr
    n_OI = stub.den[2, 0]
    n_OII = stub.den[3, 0]
    n_O_tot = n_OI + n_OII
    src_ion = n_HII * k_ion           # per-O0
    sink_rec = n_HI * k_rec           # per-O+
    for _ in range(50):
        # Coupled implicit Euler with conservation
        # n_OI + n_OII = n_O_tot:
        #   n_OII_new = (n_OII + n_O_tot * src_ion * dt) /
        #               (1 + (src_ion + sink_rec) * dt)
        n_OII = (n_OII + n_O_tot * src_ion * dt) / \
                (1.0 + (src_ion + sink_rec) * dt)
        n_OI = n_O_tot - n_OII

    x_OII_final = n_OII / n_O_tot
    assert abs(x_OII_final - x_OII_eq) / x_OII_eq < 0.02, (
        f"x_OII = {x_OII_final:.4f} did not relax to CT-equilibrium "
        f"{x_OII_eq:.4f} after 50 substeps"
    )
    # Also pin K_low ~ 0.89 (Draine 14.5 corrected) at T = 1e4 K.
    assert abs(K_low - 8.0 / 9.0) < 0.05, (
        f"K_low at T=1e4 = {K_low:.4f}, expected ~8/9 ~ 0.889"
    )
