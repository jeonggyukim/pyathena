"""Parity test: O II + H I <-> O I + H II charge transfer wiring.

Lifted from `tests/microphysics/test_OII_HII_resonance.py`. The
Phase 0 contract is that the chemistry-side path delegates to the
microphysics-side path, so this test is green by construction. As
the real `pyathena.chemistry.networks.ncr3_plus_ions16` lands in
Phase 6, the new-path callable here is rebound to it and the
delegate is removed.

Until then this test exists to prove the parity harness wiring:
both packages import cleanly, `run_both` works, `assert_close`
catches mismatches. Failure here means the harness itself is broken.

The stub builder is reused from the microphysics test rather than
copied; the parity test should not duplicate setup code because
divergent stubs would mask a real harness bug.
"""
from __future__ import annotations

import sys
import os

# Make the sibling microphysics test directory importable for the
# stub builder.
_MICROPHYSICS_TESTS = os.path.normpath(
    os.path.join(os.path.dirname(__file__),
                 '..', '..', 'microphysics'))
if _MICROPHYSICS_TESTS not in sys.path:
    sys.path.insert(0, _MICROPHYSICS_TESTS)

from test_OII_HII_resonance import _make_stub_for_CT_only  # noqa: E402

from pyathena.chemistry import _parity
from pyathena.microphysics.photchem import PhotChem


def _compute_metal_CT_old(stub):
    """Microphysics-side: call the bound stub method (delegates to
    PhotChem._compute_metal_CT_fluxes)."""
    return stub._compute_metal_CT_fluxes(stub.Tion)


def _compute_metal_CT_new(stub):
    """Chemistry-side: Phase 0 delegate. Replaced in Phase 6 by the
    real `NCRNetwork3PlusIons16.evaluate_metal_CT_prepass`.
    """
    return stub._compute_metal_CT_fluxes(stub.Tion)


def test_parity_metal_CT_fluxes_at_HII_conditions():
    """Run old and new on the same stub, assert byte-identical output.

    Phase 0: the new path delegates to the old, so this is necessarily
    exact. The point is to exercise `run_both` and `assert_close`
    end-to-end.
    """
    stub = _make_stub_for_CT_only(n_H=100.0, x_HI=0.5, x_OI=0.7, T=1.0e4)
    out_old, out_new = _parity.run_both(
        _compute_metal_CT_old, _compute_metal_CT_new, stub)
    _parity.assert_close(out_old, out_new, rtol=0.0, atol=0.0,
                         label='metal_CT_fluxes_HII')


def test_parity_metal_CT_fluxes_at_CNM_conditions():
    """A second condition (cold, mostly neutral) to confirm the
    harness works across temperature regimes too. Same delegation in
    Phase 0; same exact match.
    """
    stub = _make_stub_for_CT_only(n_H=30.0, x_HI=0.99, x_OI=0.999, T=100.0)
    out_old, out_new = _parity.run_both(
        _compute_metal_CT_old, _compute_metal_CT_new, stub)
    _parity.assert_close(out_old, out_new, rtol=0.0, atol=0.0,
                         label='metal_CT_fluxes_CNM')
