"""Unit tests for `pyathena.chemistry.config`.

Covers default values (matched against the tigris-ncr C++
`photchem_ncr.cpp` constructor defaults), `from_dict` round-trip,
boolean / enum coercion, and the minimal athinput parser fallback.
"""
from __future__ import annotations

import textwrap

import pytest

from pyathena.chemistry.config import ChemistryConfig
from pyathena.chemistry.enums import InterpMode


# ---- Defaults match C++ `photchem_ncr.cpp` constructor ----
def test_defaults_match_cpp_constructor():
    """Spot-check every default against the C++ side.

    Source of truth: `src/photchem/photchem_ncr.cpp:30-58` and
    `src/photchem/ncr_rates.hpp:218-242` in tigris-ncr.
    """
    c = ChemistryConfig()
    # photchem_ncr.cpp constructor defaults.
    assert c.cool_hyd_cie_flag is False
    assert c.h2_diss_bg_flag is False
    assert c.hi_phot_bg_flag is False
    assert c.cfl_cool_sub == pytest.approx(0.1)
    assert c.temp_hot0 == pytest.approx(20000.0)
    assert c.temp_hot1 == pytest.approx(35000.0)
    assert c.x_h2_cut == pytest.approx(0.0)
    assert c.x_hi_cut == pytest.approx(0.0)
    assert c.x_hii_cut == pytest.approx(0.0)
    assert c.temp_mu_floor == pytest.approx(2.0)
    assert c.x_floor == pytest.approx(0.0)
    assert c.b5_inv == pytest.approx(1.0 / 3.0)
    assert c.u_rad_pe_isrf_cgs == pytest.approx(7.613e-14)
    assert c.u_rad_lw_isrf_cgs == pytest.approx(1.335e-14)
    assert c.xi_diss_h2_isrf == pytest.approx(5.7e-11)
    assert c.xCstd == pytest.approx(1.6e-4)
    assert c.xOstd == pytest.approx(3.2e-4)
    assert c.temp_dust0 == pytest.approx(5.0)
    assert c.dvdr == pytest.approx(3.240779289444365e-14)

    # ncr_rates.hpp Init defaults.
    assert c.PhotDiss_flag is True
    assert c.Chem_flag is True
    assert c.PhotIon_flag is True
    assert c.CoolHISmith21_flag is True
    assert c.CoolH2rovib_flag is True
    assert c.CoolH2colldiss_flag is True
    assert c.CRPhotC_flag is True
    assert c.kgr_H2_flag is True
    assert c.CoolH2_flag is True
    assert c.HeatH2_flag is True
    assert c.iH2heating == 1
    assert c.iCII_rec_rate == 2
    assert c.iPEheating == 1
    assert c.interp_mode == InterpMode.kLogLog

    # Z and zeta defaults from the constructor (`GetOrAddReal`,
    # not `GetReal`).
    assert c.z_gas == pytest.approx(1.0)
    assert c.z_dust == pytest.approx(1.0)
    assert c.zeta_hi_phot0 == pytest.approx(0.0)
    assert c.solver_type == 'explicit'


def test_temp_mu_floor_positive_and_x_floor_nonneg():
    """Sanity: the floors are well-formed."""
    c = ChemistryConfig()
    assert c.temp_mu_floor > 0.0
    assert c.x_floor >= 0.0


# ---- from_dict ----
def test_from_dict_overrides_defaults():
    c = ChemistryConfig.from_dict({
        'temp_hot0':       1.5e4,
        'temp_hot1':       3.0e4,
        'cool_dust_flag':  True,
        'interp_mode':     0,
        'xCstd':           2.0e-4,
    })
    assert c.temp_hot0 == pytest.approx(1.5e4)
    assert c.temp_hot1 == pytest.approx(3.0e4)
    assert c.cool_dust_flag is True
    assert c.interp_mode == InterpMode.kExact
    assert c.xCstd == pytest.approx(2.0e-4)
    # Untouched defaults still hold.
    assert c.dvdr == pytest.approx(3.240779289444365e-14)


def test_from_dict_round_trip():
    """Encoding -> decode is behaviour-preserving."""
    original = ChemistryConfig.from_dict({
        'temp_hot0':       1.5e4,
        'cool_dust_flag':  True,
        'interp_mode':     2,
    })
    d = original.to_dict()
    # Round-tripping through from_dict reconstructs the same config.
    roundtrip = ChemistryConfig.from_dict(d)
    assert roundtrip == original


def test_from_dict_unknown_keys_land_in_extra():
    c = ChemistryConfig.from_dict({
        'mystery_key':  'mystery_value',
        'cfl_cool_sub': 0.2,
    })
    assert c.cfl_cool_sub == pytest.approx(0.2)
    assert c.extra == {'mystery_key': 'mystery_value'}


def test_bool_coercion_from_int_and_string():
    """athinput parsers can return 0/1 ints or 'true'/'false' strings
    for boolean keys."""
    c = ChemistryConfig.from_dict({
        'cool_dust_flag':     1,
        'cool_hyd_cie_flag':  'true',
        'h2_diss_bg_flag':    0,
        'hi_phot_bg_flag':    'false',
    })
    assert c.cool_dust_flag is True
    assert c.cool_hyd_cie_flag is True
    assert c.h2_diss_bg_flag is False
    assert c.hi_phot_bg_flag is False


def test_interp_mode_accepts_int_and_enum():
    assert (ChemistryConfig.from_dict({'interp_mode': 3}).interp_mode
            == InterpMode.kNqt1)
    assert (ChemistryConfig
            .from_dict({'interp_mode': InterpMode.kNqt2}).interp_mode
            == InterpMode.kNqt2)


# ---- from_athinput ----
def test_from_athinput_minimal_block(tmp_path):
    """Construct from a hand-rolled athinput with a `[photchem_ncr]`
    block plus extras the parser must ignore.
    """
    path = tmp_path / 'athinput.test'
    path.write_text(textwrap.dedent("""\
        # comment line
        <comment>
        problem    = sanity check

        <photchem_ncr>
        cool_dust_flag    = true
        temp_hot0         = 18000.0
        temp_hot1         = 33000.0
        cfl_cool_sub      = 0.05
        interp_mode       = 2
        xCstd             = 1.4e-4
        # trailing comment is stripped
        xOstd             = 3.0e-4

        <other_block>
        ignored_key       = 42
    """))
    c = ChemistryConfig.from_athinput(str(path))
    assert c.cool_dust_flag is True
    assert c.temp_hot0 == pytest.approx(18000.0)
    assert c.temp_hot1 == pytest.approx(33000.0)
    assert c.cfl_cool_sub == pytest.approx(0.05)
    assert c.interp_mode == InterpMode.kNqt2
    assert c.xCstd == pytest.approx(1.4e-4)
    assert c.xOstd == pytest.approx(3.0e-4)
    # Defaults untouched.
    assert c.dvdr == pytest.approx(3.240779289444365e-14)


def test_from_athinput_alternate_block(tmp_path):
    """The `block` argument selects which `[...]` section to consume."""
    path = tmp_path / 'athinput.alt'
    path.write_text(textwrap.dedent("""\
        <my_block>
        temp_hot0 = 12345.0
    """))
    c = ChemistryConfig.from_athinput(str(path), block='my_block')
    assert c.temp_hot0 == pytest.approx(12345.0)


def test_to_dict_round_trip_preserves_extra(tmp_path):
    c1 = ChemistryConfig.from_dict({
        'temp_hot0': 11000.0,
        'mystery':   'kept',
    })
    c2 = ChemistryConfig.from_dict(c1.to_dict())
    assert c2.temp_hot0 == pytest.approx(11000.0)
    assert c2.extra == {'mystery': 'kept'}
