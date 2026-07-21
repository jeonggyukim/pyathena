"""Factory helpers for assembling production-default cooling stacks.

`make_ncr_default_cooling(species, *, xi_CR, xi_diss_H2, chi_band)`
returns a `CoolingChannels` aggregator wired with the NCR production
channel set (Lya / HICollIon / HRecomb / FreeFreeH / CII / OI / CI /
H2Moseley21 / H2CollDiss / Nebular / DustGasCoupling /
GrainRecombination + heating PE / CR / H2Form / H2Diss / H2Pump). The
NCR convention is the one wired by
`pyathena.microphysics.get_cooling.py` and the C++
`PhotochemistryNCR` driver.
"""
from __future__ import annotations

from .base import CoolingChannels
from .lya import LyaCooling
from .hi_collisional_ionization import HICollisionalIonizationCooling
from .recombination_hydrogen import HRecombinationCooling
from .free_free import FreeFreeHCooling
from .cii import CIIFineStructureCooling
from .oi import OIFineStructureCooling
from .ci import CIFineStructureCooling
from .h2_moseley21 import H2Moseley21Cooling
from .h2_colldiss import H2CollDissCooling
from .nebular import NebularMetalLineCooling
from .dust import DustGasCoupling
from .grain_recombination import GrainRecombinationCooling

from ..heating.photoelectric import PhotoelectricHeating
from ..heating.cosmic_ray import CosmicRayHeating
from ..heating.h2_formation import H2FormationHeating
from ..heating.h2_photodissociation import (
    H2DissociationHeating, H2PumpHeating,
)


def make_ncr_default_cooling(
    species,
    *,
    xi_CR: float = 2.0e-16,
    xi_diss_H2: float = 0.0,
    kgr_H2: float = 3.0e-17,
    chi_band: str = 'FUV',
) -> CoolingChannels:
    """Compose the NCR production cooling + heating channel set.

    Parameters
    ----------
    species : SpeciesSet
        Must include 'HI', 'HII', 'H2', 'electron', 'CI', 'CII', 'OI',
        'OII'. The factory looks up indices via `species.idx`.
    xi_CR : float
        Cosmic-ray ionization rate per H, s^-1. Default 2e-16 (NCR).
    xi_diss_H2 : float
        H2 photodissociation rate, s^-1. Default 0 (no LW field).
    kgr_H2 : float
        H2 grain-formation rate coefficient, cm^3 / s at Z_d = 1.
        Default 3e-17.
    chi_band : str
        Radiation band name to read for PE / grain rec / pump
        heating. Default 'FUV'.
    """
    idx = species.idx
    cooling = (
        LyaCooling(i_HI=idx['HI'], i_electron=idx['electron']),
        HICollisionalIonizationCooling(
            i_HI=idx['HI'], i_electron=idx['electron']),
        HRecombinationCooling(
            i_HII=idx['HII'], i_electron=idx['electron']),
        FreeFreeHCooling(
            i_HII=idx['HII'], i_electron=idx['electron']),
        CIIFineStructureCooling(
            i_HI=idx['HI'], i_H2=idx['H2'],
            i_CII=idx['CII'], i_electron=idx['electron']),
        OIFineStructureCooling(
            i_HI=idx['HI'], i_H2=idx['H2'],
            i_OI=idx['OI'], i_electron=idx['electron']),
        CIFineStructureCooling(
            i_HI=idx['HI'], i_H2=idx['H2'],
            i_CI=idx['CI'], i_electron=idx['electron']),
        H2Moseley21Cooling(i_HI=idx['HI'], i_H2=idx['H2']),
        H2CollDissCooling(i_HI=idx['HI'], i_H2=idx['H2']),
        NebularMetalLineCooling(
            i_HII=idx['HII'], i_electron=idx['electron']),
        DustGasCoupling(),
        GrainRecombinationCooling(
            i_electron=idx['electron'], chi_band=chi_band),
    )
    heating = (
        PhotoelectricHeating(
            i_electron=idx['electron'], chi_band=chi_band),
        CosmicRayHeating(
            i_HI=idx['HI'], i_H2=idx['H2'],
            i_electron=idx['electron'], xi_CR=xi_CR),
        H2FormationHeating(
            i_HI=idx['HI'], i_H2=idx['H2'],
            kgr_H2=kgr_H2, xi_diss_H2=xi_diss_H2,
            temperature_dependent_kgr=True),
        H2DissociationHeating(
            i_H2=idx['H2'], xi_diss_H2=xi_diss_H2),
        H2PumpHeating(
            i_HI=idx['HI'], i_H2=idx['H2'],
            xi_diss_H2=xi_diss_H2, form='HM79'),
    )
    return CoolingChannels(channels=cooling, heating=heating)
