============
Microphysics
============

The ``pyathena.microphysics`` module contains classes and functions for
ISM microphysics: cooling functions, chemical equilibrium, dust properties,
photoionization cross-sections, and recombination rates.

Solar Abundances
================

.. autoclass:: pyathena.microphysics.abundance_solar.AbundanceSolar
    :members:

Cooling Functions
=================

.. autoclass:: pyathena.microphysics.cool_gnat12.CoolGnat12
    :members:

.. autoclass:: pyathena.microphysics.cool_rosen95.CoolRosen95
    :members:

.. autoclass:: pyathena.microphysics.cool_wiersma09.CoolWiersma09
    :members:

.. autofunction:: pyathena.microphysics.cool_grackle.cool_grackle

Dust
====

.. autoclass:: pyathena.microphysics.dust_draine.DustDraine
    :members:

Photoionization Cross-Sections
===============================

.. autoclass:: pyathena.microphysics.photx.PhotX
    :members:

Recombination Rates
===================

.. autoclass:: pyathena.microphysics.rec_rate.RecRate
    :members:

H2 Chemistry
============

.. autofunction:: pyathena.microphysics.h2.calc_xH2eq
