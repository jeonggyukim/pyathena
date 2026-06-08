======
Fields
======

The ``pyathena.fields`` module provides a registry of derived fields and their
visualization metadata. A *derived field* is any quantity that can be computed
from the primitive fields stored in the simulation output (density, pressure,
velocity, magnetic field, etc.).

DerivedFields
=============

.. autoclass:: pyathena.fields.fields.DerivedFields
    :members:

Derived Field Builders
======================

Each ``set_derived_fields_*`` function populates the registry for a specific
physics category. They are called automatically by :class:`~pyathena.fields.fields.DerivedFields`
based on the simulation configuration.

.. autofunction:: pyathena.fields.fields.set_derived_fields_def
.. autofunction:: pyathena.fields.fields.set_derived_fields_cooling
.. autofunction:: pyathena.fields.fields.set_derived_fields_mag
.. autofunction:: pyathena.fields.fields.set_derived_fields_rad
.. autofunction:: pyathena.fields.fields.set_derived_fields_newcool
.. autofunction:: pyathena.fields.fields.set_derived_fields_sixray
.. autofunction:: pyathena.fields.fields.set_derived_fields_xray
.. autofunction:: pyathena.fields.fields.set_derived_fields_cosmic_ray
.. autofunction:: pyathena.fields.fields.set_derived_fields_feedback_scalars

X-ray Emissivity
================

.. note::
   This module requires ``yt`` as an optional dependency. It is imported
   lazily; if ``yt`` is not installed, X-ray derived fields will be unavailable.

.. autoclass:: pyathena.fields.xray_emissivity.XrayEmissivityIntegrator
    :members:

.. autofunction:: pyathena.fields.xray_emissivity.get_xray_emissivity
