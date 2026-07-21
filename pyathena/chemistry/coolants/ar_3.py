"""Ar III 5-level cooling: 3p4 3P + 1D + 1S (O-like, np^4).

HII-region tracer. Nebular lines: [Ar III] 7136 + 7751
(1D2 -> 3P_2 + 3P_1), [Ar III] 5191 (1S0 -> 1D2), IR [Ar III] 9 um.
"""

from .base import IonCoolant

_C = IonCoolant('ar_3.txt', label='ArIII')

populations = _C.populations
cooling = _C.cooling
