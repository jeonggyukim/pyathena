"""Ar IV 5-level cooling: 3p3 4S + 2D + 2P (N-like, np^3).

HII-region high-q tracer. Nebular lines: [Ar IV] 4711 + 4740
(2D -> 4S).
"""

from .base import IonCoolant

_C = IonCoolant('ar_4.txt', label='ArIV')

populations = _C.populations
cooling = _C.cooling
