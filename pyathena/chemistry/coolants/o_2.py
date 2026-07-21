"""O II 5-level cooling: 2p3 4S + 2D + 2P (N-like, np^3).

HII-region tracer. Nebular lines: [OII] 3726 + 3729 (2D -> 4S,
n_e diagnostic), [OII] 7320 + 7330 (2P -> 2D auroral), [OII] 2470
(2P -> 4S).
"""

from .base import IonCoolant

_C = IonCoolant('o_2.txt', label='OII')

populations = _C.populations
cooling = _C.cooling
