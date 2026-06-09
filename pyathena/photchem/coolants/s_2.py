"""S II 5-level cooling: 3p3 4S + 2D + 2P (N-like, np^3).

Nebular electron-density diagnostic. Nebular lines:
[SII] 6717 + 6731 (2D -> 4S, n_e diagnostic via the doublet ratio),
[SII] 4068 + 4076 (2P -> 4S), [SII] 10287-10370 (2P -> 2D). 5-level
truncation is the standard for np^3 ions; matches CMacIonize.
"""

from .base import IonCoolant

_C = IonCoolant('s_2.txt', label='SII')

populations = _C.populations
cooling = _C.cooling
