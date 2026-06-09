"""Ne II 2-level cooling: 2p5 2P doublet (F-like, inverted).

[NeII] 12.81 um (2P_1/2 -> 2P_3/2, FS within ground term). The 2P
term is inverted (Hund's 3rd; J=3/2 is ground for the 2p hole
configuration). Two-level system is EXACT here.
"""

from .base import IonCoolant

_C = IonCoolant('ne_2.txt', label='NeII')

populations = _C.populations
cooling = _C.cooling
