"""C II 2-level cooling: 2p 2P doublet (B-like).

[CII] 158 um (2P_3/2 -> 2P_1/2) is the dominant FIR cooling line of
the PDR / CNM. The ion has only 2 levels in its ground 2P term;
5-level solver pads with high-E dummies (which contribute zero).
This is EXACT for CII at HII / PDR temperatures.
"""

from .base import IonCoolant

_C = IonCoolant('c_2.txt', label='CII')

populations = _C.populations
cooling = _C.cooling
