"""N II 5-level cooling: 2s2 2p2 3P + 1D + 1S (C-like, np^2).

Major HII-region nebular coolant. Nebular lines: [NII] 6548 + 6584
(1D2 -> 3P_1 + 3P_2), [NII] 5755 (1S0 -> 1D2, T_e diagnostic). IR
fine-structure [NII] 122 + 205 um in the 3P term. Same 5-level
limitation as O III; the dominant nebular optical lines are
captured correctly.
"""

from .base import IonCoolant

_C = IonCoolant('n_2.txt', label='NII')

populations = _C.populations
cooling = _C.cooling
