"""Ne III 5-level cooling: 2p4 3P (inv) + 1D + 1S (O-like, np^4).

HII-region nebular coolant. Nebular lines: [Ne III] 3869 + 3968
(1D2 -> 3P_2 + 3P_1), [Ne III] 15.55 um (3P_1 -> 3P_2 FS), [Ne III]
36 um (3P_0 -> 3P_1). 3P term is inverted due to Hund's 3rd
(more-than-half-filled shell): J=2 is the ground sublevel.
"""

from .base import IonCoolant

_C = IonCoolant('ne_3.txt', label='NeIII')

populations = _C.populations
cooling = _C.cooling
