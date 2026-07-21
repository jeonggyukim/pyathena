"""O I 5-level cooling: 2p4 3P (inv) + 1D + 1S (O-like, np^4).

Dominant FIR cooling in PDR / CNM via the 3P fine-structure
transitions [OI] 63 um (3P_1 -> 3P_2) and 146 um (3P_0 -> 3P_1).
Optical / UV transitions: [OI] 6300 + 6364 (1D2 -> 3P_2 + 3P_1),
[OI] 5577 (1S0 -> 1D2). The 3P term is inverted (Hund's 3rd; J=2
is ground). H I is the dominant collider in the cold neutral gas
where O I lives.
"""

from .base import IonCoolant

_C = IonCoolant('o_1.txt', label='OI')

populations = _C.populations
cooling = _C.cooling
