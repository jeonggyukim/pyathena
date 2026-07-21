"""S I 5-level cooling: 3p4 3P + 1D + 1S (O-like, np^4).

Cold neutral coolant. Nebular lines: [SI] 25.25 um (3P_1 -> 3P_2),
[SI] 56.31 um (3P_0 -> 3P_1), [SI] 1.082 um (1D2 -> 3P_2), [SI]
4589 (1S0 -> 1D2).
"""

from .base import IonCoolant

_C = IonCoolant('s_1.txt', label='SI')

populations = _C.populations
cooling = _C.cooling
