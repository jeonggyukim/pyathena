"""S III 5-level cooling: 3p2 3P + 1D + 1S (C-like, np^2).

HII-region tracer. Nebular lines: [SIII] 9069 + 9532
(1D2 -> 3P_1 + 3P_2; near-IR doublet), [SIII] 6312 (1S0 -> 1D2),
[SIII] 18.7 + 33.5 um (3P FS).
"""

from .base import IonCoolant

_C = IonCoolant('s_3.txt', label='SIII')

populations = _C.populations
cooling = _C.cooling
