"""C I 5-level cooling: 2s2 2p2 3P + 1D + 1S (C-like, np^2).

Cold neutral gas / PDR coolant. Nebular lines: [CI] 9824, 9850
(1D2 -> 3P), [CI] 8727 (1S0 -> 1D2), [CI] 370 + 609 um (3P FS,
dominant FIR coolants in dark cloud cores).
"""

from .base import IonCoolant

_C = IonCoolant('c_1.txt', label='CI')

populations = _C.populations
cooling = _C.cooling
