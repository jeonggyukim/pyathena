"""N I 5-level cooling: 2p3 4S + 2D + 2P (N-like, np^3).

Cold neutral gas coolant. Nebular lines: [NI] 5198 + 5201
(2D -> 4S), 10405 (2P -> 2D); the IR fine-structure within 2D and
2P contribute minor cooling in PDR / CNM.
"""

from .base import IonCoolant

_C = IonCoolant('n_1.txt', label='NI')

populations = _C.populations
cooling = _C.cooling
