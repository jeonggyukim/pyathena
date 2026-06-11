"""N III 2-level cooling: 2s2 2p 2P doublet (B-like).

[NIII] 57.3 um (2P_3/2 -> 2P_1/2) is the main FIR cooling line of
NIII in HII regions; same structure as [CII] 158 um. The ion has
2 levels in its ground 2P term; the 5-level solver fills the rest
with high-energy dummy levels (zero contribution). Exact for NIII
at HII temperatures.
"""

from .base import IonCoolant

_C = IonCoolant('n_3.txt', label='NIII')

populations = _C.populations
cooling = _C.cooling
