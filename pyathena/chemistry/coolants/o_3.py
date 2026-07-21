"""O III 5-level cooling: 2s2 2p2 3P + 1D + 1S.

Major HII-region nebular coolant: [OIII] 5008 + 4960 (1D2 -> 3P)
and [OIII] 4364 (1S0 -> 1D2, T_e diagnostic with the 5008+4960
sum). IR fine-structure [OIII] 52 + 88 um cooling (3P_J transitions)
is underestimated by ~factor 100 at low n_e compared to a
many-level CHIANTI calculation -- the standard nebular 5-level
truncation misses higher-level radiative cascades that populate
3P_J. This is the same limitation that CMacIonize, MOCASSIN, and
pre-Stout Cloudy share. Total cooling per OIII can be ~3x too low
at n_e <~ 1e3 cm^-3 vs ChiantiPy.populate() full populate; cooling
via 5008+4960 itself is correct.
"""

from .base import IonCoolant

_C = IonCoolant('o_3.txt', label='OIII')

populations = _C.populations
cooling = _C.cooling
