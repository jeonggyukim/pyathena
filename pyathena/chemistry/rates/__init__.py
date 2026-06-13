"""Atomic / molecular rate-coefficient modules.

Five modules:

- `photx` — Verner+96 photoionisation cross sections
- `ci_rate` — Voronov 1997 collisional ionisation
- `rec_rate` — Badnell RR + DR; Draine 2011 H case-B
- `ct_rate` — Cloudy + UGA charge-transfer database

Each is a leaf module with only `numpy`, `astropy`, and
`pyathena.chemistry.datapaths` for dependencies. Compatibility ports
of `pyathena.microphysics.{photx, ci_rate, rec_rate, ct_rate}` with
identical public API and numerical behaviour; parity tests under
`tests/chemistry/parity/` pin agreement at rtol = 1e-12.
"""

from .ci_rate import CollIonRate, CollIonRateCHIANTI
from .ct_rate import ChargeTransferRate
from .photx import PhotX, get_sigma_pi_H2
from .rec_rate import RecRate, RecRateCHIANTI

__all__ = [
    "CollIonRate",
    "CollIonRateCHIANTI",
    "ChargeTransferRate",
    "PhotX",
    "get_sigma_pi_H2",
    "RecRate",
    "RecRateCHIANTI",
]
