"""Runtime configuration for the chemistry / thermal stack.

`ChemistryConfig` mirrors the `[photchem_ncr]` block of a tigris-ncr
athinput file: one field per `GetOrAdd*` key. Defaults match the C++
`GetOrAdd*` defaults byte-for-byte where determinable, so loading an
empty athinput on the Python side yields the same configuration the
C++ solver would see.

Use one of the factory functions:

- `ChemistryConfig.from_athinput(path)` -- read an existing athinput
  file. Only the `[photchem_ncr]` block is consumed; everything else
  is ignored.
- `ChemistryConfig.from_dict(d)` -- accept a plain dict (e.g. from a
  notebook).
- `ChemistryConfig()` -- defaults only.

See `pyathena/microphysics/photchem.py` for the legacy multi-ion sweep
and `src/photchem/photchem_ncr.cpp` constructor in the tigris-ncr port
for the authoritative C++ defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from typing import Any, Mapping, Optional

from .enums import InterpMode


# Default `dvdr` from `photchem_ncr.cpp:57`. The numeric value is the
# Tigris code unit conversion `1 km/s/pc` to code units, kept here as
# a magic constant to match the C++ exactly.
_DVDR_DEFAULT: float = 3.240779289444365e-14


@dataclass
class ChemistryConfig:
    """Runtime parameters for the NCR chemistry policy.

    One field per `GetOrAdd*` key in `[photchem_ncr]`. Mutable: callers
    are free to override fields after `from_athinput` / `from_dict`
    returns. The driver reads this once at construction; live edits
    after construction do not affect the running solver.
    """

    # ---- Boolean flags ----
    # Trailing underscore on field names is dropped here because Python
    # users do not benefit from the C++ convention; the C++ port reads
    # the same input-file keys (no underscore) via `GetOrAdd*`.
    cool_dust_flag:     bool = False
    cool_hyd_cie_flag:  bool = False
    h2_diss_bg_flag:    bool = False
    hi_phot_bg_flag:    bool = False

    # Physics-model selectors (`NCRRates::Init` reads these).
    PhotDiss_flag:      bool = True
    Chem_flag:          bool = True
    PhotIon_flag:       bool = True
    CoolHISmith21_flag: bool = True
    CoolH2rovib_flag:   bool = True
    CoolH2colldiss_flag: bool = True
    CRPhotC_flag:       bool = True
    kgr_H2_flag:        bool = True
    CoolH2_flag:        bool = True
    HeatH2_flag:        bool = True

    # ---- Temperature thresholds and floors ----
    # Sigmoid blend window for the hot/cold transition; matches the
    # C++ `temp_hot0_` / `temp_hot1_`.
    temp_hot0:          float = 20000.0
    temp_hot1:          float = 35000.0
    temp_mu_floor:      float = 2.0
    temp_dust0:         float = 5.0

    # ---- Substep solver knobs ----
    cfl_cool_sub:       float = 0.1
    # nsub_max has no default in the C++ side (`GetInteger`, no
    # default); we keep an explicit Python default for ease of use,
    # but the driver should override on its way in to match the
    # production input.
    nsub_max:           int = 1000

    # ---- Abundance cutoffs ----
    x_h2_cut:           float = 0.0
    x_hi_cut:           float = 0.0
    x_hii_cut:          float = 0.0
    x_floor:            float = 0.0

    # ---- Numerical / unit-conversion parameters ----
    b5_inv:             float = 1.0 / 3.0

    # ---- ISRF / standard abundances / dust ----
    u_rad_pe_isrf_cgs:  float = 7.613e-14
    u_rad_lw_isrf_cgs:  float = 1.335e-14
    xi_diss_h2_isrf:    float = 5.7e-11
    xCstd:              float = 1.6e-4
    xOstd:              float = 3.2e-4
    dvdr:               float = _DVDR_DEFAULT

    # ---- Multi-valued selectors ----
    iH2heating:         int = 1
    iCII_rec_rate:      int = 2
    iPEheating:         int = 1
    interp_mode:        InterpMode = InterpMode.kLogLog

    # ---- Background rates / metallicities (no C++ default) ----
    # No `GetOrAdd*` defaults exist for these on the C++ side; they
    # are required inputs. We seed with sensible TIGRESS values so
    # bench tests can run unattended.
    z_gas:              float = 1.0
    z_dust:             float = 1.0
    zeta_hi_phot0:      float = 0.0

    # ---- Solver dispatch ----
    solver_type:        str = 'explicit'

    # ---- Free-form passthrough ----
    # Anything we have not yet promoted to a field stays here so the
    # config can serialise without dropping information.
    extra:              Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'ChemistryConfig':
        """Construct from a flat dict. Unknown keys land in `extra`."""
        known = {f.name for f in fields(cls) if f.name != 'extra'}
        kwargs: dict = {}
        extra: dict = {}
        for k, v in d.items():
            if k in known:
                kwargs[k] = _coerce(k, v, cls)
            else:
                extra[k] = v
        if extra:
            kwargs['extra'] = extra
        return cls(**kwargs)

    @classmethod
    def from_athinput(cls,
                      path: str,
                      block: str = 'photchem_ncr') -> 'ChemistryConfig':
        """Construct by reading `path` and consuming `[<block>]`.

        Uses `pyathena.io.athena_read.athinput` if importable so the
        parser matches the rest of pyathena. Falls back to a minimal
        block parser if the import fails (e.g., in an environment
        that ships only the chemistry subpackage).
        """
        try:
            from ..io.athena_read import athinput as _athinput
            data = _athinput(path)
        except Exception:
            data = _athinput_minimal(path)
        block_data = data.get(block, {})
        return cls.from_dict(block_data)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Round-trip helper. Enums serialise to their integer values."""
        d = asdict(self)
        # Replace InterpMode by its int value for parity with the
        # athinput key (which is an integer 0-3).
        d['interp_mode'] = int(d['interp_mode'])
        # Flatten `extra` so a round-trip through `from_dict` is
        # behaviour-preserving.
        extra = d.pop('extra', {})
        d.update(extra)
        return d


# ---- Helpers (module-private) ----------------------------------------
def _coerce(name: str, value: Any, cls: type) -> Any:
    """Coerce parser-supplied scalars to the declared field type.

    `pyathena.io.athena_read.athinput` already typecasts to int /
    float / complex / str; we only need to handle `bool` (which
    Python's `int(...)` would accept silently) and `InterpMode`.
    """
    target = {f.name: f.type for f in fields(cls)}[name]
    if target is bool or target == 'bool':
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ('true', '1', 'yes', 'on'):
                return True
            if v in ('false', '0', 'no', 'off'):
                return False
            raise ValueError(f'Cannot coerce {value!r} to bool '
                             f'for field {name}')
        return bool(int(value)) if not isinstance(value, bool) \
            else value
    if target is InterpMode or target == 'InterpMode':
        if isinstance(value, InterpMode):
            return value
        return InterpMode(int(value))
    return value


def _athinput_minimal(path: str) -> dict:
    """Tiny athinput parser for the rare case where the full
    `pyathena.io.athena_read` import fails. Only understands `<block>`
    headers and `key = value` lines; ignores everything else.
    """
    data: dict = {}
    current: Optional[str] = None
    with open(path, 'r') as fh:
        for raw in fh:
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            if line.startswith('<') and line.endswith('>'):
                current = line[1:-1].strip()
                data.setdefault(current, {})
                continue
            if current is None or '=' not in line:
                continue
            k, _, v = line.partition('=')
            k = k.strip()
            v = v.strip()
            # Lazy typecast: try int, then float, else string.
            for cast in (int, float):
                try:
                    data[current][k] = cast(v)
                    break
                except ValueError:
                    continue
            else:
                data[current][k] = v
    return data
