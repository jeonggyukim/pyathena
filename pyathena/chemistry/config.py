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
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional

from .enums import InterpMode


# Default `dvdr` from `photchem_ncr.cpp:57`. The numeric value is the
# Tigris code unit conversion `1 km/s/pc` to code units, kept here as
# a magic constant to match the C++ exactly.
_DVDR_DEFAULT: float = 3.240779289444365e-14


# ---- Solver registry ----------------------------------------------------
# Concrete solver classes register themselves at import time via
# `@register_solver('name')`. The registry is empty at config-import
# time; solver imports populate it. Phase 3 wires the driver up through
# this dict so the config file can name a solver by string and the
# driver constructs the right class without an explicit if/elif chain.
SOLVER_REGISTRY: Dict[str, type] = {}


def register_solver(name: str) -> Callable[[type], type]:
    """Class decorator: register `cls` under `name` in `SOLVER_REGISTRY`.

    Re-registering an existing name raises `ValueError` — solver names
    are expected to be globally unique within a process. The decorator
    returns the class unchanged so it can wrap a `class ...` definition
    in place.
    """
    def _decorator(cls: type) -> type:
        existing = SOLVER_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f'solver name {name!r} already registered to '
                f'{existing.__qualname__}; cannot rebind to '
                f'{cls.__qualname__}'
            )
        SOLVER_REGISTRY[name] = cls
        return cls
    return _decorator


@dataclass(frozen=True)
class SolverSpec:
    """Typed handle for the solver-side of `ChemistryConfig`.

    `name` is the registry key; `params` is a free-form mapping the
    solver's own constructor consumes (e.g., `nsub_max`,
    `cfl_cool_sub`, Rosenbrock tolerances). Defaults to the explicit
    subcycling solver with an empty params dict so legacy call sites
    that build a config without naming a solver get the same behaviour
    as the C++ default.
    """

    name: str = 'explicit_subcycling'
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Round-trip helper. `params` is materialised as a plain dict."""
        return {'name': self.name, 'params': dict(self.params)}

    @classmethod
    def from_obj(cls, obj: Any) -> 'SolverSpec':
        """Coerce a string / dict / SolverSpec into a SolverSpec.

        - `SolverSpec` -> returned as is.
        - `str` -> `SolverSpec(name=obj)`.
        - `Mapping` -> `SolverSpec(name=obj['name'], params=obj.get('params', {}))`.
        - anything else -> `TypeError`.
        """
        if isinstance(obj, SolverSpec):
            return obj
        if isinstance(obj, str):
            return cls(name=obj)
        if isinstance(obj, Mapping):
            name = obj.get('name')
            if not isinstance(name, str):
                raise TypeError(
                    f'SolverSpec dict missing string "name"; got {obj!r}'
                )
            params = obj.get('params', {})
            if params and not isinstance(params, Mapping):
                raise TypeError(
                    f'SolverSpec params must be a Mapping; got {type(params)}'
                )
            return cls(name=name, params=dict(params))
        raise TypeError(
            f'Cannot coerce {type(obj).__name__} into SolverSpec'
        )


# ---- Legacy aliases for solver-name strings -----------------------------
# `solver_type='explicit'` was the only string the pre-SolverSpec config
# understood; map it onto the new registry name so old athinput files
# do not need to change in lockstep with this PR.
_LEGACY_SOLVER_NAMES: Mapping[str, str] = MappingProxyType({
    'explicit':            'explicit_subcycling',
    'explicit_subcycling': 'explicit_subcycling',
})


# ---- Network-specific knobs --------------------------------------------
# Flat athinput keys that semantically belong to a specific network.
# `from_dict` mirrors them into the structured `network_params` mapping
# while keeping the flat attributes intact for back-compatibility, so a
# future NCRNetwork3PlusIons16 can add its own params without bloating
# the top-level schema. The flat fields remain the source of truth for
# the existing NCRNetwork3 stack.
_NCR3_NETWORK_PARAM_KEYS: tuple = (
    'xCstd', 'xOstd', 'temp_mu_floor', 'x_floor',
)


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
    # `solver` carries the registry name plus a free-form params dict.
    # Default matches the C++ legacy `solver_type='explicit'` mapped
    # through `_LEGACY_SOLVER_NAMES` to `'explicit_subcycling'`.
    solver:             SolverSpec = field(default_factory=SolverSpec)

    # ---- Network-specific knobs ----
    # Mapping consumed by the chosen NetworkPolicy. NCRNetwork3 reads
    # `xCstd`, `xOstd`, `temp_mu_floor`, `x_floor` from here; future
    # networks (NCRNetwork3PlusIons16, GOW17) add their own keys
    # without changing the top-level schema. For back-compat,
    # `from_dict` mirrors the flat top-level keys into this dict on
    # construction so callers can read either spelling.
    network_params:     Mapping[str, Any] = field(default_factory=dict)

    # ---- Free-form passthrough ----
    # Anything we have not yet promoted to a field stays here so the
    # config can serialise without dropping information.
    extra:              Mapping[str, Any] = field(default_factory=dict)

    # ---- Back-compatibility shim ----
    @property
    def solver_type(self) -> str:
        """Legacy alias for `solver.name`.

        Read-only — callers that previously set `solver_type` directly
        should construct a new `SolverSpec` and assign it to `solver`
        instead. The string values returned here are the same registry
        keys, not the legacy aliases (`'explicit'` -> `'explicit_subcycling'`).
        """
        return self.solver.name

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'ChemistryConfig':
        """Construct from a flat dict.

        Behaviour:

        - Known flat keys (e.g. `temp_hot0`, `xCstd`) populate the
          matching field on the dataclass.
        - The legacy `solver_type` key is coerced into a `SolverSpec`
          via `_LEGACY_SOLVER_NAMES`; if a `solver` key is also present
          it wins (and may itself be a string, dict, or SolverSpec).
        - Flat network-knob keys (`xCstd`, `xOstd`, `temp_mu_floor`,
          `x_floor`) are mirrored into `network_params` so a future
          NCRNetwork3PlusIons16 can read them without touching the
          flat schema. Explicit `network_params` in the dict wins on
          collision.
        - Any other unknown key falls through to `extra`.
        """
        known = {f.name for f in fields(cls) if f.name != 'extra'}
        kwargs: dict = {}
        extra: dict = {}
        legacy_solver_type: Optional[str] = None
        # Track flat NCR3 knobs that the caller actually supplied so
        # the auto-mirror into `network_params` only carries those —
        # otherwise `to_dict` -> `from_dict` would inject every flat
        # default into `network_params` on every round trip.
        explicit_ncr3_keys: set = set()
        explicit_network_params: Optional[dict] = None

        for k, v in d.items():
            if k == 'solver_type':
                # Legacy string alias; coerce later once we know whether
                # a structured `solver` was also supplied.
                legacy_solver_type = v if isinstance(v, str) else str(v)
                continue
            if k == 'network_params':
                explicit_network_params = (
                    dict(v) if isinstance(v, Mapping) else {}
                )
                continue
            if k in known:
                kwargs[k] = _coerce(k, v, cls)
                if k in _NCR3_NETWORK_PARAM_KEYS:
                    explicit_ncr3_keys.add(k)
            else:
                extra[k] = v

        # SolverSpec resolution: explicit `solver` wins; otherwise the
        # legacy `solver_type` string maps through `_LEGACY_SOLVER_NAMES`.
        if 'solver' not in kwargs and legacy_solver_type is not None:
            mapped = _LEGACY_SOLVER_NAMES.get(
                legacy_solver_type, legacy_solver_type)
            kwargs['solver'] = SolverSpec(name=mapped)

        # Mirror NCR3 network knobs into `network_params`.
        #
        # Precedence:
        # 1. If the input dict carried `network_params` (even empty),
        #    that mapping wins — the caller has opted into managing
        #    `network_params` directly, no auto-mirror happens.
        # 2. Otherwise, only the flat NCR3 keys the caller actually
        #    supplied get mirrored into a fresh `network_params`.
        #    Default values stay out so that `to_dict` -> `from_dict`
        #    is behaviour-preserving for any input that did not set
        #    `network_params` explicitly.
        if explicit_network_params is not None:
            if explicit_network_params:
                kwargs['network_params'] = dict(explicit_network_params)
        else:
            merged_np: dict = {}
            for key in _NCR3_NETWORK_PARAM_KEYS:
                if key in explicit_ncr3_keys:
                    merged_np[key] = kwargs[key]
            if merged_np:
                kwargs['network_params'] = merged_np

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
        """Round-trip helper. Enums serialise to their integer values.

        Notes
        -----
        `solver` is serialised as a plain dict (`{'name': ..., 'params': ...}`)
        so `from_dict` can rebuild it without depending on the dataclass.
        `network_params` round-trips through verbatim; the NCR3-flat-key
        mirror in `from_dict` is idempotent under serialise / deserialise.
        """
        d = asdict(self)
        # Replace InterpMode by its int value for parity with the
        # athinput key (which is an integer 0-3).
        d['interp_mode'] = int(d['interp_mode'])
        # SolverSpec -> plain dict so the result is JSON-friendly and
        # `from_dict` can parse it back via SolverSpec.from_obj.
        if isinstance(self.solver, SolverSpec):
            d['solver'] = self.solver.to_dict()
        # `network_params` always round-trips verbatim because `from_dict`
        # treats an explicit `network_params` key (even an empty dict) as
        # the caller opting out of the flat-key auto-mirror.
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
    if name == 'solver':
        return SolverSpec.from_obj(value)
    if name == 'network_params':
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(
                f'network_params must be a Mapping; got {type(value)}'
            )
        return dict(value)
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
