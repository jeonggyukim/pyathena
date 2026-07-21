"""Per-channel heating policies.

Each module in this package owns one heating mechanism (photoelectric,
cosmic-ray, H2 photoheating, ...). The abstract base
`HeatingChannel` lives in `pyathena.chemistry.cooling.base` so that
the `CoolingChannels` aggregator can compose cooling and heating
channels into the signed `net_cool = Lambda - Gamma` quantity the
solver consumes without a forward-import cycle.
"""
from __future__ import annotations
