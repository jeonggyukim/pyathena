"""Chemistry solver registry.

Importing this package side-effects the registration of every solver
class under `pyathena.chemistry.config.SOLVER_REGISTRY`. Callers can
then look up solvers by name:

    >>> from pyathena.chemistry import solvers
    >>> from pyathena.chemistry.config import SOLVER_REGISTRY
    >>> SOLVER_REGISTRY['explicit_subcycling']
    <class '...ExplicitSubcyclingSolver'>

Each concrete solver module decorates its class with
`@register_solver('name')` at module-import time.
"""
from .explicit_subcycling import ExplicitSubcyclingSolver

__all__ = ['ExplicitSubcyclingSolver']
