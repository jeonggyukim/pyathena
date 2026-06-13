"""pyathena.chemistry — typed, policy-based chemistry / thermal / opacity
stack. Replaces pyathena.microphysics over a deprecation window; see
the package README and tigris-notes/docs-claude/pyathena/
chemistry-rewrite-plan.md for the design and roadmap.
"""

from .enums import InterpMode, Regime

__all__ = ['InterpMode', 'Regime']
