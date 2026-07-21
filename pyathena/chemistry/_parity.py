"""Test-only parity harness.

`run_both(callable_old, callable_new, inputs)` invokes a function from
`pyathena.microphysics` (the frozen reference) and a function from
`pyathena.chemistry` (the rewrite) on identical inputs and returns
both outputs. Tests then assert agreement at a documented tolerance.

This module imports both packages. Production paths must NOT depend
on it. Imports are wrapped in try / except so that a partial
chemistry package (typical during in-progress module ports) does not
break microphysics-only tests.
"""
from __future__ import annotations

from typing import Any, Callable, Tuple


def run_both(
    callable_old: Callable[..., Any],
    callable_new: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """Invoke `callable_old(*args, **kwargs)` and
    `callable_new(*args, **kwargs)` and return both results as
    `(out_old, out_new)`.

    Callers handle the comparison and tolerance themselves; use the
    `assert_close` helper below for the common case (structured
    numerical output with a per-test rtol / atol).
    """
    out_old = callable_old(*args, **kwargs)
    out_new = callable_new(*args, **kwargs)
    return out_old, out_new


def assert_close(
    out_old: Any,
    out_new: Any,
    *,
    rtol: float = 1.0e-10,
    atol: float = 0.0,
    label: str = '',
) -> None:
    """Convenience comparator. Recurses into numpy arrays, lists,
    tuples, and dicts (keyed identically on both sides). Anything else
    is compared with `==`.

    Used by parity tests where outputs are simple structured data;
    tests that need a custom comparator use `run_both` directly and
    compare themselves.
    """
    import numpy as np

    if isinstance(out_old, np.ndarray) or isinstance(out_new, np.ndarray):
        np.testing.assert_allclose(
            out_new, out_old, rtol=rtol, atol=atol,
            err_msg=f'parity mismatch ({label})')
        return

    if isinstance(out_old, dict) and isinstance(out_new, dict):
        if set(out_old.keys()) != set(out_new.keys()):
            raise AssertionError(
                f'parity key mismatch ({label}): '
                f'old={sorted(out_old.keys())} '
                f'new={sorted(out_new.keys())}')
        for k in out_old:
            assert_close(out_old[k], out_new[k],
                         rtol=rtol, atol=atol,
                         label=f'{label}[{k!r}]')
        return

    if isinstance(out_old, (list, tuple)) and isinstance(out_new, (list, tuple)):
        if len(out_old) != len(out_new):
            raise AssertionError(
                f'parity length mismatch ({label}): '
                f'old={len(out_old)} new={len(out_new)}')
        for i, (a, b) in enumerate(zip(out_old, out_new)):
            assert_close(a, b, rtol=rtol, atol=atol,
                         label=f'{label}[{i}]')
        return

    if out_old != out_new:
        raise AssertionError(
            f'parity mismatch ({label}): old={out_old!r} new={out_new!r}')
