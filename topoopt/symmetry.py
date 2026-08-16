"""Mirror projection for problems whose BCs are symmetric.

The usual breaker is the optimizer's random init: ``uniform`` noise is
not invariant under a flip, so a left–right (or top–bottom) problem
grows a skewed tree or channel. Projecting the design after every
volume / port step keeps the discrete field on the symmetry subspace.
"""

from __future__ import annotations

import jax.numpy as jnp

from topoopt.config import ColdPlateParams

_AXES = {"x": 0, "y": 1}


def normalize_axes(axes) -> tuple[str, ...]:
    """Accept ``'x'``, ``'x,y'``, ``('x',)``, or ``['y']``."""
    if axes is None:
        return ()
    if isinstance(axes, str):
        parts = [p.strip().lower() for p in axes.replace(" ", "").split(",") if p.strip()]
    else:
        parts = [str(p).strip().lower() for p in axes]
    out: list[str] = []
    for axis in parts:
        if axis not in _AXES:
            raise ValueError(f"unknown symmetry axis {axis!r}; expected 'x' and/or 'y'")
        if axis not in out:
            out.append(axis)
    return tuple(out)


def axes(params: ColdPlateParams) -> tuple[str, ...]:
    return normalize_axes(params.symmetry)


def apply(field, params: ColdPlateParams):
    """Return ``0.5 (field + flip(field))`` on each requested axis."""
    out = field
    for name in axes(params):
        out = 0.5 * (out + jnp.flip(out, _AXES[name]))
    return out


def max_error(field, params: ColdPlateParams):
    """``max|field − symmetrized|``. Zero when ``params.symmetry`` is empty."""
    if not axes(params):
        return jnp.array(0.0, dtype=getattr(field, "dtype", None))
    return jnp.max(jnp.abs(field - apply(field, params)))
