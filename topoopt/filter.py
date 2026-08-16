"""Helmholtz density filter with Neumann boundaries (Lazarov & Sigmund)."""

from __future__ import annotations

import jax.numpy as jnp

from topoopt.config import ColdPlateParams
from topoopt.solvers import implicit_spd_solve


def _laplacian_neumann(x, dxs):
    """Second-order Laplacian with zero-gradient (edge-padded) boundaries."""
    lap = jnp.zeros_like(x)
    for axis, dx in enumerate(dxs):
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (1, 1)
        xp = jnp.pad(x, pad_width, mode="edge")
        sl_c = [slice(None)] * x.ndim
        sl_p = [slice(None)] * x.ndim
        sl_m = [slice(None)] * x.ndim
        sl_c[axis] = slice(1, -1)
        sl_p[axis] = slice(2, None)
        sl_m[axis] = slice(None, -2)
        d2 = (xp[tuple(sl_p)] - 2.0 * xp[tuple(sl_c)] + xp[tuple(sl_m)]) / (dx * dx)
        lap = lap + d2
    return lap


def filter_radius(params: ColdPlateParams) -> float:
    """Physical Helmholtz radius: ``r = rmin * min(dx)``."""
    return params.rmin * min(params.dx)


def helmholtz_operator(x, params: ColdPlateParams):
    """Discrete ``(-r²∇² + I) x`` with the same Neumann Laplacian as the filter."""
    r = filter_radius(params)
    return x - (r * r) * _laplacian_neumann(x, params.dx)


def helmholtz_filter(gamma_raw, params: ColdPlateParams):
    """Solve (-r^2 ∇² + I) γ̃ = γ with r = rmin * min(dx)."""
    r = filter_radius(params)
    r2 = r * r
    diag = jnp.ones_like(gamma_raw)
    for dx in params.dx:
        # Neumann Laplacian diagonal is 1/dx^2 or 2/dx^2; a safe Jacobi bound:
        diag = diag + r2 * (2.0 / (dx * dx))

    def matvec(x):
        return helmholtz_operator(x, params)

    return implicit_spd_solve(
        matvec,
        gamma_raw,
        diag,
        niter=params.filter_iters,
        tol=params.solver_tol,
    )
