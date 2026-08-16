"""Density filters: compact cone (default) and Helmholtz PDE."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax.scipy.signal import convolve

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
    """Physical filter radius: ``r = rmin * min(dx)``."""
    return params.rmin * min(params.dx)


def _cone_half_widths(params: ColdPlateParams) -> tuple[int, int]:
    """Inclusive cell offsets with positive hat weight (``d < r``)."""
    r = filter_radius(params)
    if r <= 0.0:
        return (0, 0)
    dx, dy = params.dx
    rx = max(int(math.floor((r - 1e-15) / dx)), 0)
    ry = max(int(math.floor((r - 1e-15) / dy)), 0)
    return (rx, ry)


def cone_kernel(params: ColdPlateParams):
    """Linear hat ``max(0, r − d)`` on the compact stencil."""
    rx, ry = _cone_half_widths(params)
    dx, dy = params.dx
    r = filter_radius(params)
    ix = jnp.arange(-rx, rx + 1)
    iy = jnp.arange(-ry, ry + 1)
    dist = jnp.sqrt((ix[:, None] * dx) ** 2 + (iy[None, :] * dy) ** 2)
    return jnp.maximum(r - dist, 0.0)


def cone_filter(gamma_raw, params: ColdPlateParams):
    """Normalized cone density filter (Bruns–Tortorelli / Bourdin).

    ``γ̃_i = ∑_j w_{ij} γ_j / ∑_j w_{ij}`` with
    ``w_{ij} = max(0, r − ‖x_i − x_j‖)``. The denominator is the
    in-domain weight sum, so a uniform field is unchanged at walls.
    """
    if filter_radius(params) <= 0.0:
        return gamma_raw
    kernel = cone_kernel(params)
    num = convolve(gamma_raw, kernel, mode="same", method="direct")
    den = convolve(jnp.ones_like(gamma_raw), kernel, mode="same", method="direct")
    return num / jnp.maximum(den, 1e-30)


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


def density_filter(gamma_raw, params: ColdPlateParams):
    """Apply ``params.filter_kind`` (``cone`` or ``helmholtz``)."""
    kind = params.filter_kind
    if kind == "cone":
        return cone_filter(gamma_raw, params)
    if kind == "helmholtz":
        return helmholtz_filter(gamma_raw, params)
    raise ValueError(f"unknown filter_kind {kind!r}; expected 'cone' or 'helmholtz'")
