"""Darcy flow on a 2-D cell-centered grid.

    u = −κ(γ) ∇p,    ∇·u = 0  ⇒  −∇·(κ ∇p) = 0

One inlet on the left-wall centerline and one outlet on the right-wall
centerline (``port_frac`` of the height) are held at ``p_in`` and 0.
The rest of those walls, and the top / bottom, are impermeable. Face
velocities use the same fluxes as the Poisson residual.
"""

from __future__ import annotations

import jax.numpy as jnp

from topoopt.config import ColdPlateParams
from topoopt.grid import add_axis, harmonic_faces, port_mask, set_axis, take_axis
from topoopt.heat import _diffusion_divergence
from topoopt.interpolation import darcy_kappa
from topoopt.solvers import implicit_spd_solve


def _apply_pressure_faces(div, p, kappa, params: ColdPlateParams, p_in: float):
    """Add ∇·(κ∇p) contributions from the centered inlet / outlet ports."""
    dx = params.dx[0]
    mask = port_mask(params)
    kb = take_axis(kappa, 0, 0)
    flux = mask * kb * (take_axis(p, 0, 0) - p_in) / (0.5 * dx)
    div = add_axis(div, 0, 0, -flux / dx)
    kb = take_axis(kappa, 0, -1)
    flux = mask * kb * take_axis(p, 0, -1) / (0.5 * dx)
    div = add_axis(div, 0, -1, -flux / dx)
    return div


def darcy_operator(p, kappa, params: ColdPlateParams, p_in: float):
    """Linear residual −∇·(κ∇p) including inhomogeneous port pressures."""
    div = _diffusion_divergence(p, kappa, params.dx)
    div = _apply_pressure_faces(div, p, kappa, params, p_in)
    return -div


def darcy_matvec(p, kappa, params: ColdPlateParams):
    return darcy_operator(p, kappa, params, 0.0)


def darcy_diagonal(kappa, params: ColdPlateParams):
    diag = jnp.zeros_like(kappa)
    for axis, dx in enumerate(params.dx):
        kf = harmonic_faces(kappa, axis)
        contrib = kf / (dx * dx)
        diag = add_axis(diag, axis, slice(None, -1), contrib)
        diag = add_axis(diag, axis, slice(1, None), contrib)
    dx = params.dx[0]
    mask = port_mask(params)
    diag = add_axis(diag, 0, 0, mask * take_axis(kappa, 0, 0) * 2.0 / (dx * dx))
    diag = add_axis(diag, 0, -1, mask * take_axis(kappa, 0, -1) * 2.0 / (dx * dx))
    return diag + 1e-12


def _face_velocity(p, kappa, params: ColdPlateParams):
    """MAC-like face-normal velocities u = −κ ∇p; ports only on the left / right."""
    mask = port_mask(params)
    face_vel = []
    for axis, dx in enumerate(params.dx):
        shape = list(params.n)
        shape[axis] += 1
        u = jnp.zeros(shape)
        kf = harmonic_faces(kappa, axis)
        u_int = -kf * (
            take_axis(p, axis, slice(1, None)) - take_axis(p, axis, slice(None, -1))
        ) / dx
        u = set_axis(u, axis, slice(1, -1), u_int)

        if axis == 0:
            k_in = take_axis(kappa, 0, 0)
            u_in = mask * (-k_in * (take_axis(p, 0, 0) - params.p_in) / (0.5 * dx))
            u = set_axis(u, 0, 0, u_in)
            k_out = take_axis(kappa, 0, -1)
            u_out = mask * (-k_out * (0.0 - take_axis(p, 0, -1)) / (0.5 * dx))
            u = set_axis(u, 0, -1, u_out)
        face_vel.append(u)
    return face_vel


def solve_darcy(gamma, params: ColdPlateParams):
    kappa = darcy_kappa(gamma, params)

    def matvec(p):
        return darcy_matvec(p, kappa, params)

    rhs = -darcy_operator(jnp.zeros_like(gamma), kappa, params, params.p_in)
    diag = darcy_diagonal(kappa, params)
    p = implicit_spd_solve(matvec, rhs, diag, niter=params.flow_iters, tol=params.solver_tol)
    return _face_velocity(p, kappa, params), p
