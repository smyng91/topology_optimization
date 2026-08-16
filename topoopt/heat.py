"""Cell-centered finite-volume energy equation in 2-D.

    −∇·(k ∇T) + Pe u·∇T = q

``q`` is ``q_vol`` everywhere, or only on ``q_specs`` cells, or off
(see ``uses_volume_source``). Dirichlet ``hot_specs`` prescribe *T*,
not generation. ``heat_mode`` selects which terms are active:

- ``conduction``: Pe = 0, k = k(γ), no flow
- ``convection``: Pe > 0, k = k_fluid (uniform), one left-centerline inlet
  and one right-centerline outlet, no cold patches
- ``both``: Pe > 0, k = k(γ), same ports, conjugate heat transfer
"""

from __future__ import annotations

import jax.numpy as jnp

from topoopt.config import ColdPlateParams
from topoopt.grid import add_axis, harmonic_faces, take_axis
from topoopt.interpolation import conductivity
from topoopt.regions import (
    apply_cell_dirichlet,
    apply_face_dirichlet_diffusion,
    cell_dirichlet_masks,
    face_dirichlets,
    face_heat_into_domain,
    volume_heat_from_cells,
    volume_source_field,
)
from topoopt.solvers import implicit_nonsym_solve


def _diffusion_divergence(T, k, dxs):
    """∇·(k ∇T) from interior faces only (boundary faces added by the caller)."""
    div = jnp.zeros_like(T)
    for axis, dx in enumerate(dxs):
        flux = harmonic_faces(k, axis) * (
            take_axis(T, axis, slice(1, None)) - take_axis(T, axis, slice(None, -1))
        ) / dx
        contrib = flux / dx
        div = add_axis(div, axis, slice(None, -1), contrib)
        div = add_axis(div, axis, slice(1, None), -contrib)
    return div


def _exterior_temperature(axis, side, t_cell, params: ColdPlateParams, t_in, t_hot):
    """Temperature carried by fluid entering through a domain face."""
    t_ext = t_in if (axis == 0 and side == 0) else t_cell
    temps = {"hot": t_hot, "cold": t_in}
    for patch in face_dirichlets(params):
        if patch.axis == axis and patch.side == side:
            t_ext = jnp.where(patch.mask, temps[patch.role], t_ext)
    return t_ext


def _advection_udotgrad(T, face_vel, params: ColdPlateParams, t_in, t_hot):
    """Pe u·∇T with first-order upwind, including domain-boundary faces.

    Discrete identity: u·∇T = ∇·(u T) − T ∇·u, so a constant T is a kernel
    even when the face velocities are only approximately divergence-free.
    """
    pe = params.effective_pe
    adv = jnp.zeros_like(T)
    for axis, dx in enumerate(params.dx):
        uf = face_vel[axis]
        u_int = take_axis(uf, axis, slice(1, -1))
        t0 = take_axis(T, axis, slice(None, -1))
        t1 = take_axis(T, axis, slice(1, None))
        t_up = jnp.where(u_int > 0.0, t0, t1)
        adv = add_axis(adv, axis, slice(None, -1), pe * u_int * (t_up - t0) / dx)
        adv = add_axis(adv, axis, slice(1, None), -pe * u_int * (t_up - t1) / dx)

        u_lo = take_axis(uf, axis, 0)
        t_lo = take_axis(T, axis, 0)
        t_ext_lo = _exterior_temperature(axis, 0, t_lo, params, t_in, t_hot)
        t_up_lo = jnp.where(u_lo > 0.0, t_ext_lo, t_lo)
        adv = add_axis(adv, axis, 0, -pe * u_lo * (t_up_lo - t_lo) / dx)

        u_hi = take_axis(uf, axis, -1)
        t_hi = take_axis(T, axis, -1)
        t_ext_hi = _exterior_temperature(axis, -1, t_hi, params, t_in, t_hot)
        t_up_hi = jnp.where(u_hi < 0.0, t_ext_hi, t_hi)
        adv = add_axis(adv, axis, -1, pe * u_hi * (t_up_hi - t_hi) / dx)
    return adv


def energy_operator(T, k, face_vel, params: ColdPlateParams, t_in, t_hot, q=0.0):
    """Linear residual −∇·(k∇T) + Pe u·∇T − q including inhomogeneous BCs."""
    diff = _diffusion_divergence(T, k, params.dx)
    diff = apply_face_dirichlet_diffusion(diff, T, k, params, t_hot, t_in)
    residual = -diff + _advection_udotgrad(T, face_vel, params, t_in, t_hot) - q
    return apply_cell_dirichlet(residual, T, params, t_hot, t_in)


def energy_diagonal(k, face_vel, params: ColdPlateParams):
    """Jacobi diagonal of the homogeneous energy operator."""
    diag = jnp.zeros_like(k)
    for axis, dx in enumerate(params.dx):
        kf = harmonic_faces(k, axis)
        contrib = kf / (dx * dx)
        diag = add_axis(diag, axis, slice(None, -1), contrib)
        diag = add_axis(diag, axis, slice(1, None), contrib)

        uf = face_vel[axis]
        u_w = take_axis(uf, axis, slice(None, -1))
        u_e = take_axis(uf, axis, slice(1, None))
        diag = diag + params.effective_pe * (jnp.maximum(u_w, 0.0) + jnp.maximum(-u_e, 0.0)) / dx

    for patch in face_dirichlets(params):
        dx = params.dx[patch.axis]
        sl = 0 if patch.side == 0 else -1
        kb = take_axis(k, patch.axis, sl)
        diag = add_axis(diag, patch.axis, sl, patch.mask * kb * 2.0 / (dx * dx))

    hot, cold = cell_dirichlet_masks(params)
    diag = jnp.where(hot | cold, 1.0, diag)
    return diag + 1e-8


def solve_energy(gamma, face_vel, params: ColdPlateParams, q=None):
    k = conductivity(gamma, params)

    def matvec(T):
        return energy_operator(T, k, face_vel, params, 0.0, 0.0, 0.0)

    if q is None:
        q = volume_source_field(params)
    rhs = -energy_operator(
        jnp.zeros_like(gamma), k, face_vel, params, params.t_in, params.t_hot, q
    )
    diag = energy_diagonal(k, face_vel, params)
    return implicit_nonsym_solve(
        matvec, rhs, diag, niter=params.heat_iters, tol=params.solver_tol
    )


def total_heat_transfer(gamma, T, params: ColdPlateParams):
    """Heat leaving Dirichlet sources, or −mean(T) when a volume source is on.

    With volumetric heating (uniform or ``q_specs``) ∫q dV is design-
    independent, so the objective is to *cool* the domain: maximize
    ``−mean(T)``. Dirichlet-only runs maximize heat leaving ``hot_specs``.
    """
    if params.uses_volume_source:
        return -jnp.mean(T)
    k = conductivity(gamma, params)
    heat = face_heat_into_domain(T, k, params, "hot")
    hot_cells, _ = cell_dirichlet_masks(params)
    heat = heat + volume_heat_from_cells(T, k, hot_cells, params)
    return heat
