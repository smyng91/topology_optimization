"""Manufactured and exact solutions for discrete-operator verification.

These fields satisfy the *continuous* operators on the unit box (or a
box of size ``params.L``). Tests compare them to the finite-volume
solves to check consistency and observed order.
"""

from __future__ import annotations

import jax.numpy as jnp

from topoopt.config import ColdPlateParams
from topoopt.grid import zero_face_velocity


def cell_centers(params: ColdPlateParams):
    xs = [(jnp.arange(n) + 0.5) * dx for n, dx in zip(params.n, params.dx)]
    return jnp.meshgrid(*xs, indexing="ij")


def energy_poisson_mms(params: ColdPlateParams, k: float = 1.0):
    """T = sin(πx/Lx) sin(πy/Ly) with T=0 on ∂Ω; q = −k ∇²T."""
    X, Y = cell_centers(params)
    lx, ly = params.L
    T = jnp.sin(jnp.pi * X / lx) * jnp.sin(jnp.pi * Y / ly)
    q = k * ((jnp.pi / lx) ** 2 + (jnp.pi / ly) ** 2) * T
    return T, q


def energy_advection_mms(params: ColdPlateParams, k: float = 1.0, u_west: float = 1.0):
    """Same T as the Poisson MMS plus uniform advection u=(U, 0)."""
    X, Y = cell_centers(params)
    lx, ly = params.L
    T, q_diff = energy_poisson_mms(params, k)
    pe = params.effective_pe
    q_adv = pe * u_west * (jnp.pi / lx) * jnp.cos(jnp.pi * X / lx) * jnp.sin(jnp.pi * Y / ly)
    faces = zero_face_velocity(params)
    faces[0] = jnp.full_like(faces[0], u_west)
    return T, q_diff + q_adv, faces


def darcy_linear_exact(params: ColdPlateParams):
    """Uniform-κ channel: p = p_in (1 − x/Lx), u = κ p_in / Lx."""
    X, _Y = cell_centers(params)
    lx = params.L[0]
    p = params.p_in * (1.0 - X / lx)
    u = params.kappa_max * params.p_in / lx
    return p, u


def stokes_poiseuille_exact(params: ColdPlateParams):
    """Full-height pressure-driven channel, α=0, μ=1.

    u = (Δp / (2 Lx)) y (Ly − y), v = 0, p = Δp (1 − x/Lx).
    Requires ``port_frac=1`` so both vertical walls are pressure ports.
    """
    nx, ny = params.n
    dx, dy = params.dx
    lx, ly = params.L
    dp = params.stokes_dp
    y_u = (jnp.arange(ny) + 0.5) * dy
    u = (dp / (2.0 * lx)) * y_u * (ly - y_u)
    u = jnp.broadcast_to(u, (nx + 1, ny))
    v = jnp.zeros((nx, ny + 1))
    X, _Y = cell_centers(params)
    p = dp * (1.0 - X / lx)
    return u, v, p


def helmholtz_cosine_mms(params: ColdPlateParams):
    """Neumann-compatible cosine: ``γ̃ = cos(πx/Lx) cos(πy/Ly)``.

    The continuous Helmholtz identity is ``γ = (1 + r² k²) γ̃`` with
    ``k² = (π/Lx)² + (π/Ly)²`` and ``r = rmin min(dx)``.
    """
    from topoopt.filter import filter_radius

    X, Y = cell_centers(params)
    lx, ly = params.L
    filt = jnp.cos(jnp.pi * X / lx) * jnp.cos(jnp.pi * Y / ly)
    k2 = (jnp.pi / lx) ** 2 + (jnp.pi / ly) ** 2
    r = filter_radius(params)
    raw = (1.0 + r * r * k2) * filt
    return raw, filt


def energy_variable_k_discrete_mms(params: ColdPlateParams, face_vel=None):
    """Variable-``k(γ)`` energy consistency: discrete operator as the source.

    ``T = sin(πx/Lx) sin(πy/Ly)`` and a cosine density. The manufactured
    ``q`` is the discrete residual at ``q=0``, so ``solve_energy`` should
    recover ``T`` to the Krylov tolerance (not a discretization-order test).
    """
    from topoopt.grid import zero_face_velocity
    from topoopt.heat import energy_operator
    from topoopt.interpolation import conductivity

    X, Y = cell_centers(params)
    lx, ly = params.L
    phys = 0.5 + 0.4 * jnp.cos(jnp.pi * X / lx) * jnp.cos(jnp.pi * Y / ly)
    temperature = jnp.sin(jnp.pi * X / lx) * jnp.sin(jnp.pi * Y / ly)
    faces = zero_face_velocity(params) if face_vel is None else face_vel
    k = conductivity(phys, params)
    q = energy_operator(temperature, k, faces, params, params.t_in, params.t_hot, 0.0)
    return phys, temperature, q, faces


def wall_dirichlet_specs():
    """All four walls at T=0, used by the energy MMS."""
    return ("face:left", "face:right", "face:bottom", "face:top")


def relative_l2(num, exact):
    return jnp.sqrt(jnp.mean((num - exact) ** 2)) / (jnp.sqrt(jnp.mean(exact**2)) + 1e-30)
