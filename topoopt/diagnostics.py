"""Solver and design diagnostics for the heated-box analysis.

All values are JAX scalars/arrays so they can live in ``analyze`` aux
without breaking a jitted ``value_and_grad``.
"""

from __future__ import annotations

import jax.numpy as jnp

from topoopt.config import ColdPlateParams
from topoopt.flow2d import stokes_relative_residual
from topoopt.grid import cell_divergence, port_mask
from topoopt.heat import energy_operator
from topoopt.interpolation import conductivity
from topoopt.regions import volume_source_field


def energy_residual_rms(phys, temperature, face_vel, params: ColdPlateParams):
    """RMS of the discrete energy residual at the solved temperature."""
    k = conductivity(phys, params)
    q = volume_source_field(params)
    res = energy_operator(temperature, k, face_vel, params, params.t_in, params.t_hot, q)
    return jnp.sqrt(jnp.mean(res**2))


def energy_residual_rel(phys, temperature, face_vel, params: ColdPlateParams):
    """Residual RMS scaled by the inhomogeneous-data RMS (the linear-system right-hand side)."""
    k = conductivity(phys, params)
    q = volume_source_field(params)
    res = energy_operator(temperature, k, face_vel, params, params.t_in, params.t_hot, q)
    rhs = -energy_operator(
        jnp.zeros_like(temperature), k, face_vel, params, params.t_in, params.t_hot, q
    )
    return jnp.sqrt(jnp.mean(res**2)) / (jnp.sqrt(jnp.mean(rhs**2)) + 1e-12)


def port_mass(face_vel, params: ColdPlateParams):
    """Left/right port throughput and relative mass imbalance.

    Returns zeros when flow is off. ``mass_err = |u_in − u_out| / (|u_in| + ε)``.
    """
    zero = jnp.zeros(())
    if not params.solves_flow:
        return zero, zero, zero
    mask = port_mask(params)
    u_in = jnp.sum(face_vel[0][0] * mask)
    u_out = jnp.sum(face_vel[0][-1] * mask)
    mass_err = jnp.abs(u_in - u_out) / (jnp.abs(u_in) + 1e-12)
    return u_in, u_out, mass_err


def grayness(phys, lo: float = 0.05, hi: float = 0.95):
    """Fraction of cells with intermediate physical density."""
    return jnp.mean(((phys > lo) & (phys < hi)).astype(phys.dtype))


def stokes_residual_rel(phys, face_vel, pressure, params: ColdPlateParams):
    """Residual norm relative to the inhomogeneous pressure-drive vector."""
    return stokes_relative_residual(
        (face_vel[0], face_vel[1], pressure), phys, params
    )


def field_diagnostics(phys, face_vel, pressure, temperature, speed, params: ColdPlateParams):
    """Pack residual, mass, and field scalars for ``analyze`` aux."""
    u_in, u_out, mass_err = port_mass(face_vel, params)
    if params.solves_flow and params.flow_model == "stokes":
        stokes_rel = stokes_residual_rel(phys, face_vel, pressure, params)
    else:
        stokes_rel = jnp.zeros(())
    return {
        "energy_rms": energy_residual_rms(phys, temperature, face_vel, params),
        "energy_rel": energy_residual_rel(phys, temperature, face_vel, params),
        "div_rms": jnp.sqrt(jnp.mean(cell_divergence(face_vel, params.dx) ** 2)),
        "u_in": u_in,
        "u_out": u_out,
        "mass_err": mass_err,
        "stokes_rel": stokes_rel,
        "gray": grayness(phys),
        "T_mean": jnp.mean(temperature),
        "T_max": jnp.max(temperature),
        "speed_max": jnp.max(speed),
    }
