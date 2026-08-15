"""Heated-box analysis: filter → projection → flow → energy → objective."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from topoopt.config import ColdPlateParams
from topoopt.darcy import solve_darcy
from topoopt.filter import helmholtz_filter
from topoopt.flow2d import solve_flow
from topoopt.grid import take_axis, zero_face_velocity
from topoopt.heat import solve_energy, total_heat_transfer
from topoopt.interpolation import tanh_project


def physical_density(gamma_raw, beta: float, params: ColdPlateParams):
    filtered = helmholtz_filter(gamma_raw, params)
    return tanh_project(filtered, beta, params.eta)


def solve_fields(phys, params: ColdPlateParams):
    if not params.solves_flow:
        face_vel = zero_face_velocity(params)
        pressure = jnp.zeros(params.n)
    elif params.flow_model == "darcy":
        face_vel, pressure = solve_darcy(phys, params)
    else:
        face_vel, pressure = solve_flow(phys, params)
    temperature = solve_energy(phys, face_vel, params)
    return face_vel, pressure, temperature


def analyze(gamma_raw, beta: float, params: ColdPlateParams) -> tuple[Any, dict[str, Any]]:
    """Return the figure of merit J and a dict of fields.

    With a uniform volume source, J = −mean(T) (cooler is better). With
    Dirichlet heat-source patches, J is the heat leaving those patches.
    """
    phys = physical_density(gamma_raw, beta, params)
    face_vel, pressure, temperature = solve_fields(phys, params)
    heat = total_heat_transfer(phys, temperature, params)
    speed = _cell_speed(face_vel)
    return heat, {
        "phys": phys,
        "face_vel": face_vel,
        "p": pressure,
        "T": temperature,
        "speed": speed,
        "V": jnp.mean(phys),
    }


def _cell_speed(face_vel):
    comps = []
    for axis, u_face in enumerate(face_vel):
        u0 = take_axis(u_face, axis, slice(None, -1))
        u1 = take_axis(u_face, axis, slice(1, None))
        comps.append(0.5 * (u0 + u1))
    return jnp.sqrt(sum(c * c for c in comps))
