"""Manufactured solutions and observed-order checks."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from topoopt.config import params2d
from topoopt.darcy import solve_darcy
from topoopt.flow2d import solve_stokes, stokes_relative_residual
from topoopt.grid import zero_face_velocity
from topoopt.heat import solve_energy
from topoopt.filter import helmholtz_filter, helmholtz_operator
from topoopt.mms import (
    darcy_linear_exact,
    energy_advection_mms,
    energy_poisson_mms,
    energy_variable_k_discrete_mms,
    helmholtz_cosine_mms,
    relative_l2,
    stokes_poiseuille_exact,
    wall_dirichlet_specs,
)


def _energy_params(n: int, **kwargs):
    return params2d(
        nx=n,
        ny=n,
        heat_mode="conduction",
        q_vol=0.0,
        hot_specs=(),
        cold_specs=wall_dirichlet_specs(),
        heat_iters=kwargs.pop("heat_iters", 800),
        filter_iters=20,
        **kwargs,
    )


def test_energy_poisson_mms_and_order():
    errors = []
    for n in (8, 16, 32):
        params = _energy_params(n)
        T_ex, q = energy_poisson_mms(params, k=params.k_fluid)
        T = solve_energy(jnp.zeros(params.n), zero_face_velocity(params), params, q=q)
        err = float(relative_l2(T, T_ex))
        errors.append(err)
        assert err < 0.08, (n, err)
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    rate = np.log2(errors[1] / errors[2])
    assert rate > 1.5, (errors, rate)


def test_energy_advection_mms_converges():
    errors = []
    for n in (16, 32):
        params = params2d(
            nx=n,
            ny=n,
            heat_mode="both",
            pe=2.0,
            q_vol=0.0,
            hot_specs=(),
            cold_specs=wall_dirichlet_specs(),
            heat_iters=800,
            filter_iters=20,
        )
        T_ex, q, faces = energy_advection_mms(params, k=params.k_fluid, u_west=1.0)
        T = solve_energy(jnp.zeros(params.n), faces, params, q=q)
        err = float(relative_l2(T, T_ex))
        errors.append(err)
        assert err < 0.15, (n, err)
    rate = np.log2(errors[0] / errors[1])
    assert errors[1] < errors[0]
    assert rate > 0.7, (errors, rate)


def test_darcy_recovers_linear_pressure():
    params = params2d(
        nx=16, ny=16, flow_model="darcy", port_frac=1.0, flow_iters=400, filter_iters=20
    )
    faces, p = solve_darcy(jnp.zeros(params.n), params)
    p_ex, u_ex = darcy_linear_exact(params)
    assert float(relative_l2(p, p_ex)) < 5e-3
    u_in = float(jnp.mean(faces[0][0]))
    assert abs(u_in - float(u_ex)) / float(u_ex) < 0.05


def test_stokes_poiseuille_mms_and_order():
    errors = []
    for n in (8, 16):
        params = params2d(
            nx=n,
            ny=n,
            heat_mode="both",
            flow_model="stokes",
            port_frac=1.0,
            stokes_dp=20.0,
            div_eps=1e-4,
            flow_iters=120,
            uzawa_iters=40,
            stokes_kryl_iters=200,
            heat_iters=80,
            filter_iters=20,
        )
        gamma = jnp.zeros(params.n)
        sol = solve_stokes(gamma, params)
        u, v, p = sol
        u_ex, v_ex, p_ex = stokes_poiseuille_exact(params)
        rel = float(stokes_relative_residual(sol, gamma, params))
        assert rel < 1e-5, (n, rel)
        err_u = float(relative_l2(u, u_ex))
        errors.append(err_u)
        assert err_u < 0.08, (n, err_u)
        assert float(jnp.sqrt(jnp.mean(v**2))) < 1e-3
        assert float(relative_l2(p, p_ex)) < 0.08
    assert errors[1] < errors[0]
    rate = np.log2(errors[0] / errors[1])
    assert rate > 1.2, (errors, rate)


def test_helmholtz_discrete_inverse_and_cosine_order():
    params = params2d(nx=16, ny=16, rmin=2.0, filter_iters=400, solver_tol=1e-9)
    raw, filt = helmholtz_cosine_mms(params)
    recovered = helmholtz_filter(raw, params)
    assert float(relative_l2(recovered, filt)) < 0.08

    field = jnp.cos(jnp.pi * (jnp.arange(16) + 0.5) / 16)
    field = field[:, None] * field[None, :]
    assert float(relative_l2(helmholtz_filter(helmholtz_operator(field, params), params), field)) < 5e-6

    errors = []
    for n in (8, 16, 32):
        p = params2d(nx=n, ny=n, rmin=2.0, filter_iters=400, solver_tol=1e-9)
        raw_n, filt_n = helmholtz_cosine_mms(p)
        err = float(relative_l2(helmholtz_filter(raw_n, p), filt_n))
        errors.append(err)
        assert err < 0.15, (n, err)
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]


def test_energy_variable_k_discrete_mms():
    params = params2d(
        nx=16,
        ny=16,
        heat_mode="both",
        pe=2.0,
        q_vol=0.0,
        hot_specs=(),
        cold_specs=wall_dirichlet_specs(),
        heat_iters=800,
        filter_iters=20,
    )
    faces = zero_face_velocity(params)
    faces[0] = jnp.full_like(faces[0], 0.5)
    phys, t_ex, q, faces = energy_variable_k_discrete_mms(params, faces)
    temp = solve_energy(phys, faces, params, q=q)
    assert float(relative_l2(temp, t_ex)) < 5e-5
