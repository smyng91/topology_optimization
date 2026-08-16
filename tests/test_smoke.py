"""Smoke tests for interpolation, Darcy adjoints, and Stokes residuals."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from examples.problems import TREE_SINK, conjugate_stokes, convection_darcy
from topoopt.config import HEAT_MODES, params2d
from topoopt.interpolation import brinkman_alpha, conductivity, darcy_kappa, ramp
from topoopt.problem import analyze


def test_ramp_endpoints():
    x = jnp.array([0.0, 1.0])
    y = ramp(x, 2.0, 8.0, 0.1)
    np.testing.assert_allclose(y[0], 2.0, atol=1e-12)
    np.testing.assert_allclose(y[1], 8.0, atol=1e-12)


def test_material_maps():
    params = params2d(nx=8, ny=8)
    solid = jnp.ones((8, 8))
    fluid = jnp.zeros((8, 8))
    np.testing.assert_allclose(conductivity(solid, params).mean(), params.k_solid)
    np.testing.assert_allclose(conductivity(fluid, params).mean(), params.k_fluid)
    np.testing.assert_allclose(brinkman_alpha(solid, params).mean(), params.alpha_max)
    assert float(brinkman_alpha(jnp.full((8, 8), 0.45), params).mean()) < 0.2 * params.alpha_max
    np.testing.assert_allclose(darcy_kappa(fluid, params).mean(), params.kappa_max)
    np.testing.assert_allclose(darcy_kappa(solid, params).mean(), params.kappa_min)
    conv = params2d(nx=8, ny=8, heat_mode="convection")
    np.testing.assert_allclose(conductivity(solid, conv).mean(), conv.k_fluid)
    np.testing.assert_allclose(conductivity(fluid, conv).mean(), conv.k_fluid)


def test_heat_modes_finite_objective_and_grad():
    gamma = jnp.full((10, 10), 0.45)
    for mode in HEAT_MODES:
        extra = {"cold_specs": TREE_SINK} if mode == "conduction" else {}
        params = params2d(
            nx=10,
            ny=10,
            heat_mode=mode,
            flow_model="darcy",
            flow_iters=200,
            heat_iters=200,
            filter_iters=60,
            **extra,
        )
        assert params.solves_flow == (mode != "conduction")
        assert params.effective_pe == (0.0 if mode == "conduction" else params.pe)
        j, grad = jax.value_and_grad(lambda g: analyze(g, 1.5, params)[0])(gamma)
        assert np.isfinite(float(j))
        assert grad.shape == gamma.shape
        assert np.isfinite(np.asarray(grad)).all()


def test_darcy_2d_adjoint_matches_fd():
    params = convection_darcy(nx=10, ny=10, flow_iters=250, heat_iters=250, filter_iters=80)
    gamma = jnp.clip(jnp.full(params.n, params.vol_frac) + 0.03, 0.0, 1.0)

    def obj(g):
        return analyze(g, 2.0, params)[0]

    j, grad = jax.value_and_grad(obj)(gamma)
    assert np.isfinite(float(j))
    assert grad.shape == gamma.shape
    assert np.isfinite(np.asarray(grad)).all()

    samples = [(2, 2), (4, 3), (7, 1), (5, 4)]
    x0 = np.asarray(gamma)
    eps = 2e-4
    for idx in samples:
        xp, xm = x0.copy(), x0.copy()
        xp[idx] += eps
        xm[idx] -= eps
        fd = (float(obj(jnp.asarray(xp))) - float(obj(jnp.asarray(xm)))) / (2.0 * eps)
        analytic = float(np.asarray(grad)[idx])
        if abs(analytic) + abs(fd) < 1e-6:
            continue
        rel = abs(analytic - fd) / (abs(analytic) + abs(fd) + 1e-12)
        assert np.sign(analytic) == np.sign(fd) or rel < 0.5
        assert rel < 0.55, (idx, analytic, fd, rel)


def test_stokes_2d_residual_small_on_solve():
    from topoopt.flow2d import solve_stokes, stokes_relative_residual
    from topoopt.grid import port_mask
    from topoopt.heat import solve_energy

    params = conjugate_stokes(nx=12, ny=12, flow_iters=80, uzawa_iters=40, stokes_kryl_iters=200)
    gamma = jnp.zeros(params.n)
    sol = solve_stokes(gamma, params)
    rel = float(stokes_relative_residual(sol, gamma, params))
    assert np.isfinite(rel)
    assert rel < 1e-5
    u, v, _p = sol
    mask = port_mask(params)
    u_in = float(jnp.sum(u[0] * mask))
    u_out = float(jnp.sum(u[-1] * mask))
    assert u_in > 0.05
    assert abs(u_in - u_out) / u_in < 0.03
    temp = solve_energy(gamma, [u, v], params)
    assert np.isfinite(float(temp.mean()))
    assert float(temp.min()) >= -1e-6


def test_stokes_analyze_adjoint_matches_fd():
    """Residual Stokes adjoint vs central FD on the full analyze path (filter + energy)."""
    params = conjugate_stokes(
        nx=8,
        ny=8,
        flow_iters=80,
        uzawa_iters=40,
        stokes_kryl_iters=200,
        heat_iters=200,
        filter_iters=40,
    )
    # Nearly fluid: a sharp channel edge plus a cheap Krylov cap makes
    # the forward residual large and the FD comparison meaningless.
    gamma = jnp.full(params.n, 0.10)

    def obj(g):
        return analyze(g, 2.0, params)[0]

    j, grad = jax.value_and_grad(obj)(gamma)
    assert np.isfinite(float(j))
    assert grad.shape == gamma.shape
    assert np.isfinite(np.asarray(grad)).all()

    samples = [(2, 3), (4, 4), (1, 2)]
    x0 = np.asarray(gamma)
    eps = 2e-4
    for idx in samples:
        xp, xm = x0.copy(), x0.copy()
        xp[idx] += eps
        xm[idx] -= eps
        fd = (float(obj(jnp.asarray(xp))) - float(obj(jnp.asarray(xm)))) / (2.0 * eps)
        analytic = float(np.asarray(grad)[idx])
        rel = abs(analytic - fd) / (abs(analytic) + abs(fd) + 1e-12)
        assert np.sign(analytic) == np.sign(fd)
        assert rel < 0.05, (idx, analytic, fd, rel)


def test_stokes_2d_adjoint_matches_fd():
    """Residual Stokes adjoint vs central FD on throughput, no energy/filter."""
    from topoopt.flow2d import solve_stokes

    params = conjugate_stokes(
        nx=10, ny=10, flow_iters=80, uzawa_iters=40, stokes_kryl_iters=200, filter_iters=20
    )
    gamma = jnp.clip(jnp.full(params.n, 0.10) + 0.03, 0.0, 1.0)

    def obj(g):
        u, _v, _p = solve_stokes(g, params)
        return jnp.mean(u)

    j, grad = jax.value_and_grad(obj)(gamma)
    assert np.isfinite(float(j))
    assert grad.shape == gamma.shape
    assert np.isfinite(np.asarray(grad)).all()

    samples = [(2, 2), (4, 5), (7, 3), (5, 6)]
    x0 = np.asarray(gamma)
    eps = 2e-4
    for idx in samples:
        xp, xm = x0.copy(), x0.copy()
        xp[idx] += eps
        xm[idx] -= eps
        fd = (float(obj(jnp.asarray(xp))) - float(obj(jnp.asarray(xm)))) / (2.0 * eps)
        analytic = float(np.asarray(grad)[idx])
        if abs(analytic) + abs(fd) < 1e-8:
            continue
        rel = abs(analytic - fd) / (abs(analytic) + abs(fd) + 1e-12)
        assert np.sign(analytic) == np.sign(fd)
        assert rel < 0.05, (idx, analytic, fd, rel)
