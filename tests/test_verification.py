from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from topoopt.heat import energy_operator, solve_energy
from topoopt.interpolation import conductivity, ramp
from topoopt.problem import analyze
from topoopt.problems import conduction_tree, conjugate_darcy, conjugate_stokes
from topoopt.solvers import (
    LinearSolveError,
    iterative_spd_solve_with_info,
    require_converged,
)
from topoopt.verification import (
    deterministic_direction,
    directional_taylor_test,
    fitted_log_order,
    high_contrast_design,
)


def test_standard_ramp_shape_endpoints_and_derivative():
    x = jnp.linspace(0.0, 1.0, 21)
    lo, hi, q = 2.0, 8.0, 3.0
    values = ramp(x, lo, hi, q)
    expected = lo + (hi - lo) * x / (1.0 + q * (1.0 - x))
    np.testing.assert_allclose(values, expected, rtol=1e-14, atol=1e-14)
    assert float(values[0]) == pytest.approx(lo)
    assert float(values[-1]) == pytest.approx(hi)
    assert np.all(np.diff(np.asarray(values)) > 0.0)
    np.testing.assert_allclose(ramp(x, lo, hi, 0.0), lo + (hi - lo) * x)

    derivative = jax.grad(lambda value: ramp(value, lo, hi, q))(0.37)
    exact = (hi - lo) * (1.0 + q) / (1.0 + q * (1.0 - 0.37)) ** 2
    assert float(derivative) == pytest.approx(exact, rel=1e-12)


def test_internal_dirichlet_elimination_is_symmetric_and_solved():
    from topoopt.config import params2d
    from topoopt.grid import zero_face_velocity

    params = params2d(
        nx=10,
        ny=9,
        heat_mode="conduction",
        q_vol=0.0,
        hot_specs=("box:0.2,0.4,0.2,0.5",),
        cold_specs=("box:0.7,0.9,0.6,0.9",),
        heat_iters=500,
        solver_tol=1e-10,
    )
    gamma = high_contrast_design(params.n, channel=False)
    faces = zero_face_velocity(params)
    k = conductivity(gamma, params)

    def matvec(temperature):
        return energy_operator(temperature, k, faces, params, 0.0, 0.0, 0.0)

    left = deterministic_direction(params.n, 1)
    right = deterministic_direction(params.n, 2)
    lhs = jnp.vdot(left, matvec(right))
    rhs = jnp.vdot(matvec(left), right)
    assert float(jnp.abs(lhs - rhs)) < 1e-9

    temperature = solve_energy(gamma, faces, params)
    residual = energy_operator(
        temperature, k, faces, params, params.t_in, params.t_hot, 0.0
    )
    assert float(jnp.linalg.norm(residual)) < 1e-6


def test_capped_solver_exposes_nonconvergence():
    matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = jnp.array([1.0, 2.0])
    solution, info = iterative_spd_solve_with_info(
        lambda value: matrix @ value,
        rhs,
        jnp.ones_like(rhs),
        niter=0,
        tol=1e-12,
    )
    assert np.isfinite(np.asarray(solution)).all()
    assert not bool(info["converged"])
    with pytest.raises(LinearSolveError):
        require_converged(info, name="deliberately capped CG")


@pytest.mark.parametrize(
    ("name", "factory", "shape", "channel", "seed"),
    [
        ("conduction", conduction_tree, (8, 8), False, 11),
        ("darcy", conjugate_darcy, (8, 8), True, 12),
        ("stokes", conjugate_stokes, (6, 6), True, 13),
    ],
)
def test_full_analysis_directional_taylor(name, factory, shape, channel, seed):
    overrides = dict(nx=shape[0], ny=shape[1], heat_iters=300, filter_iters=60)
    if name == "darcy":
        overrides["flow_iters"] = 300
    elif name == "stokes":
        overrides.update(
            flow_iters=120,
            uzawa_iters=80,
            stokes_kryl_iters=300,
        )
    params = factory(**overrides)
    point = high_contrast_design(params.n, channel=channel)
    direction = deterministic_direction(params.n, seed)
    beta = 4.0
    result = directional_taylor_test(
        lambda design: analyze(design, beta, params)[0],
        point,
        direction,
        (2.0e-2, 1.0e-2, 5.0e-3, 2.5e-3),
    )
    remainder_order = fitted_log_order(result["records"], "first_order_remainder")
    assert remainder_order > 1.7, (name, result)
    finest = result["records"][-1]
    scale = abs(result["directional_derivative"]) + 1e-30
    assert finest["first_order_remainder"] / scale < 5e-3, (name, result)
