"""2-D MAC Stokes–Brinkman flow with a residual discrete adjoint.

    −∇²u + α(γ) u + ∇p = 0
    ∇·u = 0

One inlet on the left-wall centerline and one outlet on the right-wall
centerline occupy ``port_frac`` of those walls. Both ports are
**pressure-driven** (same idea as Darcy): p = p_in on the left port,
p = 0 on the right port, ∂u/∂x = 0 on the openings. Off-port walls are
no-slip. There are no other inlets or outlets. Throughput is
design-dependent — block the path and the flow drops.

Solid is a Brinkman penalty rather than a geometric hole. Forward: Uzawa
warm start, then CG on the pressure Schur complement so R(u,p;γ)≈0.
Reverse: residual discrete adjoint (not an unrolled Uzawa loop).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from topoopt.config import ColdPlateParams
from topoopt.grid import port_mask
from topoopt.interpolation import brinkman_alpha
from jax.flatten_util import ravel_pytree

from topoopt.solvers import iterative_gmres_solve, iterative_spd_solve

# Same cutoff as the convective energy factor: publication Stokes meshes
# use a dense residual-Jacobian adjoint rather than a poorly conditioned
# monolithic Krylov solve of the saddle-point transpose.
STOKES_DENSE_ADJOINT_MAX_CELLS = 48 * 48


def _alpha_u(alpha):
    a = jnp.zeros((alpha.shape[0] + 1, alpha.shape[1]))
    a = a.at[0, :].set(alpha[0, :])
    a = a.at[-1, :].set(alpha[-1, :])
    a = a.at[1:-1, :].set(0.5 * (alpha[:-1, :] + alpha[1:, :]))
    return a


def _alpha_v(alpha):
    a = jnp.zeros((alpha.shape[0], alpha.shape[1] + 1))
    a = a.at[:, 0].set(alpha[:, 0])
    a = a.at[:, -1].set(alpha[:, -1])
    a = a.at[:, 1:-1].set(0.5 * (alpha[:, :-1] + alpha[:, 1:]))
    return a


def _lap_u(u, dx, dy):
    """x-faces: Neumann ∂u/∂x=0 at both ends (ports); no-slip ghosts in y."""
    lap = jnp.zeros_like(u)
    lap = lap.at[1:-1, :].set((u[2:, :] - 2.0 * u[1:-1, :] + u[:-2, :]) / (dx * dx))
    lap = lap.at[0, :].set((u[1, :] - u[0, :]) / (dx * dx))
    lap = lap.at[-1, :].set((u[-2, :] - u[-1, :]) / (dx * dx))
    u_pad = jnp.concatenate([-u[:, :1], u, -u[:, -1:]], axis=1)
    lap = lap + (u_pad[:, 2:] - 2.0 * u_pad[:, 1:-1] + u_pad[:, :-2]) / (dy * dy)
    return lap


def _lap_v(v, dx, dy):
    """y-faces: no-slip on top/bottom; no-slip (tangent) ghosts on left/right."""
    lap = jnp.zeros_like(v)
    lap = lap.at[:, 1:-1].set((v[:, 2:] - 2.0 * v[:, 1:-1] + v[:, :-2]) / (dy * dy))
    v_pad = jnp.concatenate([-v[:1, :], v, -v[-1:, :]], axis=0)
    lap = lap + (v_pad[2:, :] - 2.0 * v_pad[1:-1, :] + v_pad[:-2, :]) / (dx * dx)
    return lap


def _gradp_u(p, dx, p_in: float):
    """Pressure gradient on u-faces. Ports use a half-cell to the face pressure."""
    gp = jnp.zeros((p.shape[0] + 1, p.shape[1]))
    gp = gp.at[1:-1, :].set((p[1:, :] - p[:-1, :]) / dx)
    gp = gp.at[0, :].set((p[0, :] - p_in) / (0.5 * dx))
    gp = gp.at[-1, :].set((0.0 - p[-1, :]) / (0.5 * dx))
    return gp


def _p_drive(params: ColdPlateParams) -> float:
    return params.stokes_dp


def _gradp_v(p, dy):
    gp = jnp.zeros((p.shape[0], p.shape[1] + 1))
    gp = gp.at[:, 1:-1].set((p[:, 1:] - p[:, :-1]) / dy)
    return gp


def _div(u, v, dx, dy):
    return (u[1:, :] - u[:-1, :]) / dx + (v[:, 1:] - v[:, :-1]) / dy


def _constrain_u(u, port):
    """Eliminate off-port normal-velocity Dirichlet columns."""
    u = u.at[0, :].set(jnp.where(port, u[0, :], 0.0))
    return u.at[-1, :].set(jnp.where(port, u[-1, :], 0.0))


def _constrain_v(v):
    """Eliminate top/bottom normal-velocity Dirichlet columns."""
    return v.at[:, 0].set(0.0).at[:, -1].set(0.0)


def _diag_u(alpha_u, dx, dy, port):
    diag = alpha_u + 2.0 / (dx * dx) + 2.0 / (dy * dy)
    diag = diag.at[:, 0].add(1.0 / (dy * dy))
    diag = diag.at[:, -1].add(1.0 / (dy * dy))
    diag = diag.at[0, :].add(-1.0 / (dx * dx))
    diag = diag.at[-1, :].add(-1.0 / (dx * dx))
    diag = diag.at[0, :].set(jnp.where(port, diag[0, :], 1.0))
    diag = diag.at[-1, :].set(jnp.where(port, diag[-1, :], 1.0))
    return diag


def _diag_v(alpha_v, dx, dy):
    diag = alpha_v + 2.0 / (dx * dx) + 2.0 / (dy * dy)
    diag = diag.at[0, :].add(1.0 / (dx * dx))
    diag = diag.at[-1, :].add(1.0 / (dx * dx))
    diag = diag.at[:, 0].set(1.0)
    diag = diag.at[:, -1].set(1.0)
    return diag


def stokes_residual(sol, gamma, params: ColdPlateParams):
    u, v, p = sol
    dx, dy = params.dx
    alpha = brinkman_alpha(gamma, params)
    au, av = _alpha_u(alpha), _alpha_v(alpha)
    port = port_mask(params)

    u_free = _constrain_u(u, port)
    v_free = _constrain_v(v)
    ru = au * u_free - _lap_u(u_free, dx, dy) + _gradp_u(
        p, dx, _p_drive(params)
    )
    ru = ru.at[0, :].set(jnp.where(port, ru[0, :], u[0, :]))
    ru = ru.at[-1, :].set(jnp.where(port, ru[-1, :], u[-1, :]))
    rv = av * v_free - _lap_v(v_free, dx, dy) + _gradp_v(p, dy)
    rv = rv.at[:, 0].set(v[:, 0])
    rv = rv.at[:, -1].set(v[:, -1])
    rp = _div(u_free, v_free, dx, dy) + params.div_eps * p
    return ru, rv, rp


def stokes_relative_residual(sol, gamma, params: ColdPlateParams):
    """Linear-system residual norm scaled by the inhomogeneous pressure drive."""
    residual = stokes_residual(sol, gamma, params)
    zero = tuple(jnp.zeros_like(field) for field in sol)
    inhomogeneous = stokes_residual(zero, gamma, params)
    numerator = jnp.sqrt(sum(jnp.vdot(value, value) for value in residual))
    denominator = jnp.sqrt(
        sum(jnp.vdot(value, value) for value in inhomogeneous)
    )
    return numerator / jnp.maximum(denominator, 1e-30)


def _residual_diag(gamma, params: ColdPlateParams):
    dx, dy = params.dx
    alpha = brinkman_alpha(gamma, params)
    au, av = _alpha_u(alpha), _alpha_v(alpha)
    port = port_mask(params)
    du = _diag_u(au, dx, dy, port)
    dv = _diag_v(av, dx, dy)
    inv_uw = 1.0 / jnp.clip(du[:-1, :], 1e-8, None)
    inv_ue = 1.0 / jnp.clip(du[1:, :], 1e-8, None)
    inv_vs = 1.0 / jnp.clip(dv[:, :-1], 1e-8, None)
    inv_vn = 1.0 / jnp.clip(dv[:, 1:], 1e-8, None)
    dp = inv_uw / (dx * dx) + inv_ue / (dx * dx) + inv_vs / (dy * dy) + inv_vn / (dy * dy)
    dp = dp + params.div_eps
    return du, dv, dp


def _solve_velocity(gamma, p, params: ColdPlateParams, p_drive: float | None = None):
    dx, dy = params.dx
    alpha = brinkman_alpha(gamma, params)
    au, av = _alpha_u(alpha), _alpha_v(alpha)
    port = port_mask(params)
    drive = _p_drive(params) if p_drive is None else p_drive
    rhs_u = -_gradp_u(p, dx, drive)
    rhs_u = rhs_u.at[0, :].set(jnp.where(port, rhs_u[0, :], 0.0))
    rhs_u = rhs_u.at[-1, :].set(jnp.where(port, rhs_u[-1, :], 0.0))
    rhs_v = (-_gradp_v(p, dy)).at[:, 0].set(0.0).at[:, -1].set(0.0)

    def au_mv(u):
        u_free = _constrain_u(u, port)
        r = au * u_free - _lap_u(u_free, dx, dy)
        r = r.at[0, :].set(jnp.where(port, r[0, :], u[0, :]))
        return r.at[-1, :].set(jnp.where(port, r[-1, :], u[-1, :]))

    def av_mv(v):
        v_free = _constrain_v(v)
        r = av * v_free - _lap_v(v_free, dx, dy)
        return r.at[:, 0].set(v[:, 0]).at[:, -1].set(v[:, -1])

    u = iterative_spd_solve(
        au_mv,
        rhs_u,
        _diag_u(au, dx, dy, port),
        niter=params.flow_iters,
        tol=params.solver_tol,
    )
    v = iterative_spd_solve(
        av_mv,
        rhs_v,
        _diag_v(av, dx, dy),
        niter=params.flow_iters,
        tol=params.solver_tol,
    )
    return u, v


def _momentum_operators(gamma, params: ColdPlateParams):
    """Return symmetric eliminated momentum blocks and their diagonals."""
    dx, dy = params.dx
    alpha = brinkman_alpha(gamma, params)
    au, av = _alpha_u(alpha), _alpha_v(alpha)
    port = port_mask(params)

    def au_mv(u):
        u_free = _constrain_u(u, port)
        residual = au * u_free - _lap_u(u_free, dx, dy)
        residual = residual.at[0, :].set(
            jnp.where(port, residual[0, :], u[0, :])
        )
        return residual.at[-1, :].set(
            jnp.where(port, residual[-1, :], u[-1, :])
        )

    def av_mv(v):
        v_free = _constrain_v(v)
        residual = av * v_free - _lap_v(v_free, dx, dy)
        return residual.at[:, 0].set(v[:, 0]).at[:, -1].set(v[:, -1])

    return (
        (au_mv, av_mv),
        (_diag_u(au, dx, dy, port), _diag_v(av, dx, dy)),
    )


def _solve_momentum_rhs(gamma, rhs, params: ColdPlateParams, *, niter: int):
    operators, diagonals = _momentum_operators(gamma, params)
    return tuple(
        iterative_spd_solve(
            operator,
            value,
            diagonal,
            niter=niter,
            tol=params.solver_tol,
        )
        for operator, value, diagonal in zip(operators, rhs, diagonals)
    )


def _pressure_gradient(p, params: ColdPlateParams):
    """Unknown-pressure contribution to momentum rows (the G block)."""
    dx, dy = params.dx
    port = port_mask(params)
    gu = _gradp_u(p, dx, 0.0)
    gu = gu.at[0, :].set(jnp.where(port, gu[0, :], 0.0))
    gu = gu.at[-1, :].set(jnp.where(port, gu[-1, :], 0.0))
    gv = _gradp_v(p, dy).at[:, 0].set(0.0).at[:, -1].set(0.0)
    return gu, gv


def _velocity_divergence(velocity, params: ColdPlateParams):
    """Continuity block D after eliminating constrained velocity columns."""
    u, v = velocity
    return _div(
        _constrain_u(u, port_mask(params)),
        _constrain_v(v),
        params.dx[0],
        params.dx[1],
    )


def _uzawa_forward(gamma, params: ColdPlateParams):
    """Uzawa / pressure-correction. Pressure ports make throughput design-dependent."""
    dx, dy = params.dx
    omega = 0.6

    def body(_, p):
        u, v = _solve_velocity(gamma, p, params)
        return p - omega * _div(u, v, dx, dy)

    p = jax.lax.fori_loop(0, params.uzawa_iters, body, jnp.zeros(params.n))
    u, v = _solve_velocity(gamma, p, params)
    return u, v, p


def _schur_correct(p0, gamma, params: ColdPlateParams):
    """CG on the pressure Jacobian ``S = -D A^{-1} G + ε I``.

    Stokes–Brinkman is affine in (u, v, p) at fixed γ, so exact block
    solves would give the linear-system solution in one correction.
    Here the momentum and Schur systems are capped iterative solves, so
    the achieved residual—not "exact Newton" terminology—determines
    whether a result is acceptable.
    """
    dx, dy = params.dx
    _du, _dv, dp = _residual_diag(gamma, params)

    def schur_mv(p):
        u, v = _solve_velocity(gamma, p, params, p_drive=0.0)
        return _div(u, v, dx, dy) + params.div_eps * p

    u0, v0 = _solve_velocity(gamma, p0, params)
    rhs = -(_div(u0, v0, dx, dy) + params.div_eps * p0)
    p = p0 + iterative_spd_solve(
        schur_mv, rhs, dp, niter=params.stokes_kryl_iters, tol=params.solver_tol
    )
    u, v = _solve_velocity(gamma, p, params)
    return u, v, p


def _stokes_forward(gamma, params: ColdPlateParams):
    if params.uzawa_iters > 0:
        _u, _v, p0 = _uzawa_forward(gamma, params)
    else:
        p0 = jnp.zeros(params.n)
    if params.stokes_kryl_iters <= 0:
        u, v = _solve_velocity(gamma, p0, params)
        return u, v, p0
    return _schur_correct(p0, gamma, params)


def _stokes_adjoint_dense(gamma, sol, cotangent, params: ColdPlateParams):
    """Factor the transposed residual Jacobian on publication-sized meshes."""

    def apply_transpose(lam):
        return jax.vjp(lambda state: stokes_residual(state, gamma, params), sol)[1](lam)[0]

    rhs, unravel = ravel_pytree(cotangent)
    identity = jnp.eye(rhs.size, dtype=rhs.dtype)
    jacobian_t = jax.vmap(
        lambda column: ravel_pytree(apply_transpose(unravel(column)))[0]
    )(identity).T
    return unravel(jnp.linalg.solve(jacobian_t, rhs))


def _stokes_adjoint(gamma, sol, cotangent, params: ColdPlateParams):
    """Solve the transposed Stokes residual Jacobian for the discrete adjoint."""
    n_cells = int(params.n[0] * params.n[1])
    if n_cells <= STOKES_DENSE_ADJOINT_MAX_CELLS:
        return _stokes_adjoint_dense(gamma, sol, cotangent, params)

    momentum_iters = max(params.flow_iters, 400)
    pressure_iters = max(params.stokes_kryl_iters, 500)
    zero_velocity = (jnp.zeros_like(sol[0]), jnp.zeros_like(sol[1]))
    zero_pressure = jnp.zeros_like(sol[2])

    def divergence_transpose(value):
        return jax.vjp(
            lambda velocity: _velocity_divergence(velocity, params),
            zero_velocity,
        )[1](value)[0]

    def gradient_transpose(value):
        return jax.vjp(
            lambda pressure: _pressure_gradient(pressure, params),
            zero_pressure,
        )[1](value)[0]

    velocity_rhs = (cotangent[0], cotangent[1])
    pressure_rhs = cotangent[2]
    a_inv_rhs = _solve_momentum_rhs(
        gamma, velocity_rhs, params, niter=momentum_iters
    )
    reduced_rhs = pressure_rhs - gradient_transpose(a_inv_rhs)

    def transpose_schur(value):
        dt_value = divergence_transpose(value)
        a_inv_dt = _solve_momentum_rhs(
            gamma, dt_value, params, niter=momentum_iters
        )
        return -gradient_transpose(a_inv_dt) + params.div_eps * value

    pressure_diag = _residual_diag(gamma, params)[2]
    lambda_p = iterative_gmres_solve(
        transpose_schur,
        reduced_rhs,
        pressure_diag,
        niter=pressure_iters,
        tol=params.solver_tol,
    )
    dt_lambda = divergence_transpose(lambda_p)
    corrected_rhs = tuple(
        value - correction
        for value, correction in zip(velocity_rhs, dt_lambda)
    )
    lambda_u, lambda_v = _solve_momentum_rhs(
        gamma, corrected_rhs, params, niter=momentum_iters
    )
    return lambda_u, lambda_v, lambda_p


def solve_stokes(gamma, params: ColdPlateParams):
    """Uzawa warm start + Schur CG; reverse mode is the residual adjoint."""

    @jax.custom_vjp
    def _solve(g):
        return _stokes_forward(g, params)

    def _fwd(g):
        sol = _stokes_forward(g, params)
        return sol, (g, sol)

    def _bwd(res, gsol):
        g, sol = res

        lam = _stokes_adjoint(g, sol, gsol, params)
        g_gamma = -jax.vjp(lambda gg: stokes_residual(sol, gg, params), g)[1](lam)[0]
        return (g_gamma,)

    _solve.defvjp(_fwd, _bwd)
    return _solve(gamma)


def solve_flow(gamma, params: ColdPlateParams):
    u, v, p = solve_stokes(gamma, params)
    return [u, v], p
