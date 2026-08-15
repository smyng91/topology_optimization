"""Matrix-free Krylov solvers with implicit (adjoint) differentiation.

``jax.lax.custom_linear_solve`` applies the implicit-function theorem to
A(γ) x = b(γ). Reverse-mode then solves the adjoint system

    A(γ)^T λ = ∂J/∂x

once, independent of the number of design variables. That is the discrete
global adjoint — cheaper and more consistent than unrolling Krylov iterations
or using finite differences.

Vectors may be arrays or pytrees of arrays (used for the Stokes saddle point).
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp


Tree = Any
MatVec = Callable[[Tree], Tree]


def _dot(a, b):
    leaves = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x, y: jnp.vdot(x, y), a, b))
    return jnp.sum(jnp.stack(leaves))


def _norm(a):
    return jnp.sqrt(jnp.real(_dot(a, a)))


def _axpy(x, y, a=1.0, b=1.0):
    return jax.tree_util.tree_map(lambda u, v: a * u + b * v, x, y)


def _precond(diag, r):
    return jax.tree_util.tree_map(lambda d, v: v / jnp.clip(d, 1e-30, None), diag, r)


def _run_pcg_capped(matvec: MatVec, b: Tree, diag: Tree, niter: int, tol: float) -> Tree:
    x = _precond(diag, b)
    r = _axpy(b, matvec(x), 1.0, -1.0)
    z = _precond(diag, r)
    p = z
    rz = _dot(r, z)
    bnorm = _norm(b) + 1.0

    def body(_, state):
        x, r, p, rz, done = state

        def step(s):
            x, r, p, rz, _done = s
            ap = matvec(p)
            alpha = rz / (_dot(p, ap) + 1e-30)
            x = _axpy(x, p, 1.0, alpha)
            r = _axpy(r, ap, 1.0, -alpha)
            z = _precond(diag, r)
            rz_new = _dot(r, z)
            beta = rz_new / (rz + 1e-30)
            p = _axpy(z, p, 1.0, beta)
            done = _norm(r) <= tol * bnorm
            return x, r, p, rz_new, done

        return jax.lax.cond(done, lambda s: s, step, state)

    x, _, _, _, _ = jax.lax.fori_loop(0, niter, body, (x, r, p, rz, False))
    return x


def _run_bicgstab_capped(matvec: MatVec, b: Tree, diag: Tree, niter: int, tol: float) -> Tree:
    x = _precond(diag, b)
    r = _axpy(b, matvec(x), 1.0, -1.0)
    r0 = r
    p = r
    rho = _dot(r0, r)
    bnorm = _norm(b) + 1.0

    def body(_, state):
        x, r, p, rho, done = state

        def step(s):
            x, r, p, rho, _done = s
            v = matvec(p)
            alpha = rho / (_dot(r0, v) + 1e-30)
            svec = _axpy(r, v, 1.0, -alpha)
            t = matvec(svec)
            omega = _dot(t, svec) / (_dot(t, t) + 1e-30)
            x = _axpy(_axpy(x, p, 1.0, alpha), svec, 1.0, omega)
            r = _axpy(svec, t, 1.0, -omega)
            rho_new = _dot(r0, r)
            beta = (rho_new / (rho + 1e-30)) * (alpha / (omega + 1e-30))
            p = _axpy(r, _axpy(p, v, 1.0, -omega), 1.0, beta)
            done = _norm(r) <= tol * bnorm
            return x, r, p, rho_new, done

        return jax.lax.cond(done, lambda s: s, step, state)

    x, _, _, _, _ = jax.lax.fori_loop(0, niter, body, (x, r, p, rho, False))
    return x


def implicit_spd_solve(
    matvec: MatVec,
    b: Tree,
    diag: Tree,
    niter: int = 400,
    tol: float = 1e-8,
) -> Tree:
    """Solve a symmetric positive-definite system with a consistent adjoint."""

    def solve(mv, rhs):
        return _run_pcg_capped(mv, rhs, diag, niter, tol)

    return jax.lax.custom_linear_solve(matvec, b, solve, solve, symmetric=True)


def implicit_nonsym_solve(
    matvec: MatVec,
    b: Tree,
    diag: Tree,
    niter: int = 400,
    tol: float = 1e-8,
) -> Tree:
    """Solve a nonsymmetric system; adjoint uses A^T via JAX's automatic transpose."""

    def solve(mv, rhs):
        return _run_bicgstab_capped(mv, rhs, diag, niter, tol)

    return jax.lax.custom_linear_solve(matvec, b, solve, solve, symmetric=False)
