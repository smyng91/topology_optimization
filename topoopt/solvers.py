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
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import gmres


Tree = Any
MatVec = Callable[[Tree], Tree]


class LinearSolveError(RuntimeError):
    """A capped iterative solve did not reach its requested residual."""


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
    bnorm = jnp.maximum(_norm(b), 1e-30)

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


def _run_bicgstab_capped(
    matvec: MatVec, b: Tree, diag: Tree, niter: int, tol: float, restart: int = 50
) -> Tree:
    """Jacobi BiCGSTAB with a periodic shadow-residual restart.

    Without restart the shadow residual ``r0`` can become orthogonal to
    ``r``, after which ``ρ → 0`` and the iteration stagnates. A vanishing
    ``r0·v`` or ``t·t`` is a breakdown: the step is skipped and ``r0``
    is reset to the current residual. Restart period 50 is a compromise
    on the factory convection/Stokes cases.
    """
    x = _precond(diag, b)
    r = _axpy(b, matvec(x), 1.0, -1.0)
    r0 = r
    p = r
    rho = _dot(r0, r)
    bnorm = jnp.maximum(_norm(b), 1e-30)
    period = int(restart) if int(restart) > 0 else int(niter) + 1

    def body(i, state):
        def step(s):
            x, r, r0, p, rho, _done = s
            v = matvec(p)
            r0v = _dot(r0, v)
            alpha = rho / (r0v + 1e-30)
            svec = _axpy(r, v, 1.0, -alpha)
            t = matvec(svec)
            tt = _dot(t, t)
            omega = _dot(t, svec) / (tt + 1e-30)
            breakdown = (jnp.abs(r0v) < 1e-30) | (tt < 1e-30)

            def accept(_):
                x_new = _axpy(_axpy(x, p, 1.0, alpha), svec, 1.0, omega)
                r_new = _axpy(svec, t, 1.0, -omega)
                rho_new = _dot(r0, r_new)
                beta = (rho_new / (rho + 1e-30)) * (alpha / (omega + 1e-30))
                p_new = _axpy(r_new, _axpy(p, v, 1.0, -omega), 1.0, beta)
                done = _norm(r_new) <= tol * bnorm
                return x_new, r_new, r0, p_new, rho_new, done

            def recover(_):
                r0n = r
                rho_n = _dot(r0n, r0n)
                return x, r, r0n, r0n, rho_n, False

            return jax.lax.cond(breakdown, recover, accept, None)

        def work(s):
            x, r, r0, p, rho, done = s
            do_restart = (i > 0) & ((i % period) == 0)
            r0, p, rho = jax.lax.cond(
                do_restart,
                lambda: (r, r, _dot(r, r)),
                lambda: (r0, p, rho),
            )
            return step((x, r, r0, p, rho, done))

        return jax.lax.cond(state[-1], lambda s: s, work, state)

    x, _, _, _, _, _ = jax.lax.fori_loop(0, niter, body, (x, r, r0, p, rho, False))
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


def iterative_spd_solve(
    matvec: MatVec,
    b: Tree,
    diag: Tree,
    niter: int = 400,
    tol: float = 1e-8,
) -> Tree:
    """Forward-only Jacobi-preconditioned CG (no implicit adjoint)."""
    return _run_pcg_capped(matvec, b, diag, niter, tol)


def iterative_nonsym_solve(
    matvec: MatVec,
    b: Tree,
    diag: Tree,
    niter: int = 400,
    tol: float = 1e-8,
) -> Tree:
    """Forward-only Jacobi-preconditioned BiCGSTAB (no implicit adjoint)."""
    return _run_bicgstab_capped(matvec, b, diag, niter, tol)


def iterative_gmres_solve(
    matvec: MatVec,
    b: Tree,
    diag: Tree,
    niter: int = 400,
    tol: float = 1e-8,
    restart: int = 50,
) -> Tree:
    """Forward restarted GMRES for difficult nonsymmetric pytree systems."""
    flat_b, unravel = ravel_pytree(b)
    flat_diag, _ = ravel_pytree(diag)

    def flat_matvec(value):
        result, _ = ravel_pytree(matvec(unravel(value)))
        return result

    def precondition(value):
        return value / jnp.clip(flat_diag, 1e-30, None)

    restart = max(1, min(int(restart), int(flat_b.size)))
    cycles = max(1, int((int(niter) + restart - 1) // restart))
    solution, _info = gmres(
        flat_matvec,
        flat_b,
        tol=tol,
        atol=0.0,
        restart=restart,
        maxiter=cycles,
        M=precondition,
        solve_method="incremental",
    )
    return unravel(solution)


def linear_solve_diagnostics(matvec: MatVec, solution: Tree, rhs: Tree, tol: float) -> dict[str, Tree]:
    """Measure the achieved residual of a returned linear-system iterate."""
    residual = _axpy(rhs, matvec(solution), 1.0, -1.0)
    residual_norm = _norm(residual)
    rhs_norm = _norm(rhs)
    relative_residual = residual_norm / jnp.maximum(rhs_norm, 1e-30)
    finite = jnp.isfinite(residual_norm) & jnp.isfinite(relative_residual)
    return {
        "residual_norm": residual_norm,
        "rhs_norm": rhs_norm,
        "relative_residual": relative_residual,
        "tolerance": jnp.asarray(tol),
        "finite": finite,
        "converged": finite & (relative_residual <= tol),
    }


def iterative_spd_solve_with_info(
    matvec: MatVec,
    b: Tree,
    diag: Tree,
    niter: int = 400,
    tol: float = 1e-8,
) -> tuple[Tree, dict[str, Tree]]:
    """Forward CG together with its achieved convergence diagnostics."""
    solution = _run_pcg_capped(matvec, b, diag, niter, tol)
    return solution, linear_solve_diagnostics(matvec, solution, b, tol)


def iterative_nonsym_solve_with_info(
    matvec: MatVec,
    b: Tree,
    diag: Tree,
    niter: int = 400,
    tol: float = 1e-8,
) -> tuple[Tree, dict[str, Tree]]:
    """Forward BiCGSTAB together with its achieved convergence diagnostics."""
    solution = _run_bicgstab_capped(matvec, b, diag, niter, tol)
    return solution, linear_solve_diagnostics(matvec, solution, b, tol)


def require_converged(info: dict[str, Tree], *, name: str = "linear solve") -> None:
    """Raise on a non-finite or above-tolerance solve outside JAX transforms."""
    if not bool(info["finite"]):
        raise LinearSolveError(f"{name} produced a non-finite residual")
    if not bool(info["converged"]):
        raise LinearSolveError(
            f"{name} relative residual {float(info['relative_residual']):.3e} "
            f"exceeds tolerance {float(info['tolerance']):.3e}"
        )


def implicit_dense_solve(
    matvec: MatVec,
    b: Tree,
    diag: Tree,
    niter: int = 400,
    tol: float = 1e-8,
) -> Tree:
    """Factor the energy operator by applying it to the identity.

    On 2-D meshes with a few thousand cells this is more reliable than a
    stagnant Krylov iteration. The adjoint is still the implicit-function
    theorem. ``diag`` / ``niter`` / ``tol`` are unused; they keep the
    call signature aligned with the Krylov wrappers.
    """
    del diag, niter, tol

    def solve(mv, rhs):
        shape = rhs.shape
        eye = jnp.eye(rhs.size, dtype=rhs.dtype)
        a = jax.vmap(lambda col: mv(col.reshape(shape)).ravel())(eye).T
        return jnp.linalg.solve(a, rhs.ravel()).reshape(shape)

    return jax.lax.custom_linear_solve(matvec, b, solve, solve, symmetric=False)


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
