#!/usr/bin/env python3
"""Tutorial: manufactured / exact solutions (is the install sane?).

These are the same checks as ``tests/test_mms.py``, written as a script
you can read. Each case has a known field; we solve the discrete system
and print the relative L2 error.

* Energy Poisson — ``T = sin(πx) sin(πy)``, all walls ``T = 0``, order ≈ 2
* Helmholtz filter — Neumann cosine, continuous manufactured raw field
* Darcy — linear pressure on a full-height port (should be very small)
* Stokes–Poiseuille — full-height pressure-driven channel, ``α = 0``
  (skipped with ``--quick``)

Run from the repo root::

    python examples/06_mms_check.py
    python examples/06_mms_check.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

import jax.numpy as jnp

from _common import add_run_args
from topoopt.config import params2d
from topoopt.darcy import solve_darcy
from topoopt.flow2d import solve_stokes, stokes_residual
from topoopt.grid import zero_face_velocity
from topoopt.heat import solve_energy
from topoopt.filter import helmholtz_filter
from topoopt.mms import (
    darcy_linear_exact,
    energy_poisson_mms,
    helmholtz_cosine_mms,
    relative_l2,
    stokes_poiseuille_exact,
    wall_dirichlet_specs,
)


def _energy_error(n: int) -> float:
    params = params2d(
        nx=n,
        ny=n,
        heat_mode="conduction",
        q_vol=0.0,
        hot_specs=(),
        cold_specs=wall_dirichlet_specs(),
        heat_iters=800,
        filter_iters=20,
    )
    t_ex, q = energy_poisson_mms(params, k=params.k_fluid)
    temp = solve_energy(jnp.zeros(params.n), zero_face_velocity(params), params, q=q)
    return float(relative_l2(temp, t_ex))


def _darcy_error() -> tuple[float, float]:
    params = params2d(nx=16, ny=16, flow_model="darcy", port_frac=1.0, flow_iters=400, filter_iters=20)
    faces, pressure = solve_darcy(jnp.zeros(params.n), params)
    p_ex, u_ex = darcy_linear_exact(params)
    p_err = float(relative_l2(pressure, p_ex))
    u_err = abs(float(jnp.mean(faces[0][0])) - float(u_ex)) / float(u_ex)
    return p_err, u_err


def _stokes_error(n: int) -> tuple[float, float]:
    params = params2d(
        nx=n,
        ny=n,
        heat_mode="both",
        flow_model="stokes",
        port_frac=1.0,
        stokes_dp=20.0,
        div_eps=1e-12,
        flow_iters=120,
        uzawa_iters=40,
        stokes_kryl_iters=200,
        filter_iters=20,
    )
    gamma = jnp.zeros(params.n)
    sol = solve_stokes(gamma, params)
    u, _v, _p = sol
    u_ex, _v_ex, _p_ex = stokes_poiseuille_exact(params)
    res = stokes_residual(sol, gamma, params)
    rel = float(jnp.sqrt(sum(jnp.vdot(r, r) for r in res))) / (1.0 + float(jnp.linalg.norm(u)))
    return float(relative_l2(u, u_ex)), rel


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(p, "outputs/06_mms_check")
    args = p.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Energy Poisson  T=sin(πx)sin(πy), all-wall T=0")
    e8, e16 = _energy_error(8), _energy_error(16)
    rate = __import__("math").log2(e8 / e16)
    print(f"  n=8  rel L2={e8:.3e}")
    print(f"  n=16 rel L2={e16:.3e}  observed order ≈ {rate:.2f}  (expect ~2)")
    if e16 >= e8 or rate < 1.3:
        print("  warning: energy error did not drop at second-order rate")

    print("\nHelmholtz filter, Neumann cosine γ̃ = cos(πx) cos(πy)")
    filt_params = params2d(nx=16, ny=16, rmin=2.0, filter_iters=400)
    raw, filt = helmholtz_cosine_mms(filt_params)
    h_err = float(relative_l2(helmholtz_filter(raw, filt_params), filt))
    print(f"  n=16  rel L2={h_err:.3e}")

    print("\nDarcy linear pressure, full-height ports, uniform κ")
    p_err, u_err = _darcy_error()
    print(f"  pressure rel L2={p_err:.3e}  face-u rel err={u_err:.3e}")

    lines = [
        f"energy_n8={e8:.6e}",
        f"energy_n16={e16:.6e}",
        f"energy_order={rate:.4f}",
        f"helmholtz={h_err:.6e}",
        f"darcy_p={p_err:.6e}",
        f"darcy_u={u_err:.6e}",
    ]

    if not args.quick:
        print("\nStokes–Poiseuille, port_frac=1, α=0")
        u_err_s, res_rel = _stokes_error(12)
        print(f"  n=12  u rel L2={u_err_s:.3e}  residual rel={res_rel:.3e}")
        lines.extend([f"stokes_u={u_err_s:.6e}", f"stokes_res={res_rel:.6e}"])

    report = args.outdir / "mms_report.txt"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
