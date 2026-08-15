"""Projected-gradient topology optimization with β-continuation."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from topoopt.config import ColdPlateParams
from topoopt.grid import port_mask
from topoopt.problem import analyze, physical_density

_DIAG_KEYS = (
    "energy_rms",
    "div_rms",
    "mass_err",
    "stokes_rel",
    "gray",
    "T_mean",
    "T_max",
    "speed_max",
    "u_in",
    "u_out",
)


def project_physical_volume(gamma_raw, beta, params: ColdPlateParams, target):
    """Shift the raw field so the filtered/projected solid fraction matches ``target``."""

    def body(_, bounds):
        a, b = bounds
        mid = 0.5 * (a + b)
        mean = physical_density(jnp.clip(gamma_raw - mid, 0.0, 1.0), beta, params).mean()
        return jax.lax.cond(mean > target, lambda: (mid, b), lambda: (a, mid))

    a, b = jax.lax.fori_loop(0, 24, body, (jnp.array(-2.0), jnp.array(2.0)))
    return jnp.clip(gamma_raw - 0.5 * (a + b), 0.0, 1.0)


def keep_ports_open(gamma, params: ColdPlateParams):
    """Keep a one-cell fluid layer on Stokes ports.

    A single solid cell on a pressure port seals the opening (α is huge).
    This is a design projection, not a PDE Dirichlet, and is Stokes-only.
    """
    if not params.solves_flow or params.flow_model != "stokes":
        return gamma
    mask = port_mask(params)
    gamma = gamma.at[0, :].set(jnp.where(mask, 0.0, gamma[0, :]))
    return gamma.at[-1, :].set(jnp.where(mask, 0.0, gamma[-1, :]))


def projected_step(gamma, grad, move: float):
    """Descend ``loss`` in the mean-zero (volume-preserving) direction."""
    g = grad - jnp.mean(grad)
    g = g / (jnp.max(jnp.abs(g)) + 1e-12)
    return jnp.clip(gamma - move * g, 0.0, 1.0)


def move_limit(lr: float, beta: float) -> float:
    """β-damped move: ``lr`` at β=1, ``lr / sqrt(β)`` for β>1."""
    return lr / math.sqrt(max(float(beta), 1.0))


def beta_schedule(n_iters: int, beta_max: float) -> jnp.ndarray:
    """Piecewise-constant β continuation, doubling until ``beta_max``."""
    levels = []
    beta = 1.0
    while beta < beta_max - 1e-9:
        levels.append(beta)
        beta *= 2.0
    levels.append(float(beta_max))
    chunk = max(n_iters // len(levels), 1)
    sched = []
    for b in levels:
        sched.extend([b] * chunk)
    if len(sched) < n_iters:
        sched.extend([levels[-1]] * (n_iters - len(sched)))
    return jnp.array(sched[:n_iters])


def optimize(
    params: ColdPlateParams,
    n_iters: int = 80,
    lr: float = 0.2,
    beta_max: float = 32.0,
    seed: int = 0,
    outdir: str | Path = "outputs",
    callback=None,
):
    """Maximize J (cooler mean T, or heat leaving hot patches) at fixed solid volume."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(seed)
    noise = 0.08 * (jax.random.uniform(key, params.n) - 0.5)
    guess = params.vol_frac + noise
    if params.solves_flow:
        # Start from an open mid-height duct. Volume-preserving GD from a
        # uniform field opens an inlet cavity and dumps the solid as a dam
        # on the outlet; a through-channel is much cooler but is a jump the
        # local step will not make.
        y = (jnp.arange(params.n[1]) + 0.5) / params.n[1]
        y0 = 0.5 * (1.0 - params.port_frac)
        y1 = 0.5 * (1.0 + params.port_frac)
        guess = jnp.where((y > y0) & (y < y1), 0.08, 0.78) + noise
    gamma = keep_ports_open(
        project_physical_volume(jnp.clip(guess, 0.0, 1.0), 1.0, params, params.vol_frac),
        params,
    )
    betas = beta_schedule(n_iters, beta_max)

    def loss_fn(g, beta):
        heat, aux = analyze(g, beta, params)
        return -heat, (heat, aux)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    history = []
    print(
        f"2-D box  n={params.n}  L={params.L}  heat={params.heat_mode} "
        f"({params.heat_label})  flow={params.flow_model if params.solves_flow else 'none'}  "
        f"Pe={params.effective_pe:g}  q={params.q_vol:g}  port={params.port_frac:g}  "
        f"vol*={params.vol_frac}\n"
        f"  hot={params.hot_specs}  cold={params.cold_specs}"
    )
    print(
        f"{'it':>4}  {'beta':>6}  {'J':>12}  {'vol':>8}  {'E_rms':>9}  "
        f"{'div_rms':>9}  {'mass_err':>9}  {'gray':>6}  {'time':>7}"
    )

    aux = None
    best_J = -float("inf")
    best_iter = 0
    best_gamma = gamma
    best_aux = None
    for it in range(1, n_iters + 1):
        beta = float(betas[it - 1])
        t0 = time.time()
        gamma = keep_ports_open(project_physical_volume(gamma, beta, params, params.vol_frac), params)
        (loss, (heat, aux)), grad = value_and_grad(gamma, beta)
        move = move_limit(lr, beta)
        gamma_next = keep_ports_open(
            project_physical_volume(projected_step(gamma, grad, move), beta, params, params.vol_frac),
            params,
        )
        vol = float(aux["V"])
        dt = time.time() - t0
        rec = {
            "iter": it,
            "beta": beta,
            "J": float(heat),
            "loss": float(loss),
            "vol": vol,
            "move": move,
            "time": dt,
        }
        for key in _DIAG_KEYS:
            rec[key] = float(aux[key])
        if float(heat) > best_J:
            best_J = float(heat)
            best_iter = it
            best_gamma = gamma
            best_aux = aux
            _save_checkpoint(outdir, "best", best_gamma, best_aux, params)
        rec["is_best"] = it == best_iter
        history.append(rec)
        print(
            f"{it:4d}  {beta:6.1f}  {float(heat):12.6f}  {vol:8.4f}  "
            f"{rec['energy_rms']:9.2e}  {rec['div_rms']:9.2e}  {rec['mass_err']:9.3f}  "
            f"{rec['gray']:6.3f}  {dt:7.2f}s"
        )
        if rec["energy_rms"] > 1e-2:
            print(f"  warning: energy residual RMS {rec['energy_rms']:.3e} > 1e-2")
        if params.solves_flow and rec["mass_err"] > 0.15:
            print(f"  warning: port mass error {rec['mass_err']:.3f} > 0.15")
        if callback is not None:
            callback(it, gamma, aux, rec)
        if it == 1 or it == n_iters or it % 10 == 0:
            _save_checkpoint(outdir, it, gamma, aux, params)
        gamma = gamma_next

    for rec in history:
        rec["is_best"] = rec["iter"] == best_iter
    (outdir / "history.json").write_text(json.dumps(history, indent=2))
    _save_checkpoint(outdir, "final", gamma, aux, params)
    _save_checkpoint(outdir, "best", best_gamma, best_aux, params)
    _write_run_json(
        outdir,
        params,
        n_iters=n_iters,
        lr=lr,
        beta_max=beta_max,
        seed=seed,
        history=history,
        best_J=best_J,
        best_iter=best_iter,
    )
    return best_gamma, best_aux, history


def _params_json(params: ColdPlateParams) -> dict:
    out = {}
    for key, val in params._asdict().items():
        out[key] = list(val) if isinstance(val, tuple) else val
    return out


def _write_run_json(outdir: Path, params: ColdPlateParams, **meta):
    history = meta["history"]
    last = history[-1]
    payload = {
        "params": _params_json(params),
        "n_iters": meta["n_iters"],
        "lr": meta["lr"],
        "beta_max": meta["beta_max"],
        "seed": meta["seed"],
        "J0": history[0]["J"],
        "J_final": last["J"],
        "J_best": meta["best_J"],
        "best_iter": meta["best_iter"],
        "vol_final": last["vol"],
        "energy_rms": last["energy_rms"],
        "div_rms": last["div_rms"],
        "mass_err": last["mass_err"],
        "stokes_rel": last["stokes_rel"],
        "gray": last["gray"],
        "T_mean": last["T_mean"],
        "T_max": last["T_max"],
        "speed_max": last["speed_max"],
        "u_in": last["u_in"],
        "u_out": last["u_out"],
    }
    (outdir / "run.json").write_text(json.dumps(payload, indent=2))


def _save_checkpoint(outdir: Path, tag, gamma, aux, params: ColdPlateParams):
    payload = {
        "gamma_raw": np.asarray(gamma),
        "phys": np.asarray(aux["phys"]),
        "T": np.asarray(aux["T"]),
        "p": np.asarray(aux["p"]),
        "speed": np.asarray(aux["speed"]),
        "n": np.array(params.n),
        "L": np.array(params.L),
    }
    for i, u in enumerate(aux["face_vel"]):
        payload[f"u{i}"] = np.asarray(u)
    np.savez_compressed(outdir / f"state_{tag}.npz", **payload)
