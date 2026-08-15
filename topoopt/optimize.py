"""Projected-gradient topology optimization with β-continuation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from topoopt.config import ColdPlateParams
from topoopt.grid import port_mask
from topoopt.problem import analyze, physical_density


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
    print(f"{'it':>4}  {'beta':>6}  {'J':>12}  {'vol':>8}  {'time':>7}")

    aux = None
    for it in range(1, n_iters + 1):
        beta = float(betas[it - 1])
        t0 = time.time()
        gamma = keep_ports_open(project_physical_volume(gamma, beta, params, params.vol_frac), params)
        (loss, (heat, aux)), grad = value_and_grad(gamma, beta)
        gamma = keep_ports_open(
            project_physical_volume(projected_step(gamma, grad, lr), beta, params, params.vol_frac),
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
            "time": dt,
        }
        history.append(rec)
        print(f"{it:4d}  {beta:6.1f}  {float(heat):12.6f}  {vol:8.4f}  {dt:7.2f}s")
        if callback is not None:
            callback(it, gamma, aux, rec)
        if it == 1 or it == n_iters or it % 10 == 0:
            _save_checkpoint(outdir, it, gamma, aux, params)

    (outdir / "history.json").write_text(json.dumps(history, indent=2))
    _save_checkpoint(outdir, "final", gamma, aux, params)
    return gamma, aux, history


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
