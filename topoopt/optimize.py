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
from topoopt.symmetry import apply as apply_symmetry
from topoopt.symmetry import axes as symmetry_axes
from topoopt.symmetry import max_error as symmetry_error

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


class RunawaySolveError(RuntimeError):
    """Energy / temperature blew up, usually a sealed flow design."""


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


def project_design(gamma, beta, params: ColdPlateParams):
    """Symmetry → volume equality → Stokes port pin → symmetry."""
    gamma = apply_symmetry(jnp.clip(gamma, 0.0, 1.0), params)
    gamma = project_physical_volume(gamma, beta, params, params.vol_frac)
    gamma = keep_ports_open(gamma, params)
    return apply_symmetry(gamma, params)


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


def highest_beta_best(history):
    """Best-``J`` record at the largest β that has a finite objective.

    ``J`` is not comparable across continuation: ``physical_density``
    (and the PDE) change when the tanh projection sharpens. A mid-β
    gray field can have a larger ``J`` than a nearly 0–1 design at
    ``β_max``; that gray field is not the physical answer.
    """
    if not history:
        raise ValueError("history is empty")
    by_beta = {}
    for rec in history:
        if not math.isfinite(float(rec["J"])):
            continue
        beta = float(rec["beta"])
        prev = by_beta.get(beta)
        if prev is None or float(rec["J"]) > float(prev["J"]):
            by_beta[beta] = rec
    if not by_beta:
        return history[-1]
    return by_beta[max(by_beta)]


def upsample_field(field, new_n: tuple[int, int]):
    """Bilinear resize of a cell-centered field onto ``new_n``."""
    field = jnp.asarray(field)
    if tuple(field.shape) == tuple(new_n):
        return field
    return jax.image.resize(field.astype(jnp.float64), new_n, method="linear")


def runaway_reason(rec: dict, params: ColdPlateParams) -> str | None:
    """Why a flow solve should abort. ``None`` if the fields still look sane."""
    t_max = rec.get("T_max", float("nan"))
    t_mean = rec.get("T_mean", float("nan"))
    heat = rec.get("J", float("nan"))
    if not (math.isfinite(float(t_max)) and math.isfinite(float(t_mean)) and math.isfinite(float(heat))):
        return "non-finite temperature or objective"
    if float(t_max) > 1e3:
        return f"T_max={float(t_max):.3e} > 1e3"
    if params.solves_flow and float(rec.get("energy_rms", 0.0)) > 1e-2 and float(t_max) > 50.0:
        return (
            f"energy_rms={float(rec['energy_rms']):.3e} and T_max={float(t_max):.3g} "
            "(likely blocked flow; no conduction sink on flow modes)"
        )
    return None


def _initial_guess(params: ColdPlateParams, seed: int, start_gamma):
    if start_gamma is not None:
        guess = jnp.asarray(start_gamma)
        if tuple(guess.shape) != tuple(params.n):
            raise ValueError(f"start_gamma shape {tuple(guess.shape)} != params.n {params.n}")
        return jnp.clip(guess, 0.0, 1.0)
    key = jax.random.PRNGKey(seed)
    noise = 0.08 * (jax.random.uniform(key, params.n) - 0.5)
    noise = apply_symmetry(noise, params)
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
    return jnp.clip(guess, 0.0, 1.0)


def optimize(
    params: ColdPlateParams,
    n_iters: int = 80,
    lr: float = 0.2,
    beta_max: float = 32.0,
    seed: int = 0,
    outdir: str | Path = "outputs",
    callback=None,
    start_gamma=None,
    abort_on_runaway: bool = True,
    stall_iters: int = 8,
):
    """Maximize J (cooler mean T, or heat leaving hot patches) at fixed solid volume.

    Random init noise is symmetrized when ``params.symmetry`` is set, and
    every accepted design is projected onto that mirror. Without that
    projection a left–right or top–bottom problem grows a skewed design
    from asymmetric noise even though the PDEs and BCs are symmetric.

    The returned design is the best-``J`` iterate at the **highest β
    that ran**, not the global max ``J`` across continuation. A soft
    mid-β field is a different discrete problem (and a poor 0–1
    geometry) even when its ``J`` is larger. Stall is counted only
    inside the current β level, so entering ``β_max`` does not
    immediately stop because an earlier gray iterate was better.

    Optimizer-only arguments (not on ``ColdPlateParams``): ``n_iters``
    (default 80), ``lr`` (0.2 at β=1), ``beta_max`` (32), ``seed`` (0),
    ``outdir``, ``callback``, ``start_gamma`` (skips noise and the
    channel seed), ``abort_on_runaway`` (True), ``stall_iters`` (8; 0
    disables). Init noise amplitude is 0.08.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gamma = project_design(_initial_guess(params, seed, start_gamma), 1.0, params)
    betas = beta_schedule(n_iters, beta_max)

    def loss_fn(g, beta):
        heat, aux = analyze(g, beta, params)
        return -heat, (heat, aux)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    history = []
    sym = ",".join(symmetry_axes(params)) or "none"
    print(
        f"2-D box  n={params.n}  L={params.L}  heat={params.heat_mode} "
        f"({params.heat_label})  flow={params.flow_model if params.solves_flow else 'none'}  "
        f"Pe={params.effective_pe:g}  q={params.q_vol:g}  port={params.port_frac:g}  "
        f"filter={params.filter_kind}  rmin={params.rmin:g}  "
        f"vol*={params.vol_frac}  symmetry={sym}\n"
        f"  hot={params.hot_specs}  cold={params.cold_specs}  q_region={params.q_specs}"
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
    peak_J = -float("inf")
    peak_iter = 0
    prev_beta = None
    stopped = "completed"
    for it in range(1, n_iters + 1):
        beta = float(betas[it - 1])
        if prev_beta is not None and abs(beta - prev_beta) > 1e-12:
            # New continuation stage: do not compare J to a softer β.
            best_J = -float("inf")
        prev_beta = beta
        t0 = time.time()
        gamma = project_design(gamma, beta, params)
        (loss, (heat, aux)), grad = value_and_grad(gamma, beta)
        move = move_limit(lr, beta)
        gamma_next = project_design(projected_step(gamma, grad, move), beta, params)
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
            "sym_err": float(symmetry_error(gamma, params)),
        }
        for key in _DIAG_KEYS:
            rec[key] = float(aux[key])
        heat_f = float(heat)
        if math.isfinite(heat_f) and heat_f > peak_J:
            peak_J = heat_f
            peak_iter = it
        if math.isfinite(heat_f) and heat_f > best_J:
            best_J = heat_f
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
        reason = runaway_reason(rec, params) if abort_on_runaway else None
        if reason:
            stopped = "runaway"
            _finalize_run(
                outdir,
                params,
                n_iters=n_iters,
                lr=lr,
                beta_max=beta_max,
                seed=seed,
                history=history,
                best_J=best_J,
                best_iter=best_iter,
                peak_J=peak_J,
                peak_iter=peak_iter,
                gamma=gamma,
                aux=aux,
                best_gamma=best_gamma,
                best_aux=best_aux,
                stopped=stopped,
                start_gamma=start_gamma is not None,
                stall_iters=stall_iters,
            )
            raise RunawaySolveError(
                f"blocked or runaway solve at iter {it}: {reason}. "
                f"Returned design (iter {best_iter}, highest β with a finite J) "
                f"was written to {outdir}. "
                "Flow modes have no extra cold patch; do not add one — fix the design."
            )
        if (
            stall_iters
            and beta >= float(beta_max) - 1e-12
            and (it - best_iter) >= stall_iters
        ):
            stopped = "stall"
            print(f"  stop: no J improvement for {stall_iters} iters at β={beta:g}")
            break
        gamma = gamma_next

    _finalize_run(
        outdir,
        params,
        n_iters=n_iters,
        lr=lr,
        beta_max=beta_max,
        seed=seed,
        history=history,
        best_J=best_J,
        best_iter=best_iter,
        peak_J=peak_J,
        peak_iter=peak_iter,
        gamma=gamma,
        aux=aux,
        best_gamma=best_gamma,
        best_aux=best_aux,
        stopped=stopped,
        start_gamma=start_gamma is not None,
        stall_iters=stall_iters,
    )
    return best_gamma, best_aux, history


def optimize_hierarchy(params: ColdPlateParams, levels, **opt_kw):
    """Coarse-to-fine continuation. ``levels`` is ``((nx, ny, n_iters), ...)``."""
    levels = list(levels)
    if not levels:
        raise ValueError("levels must contain at least one (nx, ny, n_iters)")
    gamma = opt_kw.pop("start_gamma", None)
    outdir = Path(opt_kw.pop("outdir", "outputs"))
    last = None
    for i, (nx, ny, nit) in enumerate(levels):
        level_params = params._replace(n=(int(nx), int(ny)))
        if gamma is not None and tuple(gamma.shape) != tuple(level_params.n):
            gamma = upsample_field(gamma, level_params.n)
        last = optimize(
            level_params,
            n_iters=int(nit),
            start_gamma=gamma,
            outdir=outdir / f"level_{i}_{nx}x{ny}",
            **opt_kw,
        )
        gamma = last[0]
    return last


def _params_json(params: ColdPlateParams) -> dict:
    out = {}
    for key, val in params._asdict().items():
        out[key] = list(val) if isinstance(val, tuple) else val
    return out


def _finalize_run(
    outdir: Path,
    params: ColdPlateParams,
    *,
    n_iters,
    lr,
    beta_max,
    seed,
    history,
    best_J,
    best_iter,
    peak_J,
    peak_iter,
    gamma,
    aux,
    best_gamma,
    best_aux,
    stopped,
    start_gamma,
    stall_iters,
):
    chosen = highest_beta_best(history)
    best_iter = int(chosen["iter"])
    best_J = float(chosen["J"])
    for rec in history:
        rec["is_best"] = rec["iter"] == best_iter
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "history.json").write_text(json.dumps(history, indent=2))
    if aux is not None:
        _save_checkpoint(outdir, "final", gamma, aux, params)
    if best_aux is not None:
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
        peak_J=peak_J,
        peak_iter=peak_iter,
        stopped=stopped,
        start_gamma=start_gamma,
        stall_iters=stall_iters,
    )


def _write_run_json(outdir: Path, params: ColdPlateParams, **meta):
    history = meta["history"]
    last = history[-1]
    payload = {
        "params": _params_json(params),
        "n_iters": meta["n_iters"],
        "lr": meta["lr"],
        "beta_max": meta["beta_max"],
        "seed": meta["seed"],
        "stopped": meta.get("stopped", "completed"),
        "start_gamma": meta.get("start_gamma", False),
        "stall_iters": meta.get("stall_iters", 8),
        "J0": history[0]["J"],
        "J_final": last["J"],
        "J_best": meta["best_J"],
        "best_iter": meta["best_iter"],
        "J_peak": meta.get("peak_J", meta["best_J"]),
        "peak_iter": meta.get("peak_iter", meta["best_iter"]),
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
        "sym_err": last.get("sym_err", 0.0),
    }
    (outdir / "run.json").write_text(json.dumps(payload, indent=2))


def _save_checkpoint(outdir: Path, tag, gamma, aux, params: ColdPlateParams):
    outdir.mkdir(parents=True, exist_ok=True)
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
