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
    "energy_rel",
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

# 40 bisections of [-2, 2] give a shift tolerance of 4/2^40 ≈ 4e-12.
VOLUME_BISECTION_ITERS = 40

# Keep-best ignores iterates whose energy residual is not a solved PDE.
ENERGY_REL_MAX = 1.0e-3
ENERGY_RMS_MAX = 1.0e-2
VOLUME_ABS_MAX = 1.0e-8
MASS_REL_MAX = 5.0e-2
DIVERGENCE_RMS_MAX = 5.0e-3
STOKES_REL_MAX = 1.0e-5


class RunawaySolveError(RuntimeError):
    """Energy / temperature blew up, usually a sealed flow design."""


class NoTrustworthyResultError(RuntimeError):
    """No iterate met the predeclared publication evidence gates."""


def project_physical_volume(gamma_raw, beta, params: ColdPlateParams, target):
    """Shift the raw field so the filtered/projected solid fraction matches ``target``.

    Stokes port cells are pinned to fluid *inside* the residual, otherwise a
    post-hoc ``keep_ports_open`` step would drift ``mean(γ̄)`` off ``v*``.
    """

    def mean_phys(shift):
        g = keep_ports_open(jnp.clip(gamma_raw - shift, 0.0, 1.0), params)
        return physical_density(g, beta, params).mean()

    def body(_, bounds):
        a, b = bounds
        mid = 0.5 * (a + b)
        return jax.lax.cond(mean_phys(mid) > target, lambda: (mid, b), lambda: (a, mid))

    a, b = jax.lax.fori_loop(
        0, VOLUME_BISECTION_ITERS, body, (jnp.array(-2.0), jnp.array(2.0))
    )
    return keep_ports_open(jnp.clip(gamma_raw - 0.5 * (a + b), 0.0, 1.0), params)


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
    """Symmetry → volume equality (with Stokes port pin) → symmetry."""
    gamma = apply_symmetry(jnp.clip(gamma, 0.0, 1.0), params)
    gamma = project_physical_volume(gamma, beta, params, params.vol_frac)
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
    """Piecewise-constant β continuation, doubling until ``beta_max``.

    At least 40% of the steps are spent at ``β_max`` so the tanh projection
    has time to sharpen after the gray continuation stages.
    """
    levels = []
    beta = 1.0
    while beta < beta_max - 1e-9:
        levels.append(beta)
        beta *= 2.0
    levels.append(float(beta_max))
    n_last = max(1, int(math.ceil(0.4 * n_iters)))
    n_rest = max(n_iters - n_last, 0)
    n_early = max(len(levels) - 1, 1)
    chunk = max(n_rest // n_early, 1) if n_rest else 0
    sched = []
    for b in levels[:-1]:
        sched.extend([b] * chunk)
    sched.extend([levels[-1]] * max(n_iters - len(sched), n_last))
    if len(sched) < n_iters:
        sched.extend([levels[-1]] * (n_iters - len(sched)))
    return jnp.array(sched[:n_iters])


def energy_trustworthy(rec, rel_max: float = ENERGY_REL_MAX, rms_max: float = ENERGY_RMS_MAX) -> bool:
    """Whether ``T`` is a solved energy field, not a Krylov leftover."""
    if "energy_rel" in rec:
        return float(rec["energy_rel"]) <= rel_max
    return float(rec.get("energy_rms", 0.0)) <= rms_max


def result_rejection_reasons(
    rec,
    params: ColdPlateParams | None = None,
    *,
    target_vol: float | None = None,
    energy_max: float | None = None,
) -> list[str]:
    """Return every failed numerical-evidence gate for one iterate."""
    reasons = []
    required_finite = ("J",)
    if params is not None:
        required_finite += (
            "vol",
            "energy_rms",
            "energy_rel",
            "div_rms",
            "mass_err",
            "stokes_rel",
            "T_mean",
            "T_max",
            "speed_max",
        )
    for key in required_finite:
        if key not in rec or not math.isfinite(float(rec[key])):
            reasons.append(f"{key} is missing or non-finite")
    if reasons:
        return reasons
    if energy_max is not None:
        if float(rec.get("energy_rms", float("inf"))) > energy_max:
            reasons.append(f"energy_rms>{energy_max:g}")
    elif not energy_trustworthy(rec):
        reasons.append("energy residual above gate")
    if params is None:
        return reasons
    target = params.vol_frac if target_vol is None else target_vol
    if abs(float(rec["vol"]) - float(target)) > VOLUME_ABS_MAX:
        reasons.append(f"volume error>{VOLUME_ABS_MAX:g}")
    if abs(float(rec.get("sym_err", 0.0))) > VOLUME_ABS_MAX:
        reasons.append(f"symmetry error>{VOLUME_ABS_MAX:g}")
    if params.solves_flow:
        if float(rec["mass_err"]) > MASS_REL_MAX:
            reasons.append(f"mass imbalance>{MASS_REL_MAX:g}")
        if float(rec["div_rms"]) > DIVERGENCE_RMS_MAX:
            reasons.append(f"divergence RMS>{DIVERGENCE_RMS_MAX:g}")
        if params.flow_model == "stokes" and float(rec["stokes_rel"]) > STOKES_REL_MAX:
            reasons.append(f"Stokes residual>{STOKES_REL_MAX:g}")
    return reasons


def result_trustworthy(
    rec,
    params: ColdPlateParams | None = None,
    *,
    target_vol: float | None = None,
    energy_max: float | None = None,
) -> bool:
    """Whether one iterate is eligible to be returned or published."""
    return not result_rejection_reasons(
        rec, params, target_vol=target_vol, energy_max=energy_max
    )


def highest_beta_best(
    history,
    energy_max: float | None = None,
    *,
    params: ColdPlateParams | None = None,
    target_vol: float | None = None,
):
    """Best-``J`` record at the largest β that passes every evidence gate.

    ``J`` is not comparable across continuation: ``physical_density``
    (and the PDE) change when the tanh projection sharpens. Iterates that
    fail the energy, flow, mass, finite-field, symmetry, or volume gates
    cannot win. If every record fails, this function raises rather than
    returning a scientifically untrustworthy fallback. ``energy_max`` is
    an optional RMS override for focused tests.
    """
    if not history:
        raise ValueError("history is empty")

    by_beta = {}
    for rec in history:
        if not result_trustworthy(
            rec,
            params,
            target_vol=target_vol,
            energy_max=energy_max,
        ):
            continue
        beta = float(rec["beta"])
        prev = by_beta.get(beta)
        if prev is None or float(rec["J"]) > float(prev["J"]):
            by_beta[beta] = rec
    if not by_beta:
        failures = result_rejection_reasons(
            history[-1],
            params,
            target_vol=target_vol,
            energy_max=energy_max,
        )
        raise NoTrustworthyResultError(
            "no iterate passed the evidence gates; last iterate: "
            + ", ".join(failures)
        )
    return by_beta[max(by_beta)]


def _select_returned(history, snapshots, params: ColdPlateParams):
    """Map ``highest_beta_best`` onto the stored (γ, aux) snapshot."""
    chosen = highest_beta_best(history, params=params)
    gamma, aux = snapshots[int(chosen["iter"])]
    return chosen, gamma, aux


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
    that passes every evidence gate**, not the global max ``J`` across
    continuation. Gates cover finite fields, energy and flow residuals,
    mass balance, volume feasibility, and imposed symmetry. There is no
    fallback to an unconverged iterate. A soft mid-β field is a different
    discrete problem (and a poor 0–1 geometry) even when its ``J`` is
    larger. Stall is counted only inside the current β level, so entering
    ``β_max`` does not immediately stop because an earlier gray iterate
    was better.

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
        f"{'it':>4}  {'beta':>6}  {'J':>12}  {'vol':>8}  {'E_rel':>9}  "
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
    snapshots = {}
    for it in range(1, n_iters + 1):
        beta = float(betas[it - 1])
        if prev_beta is not None and abs(beta - prev_beta) > 1e-12:
            # New continuation stage: do not compare J to a softer β,
            # and restart the stall clock so β_max is not aborted because
            # a mid-β iterate was the last improvement.
            best_J = -float("inf")
            best_iter = it
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
        rejection_reasons = result_rejection_reasons(rec, params)
        evidence_ok = not rejection_reasons
        rec["evidence_ok"] = evidence_ok
        rec["rejection_reasons"] = rejection_reasons
        if evidence_ok and heat_f > peak_J:
            peak_J = heat_f
            peak_iter = it
        if evidence_ok and heat_f > best_J:
            best_J = heat_f
            best_iter = it
            best_gamma = gamma
            best_aux = aux
            _save_checkpoint(outdir, "best", best_gamma, best_aux, params)
        rec["is_best"] = it == best_iter
        snapshots[it] = (gamma, aux)
        history.append(rec)
        print(
            f"{it:4d}  {beta:6.1f}  {float(heat):12.6f}  {vol:8.4f}  "
            f"{rec['energy_rel']:9.2e}  {rec['div_rms']:9.2e}  {rec['mass_err']:9.3f}  "
            f"{rec['gray']:6.3f}  {dt:7.2f}s"
        )
        if rec["energy_rel"] > ENERGY_REL_MAX:
            print(
                f"  warning: relative energy residual {rec['energy_rel']:.3e} "
                f"> {ENERGY_REL_MAX:.0e}"
            )
        if params.solves_flow and rec["mass_err"] > MASS_REL_MAX:
            print(
                f"  warning: port mass error {rec['mass_err']:.3f} "
                f"> {MASS_REL_MAX:.2f}"
            )
        if callback is not None:
            callback(it, gamma, aux, rec)
        if it == 1 or it == n_iters or it % 10 == 0:
            _save_checkpoint(outdir, it, gamma, aux, params)
        reason = runaway_reason(rec, params) if abort_on_runaway else None
        if reason:
            stopped = "runaway"
            selected = _finalize_run(
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
                snapshots=snapshots,
                stopped=stopped,
                start_gamma=start_gamma is not None,
                stall_iters=stall_iters,
            )
            selected_text = (
                f"Trustworthy design iter {selected[0]['iter']} was preserved."
                if selected[0] is not None
                else "No iterate passed all evidence gates."
            )
            raise RunawaySolveError(
                f"blocked or runaway solve at iter {it}: {reason}. "
                f"{selected_text} Diagnostics were written to {outdir}. "
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

    chosen, best_gamma, best_aux = _finalize_run(
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
        snapshots=snapshots,
        stopped=stopped,
        start_gamma=start_gamma is not None,
        stall_iters=stall_iters,
    )
    if chosen is None:
        raise NoTrustworthyResultError(
            f"optimization finished but no iterate passed all evidence gates; "
            f"diagnostics were written to {outdir}"
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
    snapshots,
    stopped,
    start_gamma,
    stall_iters,
):
    try:
        chosen, best_gamma, best_aux = _select_returned(history, snapshots, params)
        best_iter = int(chosen["iter"])
        best_J = float(chosen["J"])
    except NoTrustworthyResultError:
        chosen = None
        best_gamma = None
        best_aux = None
        best_iter = None
        best_J = None
    for rec in history:
        rec["is_best"] = best_iter is not None and rec["iter"] == best_iter
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
    return chosen, best_gamma, best_aux


def _write_run_json(outdir: Path, params: ColdPlateParams, **meta):
    history = meta["history"]
    last = history[-1]
    best_iter = meta.get("best_iter")
    best = (
        next((record for record in history if record["iter"] == best_iter), None)
        if best_iter is not None
        else None
    )

    record_keys = (
        "iter",
        "beta",
        "J",
        "vol",
        "energy_rms",
        "energy_rel",
        "div_rms",
        "mass_err",
        "stokes_rel",
        "gray",
        "T_mean",
        "T_max",
        "speed_max",
        "u_in",
        "u_out",
        "sym_err",
        "evidence_ok",
        "rejection_reasons",
    )

    def summary(record):
        return None if record is None else {key: record.get(key) for key in record_keys}

    peak_j = meta.get("peak_J")
    if peak_j is not None and not math.isfinite(float(peak_j)):
        peak_j = None
    peak_iter = meta.get("peak_iter") if peak_j is not None else None
    payload = {
        "schema_version": 2,
        "params": _params_json(params),
        "n_iters": meta["n_iters"],
        "lr": meta["lr"],
        "beta_max": meta["beta_max"],
        "seed": meta["seed"],
        "stopped": meta.get("stopped", "completed"),
        "start_gamma": meta.get("start_gamma", False),
        "stall_iters": meta.get("stall_iters", 8),
        "published_state": "state_best.npz" if best is not None else None,
        "best": summary(best),
        "final": summary(last),
        "evidence_gates": {
            "energy_rel_max": ENERGY_REL_MAX,
            "energy_rms_max": ENERGY_RMS_MAX,
            "volume_abs_max": VOLUME_ABS_MAX,
            "mass_rel_max": MASS_REL_MAX,
            "divergence_rms_max": DIVERGENCE_RMS_MAX,
            "stokes_rel_max": STOKES_REL_MAX,
        },
        "J0": history[0]["J"],
        "J_final": last["J"],
        "J_best": None if best is None else best["J"],
        "best_iter": best_iter,
        "J_peak": peak_j,
        "peak_iter": peak_iter,
        "vol_final": last["vol"],
        "vol_best": None if best is None else best["vol"],
    }
    diagnostic_keys = (
        "energy_rms",
        "energy_rel",
        "div_rms",
        "mass_err",
        "stokes_rel",
        "gray",
        "T_mean",
        "T_max",
        "speed_max",
        "u_in",
        "u_out",
        "sym_err",
    )
    for key in diagnostic_keys:
        payload[key] = None if best is None else best.get(key)
        payload[f"{key}_final"] = last.get(key)
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
