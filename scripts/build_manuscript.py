#!/usr/bin/env python3
"""Run real MMS, adjoint, and benchmark cases; write paper figures and numbers.

This is the only supported way to produce manuscript figures. It never
synthesizes field data or convergence curves.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from topoopt.problems import (
    conduction_tree,
    convection_darcy,
    conjugate_stokes,
    custom_faces,
)
from topoopt.config import params2d
from topoopt.darcy import solve_darcy
from topoopt.experiments import (
    case_fingerprint,
    case_params,
    fingerprints_match,
    load_protocol,
)
from topoopt.filter import helmholtz_filter
from topoopt.flow2d import solve_stokes, stokes_relative_residual
from topoopt.grid import zero_face_velocity
from topoopt.heat import solve_energy
from topoopt.mms import (
    darcy_linear_exact,
    energy_advection_mms,
    energy_poisson_mms,
    helmholtz_cosine_mms,
    relative_l2,
    stokes_poiseuille_exact,
    wall_dirichlet_specs,
)
from topoopt.optimize import (
    NoTrustworthyResultError,
    RunawaySolveError,
    optimize,
    project_design,
)
from topoopt.problem import analyze
from topoopt.provenance import atomic_write_json, environment_manifest, file_digest
from topoopt.verification import (
    deterministic_direction,
    directional_taylor_test,
    fitted_log_order,
    high_contrast_design,
)

PAPER = ROOT / "paper"
FIGS = PAPER / "figures"
PROTOCOL = load_protocol()
OUT = ROOT / "outputs" / "release_v1"
FIGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "mathtext.fontset": "cm",
        "lines.linewidth": 1.4,
        "axes.linewidth": 0.7,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def _rel_l2(num, exact) -> float:
    return float(relative_l2(num, exact))


def run_mms() -> dict:
    poisson = []
    advection = []
    helmholtz = []
    stokes = []
    darcy = []

    for n in (8, 16, 32, 64):
        p = params2d(
            nx=n,
            ny=n,
            heat_mode="conduction",
            q_vol=0.0,
            hot_specs=(),
            cold_specs=wall_dirichlet_specs(),
            heat_iters=1200,
            filter_iters=20,
        )
        t_ex, q = energy_poisson_mms(p, k=p.k_fluid)
        t = solve_energy(jnp.zeros(p.n), zero_face_velocity(p), p, q=q)
        poisson.append({"n": n, "err": _rel_l2(t, t_ex)})

        p_adv = params2d(
            nx=n,
            ny=n,
            heat_mode="both",
            pe=2.0,
            q_vol=0.0,
            hot_specs=(),
            cold_specs=wall_dirichlet_specs(),
            heat_iters=1200,
            filter_iters=20,
        )
        t_ex, q, faces = energy_advection_mms(p_adv, k=p_adv.k_fluid, u_west=1.0)
        t = solve_energy(jnp.zeros(p_adv.n), faces, p_adv, q=q)
        advection.append({"n": n, "err": _rel_l2(t, t_ex)})

        p_h = params2d(nx=n, ny=n, rmin=2.0, filter_iters=800, solver_tol=1e-10)
        raw, filt = helmholtz_cosine_mms(p_h)
        helmholtz.append({"n": n, "err": _rel_l2(helmholtz_filter(raw, p_h), filt)})

        p_d = params2d(
            nx=n, ny=n, flow_model="darcy", port_frac=1.0, flow_iters=800, filter_iters=20
        )
        faces, pressure = solve_darcy(jnp.zeros(p_d.n), p_d)
        p_ex, u_ex = darcy_linear_exact(p_d)
        u_in = float(jnp.mean(faces[0][0]))
        darcy.append(
            {
                "n": n,
                "p_err": _rel_l2(pressure, p_ex),
                "u_err": abs(u_in - float(u_ex)) / float(u_ex),
            }
        )

    for n in (8, 16, 32):
        p_s = params2d(
            nx=n,
            ny=n,
            heat_mode="both",
            flow_model="stokes",
            port_frac=1.0,
            stokes_dp=20.0,
            div_eps=1e-4,
            flow_iters=200,
            uzawa_iters=80,
            stokes_kryl_iters=400,
            filter_iters=20,
        )
        gamma = jnp.zeros(p_s.n)
        sol = solve_stokes(gamma, p_s)
        u, v, p = sol
        u_ex, v_ex, p_ex = stokes_poiseuille_exact(p_s)
        rel = float(stokes_relative_residual(sol, gamma, p_s))
        stokes.append(
            {
                "n": n,
                "u_err": _rel_l2(u, u_ex),
                "p_err": _rel_l2(p, p_ex),
                "v_rms": float(jnp.sqrt(jnp.mean(v**2))),
                "res_rel": rel,
            }
        )

    def order(series, key="err"):
        records = [{"step": 1.0 / item["n"], key: item[key]} for item in series]
        return fitted_log_order(records, key)

    return {
        "poisson": poisson,
        "advection": advection,
        "helmholtz": helmholtz,
        "darcy": darcy,
        "stokes": stokes,
        "div_eps": 1e-4,
        "poisson_order": order(poisson),
        "advection_order": order(advection),
        "helmholtz_order": order(helmholtz),
        "stokes_u_order": order(stokes, "u_err"),
        "darcy_p_max": max(d["p_err"] for d in darcy),
    }


def run_adjoint() -> dict:
    params = conduction_tree(nx=16, ny=16, heat_iters=600, filter_iters=80)
    gamma = project_design(jnp.full(params.n, params.vol_frac), 2.0, params)
    direction = deterministic_direction(params.n, 21)
    result = directional_taylor_test(
        lambda design: analyze(design, 2.0, params)[0],
        gamma,
        direction,
        PROTOCOL["taylor_steps"],
    )
    return {
        "eps": PROTOCOL["taylor_steps"][-1],
        "n_samples": len(PROTOCOL["taylor_steps"]),
        "rel_max": float(
            max(
                record["centered_gradient_error"]
                / (abs(result["directional_derivative"]) + 1e-30)
                for record in result["records"]
            )
        ),
        "rel_mean": float(
            np.mean(
                [
                    record["centered_gradient_error"]
                    / (abs(result["directional_derivative"]) + 1e-30)
                    for record in result["records"]
                ]
            )
        ),
        "abs_max": max(record["centered_gradient_error"] for record in result["records"]),
        "grad_inf": result["gradient_norm"],
        "remainder_order": fitted_log_order(result["records"], "first_order_remainder"),
        "taylor": result,
    }


def run_coupled_taylor() -> dict:
    """Directional Taylor tests of the full filter–projection–flow–energy map."""
    specs = {
        "conduction": (conduction_tree, False, dict(nx=8, ny=8, heat_iters=400, filter_iters=80)),
        "darcy": (
            convection_darcy,
            True,
            dict(nx=8, ny=8, heat_iters=400, filter_iters=80, flow_iters=300),
        ),
        "stokes": (
            conjugate_stokes,
            True,
            dict(
                nx=6,
                ny=6,
                heat_iters=300,
                filter_iters=60,
                flow_iters=120,
                uzawa_iters=80,
                stokes_kryl_iters=300,
            ),
        ),
    }
    out = {}
    for name, (factory, channel, kwargs) in specs.items():
        params = factory(**kwargs)
        point = high_contrast_design(params.n, channel=channel)
        result = directional_taylor_test(
            lambda design, p=params: analyze(design, 4.0, p)[0],
            point,
            deterministic_direction(params.n, 31),
            PROTOCOL["taylor_steps"],
        )
        out[name] = {
            "remainder_order": fitted_log_order(result["records"], "first_order_remainder"),
            "value": result["value"],
            "records": result["records"],
        }
    return out


def run_fixed_design_studies() -> dict:
    """Separate discretization and solver-tolerance error from optimization."""
    study = PROTOCOL["fixed_design_studies"]
    rows = []
    for n in study["mesh_sizes"]:
        rmin = study["filter_radius_fraction"] * n
        params = conduction_tree(nx=n, ny=n, rmin=rmin, heat_iters=600, filter_iters=80)
        gamma = high_contrast_design(params.n, channel=False)
        _j, aux = analyze(gamma, 8.0, params)
        rows.append(
            {
                "n": n,
                "rmin": rmin,
                "J": float(_j),
                "T_mean": float(aux["T_mean"]),
                "energy_rel": float(aux["energy_rel"]),
            }
        )
    tols = []
    params = conduction_tree(nx=16, ny=16, heat_iters=800, filter_iters=80)
    gamma = high_contrast_design(params.n, channel=False)
    for tol in study["solver_tolerances"]:
        _j, aux = analyze(gamma, 8.0, params._replace(solver_tol=float(tol)))
        tols.append(
            {
                "solver_tol": float(tol),
                "J": float(_j),
                "energy_rel": float(aux["energy_rel"]),
            }
        )
    return {"mesh": rows, "solver_tol": tols}


def _save_npz(path: Path, aux, hist):
    np.savez_compressed(
        path,
        phys=np.asarray(aux["phys"]),
        T=np.asarray(aux["T"]),
        speed=np.asarray(aux["speed"]),
        p=np.asarray(aux["p"]),
        history=json.dumps(hist),
    )


def _summarize(name, params, aux, hist, run_json, baseline) -> dict:
    times = [h["time"] for h in hist[1:]]
    best = next(h for h in hist if h["iter"] == run_json["best_iter"])
    return {
        "name": name,
        "n": list(params.n),
        "heat_mode": params.heat_mode,
        "flow_model": params.flow_model if params.solves_flow else "none",
        "vol_frac": params.vol_frac,
        "pe": params.effective_pe,
        "n_iters": len(hist),
        "beta_max": max(h["beta"] for h in hist),
        "J0": hist[0]["J"],
        "J_best": run_json["J_best"],
        "best_iter": run_json["best_iter"],
        "best_beta": best["beta"],
        "J_peak": run_json["J_peak"],
        "peak_iter": run_json["peak_iter"],
        "stopped": run_json.get("stopped", "completed"),
        "vol_final": best["vol"],
        "vol_max_abs_err": max(abs(h["vol"] - params.vol_frac) for h in hist),
        "energy_rms_best": best["energy_rms"],
        "energy_rel_best": best.get("energy_rel", float("nan")),
        "energy_rms_final": hist[-1]["energy_rms"],
        "div_rms_final": best["div_rms"],
        "mass_err_final": best["mass_err"],
        "stokes_rel_final": best["stokes_rel"],
        "gray_final": hist[-1]["gray"],
        "gray_best": best["gray"],
        "T_mean": float(aux["T_mean"]),
        "T_max": float(aux["T_max"]),
        "u_in": float(aux["u_in"]),
        "speed_max": float(aux["speed_max"]),
        "median_step_s": float(np.median(times)) if times else hist[0]["time"],
        "baseline": baseline,
    }


def _baseline_tree(params) -> dict:
    g = project_design(jnp.full(params.n, params.vol_frac), 1.0, params)
    _j, aux_u = analyze(g, 1.0, params)
    _jf, aux_f = analyze(jnp.zeros(params.n), 1.0, params)
    return {
        "uniform_T_mean": float(aux_u["T_mean"]),
        "uniform_T_max": float(aux_u["T_max"]),
        "fluid_T_mean": float(aux_f["T_mean"]),
        "fluid_T_max": float(aux_f["T_max"]),
    }


def _baseline_custom(params) -> dict:
    g = project_design(jnp.full(params.n, params.vol_frac), 1.0, params)
    ju, _ = analyze(g, 1.0, params)
    jf, _ = analyze(jnp.zeros(params.n), 1.0, params)
    js, _ = analyze(jnp.ones(params.n), 1.0, params)
    return {
        "uniform_Q": float(ju),
        "fluid_Q": float(jf),
        "solid_Q": float(js),
    }


def _aux_from_best_dir(rundir: Path, hist, run: dict) -> dict:
    data = np.load(rundir / "state_best.npz")
    rec = next(h for h in hist if h["iter"] == run["best_iter"])
    return {
        "phys": data["phys"],
        "T": data["T"],
        "speed": data["speed"],
        "p": data["p"],
        "T_mean": rec["T_mean"],
        "T_max": rec["T_max"],
        "u_in": rec["u_in"],
        "speed_max": rec["speed_max"],
    }


def _load_completed(name, params, npz_name, rundir, baseline):
    data = np.load(OUT / npz_name, allow_pickle=False)
    hist = json.loads((OUT / rundir / "history.json").read_text(encoding="utf-8"))
    run = json.loads((OUT / rundir / "run.json").read_text(encoding="utf-8"))
    rec_best = next(h for h in hist if h["iter"] == run["best_iter"])
    aux = {
        "phys": data["phys"],
        "T": data["T"],
        "speed": data["speed"],
        "p": data["p"],
        "T_mean": float(np.mean(data["T"])),
        "T_max": float(np.max(data["T"])),
        "u_in": rec_best["u_in"],
        "speed_max": rec_best["speed_max"],
    }
    rec = _summarize(name, params, aux, hist, run, baseline)
    rec["history"] = hist
    return rec


def _try_load(name, params, npz_name, rundir, baseline, fingerprint):
    histp = OUT / rundir / "history.json"
    runp = OUT / rundir / "run.json"
    bestp = OUT / rundir / "state_best.npz"
    ident = OUT / rundir / "fingerprint.json"
    npz = OUT / npz_name
    if not (histp.is_file() and runp.is_file() and bestp.is_file() and ident.is_file()):
        return None
    saved = json.loads(ident.read_text(encoding="utf-8"))
    if not fingerprints_match(saved, fingerprint):
        return None
    run = json.loads(runp.read_text(encoding="utf-8"))
    if run.get("stopped") not in ("completed", "stall"):
        return None
    if run.get("best_iter") is None:
        return None
    if not npz.is_file():
        data = np.load(bestp)
        hist = json.loads(histp.read_text(encoding="utf-8"))
        _save_npz(
            npz,
            {"phys": data["phys"], "T": data["T"], "speed": data["speed"], "p": data["p"]},
            hist,
        )
    return _load_completed(name, params, npz_name, rundir, baseline)


def _matched_beta_baseline(params, beta: float) -> dict:
    gamma = project_design(jnp.full(params.n, params.vol_frac), beta, params)
    heat, aux = analyze(gamma, beta, params)
    return {
        "beta": float(beta),
        "J": float(heat),
        "T_mean": float(aux["T_mean"]),
        "T_max": float(aux["T_max"]),
        "energy_rel": float(aux["energy_rel"]),
        "vol": float(aux["V"]),
    }


def _run_case(name, params, *, n_iters, lr, beta_max, npz_name, rundir, baseline, seed, fingerprint):
    loaded = _try_load(name, params, npz_name, rundir, baseline, fingerprint)
    if loaded is not None:
        print(f"=== Benchmark {name} seed={seed}: reuse {rundir} ===", flush=True)
        loaded["seed"] = seed
        loaded["matched_beta"] = _matched_beta_baseline(params, loaded["best_beta"])
        loaded["published"] = True
        return loaded
    print(f"=== Benchmark {name} seed={seed}: running ===", flush=True)
    try:
        _g, aux, hist = optimize(
            params, n_iters=n_iters, lr=lr, beta_max=beta_max, seed=seed, outdir=OUT / rundir
        )
        run = json.loads((OUT / rundir / "run.json").read_text(encoding="utf-8"))
    except (RunawaySolveError, NoTrustworthyResultError):
        hist = json.loads((OUT / rundir / "history.json").read_text(encoding="utf-8"))
        run = json.loads((OUT / rundir / "run.json").read_text(encoding="utf-8"))
        if run.get("best_iter") is None:
            atomic_write_json(OUT / rundir / "fingerprint.json", fingerprint)
            raise
        aux = _aux_from_best_dir(OUT / rundir, hist, run)
    _save_npz(OUT / npz_name, aux, hist)
    atomic_write_json(OUT / rundir / "fingerprint.json", fingerprint)
    rec = _summarize(name, params, aux, hist, run, baseline)
    rec["history"] = hist
    rec["seed"] = seed
    rec["matched_beta"] = _matched_beta_baseline(params, rec["best_beta"])
    rec["published"] = True
    return rec


def run_benchmarks(model_source_sha256: str) -> dict:
    cases = {}
    seeds = list(PROTOCOL["seeds"])
    for name in PROTOCOL["cases"]:
        spec = PROTOCOL["cases"][name]
        params = case_params(PROTOCOL, name)
        if name == "tree":
            baseline = _baseline_tree(params)
        elif name == "custom":
            baseline = _baseline_custom(params)
        elif name == "darcy":
            matched = _matched_beta_baseline(params, 1.0)
            baseline = {
                "uniform_T_mean": matched["T_mean"],
                "uniform_T_max": matched["T_max"],
            }
        else:
            baseline = {}
        seed_records = []
        for seed in seeds:
            rundir = Path(name) / f"seed_{seed}"
            fingerprint = case_fingerprint(
                PROTOCOL, name, seed=seed, model_source_sha256=model_source_sha256
            )
            try:
                rec = _run_case(
                    name,
                    params,
                    n_iters=spec["n_iters"],
                    lr=spec["lr"],
                    beta_max=spec["beta_max"],
                    npz_name=str(rundir / f"{name}.npz"),
                    rundir=rundir,
                    baseline=baseline,
                    seed=seed,
                    fingerprint=fingerprint,
                )
            except (RunawaySolveError, NoTrustworthyResultError) as exc:
                print(f"=== Benchmark {name} seed={seed}: rejected ({exc}) ===", flush=True)
                seed_records.append({"seed": seed, "published": False, "reason": str(exc)})
                continue
            seed_records.append(rec)
        published = [rec for rec in seed_records if rec.get("published")]
        if not published:
            raise RuntimeError(f"protocol case {name} produced no trustworthy seed")
        canonical = next((rec for rec in published if rec.get("seed") == seeds[0]), published[0])
        canonical = dict(canonical)
        canonical["seeds"] = [
            {
                "seed": rec.get("seed"),
                "published": rec.get("published", False),
                "J_best": rec.get("J_best"),
                "best_beta": rec.get("best_beta"),
                "gray_best": rec.get("gray_best"),
                "energy_rel_best": rec.get("energy_rel_best"),
                "reason": rec.get("reason"),
            }
            for rec in seed_records
        ]
        js = [rec["J_best"] for rec in published]
        canonical["J_best_min"] = min(js)
        canonical["J_best_max"] = max(js)
        canonical["n_published_seeds"] = len(published)
        cases[name] = canonical
        src = OUT / name / f"seed_{canonical['seed']}" / f"{name}.npz"
        if src.is_file():
            (OUT / f"{name}.npz").write_bytes(src.read_bytes())
    return cases


def plot_fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.03, 0.62, 0.18, 0.28, r"Raw design" "\n" r"$\gamma\in[0,1]^N$", "#E8F0FE"),
        (0.27, 0.62, 0.20, 0.28, r"Cone filter" "\n" r"$\tilde{\gamma}=\mathcal{F}(\gamma)$", "#E8F0FE"),
        (0.53, 0.62, 0.20, 0.28, r"Tanh $+$ volume" "\n" r"$\mathrm{mean}(\bar{\gamma})=v^*$", "#E8F0FE"),
        (0.79, 0.55, 0.18, 0.38, "Forward solve\nStokes/Darcy\nenergy FV", "#FEF7E0"),
        (0.79, 0.08, 0.18, 0.28, r"$J=-\mathrm{mean}(T)$" "\n" r"or $Q_{\mathrm{hot}}$", "#E6F4EA"),
        (0.42, 0.08, 0.28, 0.30, r"Implicit adjoint" "\n" r"$A^\top\lambda=\partial J/\partial x$", "#FCE8E6"),
        (0.03, 0.08, 0.30, 0.30, r"Projected step" "\n" r"$\ell=\ell_0/\sqrt{\max(\beta,1)}$", "#E8F0FE"),
    ]
    for x, y, w, h, text, fc in boxes:
        ax.add_patch(
            mpl.patches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                facecolor=fc, edgecolor="#3C4043", linewidth=1.0,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    arrows = [
        ((0.21, 0.76), (0.27, 0.76)),
        ((0.47, 0.76), (0.53, 0.76)),
        ((0.73, 0.76), (0.79, 0.76)),
        ((0.88, 0.55), (0.88, 0.36)),
        ((0.79, 0.22), (0.70, 0.22)),
        ((0.42, 0.22), (0.33, 0.22)),
        ((0.12, 0.38), (0.12, 0.62)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="#3C4043"))
    fig.savefig(FIGS / "fig1_pipeline.pdf")
    fig.savefig(FIGS / "fig1_pipeline.png", dpi=300)
    plt.close(fig)


def plot_mms(mms: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.15))
    n_e = [d["n"] for d in mms["poisson"]]
    ax1.loglog(n_e, [d["err"] for d in mms["poisson"]], "o-", color="#1A73E8",
               label=r"Poisson conduction")
    ax1.loglog(n_e, [d["err"] for d in mms["advection"]], "s--", color="#D93025",
               label=r"Advection, $\mathrm{Pe}=2$, $u=(1,0)$")
    href = np.array([n_e[0], n_e[-1]], dtype=float)
    e0 = mms["poisson"][0]["err"]
    ax1.loglog(href, e0 * (n_e[0] / href) ** 2, "k:", lw=1.0, label=r"$O(h^2)$")
    ax1.loglog(href, mms["advection"][0]["err"] * (n_e[0] / href), "k-.", lw=1.0,
               label=r"$O(h)$")
    ax1.set_xlabel(r"$N_x=N_y$")
    ax1.set_ylabel(r"relative $L^2$ error")
    ax1.set_title("(a) Energy operator")
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend(frameon=False, fontsize=7.5)
    ax1.set_xticks(n_e)
    ax1.set_xticklabels([str(n) for n in n_e])
    ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    n_h = [d["n"] for d in mms["helmholtz"]]
    ax2.loglog(n_h, [d["err"] for d in mms["helmholtz"]], "^-", color="#137333",
               label="Helmholtz filter")
    n_s = [d["n"] for d in mms["stokes"]]
    ax2.loglog(n_s, [d["u_err"] for d in mms["stokes"]], "d-", color="#F29900",
               label="Stokes–Poiseuille $u$")
    ax2.loglog(n_h, [d["p_err"] for d in mms["darcy"]], "x--", color="#9334E6",
               label="Darcy linear $p$")
    ax2.loglog(href, mms["helmholtz"][0]["err"] * (n_h[0] / href) ** 2, "k:", lw=1.0,
               label=r"$O(h^2)$")
    ax2.set_xlabel(r"$N_x=N_y$")
    ax2.set_ylabel(r"relative $L^2$ error")
    ax2.set_title("(b) Filter and flow")
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend(frameon=False, fontsize=7.5)
    ax2.set_xticks(n_h)
    ax2.set_xticklabels([str(n) for n in n_h])
    ax2.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    fig.savefig(FIGS / "fig2_mms_convergence.pdf")
    fig.savefig(FIGS / "fig2_mms_convergence.png", dpi=300)
    plt.close(fig)


def _imshow(ax, field, title, cmap, vmin=None, vmax=None):
    im = ax.imshow(
        np.asarray(field).T, origin="lower", extent=[0, 1, 0, 1],
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal")
    return im


def _add_overlays(ax, mode):
    if mode == "tree":
        ax.plot([0.46, 0.54], [0.0, 0.0], color="deepskyblue", lw=3.0, solid_capstyle="butt")
    elif mode in ("darcy", "stokes"):
        ax.plot([0.0, 0.0], [0.25, 0.75], color="#34A853", lw=3.0, solid_capstyle="butt")
        ax.plot([1.0, 1.0], [0.25, 0.75], color="#34A853", lw=3.0, solid_capstyle="butt")
    elif mode == "custom":
        ax.plot([0.25, 0.75], [1.0, 1.0], color="crimson", lw=3.0, solid_capstyle="butt")
        ax.plot([0.25, 0.75], [0.0, 0.0], color="deepskyblue", lw=3.0, solid_capstyle="butt")


def plot_design_montage():
    """One Nature-style panel: four factory designs."""
    specs = [
        (OUT / "tree.npz", "tree", r"Tree $\bar{\gamma}$", "phys", "viridis", 0, 1),
        (OUT / "tree.npz", "tree", r"Tree $T$", "T", "inferno", None, None),
        (OUT / "custom.npz", "custom", r"Sandwich $\bar{\gamma}$", "phys", "viridis", 0, 1),
        (OUT / "custom.npz", "custom", r"Sandwich $T$", "T", "inferno", None, None),
        (OUT / "darcy.npz", "darcy", r"Darcy $\bar{\gamma}$", "phys", "viridis", 0, 1),
        (OUT / "darcy.npz", "darcy", r"Darcy $|\mathbf{u}|$", "speed", "cividis", None, None),
        (OUT / "stokes.npz", "stokes", r"Stokes $\bar{\gamma}$", "phys", "viridis", 0, 1),
        (OUT / "stokes.npz", "stokes", r"Stokes $|\mathbf{u}|$", "speed", "cividis", None, None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.55))
    letters = "abcdefgh"
    for ax, spec, let in zip(axes.ravel(), specs, letters):
        path, mode, title, key, cmap, vmin, vmax = spec
        data = np.load(path)
        im = _imshow(ax, data[key], f"({let}) {title}", cmap, vmin, vmax)
        _add_overlays(ax, mode)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
    axes[0, 0].set_ylabel(r"$y$")
    axes[1, 0].set_ylabel(r"$y$")
    for ax in axes[1]:
        ax.set_xlabel(r"$x$")
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_designs.pdf")
    fig.savefig(FIGS / "fig3_designs.png", dpi=300)
    plt.close(fig)


def plot_conduction_pair(npz_path, save_name, mode):
    data = np.load(npz_path)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.85))
    im0 = _imshow(axes[0], data["phys"], r"(a) $\bar{\gamma}$", "viridis", 0, 1)
    im1 = _imshow(axes[1], data["T"], r"(b) $T$", "inferno")
    for ax, im in zip(axes, (im0, im1)):
        _add_overlays(ax, mode)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(FIGS / f"{save_name}.pdf")
    fig.savefig(FIGS / f"{save_name}.png", dpi=300)
    plt.close(fig)


def plot_flow_quad(npz_path, save_name, mode):
    data = np.load(npz_path)
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.15))
    fields = [
        (data["phys"], r"(a) $\bar{\gamma}$", "viridis", 0, 1),
        (data["speed"], r"(b) $|\mathbf{u}|$", "cividis", None, None),
        (data["p"], r"(c) $p$", "coolwarm", None, None),
        (data["T"], r"(d) $T$", "inferno", None, None),
    ]
    for ax, (field, title, cmap, vmin, vmax) in zip(axes, fields):
        im = _imshow(ax, field, title, cmap, vmin, vmax)
        _add_overlays(ax, mode)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.set_ylabel("")
    axes[0].set_ylabel(r"$y$")
    fig.savefig(FIGS / f"{save_name}.pdf")
    fig.savefig(FIGS / f"{save_name}.png", dpi=300)
    plt.close(fig)


def plot_history(cases: dict):
    hist = cases["tree"]["history"]
    hist_s = cases["stokes"]["history"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    ax1, ax2, ax3, ax4 = axes.ravel()
    it = [h["iter"] for h in hist]
    ax1.plot(it, [h["J"] for h in hist], color="#1A73E8", lw=1.6)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel(r"$J=-\mathrm{mean}(T)$")
    ax1.set_title("(a) Conduction-tree objective")
    ax1.grid(True, ls=":", alpha=0.5)

    vols = np.array([h["vol"] for h in hist])
    ax2.plot(it, vols, color="#137333", lw=1.6)
    ax2.axhline(0.30, color="#D93025", ls="--", lw=1.0)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel(r"$\mathrm{mean}(\bar{\gamma})$")
    ax2.set_title(r"(b) Volume, $v^*=0.30$")
    ax2.grid(True, ls=":", alpha=0.5)

    tree_rel = [h.get("energy_rel", h["energy_rms"]) for h in hist]
    stokes_rel = [h.get("energy_rel", h["energy_rms"]) for h in hist_s]
    ax3.semilogy(it, tree_rel, color="#D93025", lw=1.5,
                 label=r"tree $\|R_T\|_{\mathrm{rel}}$")
    ax3.semilogy([h["iter"] for h in hist_s], stokes_rel,
                 color="#1A73E8", lw=1.5, label=r"Stokes $\|R_T\|_{\mathrm{rel}}$")
    ax3.semilogy([h["iter"] for h in hist_s], [h["div_rms"] for h in hist_s],
                 color="#F29900", lw=1.5, label=r"Stokes $\|\nabla\cdot\mathbf{u}\|_{\mathrm{rms}}$")
    ax3.axhline(1e-3, color="0.4", ls=":", lw=0.8)
    ax3.set_xlabel("iteration")
    ax3.set_ylabel("residual")
    ax3.set_title("(c) Energy and continuity residuals")
    ax3.grid(True, which="both", ls=":", alpha=0.5)
    ax3.legend(frameon=False, fontsize=7)

    ax4.plot(it, 100 * np.array([h["gray"] for h in hist]), color="#9334E6", lw=1.6)
    ax4.set_xlabel("iteration")
    ax4.set_ylabel(r"gray cells (\%)")
    ax4.set_title(r"(d) $0.05<\bar{\gamma}<0.95$")
    ax4.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig7_optimization_history.pdf")
    fig.savefig(FIGS / "fig7_optimization_history.png", dpi=300)
    plt.close(fig)


def write_numbers(mms: dict, adjoint: dict, cases: dict, taylor: dict, studies: dict):
    def sci(x):
        x = float(x)
        if not math.isfinite(x):
            return r"\mathrm{n/a}"
        mant, exp = f"{x:.2e}".split("e")
        return rf"{mant}\times 10^{{{int(exp)}}}"

    def stopped(rec):
        return str(rec.get("stopped", "completed")).replace("_", r"\_")

    t = cases["tree"]
    d = cases["darcy"]
    s = cases["stokes"]
    c = cases["custom"]
    lines = [
        rf"\newcommand{{\PoissonOrder}}{{{mms['poisson_order']:.2f}}}",
        rf"\newcommand{{\AdvectionOrder}}{{{mms['advection_order']:.2f}}}",
        rf"\newcommand{{\HelmholtzOrder}}{{{mms['helmholtz_order']:.2f}}}",
        rf"\newcommand{{\HelmholtzErrFine}}{{{sci(mms['helmholtz'][-1]['err'])}}}",
        rf"\newcommand{{\StokesUOrder}}{{{mms['stokes_u_order']:.2f}}}",
        rf"\newcommand{{\DarcyPMax}}{{{sci(mms['darcy_p_max'])}}}",
        rf"\newcommand{{\PoissonErrFine}}{{{sci(mms['poisson'][-1]['err'])}}}",
        rf"\newcommand{{\AdvErrFine}}{{{sci(mms['advection'][-1]['err'])}}}",
        rf"\newcommand{{\StokesUFine}}{{{sci(mms['stokes'][-1]['u_err'])}}}",
        rf"\newcommand{{\AdjointRelMax}}{{{sci(adjoint['rel_max'])}}}",
        rf"\newcommand{{\AdjointAbsMax}}{{{sci(adjoint['abs_max'])}}}",
        rf"\newcommand{{\TreeNx}}{{{t['n'][0]}}}",
        rf"\newcommand{{\TreeIters}}{{{t['n_iters']}}}",
        rf"\newcommand{{\TreeBeta}}{{{t['best_beta']:.0f}}}",
        rf"\newcommand{{\TreeJzero}}{{{t['J0']:.4f}}}",
        rf"\newcommand{{\TreeJbest}}{{{t['J_best']:.4f}}}",
        rf"\newcommand{{\TreeTmean}}{{{t['T_mean']:.4f}}}",
        rf"\newcommand{{\TreeTmax}}{{{t['T_max']:.4f}}}",
        rf"\newcommand{{\TreeTmeanUni}}{{{t['baseline']['uniform_T_mean']:.4f}}}",
        rf"\newcommand{{\TreeTmeanFluid}}{{{t['baseline']['fluid_T_mean']:.4f}}}",
        rf"\newcommand{{\TreeGray}}{{{100 * t['gray_best']:.1f}}}",
        rf"\newcommand{{\TreeVolErr}}{{{sci(t['vol_max_abs_err'])}}}",
        rf"\newcommand{{\TreeErms}}{{{sci(t['energy_rms_best'])}}}",
        rf"\newcommand{{\TreeErel}}{{{sci(t.get('energy_rel_best', float('nan')))}}}",
        rf"\newcommand{{\TreeStepMs}}{{{1000 * t['median_step_s']:.1f}}}",
        rf"\newcommand{{\TreeStopped}}{{{stopped(t)}}}",
        rf"\newcommand{{\DarcyNx}}{{{d['n'][0]}}}",
        rf"\newcommand{{\DarcyIters}}{{{d['n_iters']}}}",
        rf"\newcommand{{\DarcyBeta}}{{{d['best_beta']:.0f}}}",
        rf"\newcommand{{\DarcyJzero}}{{{d['J0']:.4f}}}",
        rf"\newcommand{{\DarcyJbest}}{{{d['J_best']:.4f}}}",
        rf"\newcommand{{\DarcyTmean}}{{{d['T_mean']:.4f}}}",
        rf"\newcommand{{\DarcyGray}}{{{100 * d['gray_best']:.1f}}}",
        rf"\newcommand{{\DarcyVolErr}}{{{sci(d['vol_max_abs_err'])}}}",
        rf"\newcommand{{\DarcyErms}}{{{sci(d['energy_rms_best'])}}}",
        rf"\newcommand{{\DarcyErel}}{{{sci(d.get('energy_rel_best', float('nan')))}}}",
        rf"\newcommand{{\DarcyUin}}{{{d['u_in']:.3g}}}",
        rf"\newcommand{{\DarcyStopped}}{{{stopped(d)}}}",
        rf"\newcommand{{\StokesNx}}{{{s['n'][0]}}}",
        rf"\newcommand{{\StokesIters}}{{{s['n_iters']}}}",
        rf"\newcommand{{\StokesBeta}}{{{s['best_beta']:.0f}}}",
        rf"\newcommand{{\StokesJzero}}{{{s['J0']:.4f}}}",
        rf"\newcommand{{\StokesJbest}}{{{s['J_best']:.4f}}}",
        rf"\newcommand{{\StokesTmean}}{{{s['T_mean']:.4f}}}",
        rf"\newcommand{{\StokesGray}}{{{100 * s['gray_best']:.1f}}}",
        rf"\newcommand{{\StokesVolErr}}{{{sci(s['vol_max_abs_err'])}}}",
        rf"\newcommand{{\StokesErms}}{{{sci(s['energy_rms_best'])}}}",
        rf"\newcommand{{\StokesErel}}{{{sci(s.get('energy_rel_best', float('nan')))}}}",
        rf"\newcommand{{\StokesDiv}}{{{sci(s['div_rms_final'])}}}",
        rf"\newcommand{{\StokesRel}}{{{sci(s['stokes_rel_final'])}}}",
        rf"\newcommand{{\StokesMass}}{{{sci(s['mass_err_final'])}}}",
        rf"\newcommand{{\StokesStepMs}}{{{1000 * s['median_step_s']:.1f}}}",
        rf"\newcommand{{\StokesStopped}}{{{stopped(s)}}}",
        rf"\newcommand{{\CustomNx}}{{{c['n'][0]}}}",
        rf"\newcommand{{\CustomIters}}{{{c['n_iters']}}}",
        rf"\newcommand{{\CustomBeta}}{{{c['best_beta']:.0f}}}",
        rf"\newcommand{{\CustomJpeak}}{{{c['J_peak']:.3f}}}",
        rf"\newcommand{{\CustomJzero}}{{{c['J0']:.3f}}}",
        rf"\newcommand{{\CustomJbest}}{{{c['J_best']:.3f}}}",
        rf"\newcommand{{\CustomQuni}}{{{c['baseline']['uniform_Q']:.3f}}}",
        rf"\newcommand{{\CustomGray}}{{{100 * c['gray_best']:.1f}}}",
        rf"\newcommand{{\CustomErms}}{{{sci(c['energy_rms_best'])}}}",
        rf"\newcommand{{\CustomErel}}{{{sci(c.get('energy_rel_best', float('nan')))}}}",
        rf"\newcommand{{\CustomVolErr}}{{{sci(c['vol_max_abs_err'])}}}",
        rf"\newcommand{{\CustomStopped}}{{{stopped(c)}}}",
        rf"\newcommand{{\TreeJmin}}{{{t['J_best_min']:.4f}}}",
        rf"\newcommand{{\TreeJmax}}{{{t['J_best_max']:.4f}}}",
        rf"\newcommand{{\DarcyJmin}}{{{d['J_best_min']:.4f}}}",
        rf"\newcommand{{\DarcyJmax}}{{{d['J_best_max']:.4f}}}",
        rf"\newcommand{{\StokesJmin}}{{{s['J_best_min']:.4f}}}",
        rf"\newcommand{{\StokesJmax}}{{{s['J_best_max']:.4f}}}",
        rf"\newcommand{{\CustomJmin}}{{{c['J_best_min']:.3f}}}",
        rf"\newcommand{{\CustomJmax}}{{{c['J_best_max']:.3f}}}",
        rf"\newcommand{{\TreeJmatched}}{{{t['matched_beta']['J']:.4f}}}",
        rf"\newcommand{{\DarcyJmatched}}{{{d['matched_beta']['J']:.4f}}}",
        rf"\newcommand{{\StokesJmatched}}{{{s['matched_beta']['J']:.4f}}}",
        rf"\newcommand{{\CustomJmatched}}{{{c['matched_beta']['J']:.3f}}}",
        rf"\newcommand{{\TaylorCondOrder}}{{{taylor['conduction']['remainder_order']:.2f}}}",
        rf"\newcommand{{\TaylorDarcyOrder}}{{{taylor['darcy']['remainder_order']:.2f}}}",
        rf"\newcommand{{\TaylorStokesOrder}}{{{taylor['stokes']['remainder_order']:.2f}}}",
        rf"\newcommand{{\MeshJcoarse}}{{{studies['mesh'][0]['J']:.4f}}}",
        rf"\newcommand{{\MeshJfine}}{{{studies['mesh'][-1]['J']:.4f}}}",
        rf"\newcommand{{\NPublishedSeeds}}{{{t['n_published_seeds']}}}",
        rf"\newcommand{{\StokesDivEps}}{{{sci(mms['div_eps'])}}}",
    ]
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    slim = {
        "protocol_id": PROTOCOL["protocol_id"],
        "mms": {k: v for k, v in mms.items()},
        "adjoint": {k: v for k, v in adjoint.items() if k != "taylor"},
        "taylor": taylor,
        "studies": studies,
        "cases": {
            k: {kk: vv for kk, vv in rec.items() if kk != "history"} for k, rec in cases.items()
        },
    }
    (PAPER / "results.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    return slim


def write_provenance(manifest: dict, slim: dict) -> None:
    figures = sorted(path.name for path in FIGS.glob("*") if path.suffix in {".pdf", ".png"})
    payload = {
        "manifest": manifest,
        "protocol_id": PROTOCOL["protocol_id"],
        "artifacts": {
            "results_json_sha256": file_digest(PAPER / "results.json"),
            "numbers_tex_sha256": file_digest(PAPER / "numbers.tex"),
            "figures": {name: file_digest(FIGS / name) for name in figures},
        },
        "published_cases": {
            name: {
                "seed": rec.get("seed"),
                "n_published_seeds": rec.get("n_published_seeds"),
                "J_best": rec.get("J_best"),
            }
            for name, rec in slim["cases"].items()
        },
    }
    atomic_write_json(PAPER / "provenance.json", payload)
    atomic_write_json(OUT / "provenance.json", payload)


def main():
    t0 = time.time()
    manifest = environment_manifest(ROOT, protocol_path=PAPER / "experiments.json")
    print("MMS ...", flush=True)
    mms = run_mms()
    print(json.dumps({k: mms[k] for k in mms if k not in ("poisson", "advection", "helmholtz", "darcy", "stokes")}, indent=2))
    print("Adjoint / Taylor ...", flush=True)
    adjoint = run_adjoint()
    taylor = run_coupled_taylor()
    print(json.dumps({k: {"remainder_order": v["remainder_order"]} for k, v in taylor.items()}, indent=2))
    print("Fixed-design studies ...", flush=True)
    studies = run_fixed_design_studies()
    print("Benchmarks ...", flush=True)
    cases = run_benchmarks(str(manifest["model_source_sha256"]))
    print("Figures ...", flush=True)
    plot_fig1()
    plot_mms(mms)
    plot_conduction_pair(OUT / "tree.npz", "fig3_conduction_tree", "tree")
    plot_flow_quad(OUT / "darcy.npz", "fig4_convection_darcy", "darcy")
    plot_flow_quad(OUT / "stokes.npz", "fig5_conjugate_stokes", "stokes")
    plot_conduction_pair(OUT / "custom.npz", "fig6_custom_regions", "custom")
    plot_design_montage()
    plot_history(cases)
    slim = write_numbers(mms, adjoint, cases, taylor, studies)
    write_provenance(manifest, slim)
    print(f"Done in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
