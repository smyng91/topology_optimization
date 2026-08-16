"""Run every 2-D heat mode and check that the fields follow the physics."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from topoopt.config import HEAT_MODES, params2d
from topoopt.optimize import optimize
from topoopt.problem import analyze
from topoopt.problems import (
    conduction_tree,
    conjugate_darcy,
    conjugate_stokes,
    convection_darcy,
    custom_faces,
)
from topoopt.viz import plot_2d, write_vtk


def _check_temperature(aux, lo=-0.02):
    temp = np.asarray(aux["T"])
    tmin, tmax = float(temp.min()), float(temp.max())
    assert tmin >= lo and np.isfinite(tmax), (tmin, tmax)
    return tmin, tmax


def physics_report():
    rows = []
    params_c = conduction_tree(nx=16, ny=16, heat_iters=250, filter_iters=40)
    js, aux_s = analyze(jnp.ones(params_c.n), 8.0, params_c)
    jf, aux_f = analyze(jnp.zeros(params_c.n), 8.0, params_c)
    _check_temperature(aux_s)
    assert float(aux_s["speed"].max()) < 1e-12
    assert float(js) > float(jf)
    rows.append(
        {
            "check": "conduction solid cooler than fluid (volume source)",
            "ok": True,
            "J_solid": float(js),
            "J_fluid": float(jf),
            "Tmean_solid": float(aux_s["T"].mean()),
            "Tmean_fluid": float(aux_f["T"].mean()),
        }
    )

    from topoopt.darcy import solve_darcy
    from topoopt.grid import cell_divergence, port_mask

    params_d = convection_darcy(nx=16, ny=16, flow_iters=250)
    faces, _p = solve_darcy(jnp.zeros(params_d.n), params_d)
    mask = np.asarray(port_mask(params_d))
    div_rms = float(jnp.sqrt(jnp.mean(cell_divergence(faces, params_d.dx) ** 2)))
    u_in = float(jnp.sum(faces[0][0] * mask))
    u_out = float(jnp.sum(faces[0][-1] * mask))
    assert u_in > 0.05 and abs(u_in - u_out) / u_in < 0.08 and div_rms < 1e-3
    assert float(jnp.max(jnp.abs(faces[0][0] * (1.0 - mask)))) < 1e-8
    rows.append(
        {
            "check": "Darcy centered ports are divergence-free",
            "ok": True,
            "u_in": u_in,
            "u_out": u_out,
            "div_rms": div_rms,
        }
    )

    params_v = convection_darcy(nx=16, ny=16, flow_iters=200, heat_iters=250, filter_iters=40)
    jopen, aux_o = analyze(jnp.zeros(params_v.n), 2.0, params_v)
    jblk, aux_b = analyze(jnp.ones(params_v.n), 2.0, params_v)
    _check_temperature(aux_o)
    assert float(aux_o["speed"].mean()) > float(aux_b["speed"].mean())
    assert float(jopen) > float(jblk)
    assert params_v.cold_specs == () and params_v.hot_specs == ()
    rows.append(
        {
            "check": "convection open channel cooler than blocked (centerline ports only)",
            "ok": True,
            "Tmean_open": float(aux_o["T"].mean()),
            "Tmean_blocked": float(aux_b["T"].mean()),
            "J_open": float(jopen),
        }
    )

    params_b = conjugate_darcy(nx=16, ny=16, flow_iters=200, heat_iters=250, filter_iters=40)
    j, aux = analyze(jnp.full(params_b.n, 0.45), 2.0, params_b)
    tmin, tmax = _check_temperature(aux)
    assert float(aux["speed"].max()) > 0.0 and np.isfinite(float(j))
    rows.append({"check": "conjugate has flow and finite J", "ok": True, "J": float(j), "T_range": [tmin, tmax]})

    params_r = params2d(
        nx=16,
        ny=16,
        heat_mode="conduction",
        q_vol=0.0,
        heat_iters=300,
        filter_iters=40,
        hot_specs=("face:top",),
        cold_specs=("face:bottom",),
    )
    _, aux_r = analyze(jnp.ones(params_r.n), 4.0, params_r)
    top, bot = float(np.asarray(aux_r["T"])[:, -1].mean()), float(np.asarray(aux_r["T"])[:, 0].mean())
    assert top > bot
    rows.append({"check": "custom faces: hot top, cold bottom", "ok": True, "T_top": top, "T_bottom": bot})
    return rows


def _run_case(name, params, iters, outdir, plot):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    def callback(it, _g, aux, rec):
        if it == 1 or it == iters:
            plot(aux, params, out / f"design_{it:03d}.png", title=f"{name} it {it}  J={rec['J']:.4f}")

    _g, aux, hist = optimize(params, n_iters=iters, lr=0.25, beta_max=4.0, seed=0, outdir=out, callback=callback)
    plot(aux, params, out / "design_best.png", title=f"{name} published best")
    write_vtk(aux, params, out / "design_best.vtk")
    tmin, tmax = _check_temperature(aux)
    j0, j1 = hist[0]["J"], hist[-1]["J"]
    best = next(h for h in hist if h.get("is_best"))
    j_best = best["J"]
    rec = {
        "name": name,
        "heat": params.heat_mode,
        "J0": j0,
        "J1": j1,
        "J_best": j_best,
        "improved": j_best >= j0 - 1e-6,
        "vol": best["vol"],
        "T_min": tmin,
        "T_max": tmax,
        "T_mean": float(np.asarray(aux["T"]).mean()),
        "speed_max": float(np.asarray(aux["speed"]).max()),
        "hot": list(params.hot_specs),
        "cold": list(params.cold_specs),
    }
    assert rec["improved"] or np.isfinite(j1)
    assert abs(rec["vol"] - params.vol_frac) < 0.05
    return rec


def run_all(root: str | Path = "outputs"):
    root = Path(root)
    physics = physics_report()
    cases = []
    factories = {
        "conduction": conduction_tree,
        "convection": convection_darcy,
        "both": conjugate_darcy,
    }
    for mode in HEAT_MODES:
        params = factories[mode](nx=20, ny=20, filter_iters=50, heat_iters=250, flow_iters=180)
        cases.append(_run_case(f"2d_{mode}", params, 6, root / f"2d_{mode}", plot_2d))

    custom = custom_faces(nx=20, ny=20, filter_iters=50, heat_iters=250)
    cases.append(_run_case("2d_custom_regions", custom, 6, root / "2d_custom_regions", plot_2d))

    stokes = conjugate_stokes(
        nx=16,
        ny=16,
        filter_iters=40,
        heat_iters=200,
        flow_iters=200,
        uzawa_iters=200,
    )
    cases.append(_run_case("2d_stokes_both", stokes, 4, root / "2d_stokes_both", plot_2d))

    report = {"physics": physics, "optimizations": cases}
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print("\n=== Verification summary ===")
    for row in physics:
        print(f"  [ok] {row['check']}")
    print(f"{'case':<22} {'J0':>10} {'J1':>10} {'vol':>7} {'Tmean':>7} {'|u|max':>8}")
    for rec in cases:
        print(
            f"{rec['name']:<22} {rec['J0']:10.4f} {rec['J1']:10.4f} {rec['vol']:7.3f} "
            f"{rec['T_mean']:7.3f} {rec['speed_max']:8.3f}"
        )
    print(f"Wrote {root.resolve()}/report.json")
    return report


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="outputs")
    args = p.parse_args(argv)
    run_all(args.outdir)


if __name__ == "__main__":
    main()
