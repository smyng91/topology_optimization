"""Run a 2-D gallery covering every heat mode, flow model, and region type."""

from __future__ import annotations

import json
from pathlib import Path

from topoopt.config import default_2d
from topoopt.optimize import optimize
from topoopt.viz import plot_2d, write_vtk


def _cases():
    # Fine mesh + long β continuation so conduction can grow a branching tree.
    # rmin is in cells; on 80×80 that is a much thinner physical filter than 32×32.
    common = dict(nx=80, ny=80, rmin=2.0, filter_iters=200, heat_iters=400, flow_iters=280)
    return [
        (
            "2d_conduction",
            default_2d(
                **common,
                heat_mode="conduction",
                vol_frac=0.30,
                rmin=1.5,
                cold_specs=("face:bottom:frac=0.08",),
            ),
            200,
        ),
        ("2d_convection_darcy", default_2d(**common, heat_mode="convection", flow_model="darcy"), 150),
        ("2d_both_darcy", default_2d(**common, heat_mode="both", flow_model="darcy"), 150),
        (
            "2d_both_stokes",
            default_2d(
                nx=48,
                ny=48,
                rmin=2.0,
                heat_mode="both",
                flow_model="stokes",
                filter_iters=120,
                heat_iters=320,
                flow_iters=250,
                uzawa_iters=250,
            ),
            100,
        ),
        (
            "2d_custom_faces",
            default_2d(
                **common,
                heat_mode="conduction",
                q_vol=0.0,
                hot_specs=("face:top:frac=0.5",),
                cold_specs=("face:bottom:frac=0.5",),
            ),
            180,
        ),
        (
            "2d_custom_boxes",
            default_2d(
                **common,
                heat_mode="conduction",
                q_vol=0.0,
                hot_specs=("box:0.2,0.8,0.0,0.18",),
                cold_specs=("box:0.0,0.18,0.25,0.75", "face:left"),
            ),
            180,
        ),
    ]


def run_examples(root: str | Path = "outputs/examples"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for name, params, iters in _cases():
        out = root / name
        out.mkdir(parents=True, exist_ok=True)

        def callback(it, _g, aux, rec, params=params, out=out, name=name, iters=iters):
            if it == 1 or it == iters or it % 30 == 0:
                plot_2d(aux, params, out / f"design_{it:03d}.png", title=f"{name} it {it}  J={rec['J']:.4f}")

        _g, aux, hist = optimize(
            params, n_iters=iters, lr=0.12, beta_max=32.0, seed=0, outdir=out, callback=callback
        )
        plot_2d(aux, params, out / "design_final.png", title=f"{name} final")
        write_vtk(aux, params, out / "design_final.vtk")
        rec = {
            "name": name,
            "heat": params.heat_mode,
            "flow": params.flow_model if params.solves_flow else "none",
            "n": list(params.n),
            "iters": iters,
            "J0": hist[0]["J"],
            "J1": hist[-1]["J"],
            "J_best": max(h["J"] for h in hist),
            "best_iter": next(h["iter"] for h in hist if h.get("is_best")),
            "vol": hist[-1]["vol"],
            "T_min": float(aux["T"].min()),
            "T_max": float(aux["T"].max()),
            "T_mean": float(aux["T"].mean()),
            "speed_max": float(aux["speed"].max()),
            "hot": list(params.hot_specs),
            "cold": list(params.cold_specs),
            "history": [{"iter": h["iter"], "J": h["J"], "vol": h["vol"]} for h in hist],
            "outdir": str(out),
        }
        records.append(rec)
        print(f"  saved {out / 'design_final.png'}")

    report = {"cases": records}
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print("\n=== Example gallery ===")
    print(f"{'case':<22} {'heat':<12} {'flow':<7} {'J0':>10} {'J_best':>10} {'vol':>7} {'Tmean':>7}")
    for rec in records:
        print(
            f"{rec['name']:<22} {rec['heat']:<12} {rec['flow']:<7} "
            f"{rec['J0']:10.4f} {rec['J_best']:10.4f} {rec['vol']:7.3f} {rec['T_mean']:7.3f}"
        )
    print(f"Wrote {root.resolve()}/report.json")
    return report


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="outputs/examples")
    args = p.parse_args(argv)
    run_examples(args.outdir)


if __name__ == "__main__":
    main()
