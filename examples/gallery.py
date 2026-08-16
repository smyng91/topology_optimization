"""Fine-mesh gallery of the named cases in ``examples/problems.py``.

``optimize`` uses ``lr=0.12``, ``beta_max=32``, ``seed=0``. Full meshes:
80×80 / 150–200 iters (Stokes 48×48 / 100, Uzawa 250, Schur 400,
heat 1200). ``--quick`` is 16×16 / 8 iters (Stokes 12×12 / 4).
See ``examples/README.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.problems import (
    conduction_tree,
    conjugate_darcy,
    conjugate_stokes,
    convection_darcy,
    custom_boxes,
    custom_faces,
)
from topoopt.optimize import optimize
from topoopt.symmetry import max_error
from topoopt.viz import plot_2d, write_vtk


def _cases(quick=False):
    if quick:
        coarse = dict(filter_iters=40, heat_iters=150, flow_iters=80)
        return [
            ("2d_conduction", conduction_tree(nx=16, ny=16, rmin=1.5, **coarse), 8),
            ("2d_convection_darcy", convection_darcy(nx=16, ny=16, **coarse), 8),
            ("2d_both_darcy", conjugate_darcy(nx=16, ny=16, **coarse), 8),
            (
                "2d_both_stokes",
                conjugate_stokes(
                    nx=12,
                    ny=12,
                    rmin=2.0,
                    filter_iters=40,
                    heat_iters=80,
                    flow_iters=40,
                    uzawa_iters=20,
                    stokes_kryl_iters=40,
                ),
                4,
            ),
            ("2d_custom_faces", custom_faces(nx=16, ny=16, **coarse), 8),
            ("2d_custom_boxes", custom_boxes(nx=16, ny=16, **coarse), 8),
        ]
    fine = dict(filter_iters=200, heat_iters=800, flow_iters=280)
    return [
        ("2d_conduction", conduction_tree(nx=80, ny=80, rmin=1.5, **fine), 200),
        ("2d_convection_darcy", convection_darcy(nx=80, ny=80, **fine), 150),
        ("2d_both_darcy", conjugate_darcy(nx=80, ny=80, **fine), 150),
        (
            "2d_both_stokes",
            conjugate_stokes(
                nx=48,
                ny=48,
                rmin=2.0,
                filter_iters=120,
                heat_iters=1200,
                flow_iters=250,
                uzawa_iters=250,
                stokes_kryl_iters=400,
            ),
            100,
        ),
        ("2d_custom_faces", custom_faces(nx=80, ny=80, **fine), 180),
        ("2d_custom_boxes", custom_boxes(nx=80, ny=80, **fine), 180),
    ]


def run_examples(root: str | Path = "outputs", quick: bool = False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for name, params, iters in _cases(quick=quick):
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
            "J_best": next(h["J"] for h in hist if h.get("is_best")),
            "best_iter": next(h["iter"] for h in hist if h.get("is_best")),
            "J_peak": max(h["J"] for h in hist),
            "vol": hist[-1]["vol"],
            "T_min": float(aux["T"].min()),
            "T_max": float(aux["T"].max()),
            "T_mean": float(aux["T"].mean()),
            "speed_max": float(aux["speed"].max()),
            "hot": list(params.hot_specs),
            "cold": list(params.cold_specs),
            "symmetry": list(params.symmetry),
            "sym_err": float(max_error(_g, params)),
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
    p.add_argument("--outdir", default="outputs")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Coarse mesh and few iterations (not a publishable gallery)",
    )
    args = p.parse_args(argv)
    run_examples(args.outdir, quick=args.quick)


if __name__ == "__main__":
    main()
