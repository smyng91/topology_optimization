"""Generic 2-D runner. Problem geometry and BCs come from ``--config``."""

from __future__ import annotations

import argparse
from pathlib import Path

from topoopt.config import add_heat_mode_argument, load_params, params2d
from topoopt.optimize import optimize, optimize_hierarchy
from topoopt.regions import add_region_arguments, specs_from_cli
from topoopt.viz import plot_2d, write_vtk


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=None,
        help=(
            "Problem factory (module:func / path.py:func) or a JSON/YAML file "
            "(e.g. examples.problems:conduction_tree or examples/configs/conduction_tree.json)"
        ),
    )
    p.add_argument("--nx", type=int, default=None)
    p.add_argument("--ny", type=int, default=None)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--vol", type=float, default=None)
    add_heat_mode_argument(p)
    add_region_arguments(p)
    p.add_argument("--flow", choices=("stokes", "darcy"), default=None)
    p.add_argument("--pe", type=float, default=None)
    p.add_argument(
        "--q",
        type=float,
        default=None,
        help="Volumetric heat strength (uniform, or on --q-region only)",
    )
    p.add_argument("--k-ratio", type=float, default=None)
    p.add_argument("--rmin", type=float, default=None)
    p.add_argument(
        "--port-frac",
        type=float,
        default=None,
        help="Height of the left-centerline inlet and right-centerline outlet",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=0.2,
        help="Design move limit at β=1 (decays as lr/sqrt(β))",
    )
    p.add_argument("--beta-max", type=float, default=32.0)
    p.add_argument(
        "--symmetry",
        default=None,
        help="Mirror the design: x (left–right), y (top–bottom), or x,y",
    )
    p.add_argument(
        "--mesh-schedule",
        default=None,
        help="Coarse-to-fine continuation: nx,ny,iters:nx,ny,iters",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="outputs/2d")
    return p


def _cli_overrides(args) -> dict:
    hot_specs, cold_specs, q_specs = specs_from_cli(args.hot, args.cold, args.q_region)
    out = {}
    if args.nx is not None:
        out["nx"] = args.nx
    if args.ny is not None:
        out["ny"] = args.ny
    if args.vol is not None:
        out["vol_frac"] = args.vol
    if args.heat is not None:
        out["heat_mode"] = args.heat
    if args.flow is not None:
        out["flow_model"] = args.flow
    if args.pe is not None:
        out["pe"] = args.pe
    if args.q is not None:
        out["q_vol"] = args.q
    if args.k_ratio is not None:
        out["k_solid"] = args.k_ratio
    if args.rmin is not None:
        out["rmin"] = args.rmin
    if args.port_frac is not None:
        out["port_frac"] = args.port_frac
    if args.hot:
        out["hot_specs"] = hot_specs
    if args.cold:
        out["cold_specs"] = cold_specs
    if args.q_region:
        out["q_specs"] = q_specs
    if args.symmetry is not None:
        out["symmetry"] = args.symmetry
    return out


def parse_mesh_schedule(spec: str):
    levels = []
    for part in spec.split(":"):
        nx, ny, nit = part.split(",")
        levels.append((int(nx), int(ny), int(nit)))
    return levels


def build_params(args):
    overrides = _cli_overrides(args)
    if args.config:
        return load_params(args.config, **overrides)
    return params2d(
        nx=overrides.pop("nx", 40),
        ny=overrides.pop("ny", 40),
        **overrides,
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    params = build_params(args)
    outdir = Path(args.outdir)

    def callback(it, _gamma, aux, rec):
        if it == 1 or it % 5 == 0:
            plot_2d(
                aux,
                params,
                outdir / f"design_{it:03d}.png",
                title=f"2D {params.heat_label}  it {it}  J={rec['J']:.4f}",
            )

    opt_kw = dict(lr=args.lr, beta_max=args.beta_max, seed=args.seed, outdir=outdir)
    if args.mesh_schedule:
        _gamma, aux, hist = optimize_hierarchy(params, parse_mesh_schedule(args.mesh_schedule), **opt_kw)
        last_n = parse_mesh_schedule(args.mesh_schedule)[-1][:2]
        params = params._replace(n=(int(last_n[0]), int(last_n[1])))
    else:
        _gamma, aux, hist = optimize(params, n_iters=args.iters, callback=callback, **opt_kw)
    plot_2d(aux, params, outdir / "design_final.png", title=f"2D final design ({params.heat_label})")
    write_vtk(aux, params, outdir / "design_final.vtk")
    best_rec = next(h for h in hist if h.get("is_best"))
    print(
        f"Wrote results to {outdir.resolve()}  "
        f"J_best={best_rec['J']:.6f}  (β={best_rec['beta']:g} design)"
    )


if __name__ == "__main__":
    main()
