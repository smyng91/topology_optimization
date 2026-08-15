"""2-D box topology optimization entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from topoopt.config import add_heat_mode_argument, default_2d
from topoopt.optimize import optimize
from topoopt.regions import add_region_arguments, specs_from_cli
from topoopt.viz import plot_2d, write_vtk


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=40)
    p.add_argument("--ny", type=int, default=40)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--vol", type=float, default=0.45)
    add_heat_mode_argument(p)
    add_region_arguments(p)
    p.add_argument("--flow", choices=("stokes", "darcy"), default="stokes")
    p.add_argument("--pe", type=float, default=40.0)
    p.add_argument("--q", type=float, default=1.0, help="Uniform volumetric heat source")
    p.add_argument("--k-ratio", type=float, default=100.0)
    p.add_argument("--rmin", type=float, default=2.2)
    p.add_argument(
        "--port-frac",
        type=float,
        default=0.5,
        help="Centered fraction of the left/right walls used as flow ports",
    )
    p.add_argument("--lr", type=float, default=0.2, help="Design move limit per iteration")
    p.add_argument("--beta-max", type=float, default=32.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="outputs/2d")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    hot_specs, cold_specs = specs_from_cli(args.hot, args.cold, args.heat, args.port_frac)
    params = default_2d(
        nx=args.nx,
        ny=args.ny,
        heat_mode=args.heat,
        flow_model=args.flow,
        vol_frac=args.vol,
        pe=args.pe,
        q_vol=args.q,
        k_solid=args.k_ratio,
        rmin=args.rmin,
        port_frac=args.port_frac,
        hot_specs=hot_specs,
        cold_specs=cold_specs,
    )
    outdir = Path(args.outdir)

    def callback(it, _gamma, aux, rec):
        if it == 1 or it % 5 == 0:
            plot_2d(
                aux,
                params,
                outdir / f"design_{it:03d}.png",
                title=f"2D {params.heat_label}  it {it}  J={rec['J']:.4f}",
            )

    _gamma, aux, _hist = optimize(
        params,
        n_iters=args.iters,
        lr=args.lr,
        beta_max=args.beta_max,
        seed=args.seed,
        outdir=outdir,
        callback=callback,
    )
    plot_2d(aux, params, outdir / "design_final.png", title=f"2D final design ({params.heat_label})")
    write_vtk(aux, params, outdir / "design_final.vtk")
    print(f"Wrote results to {outdir.resolve()}")


if __name__ == "__main__":
    main()
