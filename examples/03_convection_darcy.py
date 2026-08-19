#!/usr/bin/env python3
"""Tutorial: pressure-driven Darcy cooling.

Flow is on, conductivity is uniform ``k_fluid`` (the design does not
change k). There is one inlet on the left-wall centerline and one
outlet on the right-wall centerline: ``p = p_in`` on the left port,
``p = 0`` on the right. No other inlets, outlets, or cold patches.
Throughput is design-dependent — block the path and the flow drops.

``J = -mean(T)`` still: a through-channel lets cold inlet fluid sweep
the volume source. The optimizer seeds a mid-height duct; from a uniform
field the local step prefers an inlet cavity and an outlet dam.

Parameters (on top of ``convection_darcy``: ``vol=0.45``, ``pe=40``,
``rmin=2``, ``port_frac=0.5``, ``symmetry=y``, no cold patch): mesh
32×32, 50 iters, ``β_max=16``, ``lr=0.2`` (``--quick``: 16×16, 8 iters,
``β_max=8``); ``flow_iters=200``, ``heat_iters=300``,
``filter_iters=60``. Shared flags: ``--quick``, ``--outdir``, ``--seed``.

Run from the repo root::

    python examples/03_convection_darcy.py
    python examples/03_convection_darcy.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

from _common import add_run_args, optimize_and_save
from examples.problems import convection_darcy


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(p, "outputs/03_convection_darcy")
    args = p.parse_args(argv)

    n = 16 if args.quick else 32
    iters = 8 if args.quick else 50
    params = convection_darcy(nx=n, ny=n, flow_iters=200, heat_iters=300, filter_iters=60)
    print(
        f"Darcy centerline ports  frac={params.port_frac:g}  p_in={params.p_in:g}  "
        f"Pe={params.pe:g}  cold={params.cold_specs}\n"
        "Look for a left→right channel in the density and speed panels.\n"
    )
    optimize_and_save(
        params,
        args.outdir,
        iters,
        lr=0.2,
        beta_max=8.0 if args.quick else 16.0,
        seed=args.seed,
        plot_every=4 if args.quick else 10,
        title="convection / Darcy",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
