#!/usr/bin/env python3
"""Tutorial: user-defined hot / cold patches.

Setting ``hot_specs`` turns the uniform volume source *off* and changes
the objective from ``J = -mean(T)`` to the conductive heat leaving those
patches. Spec strings are the same as the CLI ``--hot`` / ``--cold``:

* ``face:top:frac=0.5`` — centered half of the top wall
* ``face:bottom:frac=0.5`` — centered half of the bottom wall
* ``box:xmin,xmax,ymin,ymax`` — volumetric region (see the gallery)

This script is a Dirichlet sandwich: hot top, cold bottom, no flow.
Solid should form conducting bridges between the two faces.

Parameters: mesh 32×32, 50 iters, ``β_max=16``, ``lr=0.2``
(``--quick``: 16×16, 8 iters, ``β_max=8``); ``heat_iters=300``,
``filter_iters=60``. Default factory ``custom_faces`` (``vol=0.40``,
``q=0``, ``symmetry=x``). ``--boxes`` switches to ``custom_boxes``
(no symmetry). Shared flags: ``--quick``, ``--outdir``, ``--seed``.

Run from the repo root::

    python examples/05_custom_regions.py
    python examples/05_custom_regions.py --quick
    python examples/05_custom_regions.py --boxes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

from _common import add_run_args, optimize_and_save
from examples.problems import custom_boxes, custom_faces


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(p, "outputs/05_custom_regions")
    p.add_argument(
        "--boxes",
        action="store_true",
        help="Use box domains instead of opposite faces",
    )
    args = p.parse_args(argv)

    n = 16 if args.quick else 32
    iters = 8 if args.quick else 50
    if args.boxes:
        params = custom_boxes(nx=n, ny=n, heat_iters=300, filter_iters=60)
        title = "custom boxes"
    else:
        params = custom_faces(nx=n, ny=n, heat_iters=300, filter_iters=60)
        title = "hot top / cold bottom"
    hot, cold = params.hot_specs, params.cold_specs
    print(
        f"J = heat leaving hot patches  hot={hot}  cold={cold}\n"
        "Crimson / sky-blue overlays on the PNG are the Dirichlet regions.\n"
    )
    optimize_and_save(
        params,
        args.outdir,
        iters,
        lr=0.2,
        beta_max=8.0 if args.quick else 16.0,
        seed=args.seed,
        plot_every=4 if args.quick else 10,
        title=title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
