#!/usr/bin/env python3
"""Tutorial: user-defined Dirichlet T patches and volumetric q regions.

``hot_specs`` / ``cold_specs`` prescribe *T*. That turns off *uniform*
``q`` (legacy) and, with no ``q_specs``, changes the objective from
``J = -mean(T)`` to the conductive heat leaving the hot patches.

``q_specs`` marks cells that *generate* heat (``q = q_vol``); *T* still
floats there. The same ``face:…`` / ``box:…`` strings work for both.
You can set Dirichlet T and a source subdomain together.

Spec strings (same as ``--hot`` / ``--cold`` / ``--q-region``):

* ``face:top:frac=0.5`` — centered half of the top wall
* ``face:bottom:frac=0.5`` — centered half of the bottom wall
* ``box:xmin,xmax,ymin,ymax`` — volumetric region

Default: Dirichlet sandwich (``custom_faces``) — hot top, cold bottom,
no flow, solid bridges the two faces.

Parameters: mesh 32×32, 50 iters, ``β_max=16``, ``lr=0.2``
(``--quick``: 16×16, 8 iters, ``β_max=8``); ``heat_iters=300``,
``filter_iters=60``. ``--boxes`` → ``custom_boxes`` (no symmetry).
``--source`` → ``localized_source`` (top-center q box, bottom sink,
``J = -mean(T)``). Shared flags: ``--quick``, ``--outdir``, ``--seed``.

Run from the repo root::

    python examples/05_custom_regions.py
    python examples/05_custom_regions.py --quick
    python examples/05_custom_regions.py --boxes
    python examples/05_custom_regions.py --source
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

from _common import add_run_args, optimize_and_save
from examples.problems import custom_boxes, custom_faces, localized_source


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(p, "outputs/05_custom_regions")
    p.add_argument(
        "--boxes",
        action="store_true",
        help="Use box Dirichlet domains instead of opposite faces",
    )
    p.add_argument(
        "--source",
        action="store_true",
        help="Volumetric q in a top-center box (T is free there)",
    )
    args = p.parse_args(argv)
    if args.boxes and args.source:
        p.error("use --boxes or --source, not both")

    n = 16 if args.quick else 32
    iters = 8 if args.quick else 50
    if args.source:
        params = localized_source(nx=n, ny=n, heat_iters=300, filter_iters=60)
        title = "localized q source"
        objective = "J = -mean(T)  (volume source on q_specs)"
    elif args.boxes:
        params = custom_boxes(nx=n, ny=n, heat_iters=300, filter_iters=60)
        title = "custom boxes"
        objective = "J = heat leaving hot patches"
    else:
        params = custom_faces(nx=n, ny=n, heat_iters=300, filter_iters=60)
        title = "hot top / cold bottom"
        objective = "J = heat leaving hot patches"
    print(
        f"{objective}  hot={params.hot_specs}  cold={params.cold_specs}  "
        f"q_region={params.q_specs}\n"
        "Crimson / sky-blue overlays are Dirichlet T; orange is volumetric q.\n"
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
