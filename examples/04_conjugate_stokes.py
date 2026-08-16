#!/usr/bin/env python3
"""Tutorial: conjugate heat transfer with Stokes–Brinkman flow.

``heat=both``: flow *and* ``k(γ)`` (solid conducts, fluid is permeable).
Stokes is pressure-driven like Darcy (``stokes_dp`` on the left-centerline
inlet, ``p = 0`` on the right-centerline outlet), not a prescribed inlet
velocity. There are no other inlets, outlets, or cold patches.

Two design projections are required for a through-channel — they are
not PDE Dirichlet conditions:

* a mid-height channel seed (otherwise GD opens an inlet cavity)
* ``keep_ports_open``: a one-cell fluid layer on the port faces
  (one solid cell there seals a pressure port)

The forward Stokes solve is Uzawa + CG on the pressure Schur complement.
The adjoint is the residual discrete adjoint, not an unrolled Uzawa loop.

This tutorial uses a modest mesh. The gallery
(``python examples/gallery.py``) runs 48×48 / 100 iterations.

Run from the repo root::

    python examples/04_conjugate_stokes.py
    python examples/04_conjugate_stokes.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

from _common import add_run_args, optimize_and_save
from examples.problems import conjugate_stokes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(p, "outputs/04_conjugate_stokes")
    args = p.parse_args(argv)

    n = 16 if args.quick else 24
    iters = 6 if args.quick else 40
    params = conjugate_stokes(
        nx=n, ny=n, uzawa_iters=40, heat_iters=280, filter_iters=60
    )
    print(
        f"Stokes–Brinkman  Δp={params.stokes_dp:g}  α_max={params.alpha_max:g}  "
        f"Schur CG={params.stokes_kryl_iters}\n"
        "Look for a through-channel; watch mass_err and stokes_rel in the log.\n"
    )
    optimize_and_save(
        params,
        args.outdir,
        iters,
        lr=0.16,
        beta_max=8.0 if args.quick else 16.0,
        seed=args.seed,
        plot_every=3 if args.quick else 8,
        title="conjugate / Stokes",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
