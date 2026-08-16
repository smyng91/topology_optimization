#!/usr/bin/env python3
"""Tutorial: volume-to-point conduction tree.

Uniform volumetric heating, a *small* centered bottom sink (8% of the
wall), and a modest solid fraction. The optimizer grows a branching
conducting tree so heat can reach the sink. A 50% sink makes parallel
fins instead — try ``--wide-sink`` to see the difference.

``J = -mean(T)``. Volume is an equality (``mean(γ̄) = v*``), enforced
by a mean-zero step plus a shift after the Helmholtz filter / tanh
projection. ``β`` doubles during the run; the move limit decays as
``lr / sqrt(β)``. The returned design is the best-``J`` iterate.

Run from the repo root::

    python examples/02_conduction_tree.py
    python examples/02_conduction_tree.py --quick
    python examples/02_conduction_tree.py --wide-sink
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

from _common import add_run_args, optimize_and_save
from examples.problems import TREE_SINK, conduction_tree


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(p, "outputs/02_conduction_tree")
    p.add_argument(
        "--wide-sink",
        action="store_true",
        help="Use a half-width bottom sink (fins, not a tree)",
    )
    args = p.parse_args(argv)

    n = 16 if args.quick else 40
    iters = 8 if args.quick else 60
    sink = ("face:bottom:frac=0.5",) if args.wide_sink else TREE_SINK
    params = conduction_tree(nx=n, ny=n, cold_specs=sink, heat_iters=300, filter_iters=80)
    print(
        "Look for a trunk into the bottom sink, then branches. "
        "Wide sink → nearly parallel fins.\n"
    )
    optimize_and_save(
        params,
        args.outdir,
        iters,
        lr=0.2,
        beta_max=8.0 if args.quick else 16.0,
        seed=args.seed,
        plot_every=5 if args.quick else 10,
        title="conduction tree",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
