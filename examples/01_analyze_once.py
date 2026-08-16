#!/usr/bin/env python3
"""Quick start: evaluate the physics once, no optimization.

This is the smallest useful ``topoopt`` program. It builds a conduction
problem (uniform volumetric heat, small bottom sink), calls ``analyze``
on solid / fluid / gray fields, and writes one plot of the gray case.

What to notice
--------------
* ``γ = 1`` is conducting solid, ``γ = 0`` is fluid (here, an insulator).
* Default objective ``J = -mean(T)``: cooler is better. Solid should beat
  fluid because heat can reach the sink.
* ``analyze`` returns ``(J, aux)``. Residuals live in ``aux``
  (``energy_rms``, ``T_mean``, …).

Run from the repo root::

    python examples/01_analyze_once.py
    python examples/01_analyze_once.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

import jax.numpy as jnp

from _common import add_run_args, print_fields
from examples.problems import conduction_tree
from topoopt.problem import analyze
from topoopt.viz import plot_2d


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(p, "outputs/01_analyze_once")
    args = p.parse_args(argv)
    n = 16 if args.quick else 32
    params = conduction_tree(nx=n, ny=n, heat_iters=300, filter_iters=40)

    print(
        f"conduction box  n={params.n}  sink={params.cold_specs}  "
        f"q={params.q_vol:g}  J = -mean(T)\n"
    )
    results = {}
    for name, gamma in (
        ("solid", jnp.ones(params.n)),
        ("fluid", jnp.zeros(params.n)),
        ("gray", jnp.full(params.n, 0.45)),
    ):
        heat, aux = analyze(gamma, 4.0, params)
        results[name] = (float(heat), aux)
        print(f"{name:6s}  J={float(heat):10.4f}")
        print_fields(aux)

    js, jf = results["solid"][0], results["fluid"][0]
    assert js > jf, "solid should be cooler (larger J) than fluid"
    print(f"\nsolid cooler than fluid by ΔJ = {js - jf:.4f}  (expected)")

    _, aux_g = results["gray"]
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_2d(aux_g, params, args.outdir / "gray.png", title="01  gray γ=0.45  conduction")
    print(f"Wrote {args.outdir.resolve() / 'gray.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
