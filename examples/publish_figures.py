#!/usr/bin/env python3
"""Write the committed snapshots under ``docs/figures/``.

These are medium-short runs (not the 80×80 gallery) so the PNGs stay
small enough to live in git. Re-run after a visual change::

    python examples/publish_figures.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

import jax.numpy as jnp

from _common import optimize_and_save
from examples.problems import conduction_tree, convection_darcy, custom_faces
from topoopt.problem import analyze
from topoopt.viz import plot_2d

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"


def _copy(src: Path, dest_name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    dest = FIGURES / dest_name
    shutil.copy2(src, dest)
    print(f"  {dest}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "publish_figures")
    args = p.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    params = conduction_tree(nx=24, ny=24, heat_iters=250, filter_iters=60)
    _j, aux = analyze(jnp.full(params.n, 0.45), 4.0, params)
    gray = args.outdir / "analyze_gray.png"
    plot_2d(aux, params, gray, title="analyze() gray field")
    _copy(gray, "analyze_gray.png")

    cases = (
        (
            "conduction_tree.png",
            conduction_tree(nx=32, ny=32, heat_iters=280, filter_iters=80),
            20,
            "conduction tree",
        ),
        (
            "convection_darcy.png",
            convection_darcy(nx=24, ny=24, flow_iters=160, heat_iters=220, filter_iters=50),
            12,
            "convection / Darcy",
        ),
        (
            "custom_faces.png",
            custom_faces(nx=24, ny=24, heat_iters=220, filter_iters=50),
            12,
            "custom faces",
        ),
    )
    for dest_name, case, iters, title in cases:
        out = args.outdir / Path(dest_name).stem
        _g, _aux, _hist = optimize_and_save(
            case,
            out,
            iters,
            lr=0.2,
            beta_max=8.0,
            seed=0,
            plot_every=iters,
            title=title,
        )
        _copy(out / "design_final.png", dest_name)
    print(f"Wrote {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
