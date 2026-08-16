"""Shared helpers for the ``examples/`` tutorials.

Scripts add this directory (and the repo root) to ``sys.path`` so
``python examples/01_analyze_once.py`` works after an editable install
or from a source checkout.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

_EXAMPLES = Path(__file__).resolve().parent
_ROOT = _EXAMPLES.parent
for _p in (str(_EXAMPLES), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from topoopt.optimize import optimize
from topoopt.viz import plot_2d, write_vtk


def add_run_args(parser: argparse.ArgumentParser, default_outdir: str) -> argparse.ArgumentParser:
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Coarse mesh and few iterations (for a smoke run)",
    )
    parser.add_argument("--outdir", type=Path, default=Path(default_outdir))
    parser.add_argument("--seed", type=int, default=0)
    return parser


def print_fields(aux, extra: dict | None = None):
    """Print the diagnostics that ``analyze`` / ``optimize`` put in aux."""
    keys = (
        "V",
        "T_mean",
        "T_max",
        "speed_max",
        "u_in",
        "u_out",
        "mass_err",
        "energy_rms",
        "div_rms",
        "stokes_rel",
        "gray",
    )
    parts = []
    for key in keys:
        if key in aux:
            parts.append(f"{key}={float(aux[key]):.4g}")
    if extra:
        parts.extend(f"{k}={v}" for k, v in extra.items())
    print("  " + "  ".join(parts))


def optimize_and_save(
    params,
    outdir: Path,
    n_iters: int,
    *,
    lr: float = 0.2,
    beta_max: float = 16.0,
    seed: int = 0,
    plot_every: int = 10,
    title: str = "",
):
    """Run projected GD and write the usual PNG / VTK / JSON artifacts."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    label = title or params.heat_label

    def callback(it, _gamma, aux, rec):
        if it == 1 or it == n_iters or it % plot_every == 0:
            plot_2d(
                aux,
                params,
                outdir / f"design_{it:03d}.png",
                title=f"{label}  it {it}  J={rec['J']:.4f}",
            )

    _gamma, aux, hist = optimize(
        params,
        n_iters=n_iters,
        lr=lr,
        beta_max=beta_max,
        seed=seed,
        outdir=outdir,
        callback=callback,
    )
    plot_2d(aux, params, outdir / "design_final.png", title=f"{label}  best-J design")
    write_vtk(aux, params, outdir / "design_final.vtk")
    j_best = max(h["J"] for h in hist)
    print(
        f"\nWrote {outdir.resolve()}\n"
        f"  J0={hist[0]['J']:.6f}  J_best={j_best:.6f}  "
        f"best_iter={next(h['iter'] for h in hist if h.get('is_best'))}  "
        f"vol={hist[-1]['vol']:.4f}"
    )
    print_fields(aux)
    return _gamma, aux, hist
