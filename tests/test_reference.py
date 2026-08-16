"""Archived factory table: symmetry, volume, and a short custom-faces run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from topoopt.config import load_params
from topoopt.optimize import optimize
from topoopt.symmetry import max_error

ROOT = Path(__file__).resolve().parents[1]
TABLE = json.loads((ROOT / "examples" / "reference.json").read_text())


def test_reference_table_matches_factories():
    for name, spec in TABLE["cases"].items():
        params = load_params(spec["factory"], nx=8, ny=8)
        assert list(params.symmetry) == spec["symmetry"], name
        assert params.vol_frac == spec["vol_frac"], name
        assert params.heat_mode == spec["heat_mode"], name
        assert params.solves_flow is spec["solves_flow"], name


def test_reference_short_custom_faces(tmp_path):
    spec = TABLE["cases"]["custom_faces"]
    short = spec["short"]
    params = load_params(
        spec["factory"],
        nx=short["nx"],
        ny=short["ny"],
        heat_iters=short["heat_iters"],
        filter_iters=short["filter_iters"],
    )
    gamma, _aux, hist = optimize(
        params,
        n_iters=short["iters"],
        lr=short["lr"],
        beta_max=short["beta_max"],
        seed=0,
        outdir=tmp_path,
    )
    j_best = max(h["J"] for h in hist)
    assert short["j_best_min"] <= j_best <= short["j_best_max"]
    assert j_best >= hist[0]["J"]
    assert abs(hist[-1]["vol"] - params.vol_frac) < 1e-3
    assert float(max_error(gamma, params)) <= short["sym_err_max"]
    assert np.isfinite(j_best)
