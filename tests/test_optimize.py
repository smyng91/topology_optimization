"""Optimizer, diagnostics, and short end-to-end runs."""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from topoopt.config import default_2d
from topoopt.optimize import beta_schedule, move_limit, optimize
from topoopt.problem import analyze


def test_analyze_diagnostics_conduction():
    params = default_2d(nx=12, ny=12, heat_mode="conduction", heat_iters=400, filter_iters=30)
    gamma = jnp.full(params.n, 0.5)
    j, aux = analyze(gamma, 4.0, params)
    assert np.isfinite(float(j))
    assert float(aux["energy_rms"]) < 1e-3
    assert float(aux["div_rms"]) < 1e-12
    assert float(aux["mass_err"]) == 0.0
    assert float(aux["u_in"]) == 0.0
    assert np.isfinite(float(aux["T_mean"]))
    assert np.isfinite(float(aux["T_max"]))
    assert float(aux["speed_max"]) < 1e-12
    assert 0.0 <= float(aux["gray"]) <= 1.0


def test_move_limit_decays_with_beta():
    assert move_limit(0.2, 1.0) == pytest.approx(0.2)
    assert move_limit(0.2, 16.0) == pytest.approx(0.05)
    assert move_limit(0.2, 32.0) == pytest.approx(0.2 / 32**0.5)
    assert move_limit(0.2, 0.25) == pytest.approx(0.2)


def test_beta_schedule_doubles_to_max():
    sched = np.asarray(beta_schedule(10, 4.0))
    assert len(sched) == 10
    assert sched[0] == pytest.approx(1.0)
    assert sched[-1] == pytest.approx(4.0)
    assert 2.0 in sched


def test_short_conduction_optimize(tmp_path):
    params = default_2d(nx=16, ny=16, heat_mode="conduction", filter_iters=40, heat_iters=250)
    _g, aux, hist = optimize(
        params, n_iters=10, lr=0.2, beta_max=4.0, seed=0, outdir=tmp_path
    )
    assert abs(hist[-1]["vol"] - params.vol_frac) < 1e-3
    assert max(h["J"] for h in hist) > hist[0]["J"]
    assert (tmp_path / "history.json").is_file()
    assert (tmp_path / "run.json").is_file()
    assert (tmp_path / "state_best.npz").is_file()
    assert (tmp_path / "state_final.npz").is_file()
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["n_iters"] == 10
    assert run["J_best"] == pytest.approx(max(h["J"] for h in hist))
    assert run["best_iter"] >= 1
    assert "params" in run
    assert isinstance(run["params"]["n"], list)
    assert np.isfinite(float(aux["energy_rms"]))
    assert sum(h["is_best"] for h in hist) == 1


def test_short_darcy_optimize(tmp_path):
    params = default_2d(
        nx=12,
        ny=12,
        heat_mode="convection",
        flow_model="darcy",
        flow_iters=120,
        heat_iters=200,
        filter_iters=30,
    )
    _g, aux, hist = optimize(
        params, n_iters=8, lr=0.2, beta_max=4.0, seed=0, outdir=tmp_path
    )
    assert abs(hist[-1]["vol"] - params.vol_frac) < 1e-3
    j_best = max(h["J"] for h in hist)
    t0 = hist[0]["T_mean"]
    assert j_best > hist[0]["J"] or hist[-1]["T_mean"] < t0
    assert float(aux["u_in"]) > 0.0
    assert (tmp_path / "run.json").is_file()
