"""Optimizer, diagnostics, and short end-to-end runs."""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from examples.problems import conduction_tree, convection_darcy
from topoopt.optimize import (
    RunawaySolveError,
    beta_schedule,
    highest_beta_best,
    move_limit,
    optimize,
    optimize_hierarchy,
    runaway_reason,
)
from topoopt.problem import analyze
from topoopt.symmetry import max_error


def test_analyze_diagnostics_conduction():
    params = conduction_tree(nx=12, ny=12, heat_iters=400, filter_iters=30)
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
    params = conduction_tree(nx=16, ny=16, filter_iters=40, heat_iters=250)
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
    best = next(h for h in hist if h["is_best"])
    assert run["n_iters"] == 10
    assert run["J_best"] == pytest.approx(best["J"])
    assert run["best_iter"] == best["iter"]
    assert best["beta"] == pytest.approx(max(h["beta"] for h in hist))
    assert run["J_peak"] == pytest.approx(max(h["J"] for h in hist))
    assert run["best_iter"] >= 1
    assert "params" in run
    assert isinstance(run["params"]["n"], list)
    assert np.isfinite(float(aux["energy_rms"]))
    assert sum(h["is_best"] for h in hist) == 1
    assert float(max_error(_g, params)) < 1e-12
    assert run["stopped"] == "completed"
    assert run["params"]["symmetry"] == ["x"]


def test_highest_beta_best_rejects_softer_projection():
    hist = [
        {"iter": 1, "beta": 1.0, "J": -0.020},
        {"iter": 42, "beta": 4.0, "J": -0.012},
        {"iter": 80, "beta": 8.0, "J": -0.014},
        {"iter": 81, "beta": 16.0, "J": -0.015},
        {"iter": 85, "beta": 16.0, "J": -0.0135},
    ]
    rec = highest_beta_best(hist)
    assert rec["iter"] == 85
    assert rec["beta"] == pytest.approx(16.0)
    assert rec["J"] == pytest.approx(-0.0135)


def test_keep_best_and_stall_are_per_beta_level(tmp_path):
    params = conduction_tree(nx=12, ny=12, filter_iters=20, heat_iters=80)
    _g, aux, hist = optimize(
        params,
        n_iters=15,
        lr=0.2,
        beta_max=4.0,
        seed=0,
        outdir=tmp_path,
        stall_iters=8,
    )
    best = next(h for h in hist if h["is_best"])
    run = json.loads((tmp_path / "run.json").read_text())
    assert best["beta"] == pytest.approx(max(h["beta"] for h in hist))
    assert run["stopped"] == "completed"
    assert run["J_peak"] == pytest.approx(max(h["J"] for h in hist))
    assert float(aux["energy_rms"]) == pytest.approx(best["energy_rms"])


def test_short_darcy_optimize(tmp_path):
    params = convection_darcy(nx=12, ny=12, flow_iters=120, heat_iters=200, filter_iters=30)
    _g, aux, hist = optimize(
        params, n_iters=8, lr=0.2, beta_max=4.0, seed=0, outdir=tmp_path
    )
    assert abs(hist[-1]["vol"] - params.vol_frac) < 1e-3
    j_best = max(h["J"] for h in hist)
    t0 = hist[0]["T_mean"]
    assert j_best > hist[0]["J"] or hist[-1]["T_mean"] < t0
    assert float(aux["u_in"]) > 0.0
    assert (tmp_path / "run.json").is_file()
    assert float(max_error(_g, params)) < 1e-12
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["params"]["symmetry"] == ["y"]


def test_runaway_reason_and_hierarchy(tmp_path):
    params = conduction_tree(nx=16, ny=16, filter_iters=20, heat_iters=80)
    rec_ok = {"J": -0.1, "T_max": 1.2, "T_mean": 0.4, "energy_rms": 1e-4}
    assert runaway_reason(rec_ok, params) is None
    rec_bad = {"J": float("nan"), "T_max": 1.0, "T_mean": 0.4, "energy_rms": 1e-4}
    assert runaway_reason(rec_bad, params) is not None
    rec_hot = {"J": -10.0, "T_max": 2e3, "T_mean": 800.0, "energy_rms": 1e-3}
    assert "T_max" in runaway_reason(rec_hot, params)
    flow = convection_darcy(nx=8, ny=8)
    rec_block = {"J": -80.0, "T_max": 80.0, "T_mean": 40.0, "energy_rms": 5e-2}
    assert "blocked" in runaway_reason(rec_block, flow)

    gamma, _aux, hist = optimize_hierarchy(
        params,
        levels=((8, 8, 3), (16, 16, 3)),
        lr=0.2,
        beta_max=2.0,
        seed=0,
        outdir=tmp_path,
        stall_iters=0,
    )
    assert gamma.shape == (16, 16)
    assert abs(hist[-1]["vol"] - params.vol_frac) < 1e-2
    assert float(max_error(gamma, params._replace(n=(16, 16)))) < 1e-12
    assert RunawaySolveError is RuntimeError or issubclass(RunawaySolveError, RuntimeError)
