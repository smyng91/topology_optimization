"""Physics checks for volume-source box problems and user-defined regions."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from examples.problems import TREE_SINK, conduction_tree, conjugate_darcy, conjugate_stokes, convection_darcy, custom_faces
from topoopt.config import params2d
from topoopt.heat import energy_operator, solve_energy
from topoopt.interpolation import conductivity
from topoopt.problem import analyze
from topoopt.regions import parse_spec, resolve_axis, specs_from_cli


def test_parse_region_specs():
    assert parse_spec("face:bottom:frac=0.5")[0] == "face"
    assert parse_spec("box:0.2,0.8,0,0.1")[1][0] == 0.2
    assert resolve_axis("bottom", 2) == (1, 0)
    assert resolve_axis("left", 2) == (0, 0)
    assert resolve_axis("top", 2) == (1, -1)
    hot, cold, q = specs_from_cli(
        ["face:bottom:frac=0.4", "box:0.1,0.3,0.0,0.2"],
        ["face:left", "face:top:frac=0.3"],
        ["box:0.3,0.7,0.7,1.0"],
    )
    assert hot == ("face:bottom:frac=0.4", "box:0.1,0.3,0.0,0.2")
    assert cold == ("face:left", "face:top:frac=0.3")
    assert q == ("box:0.3,0.7,0.7,1.0",)
    assert specs_from_cli(None, None) == ((), (), ())


def test_conduction_volume_source_solid_cools_better():
    params = conduction_tree(nx=16, ny=16, filter_iters=40, heat_iters=300)
    assert params.uses_volume_source
    assert params.cold_specs == TREE_SINK
    solid = jnp.ones(params.n)
    fluid = jnp.zeros(params.n)
    js, aux_s = analyze(solid, 8.0, params)
    jf, aux_f = analyze(fluid, 8.0, params)
    assert float(aux_s["speed"].max()) < 1e-12
    assert float(aux_f["speed"].max()) < 1e-12
    assert float(aux_s["T"].min()) >= -1e-8
    assert float(aux_s["T"].mean()) < float(aux_f["T"].mean())
    assert float(js) > float(jf)
    temp = np.asarray(aux_s["T"])
    assert temp[:, 0].mean() < temp[:, -1].mean()


def test_convection_ports_and_flow_cools():
    params = convection_darcy(nx=16, ny=16, flow_iters=200, heat_iters=250, filter_iters=40)
    solid = jnp.ones(params.n)
    fluid = jnp.zeros(params.n)
    np.testing.assert_allclose(conductivity(solid, params), conductivity(fluid, params))
    jf, aux_f = analyze(fluid, 2.0, params)
    js, aux_s = analyze(solid, 2.0, params)
    assert float(aux_f["speed"].mean()) > float(aux_s["speed"].mean())
    assert float(aux_f["T"].mean()) < float(aux_s["T"].mean())
    assert float(jf) > float(js)
    from topoopt.grid import port_mask

    mask = np.asarray(port_mask(params))
    u_left = np.asarray(aux_f["face_vel"][0][0])
    assert float(np.max(np.abs(u_left * (1.0 - mask)))) < 1e-8
    assert float(np.sum(u_left * mask)) > 0.05
    assert params.cold_specs == ()
    assert params.hot_specs == ()
    assert bool(mask[len(mask) // 2])
    assert not bool(mask[0]) and not bool(mask[-1])


def test_darcy_open_channel_is_divergence_free():
    from topoopt.darcy import solve_darcy
    from topoopt.grid import cell_divergence, port_mask

    params = convection_darcy(nx=16, ny=16, flow_iters=250, filter_iters=20)
    faces, _p = solve_darcy(jnp.zeros(params.n), params)
    mask = port_mask(params)
    div = cell_divergence(faces, params.dx)
    u_in = float(jnp.sum(faces[0][0] * mask))
    u_out = float(jnp.sum(faces[0][-1] * mask))
    assert u_in > 0.05
    assert abs(u_in - u_out) / u_in < 0.08
    assert float(jnp.sqrt(jnp.mean(div**2))) < 1e-3


def test_stokes_pressure_ports_channel_beats_block():
    from topoopt.flow2d import solve_stokes, stokes_relative_residual
    from topoopt.grid import cell_divergence, port_mask
    from topoopt.heat import solve_energy

    params = convection_darcy(
        nx=16,
        ny=16,
        flow_model="stokes",
        flow_iters=80,
        uzawa_iters=40,
        stokes_kryl_iters=200,
        heat_iters=250,
    )
    mask = port_mask(params)
    y = (jnp.arange(params.n[1]) + 0.5) / params.n[1]
    channel = jnp.where((y > 0.3) & (y < 0.7), 0.0, 1.0)
    channel = jnp.broadcast_to(channel, params.n)

    def _flow(g):
        sol = solve_stokes(g, params)
        u, v, _p = sol
        rel = float(stokes_relative_residual(sol, g, params))
        u_in = float(jnp.sum(u[0] * mask))
        u_out = float(jnp.sum(u[-1] * mask))
        div_rms = float(jnp.sqrt(jnp.mean(cell_divergence([u, v], params.dx) ** 2)))
        temp = solve_energy(g, [u, v], params)
        return u_in, u_out, div_rms, rel, float(temp.mean())

    u_open, out_open, div_o, rel_o, t_open = _flow(jnp.zeros(params.n))
    u_ch, out_ch, div_c, rel_c, t_ch = _flow(channel)
    u_blk, out_blk, div_b, rel_b, t_blk = _flow(jnp.ones(params.n))
    assert rel_o < 1e-5 and rel_c < 1e-5
    assert u_open > 0.5 and abs(u_open - out_open) / u_open < 0.03
    assert u_ch > 0.2 and abs(u_ch - out_ch) / u_ch < 0.05
    assert u_blk < 0.05 * u_open
    assert div_o < 2e-3 and div_c < 5e-3
    assert t_open < t_blk and t_ch < t_blk
    assert params.cold_specs == () and params.hot_specs == ()


def test_both_mode_has_flow_and_conduction_contrast():
    params = conjugate_darcy(nx=16, ny=16, flow_iters=200, heat_iters=250, filter_iters=40)
    gray = jnp.full(params.n, 0.45)
    j, aux = analyze(gray, 2.0, params)
    assert np.isfinite(float(j))
    assert float(aux["speed"].max()) > 0.0
    assert conductivity(jnp.ones(params.n), params).mean() > conductivity(jnp.zeros(params.n), params).mean()
    assert float(aux["T"].min()) >= -1e-8
    assert params.cold_specs == () and params.hot_specs == ()


def test_custom_faces_hot_top_cold_bottom():
    params = params2d(
        nx=16,
        ny=16,
        heat_mode="conduction",
        q_vol=0.0,
        heat_iters=300,
        filter_iters=40,
        hot_specs=("face:top",),
        cold_specs=("face:bottom",),
    )
    _, aux = analyze(jnp.ones(params.n), 4.0, params)
    temp = np.asarray(aux["T"])
    assert temp[:, -1].mean() > temp[:, 0].mean()
    assert temp[:, -1].mean() > 0.7
    assert temp[:, 0].mean() < 0.3


def test_custom_box_domains():
    params = params2d(
        nx=16,
        ny=16,
        heat_mode="conduction",
        q_vol=0.0,
        heat_iters=300,
        filter_iters=40,
        hot_specs=("box:0.3,0.7,0.0,0.2",),
        cold_specs=("box:0.0,0.2,0.6,1.0",),
    )
    j, aux = analyze(jnp.ones(params.n), 4.0, params)
    assert float(j) > 0.0
    temp = np.asarray(aux["T"])
    assert temp[5:11, 0:3].mean() > temp[0:3, 10:16].mean()


def test_flow_modes_only_centerline_ports():
    """Convection / both: one left inlet, one right outlet, no cold patches."""
    from topoopt.grid import port_mask

    factories = (
        convection_darcy,
        conjugate_darcy,
        lambda **kw: conjugate_stokes(uzawa_iters=20, stokes_kryl_iters=80, **kw),
    )
    for factory in factories:
        params = factory(nx=16, ny=16, flow_iters=80, heat_iters=80, filter_iters=20)
        assert params.hot_specs == ()
        assert params.cold_specs == ()
        mask = np.asarray(port_mask(params))
        assert bool(mask[len(mask) // 2])
        assert not bool(mask[0]) and not bool(mask[-1])
        j, aux = analyze(jnp.zeros(params.n), 2.0, params)
        assert np.isfinite(float(j))
        u_left = np.asarray(aux["face_vel"][0][0])
        u_right = np.asarray(aux["face_vel"][0][-1])
        v = np.asarray(aux["face_vel"][1])
        assert float(np.max(np.abs(u_left * (1.0 - mask)))) < 1e-8
        assert float(np.max(np.abs(u_right * (1.0 - mask)))) < 1e-8
        assert float(np.max(np.abs(v[:, 0]))) < 1e-8
        assert float(np.max(np.abs(v[:, -1]))) < 1e-8


def test_conduction_energy_residual_small():
    params = conduction_tree(nx=12, ny=12, heat_iters=400, filter_iters=30)
    gamma = jnp.full(params.n, 0.5)
    from topoopt.grid import zero_face_velocity

    faces = zero_face_velocity(params)
    temp = solve_energy(gamma, faces, params)
    k = conductivity(gamma, params)
    from topoopt.regions import volume_source_field

    q = volume_source_field(params)
    res = energy_operator(temp, k, faces, params, params.t_in, params.t_hot, q)
    assert float(jnp.sqrt(jnp.mean(res**2))) < 1e-3


def test_high_beta_conduction_energy_converges():
    """CG on Pe=0 must solve high-contrast tanh fields, not stagnate."""
    from topoopt.optimize import project_design

    params = custom_faces(nx=24, ny=24, heat_iters=400, filter_iters=40)
    gamma = project_design(jnp.full(params.n, params.vol_frac), 16.0, params)
    j, aux = analyze(gamma, 16.0, params)
    assert np.isfinite(float(j))
    assert float(aux["energy_rel"]) < 1e-4
    assert float(aux["energy_rms"]) < 1e-3


def test_stokes_channel_energy_dense():
    """Pe>0 energy on a small Stokes mesh factors densely and should solve."""
    from topoopt.optimize import project_design, _initial_guess

    params = conjugate_stokes(
        nx=24,
        ny=24,
        heat_iters=400,
        flow_iters=40,
        uzawa_iters=40,
        stokes_kryl_iters=80,
        filter_iters=20,
    )
    gamma = project_design(_initial_guess(params, 0, None), 8.0, params)
    j, aux = analyze(gamma, 8.0, params)
    assert np.isfinite(float(j))
    assert float(aux["energy_rel"]) < 1e-8
    assert float(aux["u_in"]) > 0.0


def test_localized_volume_source_heats_the_box():
    params = params2d(
        nx=20,
        ny=20,
        heat_mode="conduction",
        q_vol=1.0,
        q_specs=("box:0.25,0.75,0.70,1.0",),
        cold_specs=("face:bottom:frac=0.5",),
        heat_iters=400,
        filter_iters=40,
    )
    assert params.uses_volume_source
    j, aux = analyze(jnp.ones(params.n), 4.0, params)
    temp = np.asarray(aux["T"])
    assert float(j) == pytest.approx(-float(temp.mean()))
    assert temp[5:15, 14:20].mean() > temp[5:15, 0:6].mean()
    from topoopt.regions import source_cell_mask, volume_source_field

    mask = np.asarray(source_cell_mask(params))
    q = np.asarray(volume_source_field(params))
    assert mask.any() and not mask.all()
    assert float(q[mask].min()) == pytest.approx(1.0)
    assert float(q[~mask].max()) == pytest.approx(0.0)


def test_q_region_and_dirichlet_T_together():
    params = params2d(
        nx=16,
        ny=16,
        heat_mode="conduction",
        q_vol=1.0,
        q_specs=("box:0.25,0.75,0.55,0.85",),
        hot_specs=("face:top:frac=0.5",),
        cold_specs=("face:bottom:frac=0.5",),
        heat_iters=400,
        filter_iters=40,
    )
    assert params.uses_volume_source
    j, aux = analyze(jnp.ones(params.n), 4.0, params)
    temp = np.asarray(aux["T"])
    assert float(j) == pytest.approx(-float(temp.mean()))
    assert temp[:, -1].mean() > 0.7
    assert temp[:, 0].mean() < 0.3


def test_hot_specs_still_disable_uniform_q():
    params = params2d(
        nx=12,
        ny=12,
        heat_mode="conduction",
        q_vol=1.0,
        hot_specs=("face:top",),
        cold_specs=("face:bottom",),
        heat_iters=200,
        filter_iters=20,
    )
    assert not params.uses_volume_source
    from topoopt.regions import volume_source_field

    assert float(volume_source_field(params)) == 0.0
