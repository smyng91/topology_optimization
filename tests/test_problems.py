"""Named example problems live outside the solver package."""

from __future__ import annotations

from examples.problems import (
    PROBLEMS,
    TREE_SINK,
    conduction_tree,
    conjugate_darcy,
    conjugate_stokes,
    convection_darcy,
)
from topoopt.config import load_params, params2d


def test_params2d_does_not_infer_regions():
    cond = params2d(nx=8, ny=8, heat_mode="conduction")
    conv = params2d(nx=8, ny=8, heat_mode="convection")
    assert cond.hot_specs == () and cond.cold_specs == ()
    assert conv.hot_specs == () and conv.cold_specs == ()


def test_example_problem_boundary_conditions():
    tree = conduction_tree(nx=8, ny=8)
    assert tree.cold_specs == TREE_SINK
    assert tree.hot_specs == ()
    assert not tree.solves_flow
    assert tree.symmetry == ("x",)

    conv = convection_darcy(nx=8, ny=8)
    assert conv.cold_specs == () and conv.hot_specs == ()
    assert conv.flow_model == "darcy" and conv.solves_flow
    assert conv.symmetry == ("y",)

    both = conjugate_darcy(nx=8, ny=8)
    assert both.cold_specs == () and both.heat_mode == "both"
    assert both.symmetry == ("y",)

    stokes = conjugate_stokes(nx=8, ny=8)
    assert stokes.flow_model == "stokes" and stokes.cold_specs == ()
    assert stokes.port_frac == conv.port_frac
    assert stokes.symmetry == ("y",)


def test_load_params_module_and_overrides():
    params = load_params("examples.problems:convection_darcy", nx=12, ny=10, pe=8.0)
    assert params.n == (12, 10)
    assert params.pe == 8.0
    assert params.heat_mode == "convection"
    assert set(PROBLEMS) == {
        "conduction_tree",
        "convection_darcy",
        "conjugate_darcy",
        "conjugate_stokes",
        "custom_faces",
        "custom_boxes",
    }


def test_load_params_json_factory_and_standalone():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    tree = load_params(str(root / "examples/configs/conduction_tree.json"), nx=10, ny=8)
    assert tree.n == (10, 8)
    assert tree.heat_mode == "conduction"
    assert tree.symmetry == ("x",)

    box = load_params(str(root / "examples/configs/standalone_box.json"))
    assert box.n == (16, 16)
    assert box.cold_specs == ("face:bottom:frac=0.08",)
    assert box.symmetry == ("x",)
    assert box.vol_frac == 0.3


def test_package_config_does_not_import_examples():
    import inspect

    import topoopt.config as cfg

    source = inspect.getsource(cfg)
    assert "import examples" not in source
    assert "from examples" not in source
