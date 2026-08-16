"""Named example problems live outside the solver package."""

from __future__ import annotations

import pytest

from examples.problems import (
    PROBLEMS,
    SOURCE_BOX,
    TREE_SINK,
    conduction_tree,
    conjugate_darcy,
    conjugate_stokes,
    convection_darcy,
    localized_source,
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
    assert tree.q_specs == ()
    assert not tree.solves_flow
    assert tree.symmetry == ("x",)

    src = localized_source(nx=8, ny=8)
    assert src.q_specs == SOURCE_BOX
    assert src.hot_specs == ()
    assert src.uses_volume_source
    assert src.cold_specs == TREE_SINK

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
    params = load_params("convection_darcy", nx=12, ny=10, pe=8.0)
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
        "localized_source",
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

    src = load_params(str(root / "examples/configs/localized_source.json"))
    assert src.n == (32, 32)
    assert src.q_specs == SOURCE_BOX
    assert src.uses_volume_source


def test_package_config_does_not_import_examples():
    import inspect

    import topoopt.config as cfg

    source = inspect.getsource(cfg)
    assert "import examples" not in source
    assert "from examples" not in source


def test_config_rejects_unknown_keys_and_untrusted_python(tmp_path):
    with pytest.raises(ValueError, match="unknown configuration keys"):
        params2d(nx=8, ny=8, solver_tlo=1e-7)

    factory = tmp_path / "factory.py"
    factory.write_text(
        "from topoopt.config import params2d\n"
        "def build():\n"
        "    return params2d(nx=8, ny=8)\n",
        encoding="utf-8",
    )
    spec = f"{factory}:build"
    with pytest.raises(ValueError, match="disabled"):
        load_params(spec)
    assert load_params(spec, allow_unsafe_python=True).n == (8, 8)


@pytest.mark.parametrize(
    "overrides",
    [
        {"vol_frac": 1.0},
        {"solver_tol": 0.0},
        {"q_k": -1.0},
        {"port_frac": 0.0},
        {"heat_iters": 0},
    ],
)
def test_invalid_parameter_ranges_are_rejected(overrides):
    with pytest.raises(ValueError):
        params2d(nx=8, ny=8, **overrides)
