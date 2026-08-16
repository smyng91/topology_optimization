"""Why a symmetric problem used to grow a skewed design: init noise."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from examples.problems import conduction_tree, convection_darcy, custom_boxes, custom_faces
from topoopt.optimize import upsample_field
from topoopt.symmetry import apply, max_error, normalize_axes


def test_random_noise_is_not_symmetric():
    """This is the historical breaker: uniform noise is not a left–right mirror."""
    noise = jax.random.uniform(jax.random.PRNGKey(0), (16, 16))
    assert float(jnp.max(jnp.abs(noise - jnp.flip(noise, 0)))) > 0.1
    assert float(jnp.max(jnp.abs(noise - jnp.flip(noise, 1)))) > 0.1


def test_apply_makes_mirror():
    field = jnp.arange(16.0).reshape(4, 4)
    params = conduction_tree(nx=4, ny=4)
    assert params.symmetry == ("x",)
    sym = apply(field, params)
    np.testing.assert_allclose(np.asarray(sym), np.asarray(jnp.flip(sym, 0)))
    assert float(max_error(sym, params)) < 1e-15


def test_flow_problems_are_y_symmetric_not_x():
    conv = convection_darcy(nx=8, ny=8)
    assert conv.symmetry == ("y",)
    tree = conduction_tree(nx=8, ny=8)
    assert tree.symmetry == ("x",)
    faces = custom_faces(nx=8, ny=8)
    assert faces.symmetry == ("x",)
    boxes = custom_boxes(nx=8, ny=8)
    assert boxes.symmetry == ()
    assert float(max_error(jnp.arange(64.0).reshape(8, 8), boxes)) == 0.0


def test_normalize_axes_accepts_strings_and_lists():
    assert normalize_axes("x,y") == ("x", "y")
    assert normalize_axes(["Y"]) == ("y",)
    assert normalize_axes(()) == ()


def test_upsample_preserves_mean_roughly():
    field = jnp.full((8, 8), 0.3)
    up = upsample_field(field, (16, 16))
    assert up.shape == (16, 16)
    assert abs(float(up.mean()) - 0.3) < 1e-12
