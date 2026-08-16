"""Cone density filter and Helmholtz option."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from topoopt.config import params2d
from topoopt.filter import cone_filter, density_filter, filter_radius, helmholtz_filter
from topoopt.problem import physical_density


def _numpy_cone(gamma, params):
    nx, ny = params.n
    dx, dy = params.dx
    r = params.rmin * min(dx, dy)
    out = np.zeros((nx, ny), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            num = 0.0
            den = 0.0
            for k in range(nx):
                for ell in range(ny):
                    dist = math.hypot((i - k) * dx, (j - ell) * dy)
                    w = max(0.0, r - dist)
                    num += w * float(gamma[k, ell])
                    den += w
            out[i, j] = num / den if den else 0.0
    return out


def test_params2d_defaults_to_cone():
    params = params2d(nx=8, ny=8)
    assert params.filter_kind == "cone"


def test_filter_kind_rejects_unknown():
    with pytest.raises(ValueError, match="filter_kind"):
        params2d(nx=8, ny=8, filter_kind="sensitivity")


def test_cone_preserves_constants():
    params = params2d(nx=12, ny=10, rmin=2.2)
    field = jnp.full(params.n, 0.37)
    out = cone_filter(field, params)
    assert float(jnp.max(jnp.abs(out - 0.37))) < 1e-12


def test_cone_compact_support_and_matches_numpy():
    params = params2d(nx=10, ny=10, rmin=2.2)
    gamma = jnp.zeros(params.n)
    gamma = gamma.at[4, 5].set(1.0)
    filt = np.asarray(cone_filter(gamma, params))
    ref = _numpy_cone(np.asarray(gamma), params)
    assert np.max(np.abs(filt - ref)) < 1e-12

    r = filter_radius(params)
    dx, dy = params.dx
    xs = (np.arange(params.n[0]) + 0.5) * dx
    ys = (np.arange(params.n[1]) + 0.5) * dy
    # Cell-center distance from the spike cell (4, 5).
    x0, y0 = (4 + 0.5) * dx, (5 + 0.5) * dy
    dist = np.sqrt((xs[:, None] - x0) ** 2 + (ys[None, :] - y0) ** 2)
    assert np.all(filt[dist >= r - 1e-12] == 0.0)
    assert filt[4, 5] > 0.0


def test_density_filter_dispatches():
    params_c = params2d(nx=8, ny=8, rmin=2.0)
    params_h = params2d(nx=8, ny=8, rmin=2.0, filter_kind="helmholtz", filter_iters=80)
    spike = jnp.zeros(params_c.n).at[3, 3].set(1.0)
    cone = density_filter(spike, params_c)
    helm = density_filter(spike, params_h)
    assert float(jnp.max(jnp.abs(cone - cone_filter(spike, params_c)))) == 0.0
    assert float(jnp.max(jnp.abs(helm - helmholtz_filter(spike, params_h)))) < 1e-12
    # Helmholtz tails reach the far corner; the cone does not.
    assert float(cone[0, 0]) == 0.0
    assert float(helm[0, 0]) > 1e-8


def test_physical_density_uses_cone_by_default():
    params = params2d(nx=8, ny=8, rmin=2.0, eta=0.5)
    raw = jnp.zeros(params.n).at[3, 3].set(1.0)
    phys = physical_density(raw, 1.0, params)
    # β=1, η=0.5: projection is nearly the identity on [0, 1].
    filt = cone_filter(raw, params)
    assert float(jnp.max(jnp.abs(phys - filt))) < 0.05
    assert float(phys[0, 0]) < 0.05
