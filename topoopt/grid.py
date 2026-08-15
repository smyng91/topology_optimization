"""Grid helpers for the 2-D finite-volume discretizations."""

from __future__ import annotations

import jax.numpy as jnp

from topoopt.config import ColdPlateParams


def axis_index(ndim: int, axis: int, sl):
    idx = [slice(None)] * ndim
    idx[axis] = sl
    return tuple(idx)


def take_axis(x, axis: int, sl):
    return x[axis_index(x.ndim, axis, sl)]


def add_axis(x, axis: int, sl, val):
    return x.at[axis_index(x.ndim, axis, sl)].add(val)


def set_axis(x, axis: int, sl, val):
    return x.at[axis_index(x.ndim, axis, sl)].set(val)


def line_mask(n: int, frac: float, center: float = 0.5):
    """Centered (or offset) 1-D mask occupying ``frac`` of ``n`` cells.

    Always includes the cell nearest ``center`` so a small patch cannot
    vanish on a coarse mesh (which would leave the energy problem singular).
    """
    xi = (jnp.arange(n) + 0.5) / n
    half = 0.5 * min(max(frac, 0.0), 1.0)
    mask = (xi >= center - half) & (xi <= center + half)
    nearest = jnp.argmin(jnp.abs(xi - center))
    return mask.at[nearest].set(True)


def port_mask(params: ColdPlateParams):
    """Centered opening on a vertical face, occupying ``port_frac`` of the height."""
    return line_mask(params.n[1], params.port_frac)


def inlet_profile(params: ColdPlateParams):
    """Parabolic inlet speed on the centered left-wall port; zero on the rest of the wall."""
    _, ny = params.n
    _, ly = params.L
    _, dy = params.dx
    y = (jnp.arange(ny) + 0.5) * dy
    y0 = 0.5 * ly * (1.0 - params.port_frac)
    y1 = 0.5 * ly * (1.0 + params.port_frac)
    span = jnp.clip(y1 - y0, 1e-12, None)
    s = (y - y0) / span
    para = params.u_in_max * 4.0 * s * (1.0 - s)
    return jnp.where(port_mask(params), para, 0.0)


def zero_face_velocity(params: ColdPlateParams):
    """MAC face-normal velocities of zero, used when the flow solve is skipped."""
    faces = []
    for axis in range(params.dim):
        shape = list(params.n)
        shape[axis] += 1
        faces.append(jnp.zeros(shape))
    return faces


def harmonic_faces(k, axis: int):
    k0 = take_axis(k, axis, slice(None, -1))
    k1 = take_axis(k, axis, slice(1, None))
    return 2.0 * k0 * k1 / (k0 + k1 + 1e-30)


def cell_divergence(face_vel, dxs):
    """Discrete ∇·u from MAC face-normal velocities."""
    div = jnp.zeros_like(take_axis(face_vel[0], 0, slice(None, -1)))
    for axis, dx in enumerate(dxs):
        uf = face_vel[axis]
        div = div + (take_axis(uf, axis, slice(1, None)) - take_axis(uf, axis, slice(None, -1))) / dx
    return div
