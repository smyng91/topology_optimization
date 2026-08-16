"""User-defined heat-source and cold-plate regions (faces or volumetric boxes).

Spec strings (repeatable via ``--hot`` / ``--cold``):

- ``face:bottom`` / ``face:left`` / ``face:right`` / ``face:top``
- ``face:bottom:frac=0.5`` — centered patch
- ``face:bottom:frac=0.4:center=0.3`` — off-center patch
- ``box:xmin,xmax,ymin,ymax`` — volumetric domain

The package does not pick default patches. Research cases set
``hot_specs`` / ``cold_specs`` in ``examples/problems.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from topoopt.config import ColdPlateParams
from topoopt.grid import add_axis, harmonic_faces, line_mask, take_axis

FACE_AXES = {
    "left": (0, 0),
    "right": (0, -1),
    "bottom": (-1, 0),
    "top": (-1, -1),
}


class FacePatch(NamedTuple):
    axis: int
    side: int
    mask: object
    temperature: float
    role: str


def parse_spec(spec: str):
    """Return a Python descriptor ``('face', side, fracs, centers)`` or ``('box', bounds)``."""
    raw = spec.strip().lower()
    if not raw:
        raise ValueError("empty region spec")
    kind, _, rest = raw.partition(":")
    if kind == "face":
        parts = [p for p in rest.split(":") if p]
        if not parts:
            raise ValueError(f"face spec needs a side: {spec!r}")
        side = parts[0]
        if side not in FACE_AXES:
            raise ValueError(f"unknown face {side!r}; expected {sorted(FACE_AXES)}")
        opts: dict[str, str] = {}
        for part in parts[1:]:
            key, _, val = part.partition("=")
            if not val:
                raise ValueError(f"expected key=value in {part!r} of {spec!r}")
            opts[key] = val
        fracs = tuple(float(x) for x in opts.get("frac", "1").split(","))
        centers = tuple(float(x) for x in opts.get("center", "0.5").split(","))
        return ("face", side, fracs, centers)
    if kind == "box":
        bounds = tuple(float(x) for x in rest.split(","))
        if len(bounds) != 4:
            raise ValueError(f"box spec needs 4 numbers xmin,xmax,ymin,ymax: {spec!r}")
        return ("box", bounds)
    raise ValueError(f"region spec must start with face: or box: ({spec!r})")


def resolve_axis(side: str, dim: int) -> tuple[int, int]:
    axis, lohi = FACE_AXES[side]
    if axis < 0:
        axis = dim + axis
    if axis < 0 or axis >= dim:
        raise ValueError(f"face {side!r} is not valid in {dim}D")
    return axis, lohi


def face_patch_mask(params: ColdPlateParams, axis: int, fracs, centers):
    other = [i for i in range(params.dim) if i != axis]
    shape = tuple(params.n[i] for i in other)
    mask = jnp.ones(shape, dtype=bool)
    for j, ax in enumerate(other):
        frac = fracs[j] if j < len(fracs) else 1.0
        center = centers[j] if j < len(centers) else 0.5
        bit = line_mask(params.n[ax], frac, center)
        view = [1] * len(other)
        view[j] = params.n[ax]
        mask = mask & bit.reshape(view)
    return mask


def box_cell_mask(params: ColdPlateParams, bounds: tuple[float, ...]):
    if len(bounds) < 2 * params.dim:
        raise ValueError(f"box bounds {bounds} too short for {params.dim}D")
    mask = jnp.ones(params.n, dtype=bool)
    for ax in range(params.dim):
        lo, hi = bounds[2 * ax], bounds[2 * ax + 1]
        xc = (jnp.arange(params.n[ax]) + 0.5) * params.dx[ax]
        bit = (xc >= lo) & (xc <= hi)
        view = [1] * params.dim
        view[ax] = params.n[ax]
        mask = mask & bit.reshape(view)
    return mask


def _specs(params: ColdPlateParams, role: str) -> tuple[str, ...]:
    return params.hot_specs if role == "hot" else params.cold_specs


def _temperature(params: ColdPlateParams, role: str) -> float:
    return params.t_hot if role == "hot" else params.t_in


def face_dirichlets(params: ColdPlateParams) -> list[FacePatch]:
    patches = []
    for role in ("hot", "cold"):
        tbc = _temperature(params, role)
        for spec in _specs(params, role):
            parsed = parse_spec(spec)
            if parsed[0] != "face":
                continue
            _, side, fracs, centers = parsed
            axis, lohi = resolve_axis(side, params.dim)
            mask = face_patch_mask(params, axis, fracs, centers)
            patches.append(FacePatch(axis, lohi, mask, tbc, role))
    return patches


def cell_dirichlet_masks(params: ColdPlateParams):
    hot = jnp.zeros(params.n, dtype=bool)
    cold = jnp.zeros(params.n, dtype=bool)
    for spec in params.hot_specs:
        parsed = parse_spec(spec)
        if parsed[0] == "box":
            hot = hot | box_cell_mask(params, parsed[1])
    for spec in params.cold_specs:
        parsed = parse_spec(spec)
        if parsed[0] == "box":
            cold = cold | box_cell_mask(params, parsed[1])
    cold = cold & ~hot
    return hot, cold


def apply_face_dirichlet_diffusion(div, T, k, params: ColdPlateParams, t_hot, t_in):
    """Add ∇·(k∇T) contributions from user Dirichlet faces (uses explicit BC values)."""
    temps = {"hot": t_hot, "cold": t_in}
    for patch in face_dirichlets(params):
        dx = params.dx[patch.axis]
        sl = 0 if patch.side == 0 else -1
        tb = take_axis(T, patch.axis, sl)
        kb = take_axis(k, patch.axis, sl)
        tbc = temps[patch.role]
        flux = patch.mask * kb * (tb - tbc) / (0.5 * dx)
        div = add_axis(div, patch.axis, sl, -flux / dx)
    return div


def apply_cell_dirichlet(residual, T, params: ColdPlateParams, t_hot, t_in):
    hot, cold = cell_dirichlet_masks(params)
    residual = jnp.where(hot, T - t_hot, residual)
    residual = jnp.where(cold, T - t_in, residual)
    return residual


def face_heat_into_domain(T, k, params: ColdPlateParams, role: str = "hot"):
    """Conductive heat entering the domain through Dirichlet faces of ``role``."""
    total = 0.0
    tbc = _temperature(params, role)
    for patch in face_dirichlets(params):
        if patch.role != role:
            continue
        dx = params.dx[patch.axis]
        sl = 0 if patch.side == 0 else -1
        tb = take_axis(T, patch.axis, sl)
        kb = take_axis(k, patch.axis, sl)
        area = params.cell_volume / dx
        q = patch.mask * kb * (tbc - tb) / (0.5 * dx)
        total = total + jnp.sum(q * area)
    return total


def volume_heat_from_cells(T, k, cell_mask, params: ColdPlateParams):
    """Heat leaving Dirichlet source cells into the rest of the plate."""
    total = 0.0
    for axis, dx in enumerate(params.dx):
        area = params.cell_volume / dx
        kf = harmonic_faces(k, axis)
        t0 = take_axis(T, axis, slice(None, -1))
        t1 = take_axis(T, axis, slice(1, None))
        leave0 = kf * (t0 - t1) / dx * area
        h0 = take_axis(cell_mask, axis, slice(None, -1))
        h1 = take_axis(cell_mask, axis, slice(1, None))
        total = total + jnp.sum(jnp.where(h0 & ~h1, leave0, 0.0))
        total = total + jnp.sum(jnp.where(h1 & ~h0, -leave0, 0.0))
    return total


def overlay_segments_2d(params: ColdPlateParams):
    """Line segments ``(x0, y0, x1, y1, color)`` for 2-D plots."""
    lx, ly = params.L
    segs = []
    colors = {"hot": "crimson", "cold": "deepskyblue"}
    for patch in face_dirichlets(params):
        mask = np.asarray(patch.mask)
        if patch.axis == 1:
            y = 0.0 if patch.side == 0 else ly
            xs = (np.arange(params.n[0]) + 0.5) * params.dx[0]
            active = xs[mask]
            if active.size:
                segs.append((float(active[0] - 0.5 * params.dx[0]), y, float(active[-1] + 0.5 * params.dx[0]), y, colors[patch.role]))
        elif patch.axis == 0:
            x = 0.0 if patch.side == 0 else lx
            ys = (np.arange(params.n[1]) + 0.5) * params.dx[1]
            active = ys[mask]
            if active.size:
                segs.append((x, float(active[0] - 0.5 * params.dx[1]), x, float(active[-1] + 0.5 * params.dx[1]), colors[patch.role]))
    return segs


def overlay_boxes_2d(params: ColdPlateParams):
    boxes = []
    colors = {"hot": "crimson", "cold": "deepskyblue"}
    for role in ("hot", "cold"):
        for spec in _specs(params, role):
            parsed = parse_spec(spec)
            if parsed[0] == "box":
                b = parsed[1]
                boxes.append((b[0], b[1], b[2], b[3], colors[role]))
    return boxes


def add_region_arguments(parser) -> None:
    parser.add_argument(
        "--hot",
        action="append",
        default=None,
        metavar="SPEC",
        help=(
            "Dirichlet heat-source region (disables the uniform volume source). "
            "Repeatable. Examples: face:top, box:0.25,0.75,0,0.15"
        ),
    )
    parser.add_argument(
        "--cold",
        action="append",
        default=None,
        metavar="SPEC",
        help=(
            "Cold-plate / heat-sink region. Repeatable. "
            "Examples: face:left, face:top:frac=0.4, box:0.8,1.0,0.3,0.7"
        ),
    )


def specs_from_cli(hot, cold) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Parse CLI ``--hot`` / ``--cold`` lists. Missing lists become empty."""
    hot_specs = tuple(hot) if hot else ()
    cold_specs = tuple(cold) if cold else ()
    for spec in (*hot_specs, *cold_specs):
        parse_spec(spec)
    return hot_specs, cold_specs
