"""Named research configurations.

The ``topoopt`` package is a generic solver. Geometry, ports, and hot/cold
patches for the heated-box studies live here so you can add or swap cases
without changing the library.

Each factory returns a ``ColdPlateParams``. Extra keywords override the
case defaults (mesh, solver iters, ``vol_frac``, …).

::

    from examples.problems import conduction_tree, convection_darcy
    params = convection_darcy(nx=64, ny=64)

    python -m topoopt 2d --config examples.problems:conduction_tree
"""

from __future__ import annotations

from topoopt.config import ColdPlateParams, params2d

# Centered left inlet / right outlet, as a fraction of wall height.
CENTERLINE_PORT = 0.5
# Narrow bottom sink for the volume-to-point conduction tree.
TREE_SINK = ("face:bottom:frac=0.08",)


def conduction_tree(nx: int = 40, ny: int = 40, **kwargs) -> ColdPlateParams:
    """Uniform q, small bottom sink, no flow. ``J = -mean(T)``."""
    spec = dict(
        heat_mode="conduction",
        vol_frac=0.30,
        rmin=1.5,
        hot_specs=(),
        cold_specs=TREE_SINK,
        symmetry=("x",),
        heat_iters=400,
        filter_iters=200,
    )
    spec.update(kwargs)
    return params2d(nx, ny, **spec)


def convection_darcy(nx: int = 40, ny: int = 40, **kwargs) -> ColdPlateParams:
    """Uniform q and k, one left-centerline inlet, one right-centerline outlet."""
    spec = dict(
        heat_mode="convection",
        flow_model="darcy",
        vol_frac=0.45,
        pe=40.0,
        rmin=2.0,
        port_frac=CENTERLINE_PORT,
        hot_specs=(),
        cold_specs=(),
        symmetry=("y",),
        flow_iters=280,
        heat_iters=400,
        filter_iters=200,
    )
    spec.update(kwargs)
    return params2d(nx, ny, **spec)


def conjugate_darcy(nx: int = 40, ny: int = 40, **kwargs) -> ColdPlateParams:
    """Conjugate ``k(γ)`` with the same centerline Darcy ports as convection."""
    spec = dict(
        heat_mode="both",
        flow_model="darcy",
        vol_frac=0.45,
        pe=40.0,
        rmin=2.0,
        port_frac=CENTERLINE_PORT,
        hot_specs=(),
        cold_specs=(),
        symmetry=("y",),
        flow_iters=280,
        heat_iters=400,
        filter_iters=200,
    )
    spec.update(kwargs)
    return params2d(nx, ny, **spec)


def conjugate_stokes(nx: int = 40, ny: int = 40, **kwargs) -> ColdPlateParams:
    """Conjugate ``k(γ)`` with pressure-driven Stokes–Brinkman centerline ports."""
    spec = dict(
        heat_mode="both",
        flow_model="stokes",
        vol_frac=0.45,
        pe=40.0,
        rmin=2.0,
        port_frac=CENTERLINE_PORT,
        hot_specs=(),
        cold_specs=(),
        symmetry=("y",),
        flow_iters=80,
        uzawa_iters=80,
        stokes_kryl_iters=200,
        heat_iters=320,
        filter_iters=120,
    )
    spec.update(kwargs)
    return params2d(nx, ny, **spec)


def custom_faces(nx: int = 40, ny: int = 40, **kwargs) -> ColdPlateParams:
    """Dirichlet sandwich: hot top, cold bottom. ``J`` is heat leaving the hot face."""
    spec = dict(
        heat_mode="conduction",
        q_vol=0.0,
        vol_frac=0.40,
        rmin=2.0,
        hot_specs=("face:top:frac=0.5",),
        cold_specs=("face:bottom:frac=0.5",),
        symmetry=("x",),
        heat_iters=400,
        filter_iters=200,
    )
    spec.update(kwargs)
    return params2d(nx, ny, **spec)


def custom_boxes(nx: int = 40, ny: int = 40, **kwargs) -> ColdPlateParams:
    """Volumetric hot / cold boxes plus a left-wall sink."""
    spec = dict(
        heat_mode="conduction",
        q_vol=0.0,
        vol_frac=0.40,
        rmin=2.0,
        hot_specs=("box:0.2,0.8,0.0,0.18",),
        cold_specs=("box:0.0,0.18,0.25,0.75", "face:left"),
        heat_iters=400,
        filter_iters=200,
    )
    spec.update(kwargs)
    return params2d(nx, ny, **spec)


PROBLEMS = {
    "conduction_tree": conduction_tree,
    "convection_darcy": convection_darcy,
    "conjugate_darcy": conjugate_darcy,
    "conjugate_stokes": conjugate_stokes,
    "custom_faces": custom_faces,
    "custom_boxes": custom_boxes,
}
