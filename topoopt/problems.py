"""Packaged, registered problem factories used by examples and verification."""

from __future__ import annotations

from topoopt.config import ColdPlateParams, params2d

CENTERLINE_PORT = 0.5
TREE_SINK = ("face:bottom:frac=0.08",)
SOURCE_BOX = ("box:0.3,0.7,0.70,1.0",)


def conduction_tree(nx: int = 40, ny: int = 40, **kwargs) -> ColdPlateParams:
    """Volume-to-point conduction with uniform heating and a narrow sink."""
    spec = dict(
        heat_mode="conduction",
        vol_frac=0.30,
        rmin=1.5,
        hot_specs=(),
        cold_specs=TREE_SINK,
        symmetry=("x",),
        heat_iters=800,
        filter_iters=200,
    )
    spec.update(kwargs)
    return params2d(nx, ny, **spec)


def convection_darcy(nx: int = 40, ny: int = 40, **kwargs) -> ColdPlateParams:
    """Uniform-conductivity cooling with pressure-driven Darcy flow."""
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
        heat_iters=800,
        filter_iters=200,
    )
    spec.update(kwargs)
    return params2d(nx, ny, **spec)


def conjugate_darcy(nx: int = 40, ny: int = 40, **kwargs) -> ColdPlateParams:
    """Design-dependent conductivity with pressure-driven Darcy flow."""
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
        heat_iters=800,
        filter_iters=200,
    )
    spec.update(kwargs)
    return params2d(nx, ny, **spec)


def conjugate_stokes(nx: int = 40, ny: int = 40, **kwargs) -> ColdPlateParams:
    """Conjugate Stokes--Brinkman cooling with pressure-driven ports."""
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
        heat_iters=800,
        filter_iters=120,
    )
    spec.update(kwargs)
    return params2d(nx, ny, **spec)


def custom_faces(nx: int = 40, ny: int = 40, **kwargs) -> ColdPlateParams:
    """Hot-top/cold-bottom conduction benchmark."""
    spec = dict(
        heat_mode="conduction",
        q_vol=0.0,
        vol_frac=0.40,
        rmin=2.0,
        hot_specs=("face:top:frac=0.5",),
        cold_specs=("face:bottom:frac=0.5",),
        symmetry=("x",),
        heat_iters=800,
        filter_iters=200,
    )
    spec.update(kwargs)
    return params2d(nx, ny, **spec)


def custom_boxes(nx: int = 40, ny: int = 40, **kwargs) -> ColdPlateParams:
    """Interior hot/cold Dirichlet boxes plus a full left-wall sink."""
    spec = dict(
        heat_mode="conduction",
        q_vol=0.0,
        vol_frac=0.40,
        rmin=2.0,
        hot_specs=("box:0.2,0.8,0.0,0.18",),
        cold_specs=("box:0.0,0.18,0.25,0.75", "face:left"),
        heat_iters=800,
        filter_iters=200,
    )
    spec.update(kwargs)
    return params2d(nx, ny, **spec)


def localized_source(nx: int = 40, ny: int = 40, **kwargs) -> ColdPlateParams:
    """Localized volumetric source with a narrow bottom sink."""
    spec = dict(
        heat_mode="conduction",
        q_vol=1.0,
        vol_frac=0.30,
        rmin=1.5,
        q_specs=SOURCE_BOX,
        hot_specs=(),
        cold_specs=TREE_SINK,
        symmetry=("x",),
        heat_iters=800,
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
    "localized_source": localized_source,
}
