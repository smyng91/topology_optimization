"""Compatibility re-exports for the packaged, registered problem factories."""

from topoopt.problems import (
    CENTERLINE_PORT,
    PROBLEMS,
    SOURCE_BOX,
    TREE_SINK,
    conduction_tree,
    conjugate_darcy,
    conjugate_stokes,
    convection_darcy,
    custom_boxes,
    custom_faces,
    localized_source,
)

__all__ = [
    "CENTERLINE_PORT",
    "PROBLEMS",
    "SOURCE_BOX",
    "TREE_SINK",
    "conduction_tree",
    "conjugate_darcy",
    "conjugate_stokes",
    "convection_darcy",
    "custom_boxes",
    "custom_faces",
    "localized_source",
]
