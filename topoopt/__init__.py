"""JAX topology optimization for a 2-D heated box."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from topoopt.config import HEAT_MODES, ColdPlateParams, load_params, params2d
from topoopt.problem import analyze, physical_density

__all__ = [
    "HEAT_MODES",
    "ColdPlateParams",
    "analyze",
    "load_params",
    "params2d",
    "physical_density",
]
