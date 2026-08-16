"""RAMP interpolations from the design field to material properties.

γ = 1 is solid (conducting, impermeable). γ = 0 is fluid (coolant).
"""

from __future__ import annotations

import jax.numpy as jnp

from topoopt.config import ColdPlateParams


def ramp(x, lo, hi, q):
    """Conventional Stolpe--Svanberg RAMP map.

    ``x=0`` maps to ``lo`` and ``x=1`` maps to ``hi``. ``q=0`` is
    linear; increasing non-negative ``q`` penalizes intermediate values.
    """
    fraction = x / (1.0 + q * (1.0 - x))
    return lo + (hi - lo) * fraction


def conductivity(gamma, params: ColdPlateParams):
    """Design-dependent k, except in convection-only mode (uniform fluid k)."""
    if params.heat_mode == "convection":
        return jnp.full_like(gamma, params.k_fluid)
    return ramp(gamma, params.k_fluid, params.k_solid, params.q_k)


def brinkman_alpha(gamma, params: ColdPlateParams):
    """Borrvall–Petersson map: small q keeps intermediate γ permeable.

    Ordinary RAMP to α_max makes γ=0.45 almost impermeable (90% of α_max),
    so a channel cannot nucleate. This form stays open until γ is near 1.
    """
    q = params.q_alpha
    frac = q * gamma / (1.0 - gamma + q)
    return params.alpha_min + (params.alpha_max - params.alpha_min) * frac


def darcy_kappa(gamma, params: ColdPlateParams):
    """Permeability: high in fluid, near-zero in solid."""
    return ramp(gamma, params.kappa_max, params.kappa_min, params.q_kappa)


def tanh_project(x, beta: float, eta: float = 0.5):
    """Smooth Heaviside projection (Wang, Lazarov, Sigmund 2011)."""
    num = jnp.tanh(beta * eta) + jnp.tanh(beta * (x - eta))
    den = jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
    return num / den
