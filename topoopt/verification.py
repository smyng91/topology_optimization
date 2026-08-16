"""Reusable verification studies for the complete differentiable analysis."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def directional_taylor_test(
    objective: Callable,
    point,
    direction,
    steps: Sequence[float],
) -> dict[str, object]:
    """Run first-order remainder and centered directional-derivative tests."""
    point = jnp.asarray(point, dtype=jnp.float64)
    direction = jnp.asarray(direction, dtype=point.dtype)
    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-30)
    value, gradient = jax.value_and_grad(objective)(point)
    derivative = jnp.vdot(gradient, direction)

    records = []
    for step in steps:
        step = float(step)
        plus = objective(point + step * direction)
        minus = objective(point - step * direction)
        remainder = jnp.abs(plus - value - step * derivative)
        centered = (plus - minus) / (2.0 * step)
        gradient_error = jnp.abs(centered - derivative)
        records.append(
            {
                "step": step,
                "forward_value": float(plus),
                "first_order_remainder": float(remainder),
                "centered_gradient_error": float(gradient_error),
            }
        )

    def pairwise_orders(key: str) -> list[float | None]:
        orders: list[float | None] = []
        for left, right in zip(records[:-1], records[1:]):
            e0, e1 = float(left[key]), float(right[key])
            h0, h1 = float(left["step"]), float(right["step"])
            if e0 <= 0.0 or e1 <= 0.0 or not math.isfinite(e0 + e1):
                orders.append(None)
            else:
                orders.append(math.log(e0 / e1) / math.log(h0 / h1))
        return orders

    return {
        "value": float(value),
        "directional_derivative": float(derivative),
        "gradient_norm": float(jnp.linalg.norm(gradient)),
        "records": records,
        "first_order_orders": pairwise_orders("first_order_remainder"),
        "centered_gradient_orders": pairwise_orders("centered_gradient_error"),
    }


def fitted_log_order(records: Sequence[dict[str, float]], key: str) -> float:
    """Fit log(error) against log(step), excluding zero/non-finite errors."""
    pairs = [
        (float(record["step"]), float(record[key]))
        for record in records
        if float(record[key]) > 0.0 and math.isfinite(float(record[key]))
    ]
    if len(pairs) < 2:
        return float("nan")
    steps, errors = zip(*pairs)
    return float(np.polyfit(np.log(steps), np.log(errors), 1)[0])


def deterministic_direction(shape: tuple[int, ...], seed: int) -> jax.Array:
    """A deterministic, mean-zero perturbation for directional checks."""
    direction = jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=jnp.float64)
    return direction - jnp.mean(direction)


def high_contrast_design(shape: tuple[int, int], *, channel: bool) -> jax.Array:
    """Bounded nontrivial raw field representative of high-contrast designs."""
    nx, ny = shape
    x = (jnp.arange(nx, dtype=jnp.float64) + 0.5) / nx
    y = (jnp.arange(ny, dtype=jnp.float64) + 0.5) / ny
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    if channel:
        field = jnp.where(jnp.abs(yy - 0.5) < 0.18, 0.12, 0.82)
        field = field + 0.04 * jnp.sin(2.0 * jnp.pi * xx)
    else:
        field = 0.50 + 0.36 * jnp.sin(2.0 * jnp.pi * xx) * jnp.cos(
            2.0 * jnp.pi * yy
        )
    return jnp.clip(field, 0.08, 0.92)
