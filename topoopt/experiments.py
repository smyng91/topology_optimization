"""Versioned experiment protocol and cache identity for publication runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from topoopt.problems import PROBLEMS
from topoopt.provenance import file_digest


PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "paper" / "experiments.json"


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:
    """Load the predeclared publication protocol."""
    protocol_path = Path(path) if path is not None else PROTOCOL_PATH
    data = json.loads(protocol_path.read_text(encoding="utf-8"))
    if "protocol_id" not in data or "cases" not in data:
        raise ValueError(f"{protocol_path} is missing protocol_id or cases")
    data["_path"] = str(protocol_path)
    data["_sha256"] = file_digest(protocol_path)
    return data


def case_params(protocol: dict[str, Any], name: str):
    """Instantiate the registered factory for one protocol case."""
    spec = protocol["cases"][name]
    factory = PROBLEMS[spec["factory"]]
    nx, ny = spec["mesh"]
    return factory(nx=int(nx), ny=int(ny), **spec.get("overrides", {}))


def case_fingerprint(
    protocol: dict[str, Any],
    name: str,
    *,
    seed: int,
    model_source_sha256: str,
) -> dict[str, Any]:
    """Identity that must match before a cached run may be reused."""
    spec = protocol["cases"][name]
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["_sha256"],
        "model_source_sha256": model_source_sha256,
        "case": name,
        "factory": spec["factory"],
        "mesh": list(spec["mesh"]),
        "n_iters": spec["n_iters"],
        "lr": spec["lr"],
        "beta_max": spec["beta_max"],
        "overrides": spec.get("overrides", {}),
        "seed": int(seed),
        "acceptance": protocol.get("acceptance", {}),
    }


def fingerprints_match(saved: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Compare cache identity, ignoring write-time metadata."""
    ignore = {"created_utc", "run_dir"}
    left = {key: saved[key] for key in saved if key not in ignore}
    right = {key: expected[key] for key in expected if key not in ignore}
    return left == right
