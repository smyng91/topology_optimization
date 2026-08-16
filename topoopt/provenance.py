"""Deterministic provenance records for scientific runs and paper artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


MODEL_SOURCE_PATHS = (
    "topoopt",
    "examples/problems.py",
    "scripts/build_manuscript.py",
    "paper/experiments.json",
    "pyproject.toml",
    "requirements-lock.txt",
)
MANUSCRIPT_SOURCE_PATHS = (
    "paper/main.tex",
    "paper/references.bib",
)
_IGNORED_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_IGNORED_SUFFIXES = {".pyc", ".pyo", ".log", ".aux", ".out", ".blg", ".bbl", ".pdf", ".png"}


def _iter_files(root: Path, paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        path = root / item
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and not any(part in _IGNORED_NAMES for part in candidate.relative_to(root).parts)
                and candidate.suffix.lower() not in _IGNORED_SUFFIXES
            )
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def source_digest(root: str | Path, paths: Iterable[str] = MODEL_SOURCE_PATHS) -> str:
    """Hash relative paths and contents of the selected scientific sources."""
    root = Path(root).resolve()
    digest = hashlib.sha256()
    files = _iter_files(root, paths)
    if not files:
        raise FileNotFoundError(f"no provenance sources found below {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def file_digest(path: str | Path) -> str:
    """Return the SHA-256 digest of one artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(root: str | Path) -> dict[str, object]:
    """Return commit/dirty metadata, or an explicit unavailable record."""
    root = Path(root).resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
        return {"available": True, "commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"available": False, "commit": None, "dirty": None}


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_manifest(
    root: str | Path,
    *,
    protocol_path: str | Path | None = None,
) -> dict[str, object]:
    """Capture source, interpreter, dependency, and JAX backend identity."""
    import jax

    root = Path(root).resolve()
    protocol = Path(protocol_path).resolve() if protocol_path is not None else None
    devices = [
        {
            "id": int(device.id),
            "platform": str(device.platform),
            "kind": str(getattr(device, "device_kind", "unknown")),
        }
        for device in jax.devices()
    ]
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_source_sha256": source_digest(root, MODEL_SOURCE_PATHS),
        "manuscript_source_sha256": source_digest(root, MANUSCRIPT_SOURCE_PATHS),
        "protocol_sha256": file_digest(protocol) if protocol is not None else None,
        "git": git_identity(root),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {
            name: _version(name)
            for name in ("topoopt", "jax", "jaxlib", "numpy", "matplotlib", "scipy")
        },
        "jax": {
            "backend": jax.default_backend(),
            "enable_x64": bool(jax.config.read("jax_enable_x64")),
            "devices": devices,
        },
        "environment": {
            key: os.environ.get(key)
            for key in ("JAX_ENABLE_X64", "JAX_PLATFORM_NAME", "XLA_FLAGS")
            if os.environ.get(key) is not None
        },
    }


def atomic_write_json(path: str | Path, value: object) -> None:
    """Write JSON atomically so interrupted runs do not leave valid-looking files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
