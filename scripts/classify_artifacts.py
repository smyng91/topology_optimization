#!/usr/bin/env python3
"""Classify local result artifacts as regenerated, synthetic, or unresolved."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PAPER = ROOT / "paper"


def _exists(path: Path) -> bool:
    return path.exists()


def classify() -> dict:
    records = []
    old = OUT / "paper_runs"
    if old.exists():
        records.append(
            {
                "path": str(old.relative_to(ROOT)).replace("\\", "/"),
                "class": "provenance-unresolved",
                "reason": (
                    "Predates conventional Stolpe--Svanberg RAMP, source-digest "
                    "cache keys, and the full evidence gates. Not a publication source."
                ),
            }
        )
    release = OUT / "release_v1"
    records.append(
        {
            "path": str(release.relative_to(ROOT)).replace("\\", "/"),
            "class": "regenerated" if _exists(release) else "missing",
            "reason": "Clean namespace for protocol journal-neutral-2d-v1.",
        }
    )
    for name in ("results.json", "numbers.tex", "provenance.json"):
        path = PAPER / name
        records.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "class": "regenerated" if path.is_file() else "missing",
                "reason": "Written only by scripts/build_manuscript.py.",
            }
        )
    figs = PAPER / "figures"
    protocol_figs = (
        list(figs.glob("fig*.pdf")) + list(figs.glob("fig*.png")) if figs.exists() else []
    )
    records.append(
        {
            "path": "paper/figures",
            "class": "regenerated" if protocol_figs else "missing",
            "reason": "fig*.pdf/png generated from release_v1 fields; never synthesized.",
        }
    )
    for leftover in (
        "paper/figures/analyze_gray.png",
        "paper/figures/conduction_tree.png",
        "paper/figures/convection_darcy.png",
        "paper/figures/custom_faces.png",
    ):
        path = ROOT / leftover
        if path.is_file():
            records.append(
                {
                    "path": leftover,
                    "class": "provenance-unresolved",
                    "reason": "Predates protocol journal-neutral-2d-v1. Not a publication source.",
                }
            )
    records.append(
        {
            "path": "paper/claim_ledger.json",
            "class": "author-maintained",
            "reason": "Maps each numerical macro to a results.json path; checked by scripts/check_manuscript_integrity.py.",
        }
    )
    records.append(
        {
            "path": "docs/figures",
            "class": "not-publication",
            "reason": "Short-mesh tutorial snapshots from examples/publish_figures.py, not protocol figures.",
        }
    )
    return {
        "schema_version": 1,
        "publication_sources": [row["path"] for row in records if row["class"] == "regenerated"],
        "excluded": [row for row in records if row["class"] != "regenerated"],
        "records": records,
    }


def main() -> int:
    payload = classify()
    text = json.dumps(payload, indent=2) + "\n"
    print(text, end="")
    if PAPER.is_dir():
        target = PAPER / "artifact_classification.json"
        target.write_text(text, encoding="utf-8")
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
