#!/usr/bin/env python3
"""Fail if manuscript macros, figures, labels, citations, or hashes do not resolve."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
sys.path.insert(0, str(ROOT))

from topoopt.provenance import (  # noqa: E402
    MANUSCRIPT_SOURCE_PATHS,
    MODEL_SOURCE_PATHS,
    file_digest,
    source_digest,
)

EXPECTED_MACROS = {
    "PoissonOrder", "AdvectionOrder", "HelmholtzOrder", "HelmholtzErrFine",
    "StokesUOrder", "DarcyPMax", "PoissonErrFine", "AdvErrFine", "StokesUFine",
    "AdjointRelMax", "AdjointAbsMax", "TreeNx", "TreeIters", "TreeBeta",
    "TreeJzero", "TreeJbest", "TreeTmean", "TreeTmax", "TreeTmeanUni",
    "TreeTmeanFluid", "TreeGray", "TreeVolErr", "TreeErms", "TreeErel",
    "TreeStepMs", "TreeStopped", "DarcyNx", "DarcyIters", "DarcyBeta",
    "DarcyJzero", "DarcyJbest", "DarcyTmean", "DarcyGray", "DarcyVolErr",
    "DarcyErms", "DarcyErel", "DarcyUin", "DarcyStopped", "StokesNx",
    "StokesIters", "StokesBeta", "StokesJzero", "StokesJbest", "StokesTmean",
    "StokesGray", "StokesVolErr", "StokesErms", "StokesErel", "StokesDiv",
    "StokesRel", "StokesMass", "StokesStepMs", "StokesStopped", "CustomNx",
    "CustomIters", "CustomBeta", "CustomJpeak", "CustomJzero", "CustomJbest",
    "CustomQuni", "CustomGray", "CustomErms", "CustomErel", "CustomVolErr",
    "CustomStopped", "TreeJmin", "TreeJmax", "DarcyJmin", "DarcyJmax",
    "StokesJmin", "StokesJmax", "CustomJmin", "CustomJmax", "TreeJmatched",
    "DarcyJmatched", "StokesJmatched", "CustomJmatched", "TaylorCondOrder",
    "TaylorDarcyOrder", "TaylorStokesOrder", "MeshJcoarse", "MeshJfine",
    "NPublishedSeeds", "StokesDivEps",
}
REQUIRED_CITES = ("Yang2026", "Bradbury2018", "Stolpe2001")
LEGACY_FIGURES = {
    "analyze_gray.png",
    "conduction_tree.png",
    "convection_darcy.png",
    "custom_faces.png",
}
SI_MARKERS = (
    "supplementary information",
    "supplementary-information",
    "supplementary material",
    "sec:si",
)


def macros(text: str) -> set[str]:
    return set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", text))


def used_macros(text: str) -> set[str]:
    return set(re.findall(r"\\([A-Z][A-Za-z]+)", text))


def bib_keys(text: str) -> set[str]:
    return set(re.findall(r"@\w+\{([^,]+),", text))


def cite_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite[a-zA-Z]*\{([^}]+)\}", text):
        keys.update(part.strip() for part in match.group(1).split(",") if part.strip())
    return keys


def labels(text: str) -> set[str]:
    return set(re.findall(r"\\label\{([^}]+)\}", text))


def refs(text: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\(?:Cref|cref|ref)\{([^}]+)\}", text):
        keys.update(part.strip() for part in match.group(1).split(",") if part.strip())
    return keys


def graphics(text: str) -> list[str]:
    return re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)


COMMAND_VALUE = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}")


def parse_tex_number(text: str) -> float | str:
    raw = text.strip()
    scientific = re.sub(r"\\times 10\^{([+-]?\d+)}", r"e\1", raw)
    try:
        return float(scientific)
    except ValueError:
        return raw


def defined_macros(numbers: str) -> dict[str, float | str]:
    return {name: parse_tex_number(value) for name, value in COMMAND_VALUE.findall(numbers)}


def lookup(data: object, path: list[object]) -> object:
    value: object = data
    for part in path:
        if isinstance(part, int):
            value = value[part]  # type: ignore[index]
        else:
            value = value[part]  # type: ignore[index]
    return value


def format_claim(value: object, spec: dict) -> float | str:
    kind = spec.get("format", "float")
    if kind == "text":
        return str(value)
    number = float(value) * float(spec.get("scale", 1.0))
    if kind == "int":
        return int(round(number))
    if kind == "sci":
        return number
    digits = int(spec.get("digits", 4))
    return round(number, digits)


def first_positions(pattern: str, text: str) -> dict[str, int]:
    found: dict[str, int] = {}
    for match in re.finditer(pattern, text):
        for part in match.group(1).split(","):
            key = part.strip()
            if key and key not in found:
                found[key] = match.start()
    return found


def cite_before_definition(text: str) -> list[str]:
    cite_pos = first_positions(r"\\(?:Cref|cref|ref)\{([^}]+)\}", text)
    label_pos = first_positions(r"\\label\{([^}]+)\}", text)
    errors = []
    for key, where in label_pos.items():
        if not (key.startswith("fig:") or key.startswith("tab:")):
            continue
        cited = cite_pos.get(key)
        if cited is None:
            errors.append(f"{key} is never cited")
        elif cited > where:
            errors.append(f"{key} is cited after it is defined")
    return errors


def check_claim_ledger(results: dict, numbers: str, ledger: dict) -> list[str]:
    defined = defined_macros(numbers)
    errors: list[str] = []
    claimed = {row["macro"] for row in ledger.get("claims", [])}
    missing = sorted(set(defined) - claimed)
    extra = sorted(claimed - set(defined))
    if missing:
        errors.append(f"claim ledger missing macros: {missing}")
    if extra:
        errors.append(f"claim ledger unknown macros: {extra}")
    for row in ledger.get("claims", []):
        try:
            raw = lookup(results, row["path"])
        except (KeyError, IndexError, TypeError) as exc:
            errors.append(f"{row['id']}: results path {row['path']} ({exc})")
            continue
        if row.get("format") == "text":
            expected = str(raw)
            observed = defined.get(row["macro"])
            if observed != expected:
                errors.append(f"{row['id']}: {observed!r} != {expected!r}")
            continue
        expected = format_claim(raw, row)
        observed = defined.get(row["macro"])
        if not isinstance(observed, float) or not isinstance(expected, (int, float)):
            errors.append(f"{row['id']}: non-numeric comparison {observed!r} vs {expected!r}")
            continue
        if row.get("format") == "sci":
            scale = max(abs(expected), abs(observed), 1e-30)
            if abs(expected - observed) / scale > 0.015:
                errors.append(f"{row['id']}: {observed} != {expected}")
        elif abs(float(expected) - observed) > 0.5001 * 10 ** (-int(row.get("digits", 4))):
            errors.append(f"{row['id']}: {observed} != {expected}")
    return errors


def check_hashes(provenance: dict) -> list[str]:
    errors: list[str] = []
    manifest = provenance.get("manifest", {})
    artifacts = provenance.get("artifacts", {})
    model = source_digest(ROOT, MODEL_SOURCE_PATHS)
    if manifest.get("model_source_sha256") != model:
        errors.append("model source digest drifted from paper/provenance.json")
    protocol = ROOT / "paper" / "experiments.json"
    if manifest.get("protocol_sha256") != file_digest(protocol):
        errors.append("protocol digest drifted from paper/provenance.json")
    if artifacts.get("numbers_tex_sha256") != file_digest(PAPER / "numbers.tex"):
        errors.append("numbers.tex digest drifted from paper/provenance.json")
    if artifacts.get("results_json_sha256") != file_digest(PAPER / "results.json"):
        errors.append("results.json digest drifted from paper/provenance.json")
    manuscript = source_digest(ROOT, MANUSCRIPT_SOURCE_PATHS)
    if manifest.get("manuscript_source_sha256") != manuscript:
        errors.append("manuscript source digest drifted from paper/provenance.json")
    recorded = artifacts.get("figures", {})
    for name, digest in recorded.items():
        path = PAPER / "figures" / name
        if not path.is_file():
            errors.append(f"provenance figure missing: {name}")
        elif file_digest(path) != digest:
            errors.append(f"figure digest drifted: {name}")
    return errors


def refresh_manuscript_provenance() -> None:
    path = PAPER / "provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    figures = {
        item.name: file_digest(item)
        for item in sorted((PAPER / "figures").iterdir())
        if item.suffix.lower() in {".pdf", ".png"} and item.name not in LEGACY_FIGURES
    }
    payload["artifacts"]["figures"] = figures
    payload["artifacts"]["numbers_tex_sha256"] = file_digest(PAPER / "numbers.tex")
    payload["artifacts"]["results_json_sha256"] = file_digest(PAPER / "results.json")
    payload["manifest"]["manuscript_source_sha256"] = source_digest(ROOT, MANUSCRIPT_SOURCE_PATHS)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"refreshed manuscript/figure hashes in {path}")


def main() -> int:
    if "--refresh-manuscript" in sys.argv:
        refresh_manuscript_provenance()
        return 0
    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    numbers = (PAPER / "numbers.tex").read_text(encoding="utf-8")
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    results = json.loads((PAPER / "results.json").read_text(encoding="utf-8"))
    provenance = json.loads((PAPER / "provenance.json").read_text(encoding="utf-8"))
    ledger = json.loads((PAPER / "claim_ledger.json").read_text(encoding="utf-8"))
    defined = macros(numbers)
    used = used_macros(main_tex)
    errors: list[str] = []
    missing_macros = sorted(name for name in used if name in EXPECTED_MACROS and name not in defined)
    if missing_macros:
        errors.append(f"undefined numerical macros: {missing_macros}")
    unused = sorted(defined - used)
    if unused:
        errors.append(f"unused numerical macros: {unused}")
    missing_expected = sorted(EXPECTED_MACROS - defined)
    if missing_expected:
        errors.append(f"numbers.tex missing expected macros: {missing_expected}")
    if "protocol_id" not in results:
        errors.append("results.json missing protocol_id")
    if not any(marker in main_tex.lower() for marker in SI_MARKERS):
        errors.append("main.tex does not cite the supplementary material")
    errors.extend(cite_before_definition(main_tex))
    missing_labels = sorted(refs(main_tex) - labels(main_tex))
    if missing_labels:
        errors.append(f"undefined labels: {missing_labels}")
    keys = bib_keys(bib)
    cites = cite_keys(main_tex)
    missing_cites = sorted(cites - keys)
    if missing_cites:
        errors.append(f"undefined citations: {missing_cites}")
    uncited = sorted(keys - cites)
    if uncited:
        errors.append(f"unused bibliography keys: {uncited}")
    for required in REQUIRED_CITES:
        if required not in cites:
            errors.append(f"required citation {required} is missing")
    for graphic in graphics(main_tex):
        path = PAPER / graphic
        if not path.is_file():
            errors.append(f"missing figure asset: {graphic}")
    leftover = sorted(name for name in LEGACY_FIGURES if (PAPER / "figures" / name).is_file())
    if leftover:
        errors.append(f"legacy non-protocol figures still present: {leftover}")
    errors.extend(check_claim_ledger(results, numbers, ledger))
    errors.extend(check_hashes(provenance))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        f"ok: {len(defined)} macros, {len(labels(main_tex))} labels, "
        f"{len(cites)} citations, hashes match"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
