#!/usr/bin/env python3
"""Check bibliography keys, DOI syntax, and optional Crossref metadata."""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

BIB = Path(__file__).resolve().parents[1] / "paper" / "references.bib"
DOI_RE = re.compile(r"10\.\d{4,9}/\S+")


def _entries(text: str) -> list[dict[str, str]]:
    entries = []
    for match in re.finditer(r"@(\w+)\{([^,]+),([\s\S]*?)\n\}", text):
        body = match.group(3)
        fields = {"entry_type": match.group(1), "key": match.group(2).strip()}
        for field in re.finditer(r"(\w+)\s*=\s*[\{]([^}]*)[\}]", body):
            fields[field.group(1).lower()] = field.group(2).strip()
        entries.append(fields)
    return entries


def _crossref(doi: str) -> dict | None:
    request = urllib.request.Request(
        f"https://api.crossref.org/works/{urllib.request.quote(doi)}",
        headers={"User-Agent": "topoopt-reference-check/1.0 (mailto:smyng@gatech.edu)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))["message"]
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        return None


def main() -> int:
    text = BIB.read_text(encoding="utf-8")
    entries = _entries(text)
    errors = []
    keys = [entry["key"] for entry in entries]
    if len(keys) != len(set(keys)):
        errors.append("duplicate BibTeX keys")
    for entry in entries:
        doi = entry.get("doi")
        if doi and not DOI_RE.fullmatch(doi):
            errors.append(f"{entry['key']}: malformed DOI {doi!r}")
    online = "--online" in sys.argv
    if online:
        for entry in entries:
            doi = entry.get("doi")
            if not doi:
                continue
            message = _crossref(doi)
            if message is None:
                errors.append(f"{entry['key']}: Crossref lookup failed for {doi}")
                continue
            years = set()
            for field in ("issued", "published-print", "published-online", "published"):
                parts = message.get(field, {}).get("date-parts", [])
                if parts and parts[0] and parts[0][0]:
                    years.add(str(parts[0][0]))
            cited = entry.get("year")
            if cited and years and cited not in years:
                # Print/copyright year and Crossref ebook/online-first year
                # often differ by one (e.g. Bendsøe & Sigmund 2003).
                if not any(abs(int(cited) - int(year)) <= 1 for year in years):
                    errors.append(
                        f"{entry['key']}: year {cited} vs Crossref {sorted(years)}"
                    )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"ok: {len(entries)} bibliography entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
