#!/usr/bin/env python3
"""Compile paper/main.tex with pdflatex + bibtex."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PAPER, check=True)


def main() -> int:
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if pdflatex is None or bibtex is None:
        print("pdflatex and bibtex must be on PATH (MiKTeX or TeX Live).", file=sys.stderr)
        return 1
    _run([pdflatex, "-interaction=nonstopmode", "main.tex"])
    _run([bibtex, "main"])
    _run([pdflatex, "-interaction=nonstopmode", "main.tex"])
    _run([pdflatex, "-interaction=nonstopmode", "main.tex"])
    print(PAPER / "main.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
