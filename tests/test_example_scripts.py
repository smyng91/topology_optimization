"""Smoke the tutorial scripts so they do not rot."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SCRIPTS = (
    "01_analyze_once.py",
    "02_conduction_tree.py",
    "03_convection_darcy.py",
    "04_conjugate_stokes.py",
    "05_custom_regions.py",
    "06_mms_check.py",
    "run_all.py",
    "gallery.py",
    "publish_figures.py",
)


@pytest.mark.parametrize("name", SCRIPTS)
def test_tutorial_help(name):
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / name), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout.lower() or "quick" in proc.stdout.lower()


def test_analyze_once_quick(tmp_path):
    out = tmp_path / "01"
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / "01_analyze_once.py"), "--quick", "--outdir", str(out)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "MPLBACKEND": "Agg"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "solid cooler than fluid" in proc.stdout
    assert (out / "gray.png").is_file()


def test_mms_check_quick(tmp_path):
    out = tmp_path / "06"
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / "06_mms_check.py"), "--quick", "--outdir", str(out)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "MPLBACKEND": "Agg"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "mms_report.txt").is_file()
    text = (out / "mms_report.txt").read_text()
    assert "energy_order=" in text
    assert "helmholtz=" in text
    assert "darcy_p=" in text
