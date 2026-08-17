import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_manuscript_integrity.py"
if not SCRIPT.is_file():
    pytest.skip("manuscript check script is local-only", allow_module_level=True)

SPEC = importlib.util.spec_from_file_location(
    "check_manuscript_integrity",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_tex_number_scientific_and_plain():
    assert MODULE.parse_tex_number(r"3.80\times 10^{-6}") == 3.8e-6
    assert MODULE.parse_tex_number("-0.0206") == -0.0206
    assert MODULE.parse_tex_number("stall") == "stall"
    defined = MODULE.defined_macros(r"\newcommand{\HelmholtzErrFine}{3.80\times 10^{-6}}")
    assert defined["HelmholtzErrFine"] == 3.8e-6


def test_cite_before_definition_accepts_forward_refs():
    text = r"""
    See \Cref{fig:demo,tab:demo}.
    \begin{figure}\caption{x}\label{fig:demo}\end{figure}
    \begin{table}\caption{y}\label{tab:demo}\end{table}
    """
    assert MODULE.cite_before_definition(text) == []


def test_cite_before_definition_flags_late_or_missing_cites():
    late = r"""
    \begin{figure}\caption{x}\label{fig:late}\end{figure}
    Later \Cref{fig:late}.
    """
    missing = r"""
    \begin{figure}\caption{x}\label{fig:orphan}\end{figure}
    """
    assert "fig:late is cited after it is defined" in MODULE.cite_before_definition(late)
    assert "fig:orphan is never cited" in MODULE.cite_before_definition(missing)


def test_claim_ledger_matches_results_and_numbers():
    numbers_path = ROOT / "paper" / "numbers.tex"
    results_path = ROOT / "paper" / "results.json"
    ledger_path = ROOT / "paper" / "claim_ledger.json"
    if not (numbers_path.is_file() and results_path.is_file() and ledger_path.is_file()):
        pytest.skip("manuscript artifacts are local-only")
    numbers = numbers_path.read_text(encoding="utf-8")
    results = MODULE.json.loads(results_path.read_text(encoding="utf-8"))
    ledger = MODULE.json.loads(ledger_path.read_text(encoding="utf-8"))
    assert MODULE.check_claim_ledger(results, numbers, ledger) == []
