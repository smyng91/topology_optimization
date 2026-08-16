#!/usr/bin/env python3
"""Run every tutorial in order (01–06).

``--quick`` is forwarded to each script. ``--only 01 06`` filters by
basename. Meshes and iteration counts are in ``examples/README.md``.

From the repo root::

    python examples/run_all.py --quick
    python examples/run_all.py
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = (
    "01_analyze_once.py",
    "02_conduction_tree.py",
    "03_convection_darcy.py",
    "04_conjugate_stokes.py",
    "05_custom_regions.py",
    "06_mms_check.py",
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="Forward --quick to every tutorial")
    p.add_argument("--only", nargs="*", help="Basename filters, e.g. 01 06")
    args = p.parse_args(argv)

    extra = ["--quick"] if args.quick else []
    for name in SCRIPTS:
        if args.only and not any(tok in name for tok in args.only):
            continue
        print(f"\n======== {name} ========\n")
        sys.argv = [str(HERE / name), *extra]
        try:
            runpy.run_path(str(HERE / name), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
