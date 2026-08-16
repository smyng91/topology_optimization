"""``python -m topoopt 2d`` / ``verify``. Galleries live under ``examples/``."""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: python -m topoopt {2d|verify} [options]\n"
            "       python examples/gallery.py\n"
            "       python -m topoopt 2d --config examples.problems:conduction_tree"
        )
        return 0
    dim, rest = argv[0], argv[1:]
    if dim == "2d":
        from topoopt.run2d import main as run

        run(rest)
        return 0
    if dim == "verify":
        from topoopt.verify import main as run

        run(rest)
        return 0
    if dim == "examples":
        gallery = Path(__file__).resolve().parents[1] / "examples" / "gallery.py"
        if not gallery.is_file():
            raise SystemExit("examples/gallery.py not found; run from the repo checkout")
        import runpy

        sys.argv = [str(gallery), *rest]
        runpy.run_path(str(gallery), run_name="__main__")
        return 0
    raise SystemExit(f"unknown command {dim!r}; expected 2d, verify, or examples")


if __name__ == "__main__":
    raise SystemExit(main())
