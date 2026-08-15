"""``python -m topoopt 2d`` / ``verify`` / ``examples``."""

from __future__ import annotations

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: python -m topoopt {2d|verify|examples} [options]")
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
        from topoopt.examples import main as run

        run(rest)
        return 0
    raise SystemExit(f"unknown command {dim!r}; expected 2d, verify, or examples")


if __name__ == "__main__":
    raise SystemExit(main())
