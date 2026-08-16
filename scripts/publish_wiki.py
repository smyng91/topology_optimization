"""Build GitHub-wiki pages from README / docs and optionally push them."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/smyng91/topology_optimization"
RAW = f"{REPO}/raw/main"
BLOB = f"{REPO}/blob/main"
WIKI_GIT = "git@github.com:smyng91/topology_optimization.wiki.git"

LINK_MAP = (
    ("[docs/model.md](docs/model.md)", "[Model](Model)"),
    ("[docs/model.md](../docs/model.md)", "[Model](Model)"),
    ("[examples/README.md](examples/README.md)", "[Examples](Examples)"),
    ("[`docs/figures/`](docs/figures/)", "[Figures](Figures)"),
    ("[`docs/figures/`](../docs/figures/)", "[Figures](Figures)"),
    ("[docs/figures/README.md](docs/figures/README.md)", "[Figures](Figures)"),
    ("](docs/model.md)", "](Model)"),
    ("](../docs/model.md)", "](Model)"),
    ("](examples/README.md)", "](Examples)"),
    ("](../docs/figures/)", "](Figures)"),
    ("](docs/figures/)", "](Figures)"),
    ("](docs/figures/README.md)", "](Figures)"),
    ("](../model.md)", "](Model)"),
    ("](problems.py)", f"]({BLOB}/examples/problems.py)"),
    ("](configs/)", f"]({BLOB}/examples/configs)"),
    ("](reference.json)", f"]({BLOB}/examples/reference.json)"),
    ("](01_analyze_once.py)", f"]({BLOB}/examples/01_analyze_once.py)"),
    ("](02_conduction_tree.py)", f"]({BLOB}/examples/02_conduction_tree.py)"),
    ("](03_convection_darcy.py)", f"]({BLOB}/examples/03_convection_darcy.py)"),
    ("](04_conjugate_stokes.py)", f"]({BLOB}/examples/04_conjugate_stokes.py)"),
    ("](05_custom_regions.py)", f"]({BLOB}/examples/05_custom_regions.py)"),
    ("](06_mms_check.py)", f"]({BLOB}/examples/06_mms_check.py)"),
    ("](gallery.py)", f"]({BLOB}/examples/gallery.py)"),
    ("](publish_figures.py)", f"]({BLOB}/examples/publish_figures.py)"),
    ("](configs/conduction_tree.json)", f"]({BLOB}/examples/configs/conduction_tree.json)"),
    ("](configs/localized_source.json)", f"]({BLOB}/examples/configs/localized_source.json)"),
    ("](configs/standalone_box.json)", f"]({BLOB}/examples/configs/standalone_box.json)"),
    ("](requirements-lock.txt)", f"]({BLOB}/requirements-lock.txt)"),
    ("](examples/problems.py)", f"]({BLOB}/examples/problems.py)"),
    ("](examples/reference.json)", f"]({BLOB}/examples/reference.json)"),
    ("](examples/configs/localized_source.json)", f"]({BLOB}/examples/configs/localized_source.json)"),
)


def _rewrite(text: str) -> str:
    for old, new in LINK_MAP:
        text = text.replace(old, new)
    text = text.replace(
        "https://github.com/smyng91/topology_optimization/wiki",
        "https://github.com/smyng91/topology_optimization/wiki",
    )
    return text


def build(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    pages = {
        "Home.md": ROOT / "README.md",
        "Model.md": ROOT / "docs" / "model.md",
        "Examples.md": ROOT / "examples" / "README.md",
        "Figures.md": ROOT / "docs" / "figures" / "README.md",
    }
    for name, src in pages.items():
        (dest / name).write_text(_rewrite(src.read_text()))
    figures = ROOT / "docs" / "figures"
    extra = "\n".join(
        f"![{p.stem}]({RAW}/docs/figures/{p.name})"
        for p in sorted(figures.glob("*.png"))
    )
    fig = dest / "Figures.md"
    fig.write_text(fig.read_text().rstrip() + "\n\n## Images\n\n" + extra + "\n")
    (dest / "_Sidebar.md").write_text(
        "**topoopt**\n\n"
        "* [Home](Home)\n"
        "* [Model](Model)\n"
        "* [Examples](Examples)\n"
        "* [Figures](Figures)\n"
        f"\nSource: [{REPO}]({REPO})\n"
    )
    (dest / "_Footer.md").write_text(
        f"Synced from [`main`]({REPO}). "
        f"Edit the Markdown in the repo, not only the wiki.\n"
    )


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, text=True)


def _init_wiki(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    _git(["init", "-b", "master"], dest)
    _git(["remote", "add", "origin", WIKI_GIT], dest)


def push(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    cloned = subprocess.run(["git", "clone", WIKI_GIT, str(dest)])
    if cloned.returncode != 0:
        print("wiki clone empty; initializing a new wiki repo")
        _init_wiki(dest)
    build(dest)
    _git(["add", "-A"], dest)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=dest)
    if staged.returncode == 0:
        print("wiki already up to date")
        return
    _git(["commit", "-m", "Sync docs from main (GitHub-safe math)."], dest)
    pushed = subprocess.run(["git", "push", "-u", "origin", "HEAD"], cwd=dest)
    if pushed.returncode != 0:
        raise SystemExit(
            "Wiki git repo does not exist yet. While signed in as the owner, "
            "open https://github.com/smyng91/topology_optimization/wiki "
            "and create the first page (title Home, any text), then rerun "
            "this script with --push."
        )
    print(f"pushed {WIKI_GIT}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=Path, default=ROOT / ".wiki-build")
    p.add_argument("--push", action="store_true")
    args = p.parse_args()
    if args.push:
        push(args.dest)
    else:
        build(args.dest)
        print(f"wrote {args.dest}")
