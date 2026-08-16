# Reference snapshots

These PNGs are committed so GitHub shows a design without running the
80×80 gallery. They come from `python examples/publish_figures.py`
(medium-short meshes, not the full continuation).

| File | Case | Expected look |
|---|---|---|
| `analyze_gray.png` | `analyze()` on a uniform field | Smooth T into the bottom sink |
| `conduction_tree.png` | volume-to-point, `symmetry=x` | Trunk into the centered sink; left–right mirror |
| `convection_darcy.png` | centerline Darcy ports, `symmetry=y` | Through-channel; top–bottom mirror, not left–right |
| `custom_faces.png` | hot top / cold bottom, `symmetry=x` | Conducting bridges; left–right mirror |

The full gallery (`python examples/gallery.py`) writes under `outputs/`
and is gitignored. Archived short-run numbers are in
`examples/reference.json`.
