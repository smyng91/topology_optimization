# Reference snapshots

These PNGs are committed so GitHub shows a design without running the
80×80 gallery. They come from `python examples/publish_figures.py`
(medium-short meshes, not the full continuation). All optimizing
snapshots use `lr=0.2`, `β_max=8`, `seed=0`. The raw design is an
exact mirror on the factory axis (`sym_err = 0`); a PNG can still look
slightly uneven because of colormap interpolation.

| File | Factory | Run | Expected look |
|---|---|---|---|
| `analyze_gray.png` | `conduction_tree` 24×24 | `analyze(γ=0.45, β=4)`; heat 250, filter 60 | Smooth T into the bottom sink |
| `conduction_tree.png` | `conduction_tree` 32×32 | 20 iters; heat 280, filter 80; `symmetry=x` | Trunk into the centered sink; left–right mirror |
| `convection_darcy.png` | `convection_darcy` 24×24 | 12 iters; flow 160, heat 220, filter 50; `symmetry=y` | Through-channel; top–bottom mirror, not left–right |
| `custom_faces.png` | `custom_faces` 24×24 | 12 iters; heat 220, filter 50; `symmetry=x` | Conducting bridges; left–right mirror |

Factory physics (volume, `rmin`, patches) are the named-case defaults
in `examples/problems.py` / [docs/model.md](../model.md) §8.2. The full
gallery (`python examples/gallery.py`) writes under `outputs/` and is
gitignored. Archived short-run numbers are in `examples/reference.json`.
