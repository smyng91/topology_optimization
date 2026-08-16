# Reference snapshots

Committed PNGs from `python examples/publish_figures.py` (short meshes,
not the $80\times 80$ gallery). All optimizing snapshots use
$\ell_0=0.2$, $\beta_{\max}=8$, `seed=0`. Raw $\gamma$ is an exact
mirror on the factory axis (`sym_err=0`); a PNG can look slightly
uneven from colormap interpolation.

| File | Factory | Run | Look |
|---|---|---|---|
| `analyze_gray.png` | `conduction_tree` $24\times 24$ | `analyze`($\gamma=0.45$, $\beta=4$); heat $250$, filter $60$ | $T$ into the bottom sink |
| `conduction_tree.png` | `conduction_tree` $32\times 32$ | $20$ it; heat $280$, filter $80$; `symmetry=x` | trunk into the sink |
| `convection_darcy.png` | `convection_darcy` $24\times 24$ | $12$ it; flow $160$, heat $220$, filter $50$; `symmetry=y` | through-channel |
| `custom_faces.png` | `custom_faces` $24\times 24$ | $12$ it; heat $220$, filter $50$; `symmetry=x` | conducting bridges |

Physics defaults: [docs/model.md](../model.md) §8.2. Gallery output is
gitignored under `outputs/`. CI numbers: `examples/reference.json`.
