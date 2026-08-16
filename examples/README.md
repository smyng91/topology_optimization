# Tutorials and problem configs

Named research cases (geometry, ports, hot/cold patches, and the
solver caps those cases need) are defined in [`problems.py`](problems.py).
The `topoopt` package does not pick BCs from the heat mode. Library
defaults for every `ColdPlateParams` field are in
[docs/model.md](../docs/model.md) §8.

JSON wrappers live in [`configs/`](configs/). Short-run reference
numbers used by CI are in [`reference.json`](reference.json).

```bash
python -m topoopt 2d --config examples.problems:conduction_tree
python -m topoopt 2d --config examples/configs/convection_darcy.json
python examples/gallery.py
python examples/publish_figures.py    # committed snapshots in docs/figures/
```

Symmetric factories set `symmetry` (`x` = left–right, `y` = top–bottom)
so random init cannot skew a mirror problem. `custom_boxes` is not
symmetric and leaves that field empty.

## Named factories

Each factory is `name(nx=40, ny=40, **kwargs)`. Extra keywords override
the case defaults (mesh, `vol_frac`, solver iters, …). Fields not listed
keep the `params2d` library default (`L=(1,1)`, `q_vol=1` unless `--hot`
/ `hot_specs` is set, `k_fluid=1`, `k_solid=100`, `eta=0.5`,
`solver_tol=1e-7`, …).

| Factory | heat | flow | v* | rmin | Pe | ports / patches | symmetry | J | Solver caps |
|---|---|---|---|---|---|---|---|---|---|
| `conduction_tree` | conduction | none | 0.30 | 1.5 | 0 | cold `face:bottom:frac=0.08` | x | −mean(T) | heat 400, filter 200 |
| `convection_darcy` | convection | Darcy | 0.45 | 2.0 | 40 | `port_frac=0.5`, no hot/cold | y | −mean(T) | flow 280, heat 400, filter 200 |
| `conjugate_darcy` | both | Darcy | 0.45 | 2.0 | 40 | same ports as convection | y | −mean(T) | same as convection |
| `conjugate_stokes` | both | Stokes | 0.45 | 2.0 | 40 | same ports; `stokes_dp=20` | y | −mean(T) | flow 80, Uzawa 80, Schur 200, heat 320, filter 120 |
| `custom_faces` | conduction | none | 0.40 | 2.0 | 0 | hot `face:top:frac=0.5`, cold `face:bottom:frac=0.5`; `q=0` | x | Q_hot | heat 400, filter 200 |
| `custom_boxes` | conduction | none | 0.40 | 2.0 | 0 | hot `box:0.2,0.8,0.0,0.18`; cold `box:0.0,0.18,0.25,0.75` and `face:left`; `q=0` | none | Q_hot | heat 400, filter 200 |

`CENTERLINE_PORT = 0.5` and `TREE_SINK = ("face:bottom:frac=0.08",)` are
the shared constants. Flow factories do **not** add a conduction sink.

## JSON / YAML configs

`--config path.json` (or `.yaml` with `[yaml]`) accepts any
`ColdPlateParams` field plus:

| Key | Meaning |
|---|---|
| `factory` | Optional `module:func` to start from (then other keys override it) |
| `nx`, `ny` | Mesh; used when `n` is omitted |
| `lx`, `ly` | Box size; used when `L` is omitted |
| `n`, `L` | `(nx, ny)` and `(Lx, Ly)` lists |
| `hot_specs`, `cold_specs`, `symmetry` | Lists of strings (`["x"]`, `["face:bottom:frac=0.08"]`) |
| `comment` | Ignored |

Examples: [`configs/conduction_tree.json`](configs/conduction_tree.json)
(factory + mesh) and [`configs/standalone_box.json`](configs/standalone_box.json)
(no factory; a full conduction box).

## Tutorial scripts

Run from the **repo root** after `pip install -e ".[cpu,dev]"` on Python 3.14.

```bash
python examples/01_analyze_once.py
python examples/01_analyze_once.py --quick          # coarse mesh / few iters
python examples/run_all.py --quick                  # smoke every tutorial
```

Shared flags (every numbered script, via `_common.add_run_args`):
`--quick`, `--outdir` (default `outputs/0N_…`), `--seed` (default 0).
`--quick` is a sanity check, not a publishable design.

Each optimizing script writes PNG / VTK / `history.json` / `run.json` /
`state_*.npz` under `--outdir`. Optimizer knobs they pass to
`optimize` are `lr`, `beta_max`, `seed`, and `n_iters` (not MMA).
`stall_iters` stays at the library default (8). `analyze` / `optimize`
aux keys: `V`, `T_mean`, `T_max`, `speed_max`, `u_in`, `u_out`,
`mass_err`, `energy_rms`, `div_rms`, `stokes_rel`, `gray`.

| Script | Factory | Full mesh / iters | `--quick` | Other knobs | What it teaches |
|---|---|---|---|---|---|
| [`01_analyze_once.py`](01_analyze_once.py) | `conduction_tree` | 32×32, no TO; `β=4` | 16×16 | `heat_iters=300`, `filter_iters=40`; γ ∈ {1, 0, 0.45} | `analyze()` only |
| [`02_conduction_tree.py`](02_conduction_tree.py) | `conduction_tree` | 100×100, 100 iters, `β_max=16`, `lr=0.2` | 16×16, 8 iters, `β_max=8` | `heat_iters=300`, `filter_iters=80`; `--wide-sink` → `frac=0.5` | why the sink is 8% and `vol=0.30` |
| [`03_convection_darcy.py`](03_convection_darcy.py) | `convection_darcy` | 32×32, 50 iters, `β_max=16`, `lr=0.2` | 16×16, 8 iters, `β_max=8` | `flow_iters=200`, `heat_iters=300`, `filter_iters=60` | centerline Darcy ports, channel seed |
| [`04_conjugate_stokes.py`](04_conjugate_stokes.py) | `conjugate_stokes` | 24×24, 40 iters, `β_max=16`, `lr=0.16` | 16×16, 6 iters, `β_max=8` | `uzawa_iters=40`, `heat_iters=280`, `filter_iters=60` (Schur CG stays 200) | Stokes seed + port pin |
| [`05_custom_regions.py`](05_custom_regions.py) | `custom_faces` or `--boxes` → `custom_boxes` | 32×32, 50 iters, `β_max=16`, `lr=0.2` | 16×16, 8 iters, `β_max=8` | `heat_iters=300`, `filter_iters=60` | `J = Q_hot` |
| [`06_mms_check.py`](06_mms_check.py) | `params2d` (not a named factory) | energy n=8 and 16; Helmholtz n=16 `rmin=2`; Darcy 16×16 `port_frac=1`; Stokes n=12 `port_frac=1`, `div_eps=1e-12` | skips Stokes | energy `heat_iters=800`; Darcy `flow_iters=400`; Stokes Uzawa 40 / Schur 200 | discrete-operator checks |

[`run_all.py`](run_all.py) runs 01–06 in order. `--quick` is forwarded.
`--only 01 06` filters by basename.

## Gallery and published figures

[`gallery.py`](gallery.py) is a finer, longer sweep — not a tutorial.
`optimize` uses `lr=0.12`, `β_max=32`, `seed=0`.

| Case | Full | `--quick` |
|---|---|---|
| `2d_conduction` | 80×80, 200 iters, `rmin=1.5`, heat 400, filter 200 | 16×16, 8 iters |
| `2d_convection_darcy` | 80×80, 150 iters, flow 280 | 16×16, 8 iters |
| `2d_both_darcy` | 80×80, 150 iters, flow 280 | 16×16, 8 iters |
| `2d_both_stokes` | 48×48, 100 iters, filter 120, heat 320, flow 250, Uzawa 250 | 12×12, 4 iters, cheap Krylov |
| `2d_custom_faces` | 80×80, 180 iters | 16×16, 8 iters |
| `2d_custom_boxes` | 80×80, 180 iters | 16×16, 8 iters |

Outputs go under `outputs/` (gitignored).

[`publish_figures.py`](publish_figures.py) writes the committed PNGs in
`docs/figures/`. All use `lr=0.2`, `β_max=8`, `seed=0`:

| File | Factory | Mesh / iters | Solver caps |
|---|---|---|---|
| `analyze_gray.png` | `conduction_tree` | 24×24, `analyze` only, `β=4`, γ=0.45 | heat 250, filter 60 |
| `conduction_tree.png` | `conduction_tree` | 32×32, 20 iters | heat 280, filter 80 |
| `convection_darcy.png` | `convection_darcy` | 24×24, 12 iters | flow 160, heat 220, filter 50 |
| `custom_faces.png` | `custom_faces` | 24×24, 12 iters | heat 220, filter 50 |

## Reading order

1. **01** — fields and diagnostics, no optimizer.
2. **02** — why the sink is 8% and `vol = 0.30`.
3. **03** — one left-centerline inlet / right-centerline outlet, channel seed, `J = -mean(T)`.
4. **04** — Stokes notes (`stokes_dp`, Schur CG, port pin).
5. **05** — Dirichlet patches change both the BCs and the objective.
6. **06** — how we know the discrete operators are consistent.

The model write-up is [docs/model.md](../docs/model.md).
