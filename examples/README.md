# Tutorials and configs

Named cases (geometry, ports, Dirichlet $`T`$, volumetric $`q`$, solver caps)
are in [`problems.py`](problems.py). The package does not infer BCs from
`--heat`. Library defaults: [docs/model.md](../docs/model.md) §8.

```bash
python -m topoopt 2d --config examples.problems:conduction_tree
python -m topoopt 2d --config examples/configs/localized_source.json
python examples/run_all.py --quick
```

JSON: [`configs/`](configs/). CI numbers: [`reference.json`](reference.json).
Snapshots: [`docs/figures/`](../docs/figures/).
Wiki: https://github.com/smyng91/topology_optimization/wiki/Examples

`symmetry`: `x` = left–right, `y` = top–bottom. `custom_boxes` is none.
Flow factories do **not** add a conduction sink.

## Factories

`name(nx=40, ny=40, **kwargs)`. Unlisted fields keep `params2d` defaults
($`L=(1,1)`$, $`q_{\mathrm{vol}}=1`$ unless `hot_specs` is set without
`q_specs`, $`k_f=1`$, $`k_s=100`$, $`\eta=0.5`$, …).

Constants: `CENTERLINE_PORT=0.5`,
`TREE_SINK=("face:bottom:frac=0.08",)`,
`SOURCE_BOX=("box:0.3,0.7,0.70,1.0",)`.

| Factory | heat | flow | $`v^{*}`$ | $`r_{\min}`$ | $`\mathrm{Pe}`$ | Regions | sym | $`J`$ | Caps |
|---|---|---|---|---|---|---|---|---|---|
| `conduction_tree` | conduction | none | $`0.30`$ | $`1.5`$ | $`0`$ | cold `frac=0.08` | `x` | $`-\mathrm{mean}(T)`$ | heat $`400`$, filter $`200`$ |
| `convection_darcy` | convection | Darcy | $`0.45`$ | $`2.0`$ | $`40`$ | ports $`0.5`$ | `y` | $`-\mathrm{mean}(T)`$ | flow $`280`$, heat $`400`$, filter $`200`$ |
| `conjugate_darcy` | both | Darcy | $`0.45`$ | $`2.0`$ | $`40`$ | same ports | `y` | $`-\mathrm{mean}(T)`$ | same |
| `conjugate_stokes` | both | Stokes | $`0.45`$ | $`2.0`$ | $`40`$ | same; $`\Delta p=20`$ | `y` | $`-\mathrm{mean}(T)`$ | flow $`80`$, Uzawa $`80`$, Schur $`200`$, heat $`320`$, filter $`120`$ |
| `custom_faces` | conduction | none | $`0.40`$ | $`2.0`$ | $`0`$ | hot/cold faces `frac=0.5`; $`q=0`$ | `x` | $`Q_{\mathrm{hot}}`$ | heat $`400`$, filter $`200`$ |
| `custom_boxes` | conduction | none | $`0.40`$ | $`2.0`$ | $`0`$ | hot `box:0.2,0.8,0.0,0.18`; cold box + `face:left`; $`q=0`$ | none | $`Q_{\mathrm{hot}}`$ | heat $`400`$, filter $`200`$ |
| `localized_source` | conduction | none | $`0.30`$ | $`1.5`$ | $`0`$ | `q_specs=SOURCE_BOX`; cold `frac=0.08` | `x` | $`-\mathrm{mean}(T)`$ | heat $`400`$, filter $`200`$ |

## JSON / YAML

`--config path.json` (`.yaml` needs `[yaml]`) accepts any
`ColdPlateParams` field plus:

| Key | Meaning |
|---|---|
| `factory` | optional `module:func`; other keys override it |
| `nx`, `ny` / `n` | mesh |
| `lx`, `ly` / `L` | box size |
| `hot_specs`, `cold_specs`, `q_specs`, `symmetry` | string lists |
| `comment` | ignored |

Examples: [`configs/conduction_tree.json`](configs/conduction_tree.json),
[`configs/localized_source.json`](configs/localized_source.json),
[`configs/standalone_box.json`](configs/standalone_box.json).

## Tutorials

Install `pip install -e ".[cpu,dev]"` (Python 3.14), run from the repo
root. Shared flags: `--quick`, `--outdir` (`outputs/0N_…`), `--seed`
($`0`$). `--quick` is a smoke mesh.

Each optimizing script writes PNG / VTK / `history.json` / `run.json` /
`state_*.npz`. They pass `lr`, `beta_max`, `seed`, `n_iters` to
`optimize` (`stall_iters=8`). Aux: $`V`$, $`T_{\mathrm{mean}}`$,
$`T_{\mathrm{max}}`$, `speed_max`, $`u_{\mathrm{in}}`$, $`u_{\mathrm{out}}`$,
`mass_err`, `energy_rms`, `div_rms`, `stokes_rel`, `gray`.

```bash
python examples/01_analyze_once.py --quick
python examples/run_all.py --quick              # 01–06; --only 01 06
```

| Script | Factory | Full | `--quick` | Extra | Point |
|---|---|---|---|---|---|
| [`01`](01_analyze_once.py) | `conduction_tree` | $`32\times 32`$, $`\beta=4`$, no TO | $`16\times 16`$ | heat $`300`$, filter $`40`$; $`\gamma\in\{1,0,0.45\}`$ | `analyze()` |
| [`02`](02_conduction_tree.py) | `conduction_tree` | $`100\times 100`$, $`100`$ it, $`\beta_{\max}=16`$, $`\ell_0=0.2`$ | $`16\times 16`$, $`8`$ it, $`\beta_{\max}=8`$ | heat $`300`$, filter $`80`$; `--wide-sink` | small sink $`\Rightarrow`$ tree |
| [`03`](03_convection_darcy.py) | `convection_darcy` | $`32\times 32`$, $`50`$ it, $`\beta_{\max}=16`$ | $`16\times 16`$, $`8`$ it | flow $`200`$, heat $`300`$, filter $`60`$ | Darcy ports, channel seed |
| [`04`](04_conjugate_stokes.py) | `conjugate_stokes` | $`24\times 24`$, $`40`$ it, $`\ell_0=0.16`$ | $`16\times 16`$, $`6`$ it | Uzawa $`40`$, heat $`280`$, filter $`60`$ | Stokes seed + port pin |
| [`05`](05_custom_regions.py) | `custom_faces`; `--boxes`; `--source` | $`32\times 32`$, $`50`$ it | $`16\times 16`$, $`8`$ it | heat $`300`$, filter $`60`$ | Dirichlet $`T`$ vs $`q`$ |
| [`06`](06_mms_check.py) | `params2d` | energy $`n=8,16`$; Helmholtz $`n=16`$; Darcy $`16\times 16`$; Stokes $`n=12`$ | skips Stokes | see script | MMS |

Reading order: **01** fields $`\to`$ **02** tree $`\to`$ **03** Darcy $`\to`$
**04** Stokes $`\to`$ **05** regions $`\to`$ **06** MMS.

## Gallery and figures

[`gallery.py`](gallery.py) is a long sweep ($`\ell_0=0.12`$,
$`\beta_{\max}=32`$, `seed=0`), not a tutorial.

| Case | Full | `--quick` |
|---|---|---|
| `2d_conduction` | $`80\times 80`$, $`200`$ it, $`r_{\min}=1.5`$ | $`16\times 16`$, $`8`$ it |
| `2d_convection_darcy` | $`80\times 80`$, $`150`$ it, flow $`280`$ | $`16\times 16`$, $`8`$ it |
| `2d_both_darcy` | $`80\times 80`$, $`150`$ it | $`16\times 16`$, $`8`$ it |
| `2d_both_stokes` | $`48\times 48`$, $`100`$ it, Uzawa $`250`$ | $`12\times 12`$, $`4`$ it |
| `2d_custom_faces` | $`80\times 80`$, $`180`$ it | $`16\times 16`$, $`8`$ it |
| `2d_custom_boxes` | $`80\times 80`$, $`180`$ it | $`16\times 16`$, $`8`$ it |

[`publish_figures.py`](publish_figures.py) writes
[`docs/figures/`](../docs/figures/) ($`\ell_0=0.2`$, $`\beta_{\max}=8`$,
`seed=0`):

| File | Factory | Run |
|---|---|---|
| `analyze_gray.png` | `conduction_tree` | $`24\times 24`$, `analyze` only, $`\beta=4`$, $`\gamma=0.45`$ |
| `conduction_tree.png` | `conduction_tree` | $`32\times 32`$, $`20`$ it |
| `convection_darcy.png` | `convection_darcy` | $`24\times 24`$, $`12`$ it |
| `custom_faces.png` | `custom_faces` | $`24\times 24`$, $`12`$ it |
