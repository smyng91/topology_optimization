# topoopt

Density-based topology optimization of a 2-D heated box in JAX.
Solid ($`\gamma=1`$) conducts and blocks flow; fluid ($`\gamma=0`$) is permeable.
Sensitivities are discrete global adjoints. **2-D only.**

The discrete model is in [docs/model.md](docs/model.md)
([wiki](https://github.com/smyng91/topology_optimization/wiki/Model)).
Named cases live in [`examples/problems.py`](examples/problems.py).
Rendered docs: [GitHub wiki](https://github.com/smyng91/topology_optimization/wiki).

```math
-\nabla\cdot(k\nabla T)+\mathrm{Pe}\,\mathbf{u}\cdot\nabla T=q,
\qquad
\mathrm{mean}(\bar{\gamma})=v^{*},
\qquad
J=
\begin{cases}
-\mathrm{mean}(T) & \text{volume source on},\\
Q_{\mathrm{hot}} & \text{Dirichlet }T\text{ only}.
\end{cases}
```

Flow is Stokes–Brinkman (default) or Darcy. Heat mode `conduction` /
`convection` / `both` selects which terms are active — not the BCs.

## Quickstart

Python 3.14+. From the repo root:

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -e ".[cpu,dev]"          # [cuda] instead of [cpu] on NVIDIA
MPLBACKEND=Agg python -m pytest tests -q
python examples/02_conduction_tree.py --quick
```

`[yaml]` adds PyYAML. `jax_enable_x64` is on at import. Pins:
[`requirements-lock.txt`](requirements-lock.txt).

```bash
python -m topoopt 2d --config examples.problems:conduction_tree --iters 80
python -m topoopt 2d --heat conduction --q-region box:0.3,0.7,0.70,1.0 --cold face:bottom:frac=0.08
python -m topoopt 2d --heat conduction --hot face:top --cold face:bottom:frac=0.5
```

`--quick` on tutorials is a smoke mesh, not a publishable design. Outputs
go under `outputs/` (gitignored).

## Variables

| Symbol | Code | Meaning |
|---|---|---|
| $`\gamma`$ | `gamma_raw` | raw design in $`[0,1]`$ |
| $`\tilde{\gamma}`$ | filtered | Helmholtz-filtered design |
| $`\bar{\gamma}`$ | `phys` | tanh-projected density used in the PDEs |
| $`\beta,\eta`$ | `beta`, `eta` | projection sharpness and threshold ($`\eta=0.5`$) |
| $`r=r_{\min}\min(\Delta x,\Delta y)`$ | `rmin` | filter radius (`rmin` in cells) |
| $`k,k_f,k_s`$ | `k`, `k_fluid`, `k_solid` | conductivity; $`k=\mathrm{RAMP}(\bar{\gamma};k_f,k_s,q_k)`$ except convection ($`k\equiv k_f`$) |
| $`\alpha`$ | Brinkman | solid drag (Borrvall–Petersson) |
| $`\kappa`$ | Darcy | permeability $`\mathrm{RAMP}(\bar{\gamma};\kappa_{\max},\kappa_{\min},q_\kappa)`$ |
| $`q`$ | `q_vol` / `q_specs` | volumetric heat: uniform, or only on `--q-region` |
| $`\mathrm{Pe}`$ | `pe` | Péclet; $`0`$ in conduction |
| $`T,\mathbf{u},p`$ | `T`, `face_vel`, `p` | temperature, MAC velocity, pressure |
| $`v^{*}`$ | `vol_frac` | target $`\mathrm{mean}(\bar{\gamma})`$ |
| $`J`$ | `J` | $`-\mathrm{mean}(T)`$ if $`q\neq 0`$; else heat leaving `--hot` |
| $`\ell=\ell_0/\sqrt{\max(\beta,1)}`$ | `lr` | move limit; `--lr` is $`\ell_0`$ at $`\beta=1`$ |
| $`\varepsilon`$ | `div_eps` | Stokes continuity regularizer $`\varepsilon p`$ |

`--hot` / `--cold` prescribe Dirichlet $`T`$ (and turn off *uniform* $`q`$).
`--q-region` generates heat on a face or box; $`T`$ still floats. Specs:
`face:bottom:frac=0.5`, `face:top:frac=0.4:center=0.3`,
`box:xmin,xmax,ymin,ymax`. A face $`q`$ spec heats the adjacent cell layer.
Overlays: crimson / sky-blue $`=T`$, orange $`=q`$.

The library does **not** infer BCs from `--heat`. Geometry comes from
`--config`, JSON, or the flags below. Full field list: [docs/model.md](docs/model.md) §8.

| Flag | Field | Default |
|---|---|---|
| `--config` | factory or JSON/YAML | generic `params2d` box |
| `--nx`, `--ny` | $`n`$ | $`40,40`$ |
| `--heat` | `heat_mode` | `both` |
| `--flow {stokes,darcy}` | `flow_model` | `stokes` |
| `--vol` | $`v^{*}`$ | $`0.45`$ |
| `--pe` | $`\mathrm{Pe}`$ | $`40`$ |
| `--q` | $`q_{\mathrm{vol}}`$ | $`1`$ |
| `--k-ratio` | $`k_s`$ ($`k_f=1`$) | $`100`$ |
| `--rmin` | $`r_{\min}`$ (cells) | $`2.2`$ |
| `--port-frac` | port height | $`0.5`$ |
| `--hot`, `--cold` | Dirichlet $`T`$ | empty |
| `--q-region` | volumetric $`q`$ | empty (uniform $`q`$) |
| `--symmetry {x,y,x,y}` | design mirror | from factory, else none |
| `--iters` | design steps | $`80`$ |
| `--lr` | $`\ell_0`$ | $`0.2`$ |
| `--beta-max` | $`\beta_{\max}`$ | $`32`$ |
| `--mesh-schedule` | coarse $`\to`$ fine | unset |
| `--seed`, `--outdir` | init, artifacts | $`0`$, `outputs/2d` |

Other fields (`q_k`, `alpha_max`, `stokes_dp`, `heat_iters`, …) are set
on the factory, in JSON, or by `params2d(..., key=value)`.

## Tutorials

From the repo root after install. Details: [examples/README.md](examples/README.md).

```bash
python examples/01_analyze_once.py              # analyze() only
python examples/02_conduction_tree.py           # volume-to-point tree
python examples/03_convection_darcy.py          # Darcy ports
python examples/04_conjugate_stokes.py          # Stokes–Brinkman
python examples/05_custom_regions.py            # Dirichlet T; --source for q
python examples/06_mms_check.py                 # manufactured solutions
python examples/run_all.py --quick
```

| Script | Point |
|---|---|
| 01 | fields and diagnostics, no optimizer |
| 02 | small sink $`\Rightarrow`$ tree; $`v^{*}=0.30`$ |
| 03 | one left-centerline inlet / right-centerline outlet |
| 04 | pressure-driven Stokes, channel seed, port pin |
| 05 | `--hot`/`--cold` vs `--q-region` |
| 06 | discrete-operator MMS |

```bash
python -m topoopt 2d --config examples.problems:conduction_tree
python -m topoopt 2d --config examples/configs/localized_source.json
python examples/gallery.py                      # 80×80 sweep, not a tutorial
python examples/publish_figures.py              # docs/figures/ snapshots
```

Returned design is the best-$`J`$ iterate at the **highest $`\beta`$** that
ran (not the global max $`J`$ across continuation). Flow modes have no
extra cold patch; a sealed channel aborts.

## Validation

```bash
MPLBACKEND=Agg python -m pytest tests -q
python -m topoopt verify
python examples/06_mms_check.py
```

CI is the same pytest command on Python 3.14 (`[cpu,dev]`; no gallery).

| Check | What |
|---|---|
| Energy Poisson | $`T=\sin\pi x\sin\pi y`$, order $`\approx 2`$ |
| Advective energy | uniform $`\mathbf{u}`$, order $`\gtrsim 1`$ |
| Variable $`k`$ | discrete operator as manufactured source |
| Helmholtz | Neumann cosine; inverse recovers $`\gamma`$ to Krylov tol |
| Darcy | linear $`p`$ on a full-height port |
| Stokes | Poiseuille on a full-height $`\Delta p`$ channel |
| Adjoint | central FD on throughput and on `analyze` |
| Physics | solid cooler than fluid; localized $`q`$; Dirichlet $`T`$ |

Snapshots: [`docs/figures/`](docs/figures/). Short-run numbers:
[`examples/reference.json`](examples/reference.json).

## Stokes

Pressure-driven: $`p=\Delta p`$ (`stokes_dp`, default $`20`$) on the left
port, $`p=0`$ on the right. Forward: Uzawa warm start + CG on the
pressure Schur complement. Adjoint: residual discrete adjoint, not
unrolled Uzawa. A one-cell fluid pin on port *design* cells
(`keep_ports_open`) and a mid-height channel seed are required; they
are not PDE Dirichlet data.
