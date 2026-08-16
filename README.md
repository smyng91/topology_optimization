# topoopt

JAX package for density-based topology optimization of a 2-D heated square box.
The design field `γ` is solid (`γ = 1`, conducting, impermeable) or fluid
(`γ = 0`). Heat is a uniform volumetric source unless `--hot` sets Dirichlet
heat-source patches. Sensitivities are discrete global adjoints.

The model (physics, discretization, adjoint, optimizer) is in
[docs/model.md](docs/model.md).

## Research v0.4

Each iteration logs energy residual RMS, `div_rms`, port mass error, and
grayness (also stored in `analyze` aux). The move limit decays as
`lr / sqrt(β)` and the run keeps the best-`J` design. A run writes
`history.json`, `run.json`, `state_best.npz`, and `state_final.npz`.
At `β_max` the loop stops if `J` has not improved for `stall_iters`
(default 8). A blocked flow solve that runs `T` away **aborts** after
writing the best-`J` checkpoint — it does not add a conduction sink.

Symmetric problems stay symmetric. The historical breaker was
asymmetric random init noise; `params.symmetry` (`x` and/or `y`)
mirrors that noise and every accepted design. Named cases set this
themselves (`conduction_tree` / `custom_faces` are left–right;
centerline-port flow cases are top–bottom).

Stokes is Uzawa-warm-started then corrected with CG on the pressure
Schur complement so the discrete saddle-point residual is small. Flow
modes have a single left-centerline inlet and a single right-centerline
outlet — no extra inlets, outlets, or cold patches.
`tests/test_mms.py` covers energy Poisson, advective energy, variable-`k`
consistency, Helmholtz-filter order, Darcy linear pressure, and
Stokes–Poiseuille. Snapshots live in [`docs/figures/`](docs/figures/).

## Install

Python 3.14+. From the repo root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e ".[cpu,dev]"
```

`[cpu]` pulls the CPU JAX extra; `[dev]` adds `pytest`. For an NVIDIA
GPU use `[cuda]` (or `[gpu]`) **instead of** `[cpu]`. YAML configs need
`[yaml]`. `jax_enable_x64` is turned on when the package is imported.
Pinned versions from the development venv are in
[`requirements-lock.txt`](requirements-lock.txt).

## Problem configs

Geometry, ports, and hot/cold patches are **not** hardcoded in the
package. Named cases live in [`examples/problems.py`](examples/problems.py):

```bash
python -m topoopt 2d --config examples.problems:conduction_tree
python -m topoopt 2d --config examples.problems:convection_darcy --nx 64 --ny 64
python examples/gallery.py
```

`params2d()` builds a generic parameter object; BCs are whatever you pass.
To add a new case, write a factory in `examples/problems.py` (or any
`module:function` / `path.py:function`) or a JSON file:

```bash
python -m topoopt 2d --config examples/configs/conduction_tree.json
python -m topoopt 2d --config examples.problems:conduction_tree --mesh-schedule 20,20,20:40,40,40
```

JSON may name a `factory` or list `nx` / `ny` / `hot_specs` / `cold_specs`
/ `symmetry` directly. YAML works if PyYAML is installed. Factory
overrides (volume, `rmin`, ports, solver caps) are tabulated in
[examples/README.md](examples/README.md) and [docs/model.md](docs/model.md) §8.2.

## Tutorials

Numbered scripts in [`examples/`](examples/) teach the API. From the repo root:

```bash
python examples/01_analyze_once.py              # analyze() only, no TO
python examples/02_conduction_tree.py
python examples/03_convection_darcy.py
python examples/04_conjugate_stokes.py
python examples/05_custom_regions.py
python examples/06_mms_check.py
python examples/run_all.py --quick              # coarse smoke of every tutorial
```

`--quick` uses a coarse mesh. Outputs go under `outputs/`. See
[examples/README.md](examples/README.md) for the reading order.

## Run

```bash
python -m topoopt 2d --config examples.problems:conduction_tree --iters 80
python -m topoopt 2d --config examples.problems:convection_darcy --outdir outputs/2d_convection
python -m topoopt 2d --config examples.problems:conjugate_stokes --outdir outputs/2d_both
```

`--heat` only selects which PDE terms are on (conduction / convection / both).
Inlets, outlets, and Dirichlet patches come from `--config` or explicit
`--hot` / `--cold` / `--port-frac`.

Default objective is `J = -mean(T)`. `--hot` turns off the volume source and
sets `J` to the conductive heat leaving those patches.

```bash
python -m topoopt 2d --heat conduction --hot face:top --cold face:bottom:frac=0.5
python -m topoopt verify
python -m topoopt examples
```

Every `ColdPlateParams` field and optimizer kwarg is listed in
[docs/model.md](docs/model.md) §8. Tutorial / gallery numbers are in
[examples/README.md](examples/README.md).

| Flag | Writes | Library default if omitted |
|---|---|---|
| `--config` | factory or JSON/YAML | generic `params2d` box (no patches) |
| `--nx`, `--ny` | `n` | 40, 40 |
| `--heat` | `heat_mode` | `both` (or the factory) |
| `--flow {stokes,darcy}` | `flow_model` | `stokes` |
| `--vol` | `vol_frac` | 0.45 |
| `--pe` | `pe` | 40 |
| `--q` | `q_vol` | 1 |
| `--k-ratio` | `k_solid` (`k_fluid` stays 1) | 100 |
| `--rmin` | `rmin` (cells) | 2.2 |
| `--port-frac` | `port_frac` | 0.5 |
| `--hot`, `--cold` | `hot_specs` / `cold_specs` | empty |
| `--symmetry {x,y,x,y}` | `symmetry` | from the factory, else none |
| `--iters` | design steps | 80 |
| `--lr` | move at `β=1` | 0.2 (`ℓ = lr/√β`) |
| `--beta-max` | projection ceiling | 32 |
| `--mesh-schedule` | `optimize_hierarchy` | unset |
| `--seed` | init noise | 0 |
| `--outdir` | artifacts | `outputs/2d` |

Fields with no CLI flag (`q_k`, `alpha_max`, `stokes_dp`, `heat_iters`,
…) are set on the factory, in JSON, or by `params2d(..., key=value)`.

## Stokes notes

Stokes is **pressure-driven** (`stokes_dp` on the left port, `p = 0` on the
right). After each volume projection the optimizer pins a one-cell fluid
layer on the port *design variables* (`keep_ports_open`) — that is a design
projection, not a Dirichlet condition in the residual. A mid-height channel
seed is required: from a uniform field the local step opens an inlet cavity
and dams the outlet. Do not remove the seed or the port pin if you want a
through-channel.

The forward Stokes solve is an Uzawa warm start plus CG on the pressure
Schur complement (`stokes_kryl_iters`, default 200). The adjoint is the
residual discrete adjoint, not an unrolled Uzawa loop.

## Diagnostics

Every iteration prints `energy_rms`, `div_rms`, `mass_err`, and `gray`.
A warning is issued if `energy_rms > 1e-2` or, when flow is on,
`mass_err > 0.15`. If `T` is non-finite or `T_max > 1e3` (or flow is
blocked with a large energy residual and `T_max > 50`), the run
**aborts** after writing the best-`J` design. Flow modes have no
conduction sink — do not add one to hide a sealed channel.

## Tests

```bash
MPLBACKEND=Agg python -m pytest tests -q
```

This is the same command GitHub Actions runs (`pip install -e ".[cpu,dev]"`
on Python 3.14; the gallery is not run in CI). The suite covers
interpolation, Darcy/Stokes residuals, Stokes adjoint FD on throughput
and on the full `analyze` path, manufactured solutions and observed
order (`tests/test_mms.py`), short conduction / Darcy / custom-faces
optimizations with symmetry checks, JSON configs, and the physics
checks in `tests/test_physics.py`.
