# topoopt

JAX package for density-based topology optimization of a 2-D heated square box.
The design field `γ` is solid (`γ = 1`, conducting, impermeable) or fluid
(`γ = 0`). Heat is a uniform volumetric source unless `--hot` sets Dirichlet
heat-source patches. Sensitivities are discrete global adjoints.

The model (physics, discretization, adjoint, optimizer) is in
[docs/model.md](docs/model.md).

## Install

Python 3.10+ (3.12 recommended). From the repo root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[cpu]"
pip install pytest
```

`jax_enable_x64` is turned on when the package is imported.

## Run

```bash
python -m topoopt 2d --heat conduction --nx 40 --ny 40 --iters 80 --beta-max 32
python -m topoopt 2d --heat convection --flow darcy --outdir outputs/2d_convection
python -m topoopt 2d --heat both --flow stokes --outdir outputs/2d_both
```

`--heat` selects the physics:

- `conduction` — no flow, `k(γ)`, cold sink on the center 8% of the bottom wall
- `convection` — flow on, uniform `k_fluid`, centered 50% left/right ports, inlet `T = 0`
- `both` — conjugate `k(γ)` plus the same ports and volume source

Default objective is `J = -mean(T)`. `--hot` turns off the volume source and
sets `J` to the conductive heat leaving those patches.

```bash
python -m topoopt 2d --heat conduction --hot face:top --cold face:bottom:frac=0.5
python -m topoopt verify
python -m topoopt examples
```

Useful flags: `--heat`, `--flow {stokes,darcy}`, `--hot`, `--cold`, `--port-frac`,
`--q`, `--vol`, `--pe`, `--k-ratio`, `--rmin`, `--lr`, `--beta-max`, `--iters`,
`--seed`, `--outdir`.

Defaults: `--iters 80`, `--rmin 2.2`, `--beta-max 32`, `--vol 0.45`.

## Tests

```bash
MPLBACKEND=Agg python -m pytest tests/test_smoke.py tests/test_physics.py -q
```
