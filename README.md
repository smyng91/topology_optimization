# topoopt

JAX package for density-based topology optimization of a 2-D heated square box.
The design field `γ` is solid (`γ = 1`, conducting, impermeable) or fluid
(`γ = 0`). Heat is a uniform volumetric source unless `--hot` sets Dirichlet
heat-source patches. Sensitivities are discrete global adjoints.

The model (physics, discretization, adjoint, optimizer) is in
[docs/model.md](docs/model.md).

## Research v0.2

Each iteration logs energy residual RMS, `div_rms`, port mass error, and
grayness (also stored in `analyze` aux). The move limit decays as
`lr / sqrt(β)` and the run keeps the best-`J` design. A run writes
`history.json`, `run.json`, `state_best.npz`, and `state_final.npz`.

## Install

Python 3.10+ (3.12 recommended). From the repo root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[cpu,dev]"
```

`[cpu]` pulls the CPU JAX extra; `[dev]` adds `pytest`. `jax_enable_x64`
is turned on when the package is imported. Dependency ranges live in
`pyproject.toml` (no lockfile).

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
`--lr` is the move at `β = 1` and decays as `1/sqrt(β)`.

## Stokes notes

Stokes is **pressure-driven** (`stokes_dp` on the left port, `p = 0` on the
right). After each volume projection the optimizer pins a one-cell fluid
layer on the port *design variables* (`keep_ports_open`) — that is a design
projection, not a Dirichlet condition in the residual. A mid-height channel
seed is required: from a uniform field the local step opens an inlet cavity
and dams the outlet. Do not remove the seed or the port pin if you want a
through-channel.

## Diagnostics

Every iteration prints `energy_rms`, `div_rms`, `mass_err`, and `gray`.
A warning is issued (the run does not abort) if `energy_rms > 1e-2` or,
when flow is on, `mass_err > 0.15`. Flow modes have no conduction sink —
a blocked design can run `T` away.

## Tests

```bash
MPLBACKEND=Agg python -m pytest tests -q
```

This is the same command GitHub Actions runs (`pip install -e ".[cpu,dev]"`
on Python 3.12). The suite covers interpolation, Darcy/Stokes residuals,
a coarse Stokes adjoint FD check, short conduction and Darcy optimizations,
and the physics checks in `tests/test_physics.py`.
