# Model

2-D heated-box topology optimization as implemented in `topoopt`.
Together with the source, this is enough to reproduce the discrete
problem. There is no 3-D model.

Domain $`\Omega=[0,L_x]\times[0,L_y]`$ (default $`L=(1,1)`$), uniform mesh
$`n=(n_x,n_y)`$ (default $`40\times 40`$), $`\Delta x=L_x/n_x`$,
$`\Delta y=L_y/n_y`$, $`\Delta V=\Delta x\,\Delta y`$.

Example BCs live in the registered factories in `topoopt/problems.py`
(re-exported from `examples/problems.py`).
`--heat` only selects PDE terms. Tutorial meshes: `examples/README.md`.
Wiki: https://github.com/smyng91/topology_optimization/wiki

## 1. Notation

| Symbol | Code | Meaning |
|---|---|---|
| $`\gamma\in[0,1]`$ | `gamma_raw` | raw design |
| $`\tilde{\gamma}`$ | filtered | cone-filtered design (`--filter helmholtz` optional) |
| $`\bar{\gamma}`$ | `phys` | projected density in the PDEs |
| $`\beta,\eta`$ | `beta`, `eta` | projection sharpness; threshold $`\eta=0.5`$ |
| $`r=r_{\min}\min(\Delta x,\Delta y)`$ | `rmin` | filter radius (`rmin` in cells, default $`2.2`$) |
| $`k,k_f,k_s,q_k`$ | `k_fluid`, `k_solid`, `q_k` | conductivity and RAMP sharpness |
| $`\alpha,q_\alpha`$ | `alpha_*`, `q_alpha` | Brinkman drag and Borrvall–Petersson sharpness |
| $`\kappa,q_\kappa`$ | `kappa_*`, `q_kappa` | Darcy permeability and RAMP sharpness |
| $`q,q_{\mathrm{vol}}`$ | `q_vol`, `q_specs` | volumetric heat (uniform or on a region) |
| $`\mathrm{Pe}`$ | `pe` | Péclet; $`0`$ in conduction |
| $`T,T_{\mathrm{in}},T_{\mathrm{hot}}`$ | `T`, `t_in`, `t_hot` | temperature; inlet / hot Dirichlet values |
| $`\mathbf{u},p`$ | `face_vel`, `p` | MAC velocity, cell pressure |
| $`v^{*}`$ | `vol_frac` | target $`\mathrm{mean}(\bar{\gamma})`$ |
| $`J`$ | `J` | objective (maximize) |
| $`\ell=\ell_0/\sqrt{\max(\beta,1)}`$ | `lr` | move limit; `--lr` is $`\ell_0`$ |
| $`\varepsilon`$ | `div_eps` | Stokes continuity regularizer |
| $`\Delta p,p_{\mathrm{in}}`$ | `stokes_dp`, `p_in` | Stokes / Darcy port pressures |

Solid $`\gamma=1`$: conducting, impermeable. Fluid $`\gamma=0`$: permeable,
poor conductor (except convection-only, where $`k\equiv k_f`$).

## 2. Design, filter, projection

Default density filter is the compact linear cone (Bruns–Tortorelli /
Bourdin). Weights are renormalized on the in-domain stencil, so a
uniform field is unchanged at the walls:

```math
\tilde{\gamma}_i
=
\frac{\sum_j w_{ij}\gamma_j}{\sum_j w_{ij}},
\qquad
w_{ij}=\max\bigl(0,\,r-\lVert x_i-x_j\rVert\bigr),
\qquad
r=r_{\min}\min(\Delta x,\Delta y).
```

`--filter helmholtz` uses Lazarov & Sigmund instead (Neumann, Jacobi CG):

```math
(-r^2\nabla^2+I)\tilde{\gamma}=\gamma.
```

The same $`r`$ is the cone support and the Helmholtz length. Helmholtz
has exponential tails; the cone is zero for $`\lVert x_i-x_j\rVert\ge r`$.

Tanh / smoothed-Heaviside (Wang, Lazarov, Sigmund 2011). $`\beta`$ doubles
$`1,2,4,\ldots,\beta_{\max}`$ (default $`32`$):

```math
\bar{\gamma}
=
\frac{\tanh(\beta\eta)+\tanh\bigl(\beta(\tilde{\gamma}-\eta)\bigr)}
{\tanh(\beta\eta)+\tanh\bigl(\beta(1-\eta)\bigr)}.
```

Conventional Stolpe–Svanberg RAMP interpolation (endpoints exact; $`q=0`$ is linear):

```math
\mathrm{RAMP}(x;f_0,f_1,q)=f_0+(f_1-f_0)\,\frac{x}{1+q(1-x)}.
```

| Property | Map | Default |
|---|---|---|
| $`k`$ (conduction / both) | $`\mathrm{RAMP}(\bar{\gamma};k_f,k_s,q_k)`$ | $`k_f=1`$, $`k_s=100`$, $`q_k=1`$ |
| $`k`$ (convection) | $`k\equiv k_f`$ | $`k_f=1`$ |
| $`\alpha`$ | $`\alpha_{\min}+(\alpha_{\max}-\alpha_{\min})\dfrac{q_\alpha\bar{\gamma}}{1-\bar{\gamma}+q_\alpha}`$ | $`0`$, $`10^5`$, $`q_\alpha=0.1`$ |
| $`\kappa`$ | $`\mathrm{RAMP}(\bar{\gamma};\kappa_{\max},\kappa_{\min},q_\kappa)`$ | $`1`$, $`10^{-6}`$, $`q_\kappa=0.1`$ |

## 3. Governing equations

Nondimensional. Stokes viscosity is scaled so the Laplacian coefficient
is $`1`$.

### 3.1 Energy

```math
-\nabla\cdot(k\nabla T)+\mathrm{Pe}\,\mathbf{u}\cdot\nabla T=q
\quad\text{in }\Omega.
```

| Mode | Flow | $`k`$ | $`\mathrm{Pe}`$ |
|---|---|---|---|
| `conduction` | off | $`k(\bar{\gamma})`$ | $`0`$ |
| `convection` | on | $`k_f`$ | $`40`$ |
| `both` | on | $`k(\bar{\gamma})`$ | $`40`$ |

Default $`q=q_{\mathrm{vol}}=1`$ everywhere. `q_specs` / `--q-region`
sets $`q=q_{\mathrm{vol}}`$ on those cells and $`0`$ elsewhere ($`T`$ floats).
`--hot` without `q_specs` turns uniform $`q`$ off and fixes
$`T=T_{\mathrm{hot}}`$ on the patch. Both may be set: Dirichlet $`T`$ and a
source subdomain.

Advection is $`\mathbf{u}\cdot\nabla T`$, not $`\nabla\cdot(\mathbf{u}T)`$,
so a constant $`T`$ stays in the kernel when $`\nabla\cdot\mathbf{u}\neq 0`$.

**Thermal BCs** (examples, not library defaults):

- Conduction tree: $`T=0`$ on a centered $`8\%`$ of the bottom wall; other
  walls $`\partial T/\partial n=0`$. A wide sink (`frac=0.5`) prefers fins.
- Flow: one left-centerline inlet and one right-centerline outlet
  (height `port_frac`, default $`0.5`$). Inlet fluid carries
  $`T_{\mathrm{in}}=0`$. No other ports or cold patches — a blocked
  design can run $`T`$ away.
- Specs: `face:{left,right,bottom,top}[:frac=…][:center=…]`,
  `box:xmin,xmax,ymin,ymax`. A face $`q`$ spec heats the adjacent cell
  layer. `--hot`/`--cold` fix $`T`$; `--q-region` generates heat.

### 3.2 Stokes–Brinkman (`flow_model=stokes`)

```math
-\nabla^2\mathbf{u}+\alpha(\bar{\gamma})\,\mathbf{u}+\nabla p=\mathbf{0},
\qquad
\nabla\cdot\mathbf{u}=0.
```

Solid is a Brinkman penalty, not a cut-cell hole. Discrete continuity
adds $`\varepsilon p`$ ($`\varepsilon=10^{-4}`$).

Ports only (same openings as Darcy): left $`p=\Delta p`$ (`stokes_dp`,
default $`20`$), $`\partial u/\partial x=0`$; right $`p=0`$,
$`\partial u/\partial x=0`$; remaining walls no-slip / impermeable.
Throughput is design-dependent. Inlet *velocity* is not prescribed
(`u_in_max` is unused).

### 3.3 Darcy (`--flow darcy`)

```math
\mathbf{u}=-\kappa(\bar{\gamma})\nabla p,
\qquad
-\nabla\cdot(\kappa\nabla p)=0.
```

Face pressures $`p=p_{\mathrm{in}}=1`$ (left port) and $`p=0`$ (right port);
other walls $`\partial p/\partial n=0`$.

## 4. Discretization

Cell-centered: $`\bar{\gamma}`$, $`k`$, $`\alpha`$, $`\kappa`$, $`T`$, $`p`$.
MAC: $`u`$ on vertical faces $`(n_x+1,n_y)`$, $`v`$ on horizontal faces
$`(n_x,n_y+1)`$.

Interior conductivity / permeability is the harmonic mean
$`k_f=2k_L k_R/(k_L+k_R)`$. Interior $`\nabla\cdot(k\nabla\phi)`$ uses
$`k_f(\phi_R-\phi_L)/\Delta x_i`$. Dirichlet faces add a half-cell flux
$`k_b(\phi_b-\phi_{\mathrm{bc}})/(\tfrac12\Delta x_i)`$. Adiabatic /
impermeable walls contribute nothing.

Energy advection is first-order upwind of $`\mathrm{Pe}\,\mathbf{u}\cdot\nabla T`$,
including boundary faces. Incoming left-port fluid uses $`T_{\mathrm{in}}`$
unless a Dirichlet face overrides it. Discrete identity:
$`\mathbf{u}\cdot\nabla T=\nabla\cdot(\mathbf{u}T)-T\nabla\cdot\mathbf{u}`$.

**Stokes.** $`\alpha`$ is face-averaged. Laplacian: Dirichlet ghosts
$`(-u)`$ on no-slip, one-sided $`\partial u/\partial x=0`$ on ports.
Port $`\nabla p`$ uses a half-cell to the face value. Continuity residual

```math
R_p=\nabla\cdot\mathbf{u}+\varepsilon p.
```

Forward: Uzawa / pressure-correction warm start ($`p\leftarrow p-\omega\nabla\cdot\mathbf{u}`$,
$`\omega=0.6`$, `uzawa_iters` default $`80`$), then Jacobi CG on the
pressure correction $`S=-DA^{-1}G+\varepsilon I`$
(`stokes_kryl_iters` default $`200`$; $`0`$ skips it):

```math
S\,\mathrm{d}p=-(\nabla\cdot\mathbf{u}(p_0)+\varepsilon p_0),\qquad
p\leftarrow p_0+\mathrm{d}p.
```

An exact block solve would recover the affine Stokes--Brinkman solution
in one correction. The implementation uses capped iterative momentum
and Schur solves, so acceptance is based on the achieved residual.
Momentum blocks at frozen $`p`$ are SPD after Dirichlet elimination
(CG). Saddle-point BiCGSTAB is not used (unstable at high
$`\alpha`$ contrast).

**Linear solvers.** Filter, Darcy, Stokes momentum, the Stokes Schur,
and **energy when** $`\mathrm{Pe}=0`$: Jacobi CG. Energy when
$`\mathrm{Pe}>0`$ and $`n_x n_y\le 48^2`$: dense factorization of the
finite-volume operator (the 2-D Stokes factory mesh). Larger convective
meshes: Jacobi BiCGSTAB with a 50-iteration shadow-residual restart and
a breakdown restart when $`r_0\cdot v`$ or $`t\cdot t`$ vanishes. The
Stokes residual adjoint uses a dense transposed Jacobian on the same
$`48^2`$ cutoff. Library caps: filter $`200`$, flow $`80`$,
Schur $`200`$, heat $`800`$.
Tolerance $`10^{-7}`$. Publication runs also require the achieved
residual to meet the evidence gates in `topoopt.optimize`.

## 5. Objective and volume

Volume source on (uniform $`q`$, or $`q`$ on `q_specs`): $`\int q\,\mathrm{d}V`$
is design-independent, so

```math
J=-\frac{1}{|\Omega|}\int_\Omega T\,\mathrm{d}V=-\mathrm{mean}(T).
```

`--hot` without `q_specs` turns uniform $`q`$ off. With both `--hot` and
`--q-region`, $`J`$ stays $`-\mathrm{mean}(T)`$. Dirichlet-only:

```math
J=Q_{\mathrm{hot}},
```

the Fourier flux leaving `--hot` faces / boxes. The optimizer minimizes
$`\mathcal{L}=-J`$.

Volume equality on the **physical** density, every iteration (no
multiplier):

```math
\mathrm{mean}(\bar{\gamma})=v^{*},\qquad 0\le\gamma\le 1.
```

Library default $`v^{*}=0.45`$ (`--vol`). Enforcement: mean-zero descent
$`g\leftarrow g-\mathrm{mean}(g)`$, normalize by $`\|g\|_\infty`$, projected
step $`\gamma\leftarrow\mathrm{clip}(\gamma-\ell g,0,1)`$, then a 40-step
bisection for a shift $`c`$ with
$`\mathrm{mean}(\bar{\gamma}(\mathrm{clip}(\gamma-c,0,1)))=v^{*}`$
(Stokes port cells are held at fluid *inside* that residual so the pin
does not drift the volume).

Stokes pins a one-cell fluid layer on port *design* cells
**inside** the volume bisection (`keep_ports_open`) so the pin does not
drift $\mathrm{mean}(\bar{\gamma})$ off $v^{*}$, then mirrors again.
Neither the pin nor the mid-height channel seed is a PDE Dirichlet.
Darcy does not pin ports.

## 6. Discrete adjoint

Linear solves $`A(\bar{\gamma})x=b(\bar{\gamma})`$ (Helmholtz filter,
Darcy, energy, Stokes momentum) use Krylov in forward mode. The cone
filter is an explicit stencil; JAX differentiates it directly. Reverse mode applies the
implicit-function theorem (`jax.lax.custom_linear_solve`), not an
unrolled loop:

```math
A(\bar{\gamma})^\top\lambda=\frac{\partial J}{\partial x}.
```

For nonsymmetric energy, JAX transposes the matvec. Stokes uses a
residual `custom_vjp`:

```math
\Bigl(\frac{\partial R}{\partial(\mathbf{u},p)}\Bigr)^\top\lambda
=\frac{\partial J}{\partial(\mathbf{u},p)},
\qquad
\frac{\partial J}{\partial\bar{\gamma}}
=-\lambda^\top\frac{\partial R}{\partial\bar{\gamma}}.
```

Filter, RAMP, and tanh sit on the same `value_and_grad` tape, so
sensitivities are w.r.t. raw $`\gamma`$. `jax_enable_x64` is set in
`topoopt/__init__.py`.

## 7. Optimizer

Projected gradient with $`\beta`$-continuation. No MMA / IPOPT.

`params.symmetry` (`x` and/or `y`) mirrors init noise and every accepted
design: $`\gamma\leftarrow\tfrac12(\gamma+\mathrm{flip}(\gamma))`$.
Factories: `conduction_tree` / `custom_faces` / `localized_source` $`\to`$
`x`; centerline-port flow $`\to`$ `y`; `custom_boxes` $`\to`$ none.

1. Init $`\gamma\leftarrow v^{*}+0.08\,(\mathrm{U}[0,1]-\tfrac12)`$, then
   symmetrize. Flow replaces this with a mid-height fluid band of height
   `port_frac` (fluid $`0.08`$, solid $`0.78`$) plus the same noise.
   `start_gamma` skips both. Enforce volume at $`\beta=1`$; Stokes pins
   ports.
2. For $`i=1,\ldots,N`$ (default $`N=80`$): set $`\beta`$; project symmetry /
   volume / pin; JIT $`J`$ and $`\nabla_\gamma(-J)`$; step of size $`\ell`$;
   project again. Warn if energy residual RMS $`>10^{-2}`$ or (flow)
   port mass error $`>0.15`$. Abort (`RunawaySolveError`) if $`T`$ is
   non-finite, $`T_{\max}>10^3`$, or a flow solve has a large energy
   residual and $`T_{\max}>50`$. Flow modes get no extra cold patch.
3. Keep-best is **per $`\beta`$ level**. $`J`$ is not comparable across
   continuation ($`\bar{\gamma}(\beta)`$ changes). Iterates with relative
   energy residual $`>10^{-3}`$ (or RMS $`>10^{-2}`$ if relative is
   absent) are ignored, so an unconverged $`T`$ cannot win.
   The return value is the best-$`J`$ energy-trustworthy iterate at the
   highest $`\beta`$ whose relative energy residual is at most $`10^{-3}`$.
   `J_peak` / `peak_iter` store the global max $`J`$ among trustworthy
   iterates. Stall (`stall_iters`, default $`8`$) counts only at
   $`\beta_{\max}`$ and resets when $`\beta`$ increases.
   At least 40% of the steps are spent at $`\beta_{\max}`$.

`optimize_hierarchy` bilinear-upsamples and continues
(`--mesh-schedule nx,ny,iters:…`).

Artifacts: `history.json`, `run.json` ($`J_0`$, $`J_{\mathrm{final}}`$,
$`J_{\mathrm{best}}`$, $`J_{\mathrm{peak}}`$, `stopped`), `state_best.npz`,
`state_final.npz`, PNG, VTK of $`\bar{\gamma}`$, $`T`$, $`|\mathbf{u}|`$, $`p`$.

## 8. Parameters

`params2d(nx, ny, **kwargs)` fills $`n`$ and $`L`$. Omitted kwargs use the
library defaults. Factories override only the columns in §8.2.

### 8.1 `ColdPlateParams`

| Field | Flag / symbol | Default | Role |
|---|---|---|---|
| `n` | `--nx`, `--ny` | $`(40,40)`$ | cells |
| `L` | | $`(1,1)`$ | box; $`\Delta x=L_x/n_x`$ |
| `heat_mode` | `--heat` | `both` | `conduction` / `convection` / `both` |
| `flow_model` | `--flow` | `stokes` | ignored in conduction |
| `vol_frac` | `--vol` $`v^{*}`$ | $`0.45`$ | $`\mathrm{mean}(\bar{\gamma})`$ target |
| `pe` | `--pe` | $`40`$ | $`0`$ in conduction |
| `q_vol` | `--q` | $`1`$ | $`q`$ strength; uniform unless `q_specs` or (`hot_specs` and no `q_specs`) |
| `k_fluid` | | $`1`$ | $`k_f`$ |
| `k_solid` | `--k-ratio` | $`100`$ | $`k_s`$ |
| `q_k`, `q_alpha`, `q_kappa` | | $`1`$, $`0.1`$, $`0.1`$ | RAMP / BP sharpness |
| `alpha_min`, `alpha_max` | | $`0`$, $`10^5`$ | Brinkman $`\alpha`$ |
| `kappa_min`, `kappa_max` | | $`10^{-6}`$, $`1`$ | Darcy $`\kappa`$ (solid, fluid) |
| `p_in` | | $`1`$ | Darcy left-port $`p`$ |
| `stokes_dp` | $`\Delta p`$ | $`20`$ | Stokes left-port $`p`$ |
| `t_in`, `t_hot` | | $`0`$, $`1`$ | cold / hot Dirichlet $`T`$ |
| `rmin` | `--rmin` | $`2.2`$ | filter radius in cells (cone support / Helmholtz $`r`$) |
| `filter_kind` | `--filter` | `cone` | `cone` or `helmholtz` |
| `eta` | $`\eta`$ | $`0.5`$ | tanh threshold |
| `port_frac` | `--port-frac` | $`0.5`$ | centered port height |
| `hot_specs` | `--hot` | $`()`$ | Dirichlet $`T`$; kills uniform $`q`$ |
| `cold_specs` | `--cold` | $`()`$ | Dirichlet sinks |
| `q_specs` | `--q-region` | $`()`$ | cells with $`q=q_{\mathrm{vol}}`$ |
| `symmetry` | `--symmetry` | $`()`$ | `x` and/or `y` |
| `div_eps` | $`\varepsilon`$ | $`10^{-4}`$ | $`\varepsilon p`$ in $`R_p`$ |
| `solver_tol` | | $`10^{-7}`$ | Krylov tolerance |
| `flow_iters` | | $`80`$ | Darcy / Stokes-momentum CG |
| `uzawa_iters` | | $`80`$ | Stokes pressure warm start |
| `stokes_kryl_iters` | | $`200`$ | Schur CG; $`0`$ skips |
| `heat_iters` | | $`800`$ | energy CG ($`\mathrm{Pe}=0`$) or BiCGSTAB ($`n>48^2`$) |
| `filter_iters` | | $`200`$ | Helmholtz CG (unused by cone) |
| `u_in_max` | | $`1`$ | unused by the pressure-driven residuals |

JSON / YAML may set any field plus `nx`, `ny`, `lx`, `ly`, `factory`,
`comment` (ignored). Lists become tuples. See `examples/configs/`.

### 8.2 Factory overrides

| Factory | heat | flow | $`v^{*}`$ | $`r_{\min}`$ | $`\mathrm{Pe}`$ | BCs | sym | Caps |
|---|---|---|---|---|---|---|---|---|
| `conduction_tree` | conduction | none | $`0.30`$ | $`1.5`$ | $`0`$ | cold `face:bottom:frac=0.08` | `x` | heat $`800`$, filter $`200`$ |
| `convection_darcy` | convection | Darcy | $`0.45`$ | $`2.0`$ | $`40`$ | `port_frac=0.5` | `y` | flow $`280`$, heat $`800`$, filter $`200`$ |
| `conjugate_darcy` | both | Darcy | $`0.45`$ | $`2.0`$ | $`40`$ | same ports | `y` | same |
| `conjugate_stokes` | both | Stokes | $`0.45`$ | $`2.0`$ | $`40`$ | same; $`\Delta p=20`$ | `y` | flow $`80`$, Uzawa $`80`$, Schur $`200`$, heat $`800`$, filter $`120`$ |
| `custom_faces` | conduction | none | $`0.40`$ | $`2.0`$ | $`0`$ | hot/cold faces `frac=0.5`; $`q=0`$ | `x` | heat $`800`$, filter $`200`$ |
| `custom_boxes` | conduction | none | $`0.40`$ | $`2.0`$ | $`0`$ | hot `box:0.2,0.8,0.0,0.18`; cold box + `face:left`; $`q=0`$ | none | heat $`800`$, filter $`200`$ |
| `localized_source` | conduction | none | $`0.30`$ | $`1.5`$ | $`0`$ | `q_specs=box:0.3,0.7,0.70,1.0`; cold `frac=0.08` | `x` | heat $`800`$, filter $`200`$ |

### 8.3 Optimizer kwargs

Not on `ColdPlateParams`. Init noise amplitude $`0.08`$ (hard-coded).
Gallery uses $`\ell_0=0.12`$.

| Argument | Flag | Default | Role |
|---|---|---|---|
| `n_iters` | `--iters` | $`80`$ | design steps |
| `lr` | `--lr` $`\ell_0`$ | $`0.2`$ | move at $`\beta=1`$ |
| `beta_max` | `--beta-max` | $`32`$ | projection ceiling |
| `seed` | `--seed` | $`0`$ | ignored if `start_gamma` is set |
| `outdir` | `--outdir` | `outputs` | JSON / `npz` |
| `start_gamma` | | `None` | warm start |
| `abort_on_runaway` | | `True` | raise if $`T`$ blows up |
| `stall_iters` | | $`8`$ | stop at $`\beta_{\max}`$; $`0`$ disables |
| `callback` | | `None` | `callback(it, gamma, aux, rec)` |
| mesh schedule | `--mesh-schedule` | unset | `nx,ny,iters:…` |

### 8.4 `analyze` aux

| Key | Meaning |
|---|---|
| `phys` | $`\bar{\gamma}`$ |
| `T`, `p`, `speed`, `face_vel` | fields |
| `V` | $`\mathrm{mean}(\bar{\gamma})`$ |
| `energy_rms` | RMS energy residual |
| `energy_rel` | energy residual RMS / inhomogeneous-data RMS |
| `div_rms` | $`\|\nabla\cdot\mathbf{u}\|_{\mathrm{rms}}`$ |
| `u_in`, `u_out`, `mass_err` | port flux; $`\lvert u_{\mathrm{in}}-u_{\mathrm{out}}\rvert/(\lvert u_{\mathrm{in}}\rvert+\varepsilon)`$ |
| `stokes_rel` | relative Stokes residual ($`0`$ if not Stokes) |
| `gray` | fraction of cells with $`0.05<\bar{\gamma}<0.95`$ |
| `T_mean`, `T_max`, `speed_max` | scalars |

Each iterate also stores `J`, `vol`, `beta`, `move`, `sym_err`,
`is_best`. `run.json` adds `stopped` $`\in`$
`{completed, stall, runaway}`, $`J_{\mathrm{best}}`$ / `best_iter`,
$`J_{\mathrm{peak}}`$ / `peak_iter`.

## 9. Validation

```bash
MPLBACKEND=Agg python -m pytest tests -q
python -m topoopt verify
python examples/06_mms_check.py
```

CI: pytest on Python 3.10 and 3.12 (Linux and Windows), wheel/sdist,
and manuscript integrity when local publication artifacts are present.
Article sources are kept locally and are not in this repository.

| Check | Setup | Expect |
|---|---|---|
| Energy Poisson | $`T=\sin\pi x\sin\pi y`$, $`T=0`$ on $`\partial\Omega`$ | order $`\approx 2`$ |
| Advective energy | uniform $`\mathbf{u}`$ | order $`\gtrsim 1`$ |
| Variable $`k`$ | discrete operator as source | residual $`\to 0`$ |
| Helmholtz | $`\tilde{\gamma}=\cos\pi x\cos\pi y`$ (Neumann) | error $`\downarrow`$ under refinement; $`(-r^2\nabla^2+I)^{-1}(-r^2\nabla^2+I)`$ recovers $`\gamma`$ to Krylov tol |
| Cone | spike / constant field | compact support $`d<r`$; constants unchanged |
| Darcy | linear $`p`$, `port_frac=1` | small $`L^2`$ error |
| Stokes | Poiseuille, full-height $`\Delta p`$, $`\alpha=0`$ | small $`L^2`$ error |
| Adjoint | central FD on throughput and on `analyze` | matches discrete $`R\approx 0`$ |
| Physics | `tests/test_physics.py` | solid cooler than fluid; localized $`q`$; Dirichlet $`T`$ sandwich |

`python -m topoopt verify` is a short physics check, not a crisp-design
run. Short-run numbers: `examples/reference.json`. Snapshots:
`docs/figures/`. Gallery (`examples/gallery.py`): $`80\times 80`$
(Stokes $`48\times 48`$, Schur $`400`$, heat $`1200`$), $`\ell_0=0.12`$,
$`\beta_{\max}=32`$.

## 10. Limitations

- 2-D only. Stokes, not Navier–Stokes (no $`(\mathbf{u}\cdot\nabla)\mathbf{u}`$).
- Darcy is potential flow: no no-slip, no inertia.
- The cone support is $`r`$; Helmholtz (optional) has exponential tails
  and a thicker gray band of width $`\sim r`$. Tanh still leaves a
  gray interface of a few cells.
- Brinkman solid leaks; first-order upwind adds diffusion at high $`\mathrm{Pe}`$.
- Krylov caps: high-$`\beta`$ residuals can stay above $`10^{-2}`$ on
  convective meshes larger than $`48^2`$ (BiCGSTAB). Smaller Pe$>0$
  meshes factor the energy operator densely. Keep-best then returns
  the highest $`\beta`$ that still solved.
- Conduction + small sink chatters for $`\beta\gtrsim 16`$; return the
  high-$`\beta`$ design, not a mid-continuation gray field.
- Flow modes have no conduction sink. Do not add one to hide a block.
- $`k`$, $`\alpha`$, $`\kappa`$ do not depend on $`T`$ or $`\mathbf{u}`$.
