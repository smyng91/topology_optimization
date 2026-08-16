# Model

This document describes the 2-D heated-box topology optimization model
implemented in `topoopt`. It is written so the discrete problem can be
reproduced from the text together with the source.

The domain is a square box \(\Omega=[0,L_x]\times[0,L_y]\) with default
\(L=(1,1)\) and a uniform Cartesian mesh of \(n=(n_x,n_y)\) cells (default
\(40\times 40\)). There is no 3-D model.

## 1. Problem statement

A density field \(\gamma\in[0,1]\) distributes **solid** (\(\gamma=1\):
conducting, impermeable) and **fluid** (\(\gamma=0\): coolant / insulator).
Three heat modes share the same energy residual and differ in which terms
are active.

| Mode | Flow | Conductivity | Thermal BCs (defaults) | Source |
|---|---|---|---|---|
| `conduction` | off, \(\mathrm{Pe}=0\) | \(k(\gamma)\) | centered 8% of the bottom wall at \(T=0\); other walls adiabatic | uniform \(q\), or \(q\) on `q_specs` |
| `convection` | on | uniform \(k_\mathrm{fluid}\) | one left-centerline inlet, one right-centerline outlet; inlet \(T=0\) by advection; no other inlets, outlets, or cold patches | uniform \(q\), or \(q\) on `q_specs` |
| `both` | on | \(k(\gamma)\) | same single inlet / outlet as convection | uniform \(q\), or \(q\) on `q_specs` |

These BCs are **example configurations**, not library defaults. They are
defined in `examples/problems.py` and loaded with
`--config examples.problems:<name>`. The solver accepts any
`hot_specs` / `cold_specs` / `q_specs` / `port_frac`. Setting `--hot`
turns *uniform* \(q\) off; `--q-region` / `q_specs` keeps a volumetric
source on those cells (and can be combined with Dirichlet \(T\)).
Factory numbers and every `ColdPlateParams` field are in §8; tutorial
meshes and `--quick` settings are in `examples/README.md`.

**Notation.** \(\gamma\) is the raw design. \(\tilde\gamma\) is the Helmholtz-filtered
field. \(\bar\gamma\) is the projected physical density used in the PDEs.
In the code the physical field is named `phys`.

## 2. Design variable, filter, and projection

### 2.1 Filter

A Helmholtz density filter (Lazarov & Sigmund) with Neumann boundaries:

\[
\bigl(-r^2\nabla^2+I\bigr)\tilde\gamma=\gamma,\qquad
r=r_{\min}\min(\Delta x,\Delta y).
\]

Default \(r_{\min}=2.2\) (in cells). The Laplacian is second-order with
edge padding (zero normal gradient). The linear system is solved
matrix-free with Jacobi-preconditioned CG.

### 2.2 Projection

A tanh / smoothed-Heaviside projection (Wang, Lazarov, Sigmund 2011):

\[
\bar\gamma
=
\frac{\tanh(\beta\eta)+\tanh\bigl(\beta(\tilde\gamma-\eta)\bigr)}
{\tanh(\beta\eta)+\tanh\bigl(\beta(1-\eta)\bigr)},
\qquad \eta=0.5.
\]

\(\beta\) is continued from \(1\) up to \(\beta_{\max}\) (default \(32\))
by doubling: \(1,2,4,\ldots,\beta_{\max}\).

### 2.3 RAMP interpolation

Material properties use

\[
\mathrm{RAMP}(x;\,f_0,f_1,q)
=
f_0+(f_1-f_0)\,x\,\frac{q+1}{x+q}.
\]

Endpoints are exact: \(x=0\mapsto f_0\), \(x=1\mapsto f_1\). Large \(q\)
is nearly linear; small \(q\) rises toward \(f_1\) quickly.

| Property | Map | Defaults |
|---|---|---|
| Conductivity (conduction / both) | \(k=\mathrm{RAMP}(\bar\gamma;\,k_f,k_s,q_k)\) | \(k_f=1\), \(k_s=100\), \(q_k=1\) |
| Conductivity (convection) | \(k\equiv k_f\) | \(k_f=1\) |
| Brinkman penalty | \(\alpha=\alpha_{\min}+(\alpha_{\max}-\alpha_{\min})\,\frac{q_\alpha\bar\gamma}{1-\bar\gamma+q_\alpha}\) (Borrvall–Petersson) | \(0\), \(10^{5}\), \(q_\alpha=0.1\) |
| Darcy permeability | \(\kappa=\mathrm{RAMP}(\bar\gamma;\,\kappa_{\max},\kappa_{\min},q_\kappa)\) | \(1\), \(10^{-6}\), \(q_\kappa=0.1\) |

Solid therefore conducts and blocks flow; fluid is permeable and (except
in convection-only mode) a poor conductor.

## 3. Governing equations

All fields are nondimensional. Viscosity is absorbed into the Stokes
scaling so the Laplacian coefficient is \(1\).

### 3.1 Energy

\[
-\nabla\cdot\bigl(k\nabla T\bigr)+\mathrm{Pe}\,\mathbf{u}\cdot\nabla T=q
\quad\text{in }\Omega.
\]

- `conduction`: \(\mathrm{Pe}=0\), \(\mathbf{u}=\mathbf{0}\), \(k=k(\bar\gamma)\).
- `convection`: \(\mathrm{Pe}>0\) (default \(40\)), \(k=k_f\).
- `both`: \(\mathrm{Pe}>0\), \(k=k(\bar\gamma)\).

Default source: \(q=q_{\mathrm{vol}}=1\) everywhere. `--q-region` /
`q_specs` restricts \(q\) to those cells (\(T\) still floats). `--hot`
prescribes Dirichlet \(T=T_{\mathrm{hot}}\) and turns *uniform* \(q\)
off; with both `--hot` and `--q-region`, the patches fix \(T\) and the
source region still generates heat.

Advection is the **incompressible** form \(\mathbf{u}\cdot\nabla T\), not
the conservative flux \(\nabla\cdot(\mathbf{u}T)\). The two agree when
\(\nabla\cdot\mathbf{u}=0\); the incompressible form keeps a constant
temperature in the kernel even if the discrete velocity is only
approximately divergence-free.

**Thermal boundary conditions**

- Conduction default: \(T=0\) on a centered \(8\%\) of the bottom wall
  (a small sink so the optimizer can form a branching tree); remaining
  walls \(\partial T/\partial n=0\). A wide sink (`--cold face:bottom:frac=0.5`)
  prefers parallel fins instead. The conduction example also uses a
  lower solid fraction (\(v^\*=0.30\)) so the trunk can stay thin.
- Flow modes: one inlet on the left-wall centerline and one outlet on
  the right-wall centerline (`port_frac` of the height, default \(0.5\)).
  Fluid entering the inlet carries \(T_{\mathrm{in}}=0\). Outflow uses
  upwind interior temperature. There are **no** other inlets, outlets,
  or cold Dirichlet patches — remaining walls are adiabatic for
  diffusion and no-slip / impermeable for flow. A blocked design can
  therefore run \(T\) away.
- Optional `--hot` / `--cold` faces or boxes prescribe Dirichlet \(T\).
  `--q-region` marks cells that generate heat (\(T\) floats). Same spec
  language: `face:bottom:frac=0.5`, `face:top:frac=0.4:center=0.3`,
  `box:xmin,xmax,ymin,ymax`. A face `q` spec heats the adjacent cell layer.

### 3.2 Stokes–Brinkman (default flow)

\[
-\nabla^2\mathbf{u}+\alpha(\bar\gamma)\,\mathbf{u}+\nabla p=\mathbf{0},
\qquad
\nabla\cdot\mathbf{u}=0.
\]

Solid is a Brinkman penalty, not a geometric hole. Continuity is
regularized in the discrete residual by a small \(\varepsilon p\) term
(\(\varepsilon=10^{-4}\)).

**Velocity / pressure BCs** (one inlet and one outlet, each of height
`port_frac` on the left- and right-wall centerlines), the same
pressure-driven idea as Darcy:

- Left port: \(p=\Delta p\) (`stokes_dp`, default \(20\)) and \(\partial u/\partial x=0\).
- Right port: \(p=0\) and \(\partial u/\partial x=0\).
- Remainder of the left / right walls, and the top / bottom: no-slip.
- Off-port walls are impermeable.

Throughput is design-dependent: block the path and the flow drops. The
code does **not** force the inlet strip to fluid; the design may block
the ports. A prescribed inlet *velocity* is not used — that BC kept the
mass flux fixed and made the cheap saddle-point solve dump mass into a
non-zero \(\nabla\cdot\mathbf{u}\) once solid appeared.

### 3.3 Darcy flow (`--flow darcy`)

\[
\mathbf{u}=-\kappa(\bar\gamma)\nabla p,
\qquad
\nabla\cdot\mathbf{u}=0
\quad\Rightarrow\quad
-\nabla\cdot\bigl(\kappa\nabla p\bigr)=0.
\]

Face-pressure BCs on the ports only: \(p=p_{\mathrm{in}}=1\) on the left
port, \(p=0\) on the right port. All other walls are impermeable
(\(\partial p/\partial n=0\)). Face velocities are the same fluxes that
appear in the Poisson residual.

## 4. Discretization

### 4.1 Mesh

Uniform cell size \(\Delta x=L_x/n_x\), \(\Delta y=L_y/n_y\). Cell volume
\(\Delta V=\Delta x\,\Delta y\).

- Cell-centered: \(\bar\gamma\), \(k\), \(\alpha\), \(\kappa\), \(T\), \(p\).
- MAC face-normal velocities: \(u\) on vertical faces (shape
  \((n_x+1,n_y)\)), \(v\) on horizontal faces (shape \((n_x,n_y+1)\)).

### 4.2 Diffusion (energy and Darcy)

Interior-face conductivity (or permeability) is the harmonic mean

\[
k_f=\frac{2k_L k_R}{k_L+k_R}.
\]

The interior contribution to \(\nabla\cdot(k\nabla\phi)\) uses
\(k_f(\phi_R-\phi_L)/\Delta x_i\) on each axis. Dirichlet faces add a
half-cell flux \(k_b(\phi_b-\phi_{\mathrm{bc}})/(\tfrac12\Delta x_i)\).
Adiabatic / impermeable walls contribute nothing.

### 4.3 Advection

First-order upwind of \(\mathrm{Pe}\,\mathbf{u}\cdot\nabla T\), including
domain-boundary faces. Incoming left-port fluid uses \(T_{\mathrm{in}}\)
unless a Dirichlet face patch overrides that face. The discrete form is
written so that a constant \(T\) produces zero advection even when
\(\nabla\cdot\mathbf{u}\) is not exactly zero
(\(\mathbf{u}\cdot\nabla T=\nabla\cdot(\mathbf{u}T)-T\nabla\cdot\mathbf{u}\)).

### 4.4 Stokes

MAC Stokes–Brinkman. \(\alpha\) is averaged to faces. The Laplacian uses
Dirichlet ghosts (\(-u\)) on no-slip walls and a one-sided
\(\partial u/\partial x=0\) stencil on both ports. Port pressure
gradients use a half-cell to the face value (\(p_{\mathrm{in}}\) or
\(0\)). The discrete continuity residual is

\[
R_p=\nabla\cdot\mathbf{u}+\varepsilon p.
\]

**Forward solve:** Uzawa / pressure-correction warm start, then CG on
the pressure Schur complement \(S=DA^{-1}G+\varepsilon I\). For frozen
pressure the momentum blocks are SPD and solved with CG. Pressure is
updated \(p\leftarrow p-\omega\nabla\cdot\mathbf{u}\) with \(\omega=0.6\)
for `uzawa_iters` passes (default \(80\)). Scaled SIMPLE Richardson
updates alone stall on high-contrast Brinkman fields; the Schur CG
correction

\[
S\,\mathrm{d}p=-(\nabla\cdot\mathbf{u}(p_0)+\varepsilon p_0),\qquad
p\leftarrow p_0+\mathrm{d}p
\]

is an exact Newton step because Stokes–Brinkman is affine in
\((\mathbf{u},p)\) at fixed \(\bar\gamma\). Jacobi-preconditioned CG
(`stokes_kryl_iters`, default \(200\)) drives \(\|R\|\) to the solver
tolerance. Set `stokes_kryl_iters=0` to skip the correction. A
saddle-point BiCGSTAB correction is *not* used — it is unstable when
\(\alpha(\bar\gamma)\) jumps.

**Adjoint:** `jax.custom_vjp` on the residual \(R(\mathbf{u},p;\bar\gamma)=0\),
solved with BiCGSTAB (at least \(400\) iterations). This is the discrete
adjoint of the PDE, not an unrolled Uzawa loop. After the Schur
correction the forward state satisfies \(R\approx 0\), so the adjoint
matches a consistent discrete solve (checked against central FD on
throughput).

**Linear solvers.** Filter, Darcy, Stokes momentum, and the Stokes
pressure Schur use Jacobi-preconditioned CG. Energy and the Stokes
adjoint use Jacobi-preconditioned BiCGSTAB. Iteration caps: filter
\(200\), flow \(80\), Stokes Schur \(200\), heat \(400\). Tolerance
\(10^{-7}\).

**Verification.** `tests/test_mms.py` checks manufactured / exact
solutions and observed order: energy Poisson (\(T=\sin\pi x\sin\pi y\),
order \(\approx 2\)), energy with uniform advection (order \(\gtrsim 1\)),
variable-\(k(\gamma)\) energy consistency (discrete operator as the
source), Helmholtz filter on the Neumann cosine
\(\tilde\gamma=\cos\pi x\cos\pi y\) (error decreases under refinement;
the discrete inverse \(( -r^2\nabla^2+I)^{-1}(-r^2\nabla^2+I)\) recovers
the field to the Krylov tolerance), Darcy linear pressure on a
full-height port, and Stokes–Poiseuille on a full-height
pressure-driven channel. The Stokes adjoint is checked by central FD
on throughput and on the full `analyze` path (filter + energy).

## 5. Objective and volume constraint

### 5.1 Objective

When a volumetric source is on (uniform \(q\), or \(q=q_{\mathrm{vol}}\)
restricted to `q_specs`), the heat generated in that region is
\(\int q\,\mathrm{d}V\), independent of \(\gamma\). The design cannot
change how much heat is produced, only how hot the box runs. The
figure of merit is therefore

\[
J=-\frac{1}{|\Omega|}\int_\Omega T\,\mathrm{d}V
=-\mathrm{mean}(T).
\]

Maximizing \(J\) minimizes mean temperature. `--hot` without
`q_specs` turns *uniform* \(q\) off. With both `--hot` and
`--q-region`, Dirichlet patches fix \(T\) and the source cells still
generate heat; \(J\) stays \(-\mathrm{mean}(T)\).

If `--hot` is set and there is no volume source,

\[
J=Q_{\mathrm{hot}},
\]

the conductive heat leaving the Dirichlet heat-source faces and/or
boxes into the rest of the domain (Fourier flux on those interfaces).

The optimizer minimizes \(\mathcal{L}=-J\).

### 5.2 Volume

An equality constraint on the **physical** density:

\[
\mathrm{mean}(\bar\gamma)=v^\*,\qquad v^\*=0.45
\]

(`--vol`), together with box bounds \(0\le\gamma\le 1\).

This is enforced every iteration, not with a Lagrange multiplier:

1. The descent direction is mean-zero: \(g\leftarrow g-\mathrm{mean}(g)\),
   then normalized by \(\|g\|_\infty\).
2. A projected step \(\gamma\leftarrow\mathrm{clip}(\gamma-\ell\,g,0,1)\)
   with a \(\beta\)-damped move limit \(\ell=\ell_0/\sqrt{\max(\beta,1)}\)
   (`--lr` is \(\ell_0\), the \(\beta=1\) move; default \(0.2\)).
3. A 24-step bisection finds a shift \(c\) such that
   \(\mathrm{mean}\bigl(\bar\gamma(\mathrm{clip}(\gamma-c,0,1))\bigr)=v^\*\).

Darcy does **not** force port cells to fluid. Stokes keeps a one-cell
fluid layer on the pressure ports after each volume projection — a
single solid cell there seals the opening. That pin, together with a
mid-height channel seed, is required for a through-channel: from a
uniform field the local step prefers an inlet cavity.

## 6. Discrete adjoint and JAX

Let \(A(\bar\gamma)x=b(\bar\gamma)\) be a linear solve (filter, Darcy,
energy, Stokes momentum). Forward mode uses a Krylov method.
Reverse mode does **not** unroll those iterations.
`jax.lax.custom_linear_solve` applies the implicit-function theorem:
one adjoint solve

\[
A(\bar\gamma)^\top\lambda=\frac{\partial J}{\partial x}
\]

gives \(\mathrm{d}J/\mathrm{d}\bar\gamma\) consistent with the discrete
operator. For the nonsymmetric energy operator, JAX supplies \(A^\top\)
by automatic transposition of the matvec.

Stokes uses a residual `custom_vjp`: the forward Uzawa + Schur-CG
correction is not unrolled. Reverse mode solves

\[
\Bigl(\frac{\partial R}{\partial(\mathbf{u},p)}\Bigr)^\top\lambda
=\frac{\partial J}{\partial(\mathbf{u},p)},
\qquad
\frac{\partial J}{\partial\bar\gamma}
=
-\lambda^\top\frac{\partial R}{\partial\bar\gamma}.
\]

Filter, RAMP, and tanh projection are included in the same
`jax.value_and_grad` tape, so sensitivities are with respect to the raw
design \(\gamma\).

`jax_enable_x64` is set in `topoopt/__init__.py`.

## 7. Optimizer

Projected gradient descent with \(\beta\)-continuation, a decaying move
limit, keep-best, optional mesh continuation, and a symmetry projection.

**Why a symmetric problem used to look skewed.** The PDEs, mesh, and
example BCs are mirrors: a centered bottom sink is left–right
symmetric; centerline ports are top–bottom symmetric (not left–right —
inlet \(\neq\) outlet). The breaker was the init
\(\gamma\leftarrow v^\*+0.08\,(\mathrm{U}[0,1]-\tfrac12)\). That noise
is not invariant under a flip, so the first gradient already prefers
one side and the tree or channel grows crooked. `params.symmetry`
(`x` and/or `y`) mirrors the noise and every accepted design,
\(\gamma\leftarrow\tfrac12(\gamma+\mathrm{flip}(\gamma))\). Named
factories set this (`conduction_tree` / `custom_faces`: `x`;
centerline-port flow cases: `y`; `custom_boxes`: none). Krylov
roundoff can still seed a tiny asymmetry; the projection kills it
each step.

1. Initialize \(\gamma\) near \(v^\*\) plus uniform noise of amplitude
   \(0.08\), then symmetrize. Flow problems replace that with a
   mid-height open duct (fluid band of height `port_frac`) plus the
   same (symmetrized) noise — required for Stokes, and used for Darcy
   as well. A caller may pass `start_gamma` instead (used by
   coarse-to-fine continuation). Then enforce the volume constraint
   at \(\beta=1\), and (Stokes only) pin port cells to fluid.
2. For iteration \(i=1,\ldots,N\) (default \(N=80\)):
   - set \(\beta\) from the continuation schedule;
   - enforce symmetry, volume, and the Stokes port pin;
   - evaluate \(J\) and \(\nabla_\gamma(-J)\) with a JIT
     `value_and_grad`;
   - take a mean-zero projected step of size
     \(\ell=\ell_0/\sqrt{\max(\beta,1)}\);
   - enforce symmetry / volume / pin again.
   Diagnostics from `analyze` aux are printed each iteration: energy
   residual RMS, \(\|\nabla\cdot\mathbf{u}\|_{\mathrm{rms}}\), port mass
   error, and grayness. A warning is issued if the energy residual RMS
   exceeds \(10^{-2}\) or, when flow is on, the port mass error exceeds
   \(0.15\). If \(T\) is non-finite, \(T_{\max}>10^3\), or a flow solve
   has both a large energy residual and \(T_{\max}>50\), the run
   **aborts** after writing the best-\(J\) checkpoint
   (`RunawaySolveError`). Flow modes have no extra cold patch.
3. Keep the design with the largest \(J\) **inside the current \(\beta\)
   level**. \(J\) is not comparable across continuation:
   \(\bar\gamma(\beta)\) and the discrete energy operator change when
   the tanh projection sharpens, so a gray field at \(\beta=4\) can
   beat a nearly 0–1 tree at \(\beta_{\max}\) on raw \(J\) and still
   be a poor physical design. The returned field is the best-\(J\)
   iterate at the **highest \(\beta\) that ran**. `run.json` also
   stores \(J_{\mathrm{peak}}\) / `peak_iter` (global max \(J\), any
   \(\beta\)) for diagnostics. At \(\beta_{\max}\), stop early if
   \(J\) has not improved **at that \(\beta\)** for `stall_iters`
   iterations (default 8). The stall clock resets when \(\beta\)
   increases, so an early gray best does not abort the sharp stage
   on the first \(\beta_{\max}\) iterate. Write `history.json`,
   `run.json` (serializable params, \(J_0\), \(J_{\mathrm{final}}\),
   \(J_{\mathrm{best}}\), \(J_{\mathrm{peak}}\), `stopped`, last
   diagnostics), `state_best.npz`, `state_final.npz`, PNG slices, and
   a VTK file of cell-centered \(\bar\gamma\), \(T\), \(|\mathbf{u}|\),
   and \(p\). The optimize return value is that high-\(\beta\) design.

`optimize_hierarchy` resizes the best field with bilinear interpolation
and continues on a finer mesh (`--mesh-schedule nx,ny,iters:…`).

There is no MMA / IPOPT / optimality-criteria loop. The volume equality
is a hard projection, not a penalty.

## 8. Parameters

Every physics field lives on `ColdPlateParams` (`topoopt/config.py`).
`params2d(nx, ny, **kwargs)` fills `n` and `L`; omitted kwargs use the
library defaults below. Named factories in `examples/problems.py`
override a subset — they do **not** change fields that are blank in
§8.2. Tutorial meshes, iteration counts, and `--quick` settings are
tabulated in `examples/README.md`.

### 8.1 `ColdPlateParams` (library defaults)

| Field | Symbol / flag | Default | Role |
|---|---|---|---|
| `n` | `--nx`, `--ny` | \((40,40)\) | Cells \((n_x,n_y)\) |
| `L` | | \((1,1)\) | Box size. \(\Delta x=L_x/n_x\) |
| `heat_mode` | `--heat` | `both` | `conduction` / `convection` / `both` |
| `flow_model` | `--flow` | `stokes` | `stokes` or `darcy`; ignored in conduction |
| `vol_frac` | `--vol` \(v^\*\) | \(0.45\) | Target \(\mathrm{mean}(\bar\gamma)\) |
| `pe` | `--pe` | \(40\) | Péclet; forced to \(0\) in conduction |
| `q_vol` | `--q` | \(1\) | Volumetric source strength. Uniform if `q_specs` is empty and `hot_specs` is empty; on `q_specs` only otherwise. Off if \(q=0\), or if `hot_specs` is set and `q_specs` is empty |
| `k_fluid` | | \(1\) | Fluid conductivity |
| `k_solid` | `--k-ratio` | \(100\) | Solid conductivity (`--k-ratio` writes this field; \(k_f\) stays \(1\)) |
| `q_k` | | \(1\) | RAMP sharpness for \(k\) |
| `q_alpha` | | \(0.1\) | Borrvall–Petersson sharpness for \(\alpha\) |
| `q_kappa` | | \(0.1\) | RAMP sharpness for \(\kappa\) |
| `alpha_min` | | \(0\) | Brinkman drag in fluid |
| `alpha_max` | | \(10^{5}\) | Brinkman drag in solid |
| `kappa_min` | | \(10^{-6}\) | Darcy permeability in solid |
| `kappa_max` | | \(1\) | Darcy permeability in fluid |
| `p_in` | | \(1\) | Darcy left-port pressure |
| `stokes_dp` | | \(20\) | Stokes left-port pressure \(\Delta p\) |
| `t_in` | | \(0\) | Inlet / cold Dirichlet temperature |
| `t_hot` | | \(1\) | Hot-patch Dirichlet temperature |
| `rmin` | `--rmin` | \(2.2\) | Helmholtz radius in **cells**; \(r=r_{\min}\min(\Delta x,\Delta y)\) |
| `eta` | | \(0.5\) | Tanh-projection threshold |
| `port_frac` | `--port-frac` | \(0.5\) | Centered height of both vertical ports |
| `hot_specs` | `--hot` | `()` | Dirichlet \(T\) patches; turns *uniform* \(q\) off |
| `cold_specs` | `--cold` | `()` | Dirichlet sinks |
| `q_specs` | `--q-region` | `()` | Cells that receive \(q_{\mathrm{vol}}\) (\(T\) floats). Same `face:` / `box:` language; a face marks the adjacent cell layer |
| `symmetry` | `--symmetry` | `()` | Design mirrors: `x` and/or `y` |
| `div_eps` | \(\varepsilon\) | \(10^{-4}\) | Stokes continuity regularizer \(\varepsilon p\) |
| `solver_tol` | | \(10^{-7}\) | Krylov residual tolerance |
| `flow_iters` | | \(80\) | Darcy CG / Stokes momentum CG |
| `uzawa_iters` | | \(80\) | Stokes pressure-correction warm start |
| `stokes_kryl_iters` | | \(200\) | Pressure-Schur CG (`0` skips it) |
| `heat_iters` | | \(400\) | Energy BiCGSTAB |
| `filter_iters` | | \(200\) | Helmholtz CG |
| `u_in_max` | | \(1\) | Peak of `grid.inlet_profile` only. **Not** used by the pressure-driven Darcy/Stokes residuals |

Region spec strings: `face:{left,right,bottom,top}`,
`face:bottom:frac=0.5`, `face:top:frac=0.4:center=0.3`,
`box:xmin,xmax,ymin,ymax`.

JSON / YAML (`--config file.json`) may set any field above plus `nx`,
`ny`, `lx`, `ly`, `factory`, and `comment` (ignored). Lists become
tuples. See `examples/configs/`.

### 8.2 Named factory overrides

| Factory | heat | flow | \(v^\*\) | \(r_{\min}\) | Pe | BCs | symmetry | Solver caps |
|---|---|---|---|---|---|---|---|---|
| `conduction_tree` | conduction | none | \(0.30\) | \(1.5\) | \(0\) | cold `face:bottom:frac=0.08` | `x` | heat 400, filter 200 |
| `convection_darcy` | convection | Darcy | \(0.45\) | \(2.0\) | \(40\) | `port_frac=0.5`, no patches | `y` | flow 280, heat 400, filter 200 |
| `conjugate_darcy` | both | Darcy | \(0.45\) | \(2.0\) | \(40\) | same ports | `y` | same as convection |
| `conjugate_stokes` | both | Stokes | \(0.45\) | \(2.0\) | \(40\) | same ports | `y` | flow 80, Uzawa 80, Schur 200, heat 320, filter 120 |
| `custom_faces` | conduction | none | \(0.40\) | \(2.0\) | \(0\) | hot top / cold bottom, each `frac=0.5`; \(q=0\) | `x` | heat 400, filter 200 |
| `custom_boxes` | conduction | none | \(0.40\) | \(2.0\) | \(0\) | hot `box:0.2,0.8,0.0,0.18`; cold `box:0.0,0.18,0.25,0.75` and `face:left`; \(q=0\) | none | heat 400, filter 200 |
| `localized_source` | conduction | none | \(0.30\) | \(1.5\) | \(0\) | `q_specs=box:0.3,0.7,0.70,1.0`; cold `face:bottom:frac=0.08`; no Dirichlet hot | `x` | heat 400, filter 200 |

### 8.3 Optimizer kwargs (`optimize` / `optimize_hierarchy`)

These are **not** on `ColdPlateParams`.

| Argument | Flag | Default | Role |
|---|---|---|---|
| `n_iters` | `--iters` | \(80\) | Design steps (ignored if `--mesh-schedule` is set) |
| `lr` | `--lr` \(\ell_0\) | \(0.2\) | Move at \(\beta=1\); \(\ell=\ell_0/\sqrt{\max(\beta,1)}\) |
| `beta_max` | `--beta-max` | \(32\) | Projection continuation ceiling |
| `seed` | `--seed` | \(0\) | Init-noise PRNG. Ignored when `start_gamma` is passed |
| `outdir` | `--outdir` | `outputs` | `history.json`, `run.json`, `state_*.npz` |
| `start_gamma` | | `None` | Warm start; skips random noise and the channel seed |
| `abort_on_runaway` | | `True` | Raise `RunawaySolveError` if \(T\) blows up |
| `stall_iters` | | \(8\) | Stop after this many non-improving steps at \(\beta_{\max}\) (`0` disables) |
| `callback` | | `None` | `callback(it, gamma, aux, rec)` |
| mesh schedule | `--mesh-schedule` | unset | `nx,ny,iters:nx,ny,iters` for `optimize_hierarchy` |

Init noise amplitude is \(0.08\) (hard-coded). Flow problems add a
mid-height duct of height `port_frac` (fluid \(0.08\), solid \(0.78\))
before the noise. Gallery runs use \(\ell_0=0.12\).

### 8.4 `analyze` aux

| Key | Meaning |
|---|---|
| `phys` | \(\bar\gamma\) after filter + projection |
| `T`, `p`, `speed`, `face_vel` | Fields |
| `V` | \(\mathrm{mean}(\bar\gamma)\) |
| `energy_rms` | RMS of the discrete energy residual |
| `div_rms` | \(\|\nabla\cdot\mathbf{u}\|_{\mathrm{rms}}\) |
| `u_in`, `u_out`, `mass_err` | Port throughput; \(\lvert u_{\mathrm{in}}-u_{\mathrm{out}}\rvert/(\lvert u_{\mathrm{in}}\rvert+\varepsilon)\) |
| `stokes_rel` | Relative Stokes residual (0 if not Stokes) |
| `gray` | Fraction of cells with \(0.05<\bar\gamma<0.95\) |
| `T_mean`, `T_max`, `speed_max` | Field scalars |

Each optimize iterate also stores `J`, `vol`, `beta`, `move`, `sym_err`,
`is_best` (the returned high-\(\beta\) iterate). `run.json` adds
`stopped` (`completed` / `stall` / `runaway`), \(J_{\mathrm{best}}\) /
`best_iter` (same iterate), and \(J_{\mathrm{peak}}\) / `peak_iter`
(largest \(J\) at any \(\beta\)).

Example gallery (`python examples/gallery.py`, also
`python -m topoopt examples`) uses \(80\times 80\)
(Stokes \(48\times 48\)), \(150\)–\(200\) iterations (Stokes \(100\)),
\(\ell_0=0.12\), \(\beta_{\max}=32\). Outputs go under `outputs/`
(gitignored). Committed snapshots from shorter runs are in
`docs/figures/`. `python -m topoopt verify` is a short physics check,
not a crisp-design run. Short-run reference numbers used by CI are in
`examples/reference.json`. Per-script meshes and `--quick` values:
`examples/README.md`.

## 9. Limitations

- **2-D only.** There is no 3-D discretization.
- **Stokes, not Navier–Stokes.** Momentum has no \((\mathbf{u}\cdot\nabla)\mathbf{u}\)
  term. The model is a low-Reynolds Brinkman flow.
- **Darcy** is a potential-flow surrogate: no no-slip, no inertia.
- **Gray interfaces.** The Helmholtz filter keeps a band of intermediate
  \(\bar\gamma\) of width \(\sim r\), even at large \(\beta\).
- **Brinkman solid.** Solid is a large drag, not a cut-cell no-slip wall.
  Some leakage through “solid” remains.
- **First-order upwind** adds numerical diffusion at high \(\mathrm{Pe}\).
- **Iterative solves** are capped. Poorly conditioned high-\(\beta\)
  designs can leave a nonzero residual; the optimizer warns if the
  energy residual RMS exceeds \(10^{-2}\) or the port mass error
  exceeds \(0.15\).
- **Conduction at high \(\beta\).** With a volume source and a small
  sink, \(J=-\mathrm{mean}(T)\) can still oscillate once the projection
  becomes sharp (\(\beta\gtrsim 16\)). The \(\beta\)-damped move limit
  and per-level keep-best reduce the chatter that reaches the returned
  **sharp** design; volume stays at \(v^\*\). Do not publish a mid-β
  gray field just because its residual or \(J\) looked better.
- **Blocked flow, no conduction sink.** Flow modes have only the
  centerline ports. A sealed design can run \(T\) away; the energy
  residual warns and the optimizer aborts if \(T\) blows up. Do not
  add a cold face to hide a blocked channel.
- **Volumetric \(q\).** Heat generated in the source region (the whole
  box, or `q_specs`) is design-independent, so “maximize heat transfer”
  is the wrong default objective whenever a volume source is on;
  the objective is \(J=-\mathrm{mean}(T)\). Dirichlet-only runs use
  heat leaving `hot_specs`.
- **No property coupling** beyond \(\gamma\): \(k\), \(\alpha\), and
  \(\kappa\) do not depend on \(T\) or \(\mathbf{u}\).
