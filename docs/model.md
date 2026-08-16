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
| `conduction` | off, \(\mathrm{Pe}=0\) | \(k(\gamma)\) | centered 8% of the bottom wall at \(T=0\); other walls adiabatic | uniform \(q\) |
| `convection` | on | uniform \(k_\mathrm{fluid}\) | one left-centerline inlet, one right-centerline outlet; inlet \(T=0\) by advection; no other inlets, outlets, or cold patches | uniform \(q\) |
| `both` | on | \(k(\gamma)\) | same single inlet / outlet as convection | uniform \(q\) |

These BCs are **example configurations**, not library defaults. They are
defined in `examples/problems.py` and loaded with
`--config examples.problems:<name>`. The solver accepts any
`hot_specs` / `cold_specs` / `port_frac`. Setting `--hot` turns the
volume source off.

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

Default source: \(q=q_{\mathrm{vol}}=1\) everywhere. If `--hot` is set,
\(q=0\) and those patches are Dirichlet sources at \(T_{\mathrm{hot}}=1\).

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
- Optional `--hot` / `--cold` faces or boxes override the defaults.
  Specs: `face:bottom:frac=0.5`, `face:top:frac=0.4:center=0.3`,
  `box:xmin,xmax,ymin,ymax`.

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

With the default uniform source, the heat that eventually leaves the box
is \(\int_\Omega q\,\mathrm{d}V\), independent of \(\gamma\). The design
cannot change how much heat is rejected, only how hot the box runs.
The figure of merit is therefore

\[
J=-\frac{1}{|\Omega|}\int_\Omega T\,\mathrm{d}V
=-\mathrm{mean}(T).
\]

Maximizing \(J\) minimizes mean temperature.

If `--hot` is set, \(q=0\) and

\[
J=Q_{\mathrm{hot}},
\]

the conductive heat leaving the Dirichlet heat-source faces and/or
source boxes into the rest of the domain (Fourier flux on those
interfaces).

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
3. Keep the design with the largest \(J\). At \(\beta_{\max}\), stop
   early if \(J\) has not improved for `stall_iters` iterations
   (default 8) so high-\(\beta\) chatter does not run out the budget.
   Write `history.json`, `run.json` (serializable params, \(J_0\),
   \(J_{\mathrm{final}}\), \(J_{\mathrm{best}}\), `stopped`, last
   diagnostics), `state_best.npz`, `state_final.npz`, PNG slices, and
   a VTK file of cell-centered \(\bar\gamma\), \(T\), \(|\mathbf{u}|\),
   and \(p\). The optimize return value is the **best-\(J\)** design.

`optimize_hierarchy` resizes the best field with bilinear interpolation
and continues on a finer mesh (`--mesh-schedule nx,ny,iters:…`).

There is no MMA / IPOPT / optimality-criteria loop. The volume equality
is a hard projection, not a penalty.

## 8. Default parameters

| Quantity | Symbol / flag | Default |
|---|---|---|
| Domain | \(L\), \(n\) | \((1,1)\), \((40,40)\) |
| Heat mode | `--heat` | `both` |
| Flow model | `--flow` | `stokes` |
| Volume fraction | `--vol` \(v^\*\) | \(0.45\) |
| Péclet number | `--pe` | \(40\) |
| Volume source | `--q` | \(1\) |
| \(k_s/k_f\) | `--k-ratio` | \(100\) |
| Filter radius (cells) | `--rmin` | \(2.2\) |
| Projection threshold | \(\eta\) | \(0.5\) |
| Max projection | `--beta-max` | \(32\) |
| Iterations | `--iters` | \(80\) |
| Move limit at \(\beta=1\) | `--lr` \(\ell_0\) | \(0.2\) (\(\ell=\ell_0/\sqrt{\beta}\)) |
| Port / sink fraction | `--port-frac` | \(0.5\) |
| Inlet / Darcy pressure | \(p_{\mathrm{in}}\) | \(1\) |
| Stokes pressure drop | `stokes_dp` | \(20\) |
| Uzawa warm-start passes | `uzawa_iters` | \(80\) |
| Stokes Schur CG | `stokes_kryl_iters` | \(200\) |
| Inlet / hot temperatures | \(T_{\mathrm{in}}\), \(T_{\mathrm{hot}}\) | \(0\), \(1\) |
| Brinkman \(\alpha_{\max}\) | | \(10^{5}\) |
| Darcy \(\kappa_{\min}\) | | \(10^{-6}\) |
| Continuity regularizer | \(\varepsilon\) | \(10^{-4}\) |
| Solver tolerance | | \(10^{-7}\) |
| Random seed | `--seed` | \(0\) |
| Design symmetry | `--symmetry` | from the named factory (`x`, `y`, or none) |

Example gallery (`python examples/gallery.py`, also
`python -m topoopt examples`) uses \(80\times 80\)
(Stokes \(48\times 48\)), \(150\)–\(200\) iterations (Stokes \(100\)),
\(\beta_{\max}=32\), \(r_{\min}=2.0\) (conduction \(1.5\)). Outputs go
under `outputs/` (gitignored). The conduction case uses a small bottom
sink and \(v^\*=0.30\) so a branching tree can form. Committed
snapshots from shorter runs are in `docs/figures/`. JSON problem files
are in `examples/configs/`. `python -m topoopt verify` is a short
physics check, not a crisp-design run. Short-run reference numbers
used by CI are in `examples/reference.json`.

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
  and keep-best reduce the chatter that reaches the returned design;
  volume stays at \(v^\*\).
- **Blocked flow, no conduction sink.** Flow modes have only the
  centerline ports. A sealed design can run \(T\) away; the energy
  residual warns and the optimizer aborts if \(T\) blows up. Do not
  add a cold face to hide a blocked channel.
- **Uniform \(q\).** Heat rejected is design-independent, so “maximize
  heat transfer” is the wrong default objective for this setup.
- **No property coupling** beyond \(\gamma\): \(k\), \(\alpha\), and
  \(\kappa\) do not depend on \(T\) or \(\mathbf{u}\).
