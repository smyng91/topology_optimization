# Tutorials and problem configs

Named research cases (geometry, ports, hot/cold patches) are defined in
[`problems.py`](problems.py). The `topoopt` package does not pick BCs
from the heat mode. JSON wrappers live in [`configs/`](configs/).
Short-run reference numbers are in [`reference.json`](reference.json).

```bash
python -m topoopt 2d --config examples.problems:conduction_tree
python -m topoopt 2d --config examples/configs/convection_darcy.json
python examples/gallery.py
python examples/publish_figures.py    # committed snapshots in docs/figures/
```

Symmetric factories set `symmetry` (`x` = left–right, `y` = top–bottom)
so random init cannot skew a mirror problem. `custom_boxes` is not
symmetric and leaves that field empty.

Short scripts that teach the API. Run them from the **repo root** after
`pip install -e ".[cpu,dev]"`.

```bash
python examples/01_analyze_once.py
python examples/01_analyze_once.py --quick          # coarse mesh / few iters
python examples/run_all.py --quick                  # smoke every tutorial
```

Each script writes PNG / VTK / JSON under `outputs/`.
`--quick` is for a sanity check, not a publishable design.

| Script | What it teaches | Typical time |
|---|---|---|
| [`01_analyze_once.py`](01_analyze_once.py) | `analyze()` only: solid vs fluid vs gray | < 30 s |
| [`02_conduction_tree.py`](02_conduction_tree.py) | Volume-to-point tree (`J = -mean(T)`, small sink) | a few min |
| [`03_convection_darcy.py`](03_convection_darcy.py) | Pressure-driven Darcy channel | a few min |
| [`04_conjugate_stokes.py`](04_conjugate_stokes.py) | Conjugate Stokes–Brinkman (seed + port pin) | several min |
| [`05_custom_regions.py`](05_custom_regions.py) | `--hot` / `--cold` faces or boxes; `J = Q_hot` | a few min |
| [`06_mms_check.py`](06_mms_check.py) | Manufactured / exact solutions | < 1 min (`--quick` skips Stokes) |

The long gallery (`python examples/gallery.py`) is a finer, longer
sweep of the factories in `problems.py` — not a tutorial.

## Reading order

1. **01** — fields and diagnostics, no optimizer.
2. **02** — why the sink is 8% and `vol = 0.30`.
3. **03** — one left-centerline inlet / right-centerline outlet, channel seed, `J = -mean(T)`.
4. **04** — Stokes notes (`stokes_dp`, Schur CG, port pin).
5. **05** — Dirichlet patches change both the BCs and the objective.
6. **06** — how we know the discrete operators are consistent.

The model write-up is [docs/model.md](../docs/model.md).
