"""Generic 2-D parameter object and loader.

Research cases (geometry, ports, hot/cold patches) live in
``examples/problems.py``. This module does not infer BCs from ``heat_mode``.

``load_params`` accepts a Python factory (``module:func`` / ``path.py:func``)
or a JSON / YAML file. The package does not import ``examples.problems``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple

HeatMode = Literal["conduction", "convection", "both"]
HEAT_MODES: tuple[HeatMode, ...] = ("conduction", "convection", "both")
FilterKind = Literal["cone", "helmholtz"]
FILTER_KINDS: tuple[FilterKind, ...] = ("cone", "helmholtz")
HEAT_MODE_LABELS = {
    "conduction": "fully conductive",
    "convection": "fully convective",
    "both": "conductive + convective",
}


class ColdPlateParams(NamedTuple):
    """Nondimensional 2-D box. All geometry and BCs are caller-supplied.

    Library defaults below are what ``params2d()`` uses when a field is
    omitted. Named research cases in ``examples/problems.py`` override a
    subset (volume, filter radius, ports, patches, solver caps). See
    ``docs/model.md`` §8 and ``examples/README.md``.

    Geometry
        n, L — cells and box size. ``dx = L/n``.
    Physics mode
        heat_mode — ``conduction`` (Pe=0, k(γ)), ``convection`` (flow,
        uniform k_fluid), ``both`` (flow and k(γ)).
        flow_model — ``stokes`` or ``darcy``; ignored when heat_mode is
        conduction.
    Design / interpolation
        vol_frac — target mean(γ̄). rmin — filter radius in cells
        (cone support, or Helmholtz r). filter_kind — ``cone`` (default)
        or ``helmholtz``.
        eta — tanh-projection threshold. q_k, q_alpha, q_kappa — RAMP /
        Borrvall–Petersson sharpness. k_fluid, k_solid, alpha_*, kappa_*.
    Forcing / BCs
        q_vol — volumetric heat strength. Uniform when q_specs is empty
        and hot_specs is empty; restricted to q_specs otherwise. Off
        when q_vol=0, or when hot_specs is set and q_specs is empty.
        q_specs — ``face:…`` / ``box:…`` cells that receive q (T still
        floats). pe — Péclet (zero in conduction). p_in — Darcy
        left-port pressure. stokes_dp — Stokes left-port pressure.
        t_in, t_hot — Dirichlet temperatures. port_frac — centered
        height of both vertical ports. hot_specs, cold_specs —
        ``face:…`` / ``box:…`` Dirichlet T patches. symmetry — ``x``
        and/or ``y`` mirror after every design step.
    Solvers
        div_eps — Stokes continuity regularizer εp. solver_tol — Krylov
        tolerance. flow_iters — Darcy CG / Stokes momentum CG. uzawa_iters
        — Stokes pressure-correction warm start. stokes_kryl_iters —
        pressure-Schur CG (0 skips the correction). heat_iters —
        energy BiCGSTAB. filter_iters — Helmholtz CG (unused by cone).
    Unused by the PDE
        u_in_max — peak of ``grid.inlet_profile`` (parabolic). Current
        Darcy/Stokes solves are pressure-driven and do not prescribe it.
    """

    n: tuple[int, int]
    L: tuple[float, float]
    flow_model: Literal["stokes", "darcy"] = "stokes"
    heat_mode: HeatMode = "both"
    vol_frac: float = 0.45
    pe: float = 40.0
    k_fluid: float = 1.0
    k_solid: float = 100.0
    q_k: float = 1.0
    q_alpha: float = 0.1
    q_kappa: float = 0.1
    q_vol: float = 1.0
    alpha_min: float = 0.0
    alpha_max: float = 1.0e5
    kappa_min: float = 1.0e-6
    kappa_max: float = 1.0
    p_in: float = 1.0
    stokes_dp: float = 20.0
    u_in_max: float = 1.0
    t_in: float = 0.0
    t_hot: float = 1.0
    rmin: float = 2.2
    filter_kind: FilterKind = "cone"
    eta: float = 0.5
    port_frac: float = 0.5
    hot_specs: tuple[str, ...] = ()
    cold_specs: tuple[str, ...] = ()
    q_specs: tuple[str, ...] = ()
    symmetry: tuple[str, ...] = ()
    div_eps: float = 1.0e-4
    solver_tol: float = 1.0e-7
    flow_iters: int = 80
    uzawa_iters: int = 80
    stokes_kryl_iters: int = 200
    heat_iters: int = 400
    filter_iters: int = 200

    @property
    def dim(self) -> int:
        return len(self.n)

    @property
    def solves_flow(self) -> bool:
        return self.heat_mode != "conduction"

    @property
    def effective_pe(self) -> float:
        return 0.0 if self.heat_mode == "conduction" else self.pe

    @property
    def uses_volume_source(self) -> bool:
        """Nonzero q, either uniform or restricted to ``q_specs``.

        Dirichlet ``hot_specs`` still turn off *uniform* q (legacy).
        Set ``q_specs`` to heat a subdomain while keeping prescribed T.
        """
        if self.q_vol == 0.0:
            return False
        return bool(self.q_specs) or not self.hot_specs

    @property
    def heat_label(self) -> str:
        return HEAT_MODE_LABELS[self.heat_mode]

    @property
    def dx(self) -> tuple[float, float]:
        return (self.L[0] / self.n[0], self.L[1] / self.n[1])

    @property
    def cell_volume(self) -> float:
        hx, hy = self.dx
        return hx * hy


def add_heat_mode_argument(parser) -> None:
    parser.add_argument(
        "--heat",
        choices=HEAT_MODES,
        default=None,
        help=(
            "Which energy terms are active: conduction (Pe=0), convection "
            "(uniform k, flow), or both (conjugate). Omit when --config sets it."
        ),
    )


_TUPLE_KEYS = ("n", "L", "hot_specs", "cold_specs", "q_specs", "symmetry")
_FILE_KEYS = set(ColdPlateParams._fields) | {
    "nx",
    "ny",
    "lx",
    "ly",
    "n",
    "L",
    "factory",
}


def coerce_param_kwargs(data: dict) -> dict:
    """Turn JSON lists / comma-strings into the tuples ``ColdPlateParams`` wants."""
    from topoopt.symmetry import normalize_axes

    out = dict(data)
    if "symmetry" in out:
        out["symmetry"] = normalize_axes(out["symmetry"])
    if "filter_kind" in out:
        kind = str(out["filter_kind"]).lower()
        if kind not in FILTER_KINDS:
            raise ValueError(
                f"filter_kind must be one of {FILTER_KINDS}, got {out['filter_kind']!r}"
            )
        out["filter_kind"] = kind
    for key in _TUPLE_KEYS:
        if key in out and isinstance(out[key], list):
            out[key] = tuple(out[key])
    return out


def params2d(
    nx: int = 40,
    ny: int = 40,
    lx: float = 1.0,
    ly: float = 1.0,
    **kwargs,
) -> ColdPlateParams:
    """Assemble a 2-D parameter set. BCs are not inferred from ``heat_mode``."""
    return ColdPlateParams(n=(nx, ny), L=(lx, ly), **coerce_param_kwargs(kwargs))


def params_from_dict(data: dict, **overrides) -> ColdPlateParams:
    """Build params from a JSON/YAML object. Unknown keys (e.g. ``comment``) are ignored."""
    merged = {k: v for k, v in {**data, **overrides}.items() if k in _FILE_KEYS}
    merged = coerce_param_kwargs(merged)
    factory = merged.pop("factory", None)
    if factory:
        return load_params(str(factory), **merged)
    nx = merged.pop("nx", None)
    ny = merged.pop("ny", None)
    n = merged.pop("n", None)
    L = merged.pop("L", None)
    lx = merged.pop("lx", None)
    ly = merged.pop("ly", None)
    if n is not None:
        if L is None:
            L = (lx if lx is not None else 1.0, ly if ly is not None else 1.0)
        return ColdPlateParams(n=tuple(n), L=tuple(L), **merged)
    return params2d(
        nx=40 if nx is None else int(nx),
        ny=40 if ny is None else int(ny),
        lx=1.0 if lx is None else float(lx),
        ly=1.0 if ly is None else float(ly),
        **merged,
    )


def _read_params_file(path: Path) -> dict:
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix == ".json":
        import json

        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required for .yaml configs (pip install pyyaml)") from exc
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"unsupported config suffix {suffix!r} for {path}")
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a mapping, not {type(data).__name__}")
    return data


def load_params(spec: str, **overrides) -> ColdPlateParams:
    """Load a problem factory or a JSON/YAML file.

    * ``module:callable`` — ``examples.problems:convection_darcy``
    * ``path.py:callable`` — ``examples/problems.py:convection_darcy``
    * ``path.json`` / ``path.yaml`` — a parameter object, optionally with
      a ``factory`` key that is itself a ``module:callable``

    Extra keywords override the file or are forwarded to the factory.
    """
    import importlib
    import importlib.util

    path = Path(spec)
    if spec.endswith((".json", ".yaml", ".yml")) or path.suffix.lower() in {".json", ".yaml", ".yml"}:
        return params_from_dict(_read_params_file(path), **overrides)
    if ":" not in spec:
        raise ValueError(
            "expected module:function, path.py:function, or a .json/.yaml file, got %r" % spec
        )
    target, name = spec.rsplit(":", 1)
    if target.endswith(".py"):
        path = Path(target).resolve()
        loader = importlib.util.spec_from_file_location(f"_topoopt_cfg_{path.stem}", path)
        if loader is None or loader.loader is None:
            raise FileNotFoundError(path)
        mod = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    obj = getattr(mod, name)
    if not callable(obj):
        if overrides:
            raise TypeError(f"{spec} is not callable; cannot apply overrides")
        return obj
    return obj(**coerce_param_kwargs(overrides))
