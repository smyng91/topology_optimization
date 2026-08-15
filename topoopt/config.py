"""Problem parameters for 2-D box topology optimization."""

from __future__ import annotations

from typing import Literal, NamedTuple

HeatMode = Literal["conduction", "convection", "both"]
HEAT_MODES: tuple[HeatMode, ...] = ("conduction", "convection", "both")
HEAT_MODE_LABELS = {
    "conduction": "fully conductive",
    "convection": "fully convective",
    "both": "conductive + convective",
}


class ColdPlateParams(NamedTuple):
    """Nondimensional 2-D box: uniform volumetric heating, optional flow ports."""

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
    eta: float = 0.5
    port_frac: float = 0.5
    hot_specs: tuple[str, ...] = ()
    cold_specs: tuple[str, ...] = ()
    div_eps: float = 1.0e-4
    solver_tol: float = 1.0e-7
    flow_iters: int = 80
    uzawa_iters: int = 200
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
        """Uniform volumetric heating when no Dirichlet heat-source patches exist."""
        return self.q_vol != 0.0 and not self.hot_specs

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
        default="both",
        help=(
            "Heat-transfer physics: 'conduction' (volume source, small bottom sink), "
            "'convection' (volume source, centered ports), "
            "or 'both' (conjugate). Default: both"
        ),
    )


def default_regions(heat_mode: HeatMode, port_frac: float = 0.5) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Mode-specific default hot / cold specs (empty hot ⇒ uniform volume source)."""
    if heat_mode == "conduction":
        # Narrow sink: a wide patch makes parallel fins, not a branching tree.
        return (), ("face:bottom:frac=0.08",)
    return (), ()


def default_2d(
    nx: int = 40,
    ny: int = 40,
    lx: float = 1.0,
    ly: float = 1.0,
    **kwargs,
) -> ColdPlateParams:
    mode = kwargs.get("heat_mode", "both")
    port = kwargs.get("port_frac", 0.5)
    hot_d, cold_d = default_regions(mode, port)
    kwargs.setdefault("hot_specs", hot_d)
    kwargs.setdefault("cold_specs", cold_d)
    return ColdPlateParams(n=(nx, ny), L=(lx, ly), **kwargs)
