"""Plotting and VTK export for 2-D box designs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _draw_region_overlays(ax, params):
    from matplotlib.patches import Rectangle

    from topoopt.regions import overlay_boxes_2d, overlay_segments_2d

    for x0, y0, x1, y1, color in overlay_segments_2d(params):
        ax.plot([x0, x1], [y0, y1], color=color, lw=2.5, solid_capstyle="butt")
    for xmin, xmax, ymin, ymax, color in overlay_boxes_2d(params):
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor=color,
                lw=1.5,
                linestyle="--",
            )
        )
    if params.solves_flow:
        lx, ly = params.L
        y0 = 0.5 * ly * (1.0 - params.port_frac)
        y1 = 0.5 * ly * (1.0 + params.port_frac)
        ax.plot([0.0, 0.0], [y0, y1], color="limegreen", lw=2.5, solid_capstyle="butt")
        ax.plot([lx, lx], [y0, y1], color="limegreen", lw=2.5, solid_capstyle="butt")


def plot_2d(aux, params, path: str | Path, title: str = ""):
    import matplotlib.pyplot as plt

    phys = np.asarray(aux["phys"])
    temp = np.asarray(aux["T"])
    speed = np.asarray(aux["speed"])
    lx, ly = params.L
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    fields = (
        (phys, "Solid density γ", "viridis", 0.0, 1.0),
        (temp, "Temperature", "inferno", None, None),
        (speed, "Speed |u|", "cividis", 0.0, None),
    )
    for ax, (field, label, cmap, vmin, vmax) in zip(axes, fields):
        im = ax.imshow(
            field.T,
            origin="lower",
            extent=[0.0, lx, 0.0, ly],
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        _draw_region_overlays(ax, params)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_vtk(aux, params, path: str | Path):
    """Legacy VTK structured points at cell centers (open in ParaView)."""
    phys = np.asarray(aux["phys"])
    temp = np.asarray(aux["T"])
    speed = np.asarray(aux["speed"])
    pressure = np.asarray(aux["p"])
    phys = phys[:, :, None]
    temp = temp[:, :, None]
    speed = speed[:, :, None]
    pressure = pressure[:, :, None]
    dx = (*params.dx, min(params.dx))
    origin = (0.5 * params.dx[0], 0.5 * params.dx[1], 0.0)
    nx, ny, nz = phys.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write("box topology optimization\n")
        fh.write("ASCII\n")
        fh.write("DATASET STRUCTURED_POINTS\n")
        fh.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        fh.write(f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n")
        fh.write(f"SPACING {dx[0]} {dx[1]} {dx[2]}\n")
        fh.write(f"POINT_DATA {nx * ny * nz}\n")
        for name, field in (
            ("density", phys),
            ("temperature", temp),
            ("speed", speed),
            ("pressure", pressure),
        ):
            fh.write(f"SCALARS {name} double 1\n")
            fh.write("LOOKUP_TABLE default\n")
            np.savetxt(fh, field.ravel(order="F"), fmt="%.8e")
