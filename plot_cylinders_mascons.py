# ---------------- MINIMAL PATCHES ONLY ----------------
# 1) add per-mascon colors + per-mascon legend entries ("Mascon 1", "Mascon 2", ...)
# 2) set alternate view (azim=-14, elev=15) via default view_elev_azim
#
# Everything else unchanged.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Sequence, List

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass(frozen=True)
class CylinderSpec:
    center: np.ndarray
    radius: float
    height: float
    rotation: np.ndarray
    name: str = ""


def set_pub_style(usetex: bool = True, font_size: int = 14) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "serif",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 2,
            "legend.fontsize": font_size - 1,
            "text.usetex": bool(usetex),
        }
    )


def _paper_3d_axes(
    ax, grid: bool = True, pane_alpha: float = 0.0, grid_alpha: float = 0.18
):
    try:
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_alpha(pane_alpha)
            a.pane.set_edgecolor("0.85")
    except Exception:
        pass

    ax.grid(bool(grid))
    if grid:
        try:
            for a in (ax.xaxis, ax.yaxis, ax.zaxis):
                a._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, grid_alpha)
                a._axinfo["grid"]["linewidth"] = 0.6
        except Exception:
            pass


def _set_axes_equal_3d(ax, xyz: np.ndarray, pad: float = 0.06) -> None:
    xyz = np.asarray(xyz, float)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    c = 0.5 * (mins + maxs)
    spans = maxs - mins
    L = float(np.max(spans))
    if not np.isfinite(L) or L <= 0:
        L = 1.0

    half = 0.5 * L * (1.0 + pad)
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_zlim(c[2] - half, c[2] + half)

    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


def _as3(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(3)


def _ensure_3x3(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"rotation must be (3,3), got {R.shape}")
    return R


def _local_to_world(P_local: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ P_local.T).T + t[None, :]


def _cylinder_surface_local(radius: float, height: float, n_theta: int, n_z: int):
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    z = np.linspace(0.0, height, n_z)  # <<< PATCH (was -0.5*h .. +0.5*h)
    TH, ZZ = np.meshgrid(theta, z)
    XX = radius * np.cos(TH)
    YY = radius * np.sin(TH)
    return XX, YY, ZZ


def _circle_local(radius: float, z: float, n: int = 160) -> np.ndarray:
    th = np.linspace(0.0, 2.0 * np.pi, n)
    return np.column_stack(
        [radius * np.cos(th), radius * np.sin(th), np.full_like(th, z)]
    )


def plot_shape_cylinders_mascons(
    vertices: np.ndarray,
    faces: np.ndarray,
    cylinders: Iterable[CylinderSpec],
    mascons: Optional[np.ndarray] = None,
    mascon_labels: Optional[Sequence[str]] = None,
    title: str = "",
    # performance
    fast: bool = True,
    decimate_faces: int = 10,
    # SHAPE LOOK
    mesh_face: str = "0.65",
    mesh_alpha: float = 0.65,
    mesh_edge_rgba=(0, 0, 0, 0.25),
    mesh_edge_lw: float = 0.25,
    # cylinders
    cyl_alpha: float = 0.20,
    show_caps: bool = False,
    # mascons (BIGGER)
    mascon_size: float = 420.0,
    mascon_color: str = "k",  # kept for backwards-compat; ignored if per-mascon colors used
    mascon_edge_color: str = "w",
    mascon_edge_lw: float = 1.8,
    mascon_alpha: float = 1.0,
    mascon_colors: Optional[Sequence[str]] = None,  # <<< PATCH: per-mascon colors
    # axes
    show_axis: bool = True,
    show_grid: bool = True,
    view_elev_azim: Tuple[float, float] = (
        15.0,
        -14.0,
    ),  # <<< PATCH: alternate view (elev=15, azim=-14)
    figsize: Tuple[float, float] = (7.4, 5.8),
    legend_loc: str = "upper left",
    savepath: Optional[str] = None,
):
    V = np.asarray(vertices, float)
    F = np.asarray(faces, int)

    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3), got {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"faces must be (M,3), got {F.shape}")

    cylinders = list(cylinders)

    if fast:
        n_theta, n_z, rim_n = 44, 10, 120
        shade, antialias = False, False
    else:
        n_theta, n_z, rim_n = 96, 20, 240
        shade, antialias = True, True

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # ---------------- Shape mesh ----------------
    decimate_faces = max(1, int(decimate_faces))
    tris = V[F[::decimate_faces]]

    mesh = Poly3DCollection(
        tris,
        facecolor=mesh_face,
        edgecolor=mesh_edge_rgba,
        linewidths=mesh_edge_lw,
        alpha=mesh_alpha,
    )
    try:
        mesh.set_zsort("min")
    except Exception:
        pass
    ax.add_collection3d(mesh)

    # ---------------- Cylinders ----------------
    cyl_colors = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#F0E442",  # yellow
        "#56B4E9",  # sky blue
        "#E69F00",  # orange
        "#000000",  # black (anchor / reference)
    ]
    cyl_pts_all: List[np.ndarray] = []
    legend_handles: List[Line2D] = []

    for k, spec in enumerate(cylinders):
        c = _as3(spec.center)
        R = _ensure_3x3(spec.rotation)
        r = float(spec.radius)
        h = float(spec.height)
        color = cyl_colors[k % len(cyl_colors)]
        name = spec.name if spec.name else f"Cylinder {k+1}"

        Xl, Yl, Zl = _cylinder_surface_local(r, h, n_theta=n_theta, n_z=n_z)
        Pl = np.column_stack([Xl.ravel(), Yl.ravel(), Zl.ravel()])
        Pw = _local_to_world(Pl, R, c)
        cyl_pts_all.append(Pw)

        Xw = Pw[:, 0].reshape(Xl.shape)
        Yw = Pw[:, 1].reshape(Yl.shape)
        Zw = Pw[:, 2].reshape(Zl.shape)

        ax.plot_surface(
            Xw,
            Yw,
            Zw,
            color=color,
            alpha=cyl_alpha,
            linewidth=0.0,
            antialiased=antialias,
            shade=shade,
            zorder=2,
        )

        # rims
        z_top, z_bot = h, 0.0  # <<< PATCH (was ±0.5*h)
        rim_top = _local_to_world(_circle_local(r, z_top, n=rim_n), R, c)
        rim_bot = _local_to_world(_circle_local(r, z_bot, n=rim_n), R, c)
        ax.plot(
            rim_top[:, 0], rim_top[:, 1], rim_top[:, 2], color=color, lw=1.1, alpha=0.95
        )
        ax.plot(
            rim_bot[:, 0], rim_bot[:, 1], rim_bot[:, 2], color=color, lw=1.1, alpha=0.95
        )

        legend_handles.append(Line2D([0], [0], color=color, lw=3.0, label=name))

    # ---------------- Mascons ----------------
    M = None
    if mascons is not None:
        M = np.asarray(mascons, float)
        if M.ndim != 2 or M.shape[1] != 3:
            raise ValueError(f"mascons must be (Nm,3), got {M.shape}")

        # <<< PATCH: draw each mascon with its own color and its own legend entry
        default_mascon_colors = [
            "#0072B2",  # blue
            "#D55E00",  # vermillion
            "#009E73",  # bluish green
            "#CC79A7",  # reddish purple
            "#F0E442",  # yellow
            "#56B4E9",  # sky blue
            "#E69F00",  # orange
            "#000000",  # black (anchor / reference)
        ]

        colors = (
            list(mascon_colors) if mascon_colors is not None else default_mascon_colors
        )

        for i in range(M.shape[0]):
            ci = colors[i % len(colors)]
            ax.scatter(
                M[i, 0],
                M[i, 1],
                M[i, 2],
                s=mascon_size,
                c=ci,
                edgecolors=mascon_edge_color,
                linewidths=mascon_edge_lw,
                alpha=mascon_alpha,
                depthshade=False,
                zorder=10,
            )

            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=ci,
                    markeredgecolor=mascon_edge_color,
                    markeredgewidth=mascon_edge_lw,
                    markersize=10,
                    label=f"Mascon {i+1}",
                )
            )

        # keep your optional text labels (m_1, m_2, ...) unchanged
        if mascon_labels is not None:
            if len(mascon_labels) != M.shape[0]:
                raise ValueError("mascon_labels length must match number of mascons.")
            span = float(np.max(np.ptp(V, axis=0)))
            d = 0.015 * span
            for i, lab in enumerate(mascon_labels):
                ax.text(
                    M[i, 0] + d,
                    M[i, 1] + d,
                    M[i, 2] + d,
                    str(lab),
                    fontsize=mpl.rcParams["font.size"] - 2,
                    zorder=11,
                )

    # ---------------- View / axes ----------------
    ax.view_init(elev=view_elev_azim[0], azim=view_elev_azim[1])
    try:
        ax.dist = 9.5
    except Exception:
        pass

    _paper_3d_axes(ax, grid=show_grid, pane_alpha=0.0, grid_alpha=0.18)

    pts = [V]
    if cyl_pts_all:
        pts.append(np.vstack(cyl_pts_all))
    if M is not None:
        pts.append(M)
    pts = np.vstack(pts)
    _set_axes_equal_3d(ax, pts, pad=0.06)

    if title:
        ax.set_title(title)

    if not show_axis:
        ax.set_axis_off()
    else:
        ax.set_xlabel(r"$x \, [LU]$")
        ax.set_ylabel(r"$y\, [LU]$")
        ax.set_zlabel(r"$z\, [LU]$")

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",  # anchor point on legend box
            bbox_to_anchor=(0.85, 0.8),  # place it just outside right
            borderaxespad=0.0,
            framealpha=0.85,
        )

    if savepath is not None:
        fig.savefig(savepath, dpi=600, bbox_inches="tight")

    return fig, ax


if __name__ == "__main__":
    import mesh_utility

    set_pub_style(usetex=True, font_size=14)

    vertices, faces = mesh_utility.read_pk_file("3dmeshes/eros.pk")
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces, int)

    cyls = [
        CylinderSpec(
            center=np.array([0.00, 0.00, 0.28]),
            radius=0.10,
            height=0.50,
            rotation=np.eye(3),
            name="Cylinder A",
        ),
        CylinderSpec(
            center=np.array([-0.10, -0.28, 0.00]),
            radius=0.10,
            height=0.50,
            rotation=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
            name="Cylinder B",
        ),
        CylinderSpec(
            center=np.array([-0.76, 0.00, 0.00]),
            radius=0.10,
            height=0.50,
            rotation=np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
            name="Cylinder C",
        ),
        CylinderSpec(
            center=np.array([0.00, 0.00, -0.29]),
            radius=0.10,
            height=0.50,
            rotation=np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            name="Cylinder D",
        ),
        CylinderSpec(
            center=np.array([0.00, 0.24, 0.00]),
            radius=0.10,
            height=0.50,
            rotation=np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            name="Cylinder E",
        ),
        CylinderSpec(
            center=np.array([0.79, 0.00, 0.00]),
            radius=0.10,
            height=0.50,
            rotation=np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
            name="Cylinder F",
        ),
    ]

    mascons = np.vstack([np.array([0.35, 0.0, 0.0]), np.array([-0.35, 0.0, 0.0])])
    mascon_labels = [r"$m_1$", r"$m_2$"]

    fig, ax = plot_shape_cylinders_mascons(
        vertices,
        faces,
        cyls,
        mascons=mascons,
        mascon_labels=mascon_labels,
        fast=False,
        decimate_faces=10,
        mascon_size=650,  # bigger
        mascon_colors=[
            "#33BB44",
            "#AA33CC",
        ],  # optional explicit colors
        view_elev_azim=(35.0, -130.0),  # requested view
        show_grid=True,
        show_axis=True,
        savepath=None,
    )
    plt.show()
