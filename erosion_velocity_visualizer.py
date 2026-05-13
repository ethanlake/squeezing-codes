#!/usr/bin/env python3
"""
Visualize the angle-dependent erosion velocity v_θ for the three squeezing
rules R, F, M. For each rule, sample 50 points evenly around a unit circle
and draw an arrow normal to the circle whose signed length is v_θ at that
angle: positive v_θ → outward arrow, negative → inward.

Usage:
    python3 erosion_velocity_visualizer.py --h 0.1 --lam 0.5 --gam 0.3
"""

import argparse
import numpy as np

# Shim for matplotlib >= 3.9, which removed matplotlib.cbook._Stack that older
# interactive widgets still reference. Must run before pyplot is imported.
try:
    import matplotlib.cbook
    if not hasattr(matplotlib.cbook, "_Stack"):
        class _Stack(list):
            def push(self, item):
                self.append(item)
                return item
            def pop(self):
                return super().pop() if self else None
            def current(self):
                return self[-1] if self else None
            def forward(self):
                pass
            def back(self):
                pass
        matplotlib.cbook._Stack = _Stack
except Exception:
    pass

import matplotlib.pyplot as plt


def v_R(theta, h, lam, gam):
    return -h - lam * np.cos(2 * theta)


def v_F(theta, h, lam, gam):
    return -h - lam * np.sin(np.pi / 4 + theta) - gam * np.sin(np.pi / 4 - theta)


def v_M(theta, h, lam, gam):
    return -h - lam * np.sin(theta) - gam * np.cos(theta)


def _draw_panel(ax, v_func, color, h, lam, gam, n_points=50):
    thetas = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)
    cx = np.cos(thetas)
    cy = np.sin(thetas)
    v = v_func(thetas, h, lam, gam)

    # Outward unit normal at each point on the unit circle is just (cos, sin).
    # Sign of v handles inward arrows automatically when v < 0.
    dx = v * cx
    dy = v * cy

    circle = plt.Circle((0, 0), 1.0, fill=False, ec="k", lw=1.5)
    ax.add_patch(circle)
    ax.quiver(cx, cy, dx, dy,
              angles="xy", scale_units="xy", scale=1.0,
              color=color, width=0.006)

    extent = 1.0 + max(1e-3, np.max(np.abs(v))) + 0.2
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h",   type=float, required=True,
                    help=r"$\widetilde h$, the constant offset.")
    ap.add_argument("--lam", type=float, required=True,
                    help=r"$\widetilde\lambda$, the cos(2θ) / sin coefficient.")
    ap.add_argument("--gam", type=float, required=True,
                    help=r"$\widetilde\gamma$, the second-anisotropy coefficient "
                         "(unused by R, used by F and M).")
    args = ap.parse_args()

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern Roman"] + plt.rcParams.get(
        "font.serif", [])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, v_func, color, label in zip(
            axes,
            (v_R, v_F, v_M),
            ("purple", "red", "blue"),
            (r"$\mathsf{R}$", r"$\mathsf{F}$", r"$\mathsf{M}$"),
        ):
        _draw_panel(ax, v_func, color, args.h, args.lam, args.gam)
        ax.set_title(label, fontsize=14)

    fig.suptitle(rf"$\widetilde h={args.h:g},\ \widetilde\lambda={args.lam:g},"
                 rf"\ \widetilde\gamma={args.gam:g}$", fontsize=12)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
