"""
Diamond - render the 2D angle map (metric r2 over azimuth x elevation) as a
matplotlib heatmap for the invention-disclosure figure (Fig. 2).

Reads angle_map_2d.csv (produced by analysis/angle_map_2d.py) so the figure
always tracks the actual validation numbers. Blue sequential colormap for r2.

Layout:
  - azimuth 0..180 (0=side, 90=front, 180=mirror side). Orthographic projection
    is front-back symmetric, so az=0 and az=180 are identical and az=45/135 are
    near-mirrors; the full 0..180 span is shown for completeness.
  - elevation 0/30/60/85 (overhead row = 85, the validated regime; true 90 is the
    geometric singularity and is intentionally NOT sampled).
  - rows grouped into (1) ground-camera metrics and (2) overhead-only metrics.
  - distance/speed metrics whose vertical (stature) normalization collapses at
    elevation are drawn dashed + faded (disregard those cells).

Run:
  conda activate diamond
  cd src\viz
  python angle_map_2d_fig.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

AZ = [0, 45, 90, 135, 180, 225, 270, 315]   # full 360deg orbit
EL = [0, 30, 60, 85]
NAZ = len(AZ)
GAP = 0.6

# CSV metric label -> (display name, normalizer-confounded at high elevation, region)
ORDER = [
    ("Lead Knee Angle [O]",  "Lead knee angle",   False, "ground"),
    ("Stride (anchor) [O]",  "Stride length",     True,  "ground"),
    ("Trunk Tilt (ant) [O]", "Trunk tilt (ant.)", False, "ground"),
    ("Knee Ext Velo BR [O]", "Knee ext. velo",    False, "ground"),
    ("Wrist Speed [O]",      "Wrist speed",        True,  "ground"),
    ("Release Height [O]",   "Release height",     True,  "ground"),
    ("Arm Slot [O]",         "Arm slot",          False, "ground"),
    ("Hip-Shoulder Sep [O]", "Hip-shoulder sep",  False, "overhead"),
    ("Pelvis Rot Velo [O]",  "Pelvis rot. velo",  False, "overhead"),
]
CMAP = LinearSegmentedColormap.from_list(
    "diamondblue",
    ["#E6F1FB", "#B5D4F4", "#85B7EB", "#378ADD", "#185FA5", "#0C447C", "#042C53"])
GROUP_TITLES = ["Side · el 0°", "el 30°", "el 60°", "Overhead · el 85°"]


def fmt(v):
    if not np.isfinite(v):
        return ""
    if v >= 0.995:
        return "1.0"
    return f"{v:.2f}"[1:]          # ".94"


def xpos(block, a):               # left x of a cell
    return block * (NAZ + GAP) + a


def main():
    csv = os.path.join(config.OBP_VALIDATION_DIR, "angle_map_2d.csv")
    df = pd.read_csv(csv)
    n = len(ORDER)
    n_over = sum(1 for *_, reg in ORDER if reg == "overhead")

    fig, ax = plt.subplots(figsize=(23, 7))
    norm = Normalize(0, 1)

    grid_r = xpos(len(EL) - 1, 0) + NAZ            # right edge of the grid

    # region background band behind the overhead rows (rows sit at the bottom)
    ax.add_patch(Rectangle((-0.15, 0), grid_r + 0.15, n_over,
                 facecolor="#F6E9CC", edgecolor="none", alpha=0.55, zorder=0))

    for ri, (key, disp, confd, region) in enumerate(ORDER):
        ybot = n - 1 - ri                       # row 0 at top
        sub = df[df["metric"] == key]
        ax.text(-0.35, ybot + 0.5, disp, ha="right", va="center",
                fontsize=11, fontweight="medium", color="#2C2C2A", zorder=3)
        for b, el in enumerate(EL):
            for a, az in enumerate(AZ):
                row = sub[(sub["el"] == el) & (sub["az"] == az)]
                v = float(row["r2"].iloc[0]) if len(row) else np.nan
                x = xpos(b, a)
                dashed = confd and el > 0
                fc = CMAP(norm(v)) if np.isfinite(v) else "#F1EFE8"
                ax.add_patch(Rectangle((x, ybot), 1, 1, facecolor=fc,
                             edgecolor="white", linewidth=1.5, zorder=2,
                             alpha=0.5 if dashed else 1.0))
                if dashed:
                    ax.add_patch(Rectangle((x + 0.04, ybot + 0.04), 0.92, 0.92,
                                 facecolor="none", edgecolor="#808080",
                                 linewidth=1.3, linestyle=(0, (3, 2)), zorder=2))
                tc = "white" if (np.isfinite(v) and v >= 0.45 and not dashed) else "#042C53"
                ax.text(x + 0.5, ybot + 0.5, fmt(v), ha="center", va="center",
                        fontsize=10, color=tc, zorder=3)

    # divider between ground and overhead regions
    ax.plot([-0.15, grid_r], [n_over, n_over], color="#C9A24B",
            linewidth=1.4, zorder=4)

    # region tags (right edge)
    ax.text(grid_r + 0.15, n - 0.5, "① GROUND CAMERA", ha="left", va="center",
            rotation=90, fontsize=9.5, fontweight="bold", color="#5F5E5A")
    ax.text(grid_r + 0.15, n_over / 2, "② OVERHEAD", ha="left", va="center",
            rotation=90, fontsize=9.5, fontweight="bold", color="#B07D18")

    # azimuth subheaders + elevation group headers
    for b, title in enumerate(GROUP_TITLES):
        ax.text(xpos(b, 0) + NAZ / 2, n + 0.85, title, ha="center", va="center",
                fontsize=11.5, fontweight="medium", color="#2C2C2A")
        ax.plot([xpos(b, 0), xpos(b, 0) + NAZ], [n + 0.5, n + 0.5],
                color="#D3D1C7", linewidth=0.8)
        for a, az in enumerate(AZ):
            ax.text(xpos(b, a) + 0.5, n + 0.15, f"{az}°",
                    ha="center", va="center", fontsize=8.5, color="#5F5E5A")
    ax.text(-0.35, n + 0.15, "azimuth →", ha="right", va="center",
            fontsize=10, color="#5F5E5A")

    ax.set_xlim(-3.4, grid_r + 1.4)
    ax.set_ylim(-1.5, n + 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # colorbar
    sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035,
                      pad=0.03, aspect=45)
    cb.ax.tick_params(labelsize=9)
    cb.set_label("coefficient of determination  r²", fontsize=9.5, color="#2C2C2A")

    # dashed-cell legend
    ax.add_patch(Rectangle((-0.15, -1.05), 0.7, 0.6, facecolor="none",
                 edgecolor="#808080", linewidth=1.3, linestyle=(0, (3, 2))))
    ax.text(0.75, -0.75, "normaliser confounded — disregard",
            ha="left", va="center", fontsize=9, color="#5F5E5A")

    out = os.path.join(config.OBP_VALIDATION_DIR, "angle_map_2d.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
