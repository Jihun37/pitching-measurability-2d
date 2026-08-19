"""
⚠ SUPERSEDED 2026-07-28 — SUPPLEMENT ONLY, NOT A BODY FIGURE.

This draws the 16-metric r2 ZONE map. The paper's map is now GRADED and has
**35 rows** (29 strong-capable + 6 moderate-only); the r2 layer is only a
SCREENING pre-filter. ORDER below is frozen at the 2026-07-24 16-metric
convention and therefore CANNOT draw the current map.

Body figure = viz/fig_graded_map.py (fig_graded_map.png / fig_graded_strip.png).
Old renders archived: data/outputs/obp_validation/archive_stale_20260728/.
Authority for counts: docs/legacy_pre_dedup/FINAL_GATE_MAP_394.md.
"""
"""
Diamond - render the PRECISE angle map (15deg az x 7 el) in the "numbers-in-
boxes" heatmap style (same look as angle_map_2d_fig.py, but on the fine grid).

Reads angle_zone_sweep_gt_clean.csv - the GT-event map on the clean population
(n=394, 16 metrics, 2026-07-24 convention). The metric list is shared with
angle_zone_fig.py via map_metrics.py and is verified against the CSV at render
time. One figure, one
horizontal band per metric; inside each band the 7 elevation rows (85 at top ->
0 at bottom) x 24 azimuth columns (0..345 step 15). r2 printed in every cell.
Blue sequential colormap. Normalizer-confounded metrics (stride / wrist speed /
release height) are hatched + faded at el>0 (disregard those cells).

Run:
  conda activate diamond
  cd src\viz
  python angle_zone_map_numbers.py
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

AZ = list(range(0, 360, 15))            # 0..345, 24 columns
EL = [85, 75, 60, 45, 30, 15, 0]        # top row = overhead, bottom = ground
NAZ = len(AZ)
NEL = len(EL)
BAND_GAP = 0.7                          # vertical gap between metric bands

from map_metrics import ORDER, LABEL_COLOR, sweep_path, check_coverage

CMAP = LinearSegmentedColormap.from_list(
    "diamondblue",
    ["#E6F1FB", "#B5D4F4", "#85B7EB", "#378ADD", "#185FA5", "#0C447C", "#042C53"])


def fmt(v):
    if not np.isfinite(v):
        return ""
    if v >= 0.995:
        return "1.0"
    return f"{v:.2f}"[1:]              # ".94"


def band_ybot(ri):
    """bottom y of metric-band ri (band 0 at top)."""
    n = len(ORDER)
    return (n - 1 - ri) * (NEL + BAND_GAP)


def main():
    df = pd.read_csv(sweep_path())
    check_coverage(df)
    n = len(ORDER)
    norm = Normalize(0, 1)

    top_y = band_ybot(0) + NEL
    # height scales with the metric count so the cells keep their size as the
    # adopted set grows (13 bands fitted 22in; 16 bands need ~27)
    fig, ax = plt.subplots(figsize=(15, 1.7 * n))

    for ri, (key, disp, confd, region) in enumerate(ORDER):
        sub = df[df["metric"] == key]
        ybot = band_ybot(ri)

        # metric name (left)
        col = LABEL_COLOR.get(region, "#2C2C2A")
        ax.text(-2.6, ybot + NEL / 2, disp, ha="right", va="center",
                fontsize=12, fontweight="bold", color=col, zorder=3)

        for r, el in enumerate(EL):
            yrow = ybot + (NEL - 1 - r)          # el 85 at top of band
            ax.text(-0.4, yrow + 0.5, f"{el}", ha="right", va="center",
                    fontsize=7.5, color="#5F5E5A", zorder=3)
            for a, az in enumerate(AZ):
                row = sub[(sub["el"] == el) & (sub["az"] == az)]
                v = float(row["r2"].iloc[0]) if len(row) else np.nan
                dashed = confd and el > 0
                fc = CMAP(norm(v)) if np.isfinite(v) else "#F1EFE8"
                ax.add_patch(Rectangle((a, yrow), 1, 1, facecolor=fc,
                             edgecolor="white", linewidth=0.8, zorder=2,
                             alpha=0.45 if dashed else 1.0))
                if dashed:
                    ax.add_patch(Rectangle((a + 0.03, yrow + 0.03), 0.94, 0.94,
                                 facecolor="none", edgecolor="#8A8A8A",
                                 linewidth=0.7, linestyle=(0, (2, 1.6)), zorder=2))
                tc = "white" if (np.isfinite(v) and v >= 0.45 and not dashed) else "#042C53"
                ax.text(a + 0.5, yrow + 0.5, fmt(v), ha="center", va="center",
                        fontsize=6.2, color=tc, zorder=3)

        # "el" axis tag on the left of each band
        ax.text(-1.15, ybot + NEL / 2, "el", ha="center", va="center",
                rotation=90, fontsize=7.5, color="#9A9890", zorder=3)

    # azimuth header (top) + bottom
    for yy, va in [(top_y + 0.35, "bottom")]:
        for a, az in enumerate(AZ):
            ax.text(a + 0.5, yy, f"{az}", ha="center", va=va,
                    fontsize=7, color="#5F5E5A", rotation=0)
        ax.text(NAZ / 2, yy + 1.0, "azimuth  (deg)  0=open side · 90=front · 180=glove side · 270=behind",
                ha="center", va="bottom", fontsize=10, color="#2C2C2A")

    ax.set_xlim(-3.2, NAZ + 0.4)
    ax.set_ylim(-1.8, top_y + 2.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # colorbar
    sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.02,
                      pad=0.015, aspect=50)
    cb.ax.tick_params(labelsize=9)
    cb.set_label("coefficient of determination  r²", fontsize=10, color="#2C2C2A")

    # confounded legend
    ax.add_patch(Rectangle((-0.1, -1.35), 0.75, 0.6, facecolor="none",
                 edgecolor="#8A8A8A", linewidth=0.9, linestyle=(0, (2, 1.6))))
    ax.text(0.9, -1.05, "normaliser confounded (stature-scaled metrics at el>0) — disregard",
            ha="left", va="center", fontsize=9, color="#5F5E5A")

    out = os.path.join(config.OBP_VALIDATION_DIR, "angle_zone_map_numbers_gt.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
