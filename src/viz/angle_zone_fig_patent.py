"""Figure (도 4, PATENT COPY of angle_zone_fig): 15-metric DEPLOYMENT reliability map.

Identical rendering to the paper figure (angle_zone_fig.py), but drawn for the
DEPLOYMENT metric set and map instead of the paper's GT-clean one:

  - metrics : the 15 deployment metrics = the paper's 16 minus Elbow Flex @MER
              (measurable but frame-rate-excluded from per-pitch deployment), so
              ORDER is derived from map_metrics.ORDER by dropping that one row.
  - source  : angle_zone_sweep.csv (the DETECTED-event deployment sweep, 15
              metrics, full population), NOT angle_zone_sweep_gt_clean.csv.
  - output  : angle_zone_map_patent.png (never overwrites the paper png).

Everything else (tiers, confounded-cell fade, point markers, two-row elevation
wrap) is inherited from angle_zone_fig so the two figures stay visually identical.
The neighbouring paper script and map_metrics.py are left untouched.

Run:
  conda activate diamond
  cd src\\viz
  python angle_zone_fig_patent.py
"""
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from angle_map_2d_fig import CMAP
from map_metrics import ORDER as _ORDER16, POINT_ONLY, LABEL_COLOR  # noqa: F401
from angle_zone_fig import (AZ, EL_ROWS, NAZ, GAP, ROW_GAP, TIERS,
                            block_title, xpos, cell_runs, draw_metric_block)

# 15-metric DEPLOYMENT set = paper's 16 minus Elbow Flex @MER (frame-rate wall).
ORDER = [r for r in _ORDER16 if r[0] != "Elbow Flex @MER [O]"]
SWEEP_CSV = "angle_zone_sweep.csv"   # detected-event deployment sweep, 15 metrics


def sweep_path():
    return os.path.join(config.OBP_VALIDATION_DIR, SWEEP_CSV)


def check_coverage(df):
    """Same guard as map_metrics.check_coverage, against the 15-metric set."""
    drawn = {k for k, *_ in ORDER}
    present = set(df["metric"].unique())
    missing = present - drawn
    absent = drawn - present
    if missing:
        raise SystemExit(
            f"{SWEEP_CSV} has {len(missing)} metric(s) not in ORDER: "
            f"{sorted(missing)}")
    if absent:
        raise SystemExit(
            f"ORDER lists {len(absent)} metric(s) absent from {SWEEP_CSV}: "
            f"{sorted(absent)}")


def main():
    df = pd.read_csv(sweep_path())
    check_coverage(df)
    n = len(ORDER)
    n_strips = len(EL_ROWS)
    max_blocks = max(len(r) for r in EL_ROWS)
    grid_r = xpos(max_blocks - 1, 0) + NAZ

    fig, ax = plt.subplots(figsize=(19, 14.5))
    norm = Normalize(0, 1)

    for s, els in enumerate(EL_ROWS):
        ybase = (n_strips - 1 - s) * (n + ROW_GAP)      # top strip highest
        for ri, (key, disp, confd, _region) in enumerate(ORDER):
            ybot = ybase + (n - 1 - ri)
            sub = df[df["metric"] == key]
            tag = "  · point" if key in POINT_ONLY else ""
            ax.text(-0.8, ybot + 0.5, disp + tag, ha="right", va="center",
                    fontsize=11, fontweight="medium",
                    color="#8A8880" if key in POINT_ONLY
                          else LABEL_COLOR.get(_region, "#2C2C2A"), zorder=3)
            for b, el in enumerate(els):
                draw_metric_block(ax, sub, key, confd, b, el, ybot, norm)
        # per-block elevation headers + azimuth ticks, above this strip
        for b, el in enumerate(els):
            ax.text(xpos(b, 0) + NAZ / 2, ybase + n + 0.95, block_title(el),
                    ha="center", va="center", fontsize=11.5,
                    fontweight="medium", color="#2C2C2A")
            ax.plot([xpos(b, 0), xpos(b, 0) + NAZ], [ybase + n + 0.55] * 2,
                    color="#D3D1C7", linewidth=0.8)
            for a, az in enumerate(AZ):
                if az % 45 == 0:
                    ax.text(xpos(b, a) + 0.5, ybase + n + 0.18, f"{az}°",
                            ha="center", va="center", fontsize=8, color="#5F5E5A")
        ax.text(-0.8, ybase + n + 0.18, "azimuth →", ha="right", va="center",
                fontsize=10, color="#5F5E5A")

    # legend row (below the bottom strip)
    ly = -1.5
    ax.add_patch(Rectangle((0, ly), 2.2, 0.6, facecolor="none",
                 edgecolor=TIERS[0][2], linewidth=2.6))
    ax.text(2.7, ly + 0.3, "usable zone  r² ≥ 0.6", ha="left", va="center",
            fontsize=10, color="#5F5E5A")
    ax.add_patch(Rectangle((26, ly), 2.2, 0.6, facecolor="none",
                 edgecolor=TIERS[1][2], linewidth=1.8))
    ax.text(28.7, ly + 0.3, "reliable zone  r² ≥ 0.8", ha="left", va="center",
            fontsize=10, color="#5F5E5A")
    ax.plot(52.5, ly + 0.3, "*", markersize=15, markerfacecolor="#FFD75E",
            markeredgecolor="#2C2C2A", markeredgewidth=1.1)
    ax.text(53.3, ly + 0.3, "point rec. (velocity — zones not noise-robust)",
            ha="left", va="center", fontsize=10, color="#5F5E5A")
    ax.add_patch(Rectangle((82, ly), 2.2, 0.6, facecolor="none",
                 edgecolor="#808080", linewidth=1.2, linestyle=(0, (3, 2))))
    ax.text(84.7, ly + 0.3, "normaliser confounded — disregard",
            ha="left", va="center", fontsize=10, color="#5F5E5A")

    ax.set_xlim(-10, grid_r + 1.0)
    ax.set_ylim(-2.4, n_strips * (n + ROW_GAP) - ROW_GAP + 1.4)
    ax.axis("off")

    sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.025,
                      pad=0.02, aspect=50)
    cb.ax.tick_params(labelsize=9)
    cb.set_label("coefficient of determination  r²", fontsize=9.5, color="#2C2C2A")

    out = os.path.join(config.OBP_VALIDATION_DIR, "angle_zone_map_patent.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
