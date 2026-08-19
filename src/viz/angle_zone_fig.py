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
Diamond - render the fine-resolution reliability map with VALID ZONES.

Successor to angle_map_2d_fig.py (blue r2 heatmap, confounded-cell fade),
upgraded to the 15deg sweep (angle_zone_sweep.csv) and the threshold-based zone
framing. Two readability changes (2026-07-10): the 7 elevation blocks WRAP onto
TWO rows (top el 0/15/30/45, bottom el 60/75/85) instead of one wide strip, and
the ground/overhead background band is removed (its cream tone clashed with the
gold usable-zone outline and carried no information).

  - gold outline  = usable zone   (r2 >= 0.60, the floor defended via HSS 0.63)
  - red outline   = reliable zone (r2 >= 0.80)
  - star marker   = point recommendation for the velocity metrics, whose zones
    did NOT survive the deployment-chain noise probe (zone_edge_noise_velo):
    they get single validated viewpoints, not arcs.
  - dashed fade   = normaliser-confounded cells (stride / wrist speed / release
    height at el>0), carried over from angle_map_2d_fig; NO zone outlines are
    drawn there even where raw r2 clears the floor.

Cells carry no numbers at this resolution - quote the ledger table instead.

Source (2026-07-24): angle_zone_sweep_gt_clean.csv - the GT-event map on the
clean population (n=394, 16 metrics). The metric list lives in map_metrics.py and
is checked against the CSV at render time, so a newly adopted metric can no
longer be silently dropped.

Run:
  conda activate diamond
  cd src\\viz
  python angle_zone_fig.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from angle_map_2d_fig import CMAP
from map_metrics import ORDER, POINT_ONLY, LABEL_COLOR, sweep_path, check_coverage

AZ = list(range(0, 360, 15))
EL_ROWS = [[0, 15, 30, 45], [60, 75, 85]]   # wrapped onto two rows
NAZ = len(AZ)
GAP = 1.5           # gap between elevation blocks within a row
ROW_GAP = 3.2       # vertical gap between the two elevation rows
TIERS = [("usable", 0.60, "#C9A24B", 2.6), ("reliable", 0.80, "#C0392B", 1.8)]


def block_title(el):
    if el == 0:
        return "Side · el 0°"
    if el == 85:
        return "Overhead · el 85°"
    return f"el {el}°"

def xpos(block, a):
    return block * (NAZ + GAP) + a


def cell_runs(mask):
    """Contiguous True runs over the linear az axis -> [(i0, i1) inclusive]."""
    runs, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            runs.append((i, j)); i = j + 1
        else:
            i += 1
    return runs


def draw_metric_block(ax, sub, key, confd, b, el, ybot, norm):
    """One metric row of one elevation block: cells + dashed/point/zone marks."""
    g = sub[sub["el"] == el].set_index("az").reindex(AZ)
    vals = g["r2"].to_numpy(float)
    dashed = confd and el > 0
    for a in range(NAZ):
        v = vals[a]
        fc = CMAP(norm(v)) if np.isfinite(v) else "#F1EFE8"
        ax.add_patch(Rectangle((xpos(b, a), ybot), 1, 1, facecolor=fc,
                     edgecolor="white", linewidth=0.6, zorder=2,
                     alpha=0.45 if dashed else 1.0))
    if dashed:
        ax.add_patch(Rectangle((xpos(b, 0) + 0.1, ybot + 0.08), NAZ - 0.2, 0.84,
                     facecolor="none", edgecolor="#808080", linewidth=1.2,
                     linestyle=(0, (3, 2)), zorder=3))
        return
    if key in POINT_ONLY:
        for pel, paz in POINT_ONLY[key]:
            if pel == el:
                ax.plot(xpos(b, AZ.index(paz)) + 0.5, ybot + 0.5, "*",
                        markersize=15, markerfacecolor="#FFD75E",
                        markeredgecolor="#2C2C2A", markeredgewidth=1.1, zorder=5)
        return
    for _tier, thr, color, lw in TIERS:
        mask = np.isfinite(vals) & (vals >= thr)
        for i0, i1 in cell_runs(list(mask)):
            ax.add_patch(Rectangle((xpos(b, i0) + 0.06, ybot + 0.06),
                         (i1 - i0 + 1) - 0.12, 0.88, facecolor="none",
                         edgecolor=color, linewidth=lw, zorder=4))


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

    out = os.path.join(config.OBP_VALIDATION_DIR, "angle_zone_map_gt.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
