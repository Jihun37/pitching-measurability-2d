"""GT-event (oracle) map vs the DEPLOYABLE map, side by side.

Two figures, deliberately mirroring fig_graded_map.py's palette and helpers so the
pair reads as one family and the reader can overlay them mentally:

  fig_deploy_strip.png    every map row, azimuth on x at the metric's best
                          elevation, drawn as a PAIR of bands: the GT-event grade
                          on top, the deployed grade (FP-specific LOPO routing) beneath.
                          This is the figure that answers "where can I actually
                          stand", against "where could I stand if the events were
                          given".
  fig_deploy_loss.png     small multiples: per metric, the per-cell grade LOSS
                          (oracle rank - deployed rank) over azimuth x elevation.
                          Shows WHERE detection cost the map, not just how much.

Reads deploy_map_cells.csv / deploy_map_summary.csv, written by
analysis/deploy_map.py. The frozen map is never touched.

Run:  conda activate diamond; cd src\\viz; python fig_deploy_map.py
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../analysis"):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
from fig_graded_map import short, INK, MUTE, C_STRONG, C_MOD, C_OFF

CELLS = os.path.join(config.OBP_VALIDATION_DIR, "deploy_map_cells.csv")
SUMM = os.path.join(config.OBP_VALIDATION_DIR, "deploy_map_summary.csv")
OUT = os.path.join(config.ROOT, "data", "outputs", "viz")
RANK = {"limited": 0, "moderate": 1, "strong": 2}


def colour(gr):
    return C_STRONG if gr == "strong" else (C_MOD if gr == "moderate" else C_OFF)


def strip(cells, summ, out_png):
    """Paired bands per metric: oracle on top, deployed underneath."""
    rows = summ[summ.oracle_gate_cells > 0].sort_values(
        ["dep_strong", "oracle_gate_cells"], ascending=True)
    n = len(rows)
    az = sorted(cells.az.unique())
    # Authored at the IEEE two-column width; a tight bbox would rescale every font.
    fig, ax = plt.subplots(figsize=(7.16, 0.21 * n + 1.15))
    for i, r in enumerate(rows.itertuples(index=False)):
        sub = cells[(cells.metric == r.metric) & (cells.source == r.source)]
        onmap = sub[sub.onmap_gt]
        if onmap.empty:
            continue
        # the metric's best elevation under the GT map, same convention as
        # fig_graded_map's strip
        el = int(onmap.loc[onmap.ccc_gt.idxmax()].el)
        row = sub[sub.el == el].set_index("az")
        for j, a in enumerate(az):
            if a not in row.index:
                continue
            for k, (col, dy) in enumerate((("grade_gt", 0.06),
                                           ("grade_lopo", -0.42))):
                gr = row.loc[a, col]
                ax.add_patch(plt.Rectangle((j - 0.5, i + dy), 1, 0.36,
                                           facecolor=colour(gr),
                                           edgecolor="white", lw=0.35))
        ax.text(len(az) - 0.4, i + 0.24, "GT", fontsize=4.4, color=MUTE, va="center")
        ax.text(len(az) - 0.4, i - 0.24, "dep", fontsize=4.4, color=MUTE, va="center")
    ax.set_xlim(-0.5, len(az) + 0.6); ax.set_ylim(-0.8, n - 0.2)
    ax.set_xticks(range(len(az)))
    ax.set_xticklabels([str(a) for a in az], fontsize=5.2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([short(r.metric) for r in rows.itertuples(index=False)],
                       fontsize=5.4)
    # axis NAME only -- the az0/az90 convention is caption text (see fig_graded_map)
    ax.set_xlabel("camera azimuth (deg)", fontsize=6.6)
    ax.set_title("GT-event measurement grades vs DETECTED-EVENT measurement "
                 "grades\nupper band = GT events, lower band = detected events "
                 "(FP-specific LOPO routing), at each estimator's best elevation",
                 fontsize=8.0, weight="bold", color=INK, pad=5)
    ax.legend(handles=[Patch(facecolor=C_STRONG, label="strong (CCC ≥ 0.80)"),
                       Patch(facecolor=C_MOD, label="moderate (0.75–0.80)"),
                       Patch(facecolor=C_OFF, label="off the map")],
              loc="lower center", bbox_to_anchor=(0.5, -0.10 - 1.1 / n),
              ncol=3, frameon=False, fontsize=6.0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.tight_layout(pad=0.35)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print("saved ->", out_png)


def loss_panels(cells, summ, out_png, ncol=5):
    """Per-metric field of grade loss (oracle rank - deployed rank)."""
    rows = summ[summ.oracle_gate_cells > 0].sort_values(
        "oracle_gate_cells", ascending=False)
    n = len(rows); nrow = int(np.ceil(n / ncol))
    cmap = ListedColormap(["#EDF2F7", "#F6C177", "#D92B4B"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(7.16, 1.02 * nrow + 0.70))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for i, r in enumerate(rows.itertuples(index=False)):
        ax = axes[i]
        sub = cells[(cells.metric == r.metric) & (cells.source == r.source)]
        az = sorted(sub.az.unique()); el = sorted(sub.el.unique())
        M = np.full((len(el), len(az)), np.nan)
        for c in sub.itertuples(index=False):
            if not c.onmap_gt:
                continue
            M[el.index(c.el), az.index(c.az)] = (
                RANK.get(c.grade_gt, 0) - RANK.get(c.grade_lopo, 0))
        ax.imshow(np.clip(M, 0, 2), origin="lower", aspect="auto", cmap=cmap,
                  norm=norm, extent=[az[0], az[-1], el[0], el[-1]])
        # moderate-only rows have gt_strong == 0, and "0/0 strong kept" reads as a
        # total failure when the row simply never had a strong cell to lose
        if r.gt_strong:
            lab = f"{r.dep_strong}/{r.gt_strong} strong kept"
        else:
            kept = r.dep_strong + r.dep_moderate
            lab = f"moderate-only · {kept}/{r.oracle_gate_cells} cells kept"
        ax.set_title(f"{short(r.metric)}\n{lab}", fontsize=4.8, color=INK)
        ax.set_xticks([0, 90, 180, 270]); ax.tick_params(labelsize=6.5)
        ax.set_yticks([0, 30, 60, 85])
    fig.suptitle("What detection costs, per cell — grade lost from the GT-event "
                 "map to the detected-event map", fontsize=8.0, weight="bold", color=INK,
                 y=0.995)
    fig.text(0.5, 0.965, "grey = grade held · amber = dropped one grade · "
             "red = fell off the map.  blank = not on the GT map.  "
             "FP-specific LOPO routing, n = 394",
             ha="center", va="top", fontsize=5.2, color=MUTE)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.04,
                        wspace=0.32, hspace=0.85)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print("saved ->", out_png)


def main():
    if not os.path.exists(CELLS):
        sys.exit("run analysis/deploy_map.py first")
    os.makedirs(OUT, exist_ok=True)
    cells = pd.read_csv(CELLS); summ = pd.read_csv(SUMM)
    strip(cells, summ, os.path.join(OUT, "fig_deploy_strip.png"))
    loss_panels(cells, summ, os.path.join(OUT, "fig_deploy_loss.png"))


if __name__ == "__main__":
    main()
