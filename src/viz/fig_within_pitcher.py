"""Body figure: pooled r2 vs within-pitcher r2, over every row of the map.

Most of the truth's variance lies BETWEEN pitchers, so a pooled r2 largely measures
pitcher RANKING. Removing the pitcher mean (within-pitcher r2) shows which rows also
track one athlete's pitch-to-pitch change. Some hold both; several are pooled-oversells.

⚠ REPOINTED 2026-07-30. This used to read `within_pitcher_agreement_gt_clean.csv`,
which covered the 16 adopted metrics at 16 hand-pinned anchor viewpoints. That table
could not be joined to the graded map and four of its keys stopped existing at the
2026-07-29 dedup. It now reads `accuracy_bestcell_gt_clean.csv`
(`analysis/accuracy_map.py`), which covers every map row at that row's best-CCC cell
and is cross-checked against `gate_map.csv` cell by cell.

⚠ RE-AUTHORED 2026-07-30 for BODY placement (it was supplement S5). The old layout was
8.6 x 12.7 in with paired bars, which at `\\includegraphics[width=7.16in]` came out
taller than a page and shrank every label. Now:
  - authored at BODY_W, saved without a tight bbox, so the type prints at its stated size
  - a dumbbell replaces the paired bars: at one row per 0.107 in two bars are 3 pt tall
    each, while a connector plus two dots stays legible AND draws the eye to the GAP,
    which is the whole claim
  - sorted by within-pitcher r2 descending, so the 0.60 reference line cuts the sorted
    list at ONE place and the count above it is readable off the figure
No number changed: both columns come straight from the CSV.

⚠ The VIEW is the same-sample CCC argmax, so it carries the map's own selection caveat;
state that wherever this figure is cited.

⚠ CAPTION MUST CARRY THIS. A low within-pitcher r2 mixes two causes: the estimator
failing to track an individual, and the TRUTH itself barely moving within a pitcher.
`truth_icc` is 0.80-0.98 over these rows, so the within-pitcher signal is genuinely
small; the script prints it per row for exactly this reason.

Data: accuracy_bestcell_gt_clean.csv. Clean-projection ceilings under GT events.

Run:  conda activate diamond
      cd src\\viz
      python fig_within_pitcher.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", ".."):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
from fig_graded_map import short, INK, MUTE, BODY_W

POOLED_C = "#0FA3B1"         # ranks pitchers
WITHIN_C = "#7C3AED"         # tracks one pitcher's change
GAP_C = "#E4572E"            # flagged: pooled oversells
IDENT_C = "#9AA3B2"          # arm slot: shown, but flagged as a synthetic identity
IDENTITY = {"Arm Slot"}      # 2D coronal definition == 3D-direct definition at az90
FLOOR = 0.60
OVERSELL = 0.20              # pooled - within at or above this is flagged
OUT = os.path.join(config.ROOT, "data", "outputs", "viz")

# 2026-08-03 cleanup, same faults as fig_arc_width had:
#   - the legend sat at lower LEFT, straight on top of the bottom rows' dumbbells. It
#     moves to the top left, the one quadrant with no data in it (rows are sorted by
#     within-pitcher r2, so the top rows all sit at x >= 0.75).
#   - "r^2 = 0.60" was floating text with its own dashed rule drawn through it. It is a
#     legend entry now, so nothing on the canvas is overprinted.
#   - "17 of 34 above" is caption text and is gone from the canvas; the script prints it.
#   - the horizontal divider marking that crossing is gone too: the rows are SORTED by
#     within-pitcher r2, so where the purple dots cross the 0.60 rule is the crossing,
#     and a second mark for it was one more pale grey line to confuse with the x grid.
#   - the x grid is gone for that same reason.
#   - red y tick labels duplicated the orange connector; the connector keeps the flag.
FS_ROW, FS_TICK, FS_AXLAB, FS_LEG = 5.4, 5.6, 6.2, 5.2
ROW_IN = 0.105


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR,
                                  "accuracy_bestcell_gt_clean.csv"))
    df["metric"] = df.metric.str.replace(" [O]", "", regex=False)
    df["gap"] = df.pooled_r2 - df.within_r2
    df["ident"] = df.metric.isin(IDENTITY)
    # descending within-pitcher r2: the FLOOR line then crosses the list exactly once
    df = df.sort_values("within_r2", ascending=False).reset_index(drop=True)

    real = df[~df.ident]
    n_keep = int((real.within_r2 >= FLOOR).sum())
    print(f"rows {len(df)}  ({int(df.ident.sum())} synthetic identity excluded from "
          f"the count)")
    print(f"within-pitcher r2 >= {FLOOR:.2f}:  {n_keep} of {len(real)}")
    print(f"pooled oversells by >= {OVERSELL:.2f}:  "
          f"{int((real.gap >= OVERSELL).sum())} of {len(real)}")
    w = real.loc[real.gap.idxmax()]
    print(f"widest drop: {w.metric}  pooled {w.pooled_r2:.3f} -> within "
          f"{w.within_r2:.3f}  (truth_icc {w.truth_icc:.3f})")
    print("per row (within, pooled, gap, truth_icc):")
    for r in df.itertuples(index=False):
        print(f"  {r.metric:<42s} {r.within_r2:6.3f} {r.pooled_r2:6.3f} "
              f"{r.gap:+6.3f}  icc {r.truth_icc:.3f}")

    n = len(df)
    # SINGLE COLUMN (2026-08-01). A dumbbell survives the halving that a paired bar
    # chart would not: the two values stay on one x axis and only the connector shortens.
    COL_W = BODY_W / 2
    L, R, TOP, BOT_IN = 0.292, 0.972, 0.995, 0.38
    fig_h = ROW_IN * n + BOT_IN + 0.06
    fig, ax = plt.subplots(figsize=(COL_W, fig_h))
    fig.subplots_adjust(left=L, right=R, top=TOP, bottom=BOT_IN / fig_h)
    y = np.arange(n)[::-1]                      # first (highest within) at the top

    # the floor rule, behind the dumbbells: it is a reference, and unlike fig_arc_width
    # nothing here is drawn in its colour, so it reads without being lifted over the data
    ax.axvline(FLOOR, ls="--", color="0.45", lw=0.9, zorder=1)

    for yy, r in zip(y, df.itertuples(index=False)):
        pc, wc = (IDENT_C, IDENT_C) if r.ident else (POOLED_C, WITHIN_C)
        lc = IDENT_C if r.ident else (GAP_C if r.gap >= OVERSELL else "#C7D2DE")
        ax.plot([r.within_r2, r.pooled_r2], [yy, yy], color=lc,
                lw=1.6 if r.gap >= OVERSELL and not r.ident else 1.1,
                solid_capstyle="round", zorder=2)
        ax.plot(r.pooled_r2, yy, "o", ms=3.0, color=pc, zorder=3)
        ax.plot(r.within_r2, yy, "o", ms=3.0, color=wc, zorder=4)

    labels = [f"{r.metric}  (identity)" if r.ident else short(r.metric)
              for r in df.itertuples(index=False)]
    ax.set_yticks(y)
    ax.set_yticklabels([l[:26] for l in labels], fontsize=FS_ROW)
    for i, tick in enumerate(ax.get_yticklabels()):
        if df.ident.iloc[i]:
            tick.set_color(IDENT_C)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.tick_params(axis="x", labelsize=FS_TICK, length=2, pad=1.5)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlim(0, 1.06)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # axis NAME only. "at each row's best-CCC cell" is which cell was read, i.e. method,
    # and the caption states it -- along with the selection caveat that comes with it.
    ax.set_xlabel("$r^2$", fontsize=FS_AXLAB, labelpad=1.5)
    # top left: the rows are sorted, so the only empty quadrant is up there
    ax.legend(handles=[
        # decode the marks, nothing more. "(ranks pitchers)" / "(tracks change)" is the
        # paper's reading of them and belongs to the caption, not the key.
        Line2D([0], [0], marker="o", ls="none", color=POOLED_C, ms=3.6,
               label="pooled"),
        Line2D([0], [0], marker="o", ls="none", color=WITHIN_C, ms=3.6,
               label="within-pitcher"),
        Line2D([0], [0], color=GAP_C, lw=1.6,
               label=f"pooled − within ≥ {OVERSELL:.2f}"),
        Line2D([0], [0], ls="--", color="0.45", lw=0.9,
               label=f"$r^2$ = {FLOOR:.2f}")],
        loc="upper left", bbox_to_anchor=(-0.005, 1.005), frameon=False,
        fontsize=FS_LEG, ncol=1, handlelength=1.3, handletextpad=0.4,
        labelspacing=0.30, borderpad=0.1)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")

    out = os.path.join(OUT, "fig_within_pitcher_gt.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved -> {out}   {COL_W:.2f} x {fig_h:.2f} in")


if __name__ == "__main__":
    main()
