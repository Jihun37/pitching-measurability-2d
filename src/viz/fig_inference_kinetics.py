"""Act 1 §4.6 — back-calculation of kinetics: a ground-truth-input UPPER-BOUND
DIAGNOSTIC.

⚠ WORDING. This figure must NOT claim theoretical impossibility. It reports what was
observed under the tested inputs and models. It is an upper-bound diagnostic because
the inputs are ground-truth 3D kinematics rather than our 2D pipeline; it is NOT a
theoretical ceiling, and words like "closed" or "not recoverable" do not belong on it.

Reads ONLY the canonical n=394 table `inference_trajectory.csv` (the layer whose
population matches the map). `inference_final.csv` is the n=403 scalar run and must
NOT be used for paper numbers.

Two panels, reported as VALUES with NMAE (never pass/fail):
  left   best CV R2 per kinetic target, against the 0.60 screening floor
  right  NMAE of the same target (MAE / population SD of the truth); 1.0 = as wrong as
         guessing the population mean

The inputs are the ground-truth 3D poi scalars and full_sig trajectories, not our 2D
pipeline, so the result bounds what these models achieve from ideal kinematics.
Under those inputs and models, no target met the prespecified R2 = 0.60 screening
threshold.

The two peak_rfd targets are marked because they are a trap: they hold the LOWEST
NMAE in the whole table while their R2 is NEGATIVE, i.e. the model sits near the
median and tracks nothing. Never quote their NMAE alone.

Run:  conda activate diamond
      cd src\\viz
      python fig_inference_kinetics.py
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", ".."):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from fig_graded_map import INK, MUTE, C_STRONG, C_MOD, BODY_W

SRC = os.path.join(config.OBP_VALIDATION_DIR, "inference_trajectory.csv")
OUT = os.path.join(config.ROOT, "data", "outputs", "viz",
                   "fig_inference_kinetics.png")
FLOOR = 0.60
# COLUMN width, not page width. The figure carries one panel and 34 short labels,
# and a page-wide bar chart of one series is mostly white.
COL_W = BODY_W / 2

# House palette, from fig_graded_map like every sibling figure. Two corrections here:
# the right panel was drawn in "#94A3B8", a slate that is in no other figure, and the
# SAME red was carrying the 0.60 line and the trap bars, so one colour meant two things.
# Now the two panels are the two-tone blue ramp -- one quantity, two measures -- and the
# traps take the house caution amber (fig_hss_elevation, fig_event_tolerance).
# The floor line KEEPS the red of fig_arc_width.FLOOR_C, and must keep a colour far from
# the bars: the top bar ends 0.003 short of it, so a dark line there would merge with it.
R2_C = C_STRONG
# far from the bars: the top bar ends 0.003 short of the line
FLOOR_C = "#D92B4B"

# 2026-08-03 rebuild. This was the last figure still AUTHORED WIDER THAN IT IS PLACED:
# 12.4 in saved at dpi 170, then shrunk to 7.16 by \includegraphics, so its 7.6 pt row
# labels printed at 4.4 pt and its 13 pt title at 7.5. It is authored at BODY_W now and
# saved at dpi 300 like every other body figure.
#
# And the whole top of it was the caption, drawn on the canvas: a two-line bold suptitle
# stating the result, and a three-line grey block giving the inputs, the population and
# the method. Both are gone. So are the two panel titles, which said what the x axis
# labels underneath them already said, and the free-floating notes on the two reference
# lines -- those notes were clipped by the neighbouring panel anyway, and a reference
# line belongs in a legend.
FS_ROW, FS_TICK, FS_AXLAB, FS_LEG = 5.4, 5.6, 6.2, 5.2
ROW_IN = 0.105


def label(t):
    return (t.replace("_", " ").replace("transfer", "transf")
             .replace("rotation", "rot").replace("internal", "int"))


def main():
    d = pd.read_csv(SRC)
    n_pitch = int(d.n.iloc[0])
    d["nmae_best"] = d[["nmae_anthro", "nmae_scalar", "nmae_traj"]].min(axis=1)
    d = d.sort_values("best_r2").reset_index(drop=True)
    n = len(d)
    assert d.target.nunique() == n

    # The top margin is an inch count, not a fraction. At top=0.995 there was 0.02 in
    # above the axes, so the panel letters were drawn off the canvas and lost.
    # Reserving it in inches also keeps the row pitch at exactly ROW_IN for any n.
    # L is a fraction, and the labels it must clear are an inch count, so halving
    # the figure width doubles the fraction they need.
    L, R, BOT_IN, TOP_IN = 0.34, 0.985, 0.38, 0.10
    fig_h = ROW_IN * n + BOT_IN + TOP_IN
    TOP = 1 - TOP_IN / fig_h
    fig, a1 = plt.subplots(figsize=(COL_W, fig_h))
    fig.subplots_adjust(left=L, right=R, top=TOP, bottom=BOT_IN / fig_h)
    y = np.arange(n)

    # One colour: NOTHING reaches the floor, so a conditional colour would be a branch
    # that never fires. The assert keeps that honest -- if the table ever changes this
    # fails instead of quietly drawing every bar as though it had missed.
    assert (d.best_r2 < FLOOR).all(), "a target reaches the floor; the bar colour lies"
    a1.barh(y, d.best_r2, height=0.72, color=R2_C)
    a1.axvline(FLOOR, color=FLOOR_C, lw=1.2, ls="--", zorder=3)
    a1.axvline(0, color=MUTE, lw=0.8)
    a1.set_yticks(y)
    a1.set_yticklabels([label(t) for t in d.target], fontsize=FS_ROW)
    a1.set_xlabel("best cross-validated $R^2$", fontsize=FS_AXLAB, labelpad=1.5)
    a1.set_xlim(-0.26, 0.66)
    # ONE legend, in the left panel: its bottom rows are all negative, so the whole
    # lower right of it is empty. The right panel has no such gap -- every bar there is
    # 0.34 to 0.80 wide -- and a legend placed in it sat on the bars.
    # Flush right (1.005) put the legend UNDER the 0.60 rule, which crosses the axes at
    # 0.93 of its width: the dashed line ran straight through both labels. It is pulled
    # left to clear the rule, into the band between the staircase and the line, and
    # LEG_ASSERT below fails if either clearance is ever lost.
    # the manuscript's own term for this level; "screening" is internal vocabulary
    # and prints nowhere in the paper
    leg = a1.legend(handles=[Line2D([], [], color=FLOOR_C, lw=1.2, ls="--",
                                    label=f"$R^2$ = {FLOOR:.2f} reference level")],
                    loc="lower right", bbox_to_anchor=(0.905, -0.004), frameon=False,
                    fontsize=FS_LEG, handlelength=1.6, labelspacing=0.30, borderpad=0.1)

    # No panel letters: there is one panel, and the caption has nothing to point at.
    for sp in ("top", "right"):
        a1.spines[sp].set_visible(False)
    a1.tick_params(axis="x", labelsize=FS_TICK, length=2, pad=1.5)
    a1.tick_params(axis="y", length=0, pad=1.5)
    a1.set_ylim(-0.8, n - 0.2)

    # LEG_ASSERT. The legend is placed in axes fractions, but what it must clear -- the
    # 0.60 rule and the bar staircase -- lives in data coordinates, and the mapping
    # between them moves with the row count, the font sizes and the label texts. So the
    # clearance is measured off the drawn legend rather than assumed from the anchor.
    fig.canvas.draw()
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    (lx0, ly0), (lx1, ly1) = a1.transData.inverted().transform(
        [(bb.x0, bb.y0), (bb.x1, bb.y1)])
    assert lx1 < FLOOR - 0.01, f"legend right edge {lx1:.3f} touches the {FLOOR} rule"
    over = [t for t, r, yy in zip(d.target, d.best_r2, y)
            if ly0 - 0.5 <= yy <= ly1 + 0.5 and max(r, 0.0) > lx0]
    assert not over, f"legend sits on the bars for {over}"

    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print(f"targets {n}  n={n_pitch}  reach{FLOOR}: {int((d.best_r2>=FLOOR).sum())}"
          f"  best {d.best_r2.max():.4f} ({d.target.iloc[-1]})"
          f"  NMAE {d.nmae_best.min():.3f}..{d.nmae_best.max():.3f}")
    print(f"saved -> {OUT}   {COL_W:.2f} x {fig_h:.2f} in")


if __name__ == "__main__":
    main()
