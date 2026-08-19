"""Body figure: how much camera-placement slack each measurement has.

The graded map is drawn on a 15-degree azimuth grid, but a camera cannot be placed to
that precision. What matters operationally is the WIDTH of the contiguous azimuthal arc
a row holds -- how far the camera may sit from the ideal line and still return a graded
reading. `continuity_rows.csv` (analysis/continuity_map.py) carries the widest arc per
row and the elevation it occurs at.

This figure is the scalar companion to the map: fig_graded_strip shows THAT the graded
cells form runs, this one says HOW WIDE the runs are and splits them by quantity kind.
Three tiers fall out of the data, not out of a chosen cut:

  azimuth-independent   the arc spans the whole orbit (24 of 24 bins)
  arc-limited           a contiguous run of >= 3 bins
  point-like            below 3 bins -- narrower than a camera can be aimed

Say "contiguous azimuthal arc", never "zone": `zone` names the retired r2-threshold
construct. The elevation is printed beside every bar because a full-orbit arc at el 75
is not a ground camera.

Every count, median and tier boundary is read from the CSV and printed on each run.

Run:  conda activate diamond
      cd src\\viz
      python fig_arc_width.py
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", ".."):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
from fig_graded_map import short, INK, MUTE, BODY_W

ROWS = os.path.join(config.OBP_VALIDATION_DIR, "continuity_rows.csv")
ARCS = os.path.join(config.OBP_VALIDATION_DIR, "continuity_arcs.csv")
GATE = os.path.join(config.OBP_VALIDATION_DIR, "gate_map.csv")
OUT = os.path.join(config.ROOT, "data", "outputs", "viz")

KIND_C = {"angle/posture": "#0C447C", "velocity": "#0FA3B1"}
FLOOR_C = "#D92B4B"
# ONE RULE ONLY. The two per-kind median rules were dropped 2026-08-03: a median is a
# summary of the bars already drawn, not a boundary, and the reader can see the two
# groups' spread directly. They are still computed and PRINTED on every run, so the
# caption keeps its number (105 deg angle/posture vs 90 deg velocity) -- and the caption
# is where a six-row median belongs, not on the canvas as a rule of equal weight to a
# threshold. The 45 deg floor stays because it is a boundary: it defines the tier.
FULL_BINS = 24          # a full orbit on the swept grid
ARC_MIN = 3             # the aimability floor used by continuity_map.py
BIN_DEG = 15.0

# 2026-08-03 rebuild. The single-column squeeze of 2026-08-01 had produced the worst
# figure in the paper, and all of it followed from two decisions:
#
#   1. Value labels were drawn OUTSIDE every bar under 300 deg. Five rows hold exactly
#      90 deg and the two median rules stand at 90 and 105, so those labels were printed
#      ON the rules. The rules are at zorder 0 and hide behind the bars, so the ONLY
#      place they collide with anything is the strip just past a bar end -- which is
#      exactly where the labels were put. They now go INSIDE the bar wherever they fit,
#      which is everywhere but the narrowest row.
#   2. The three rules carried floating text: two median labels stacked above the top
#      row and the floor label below the bottom one, none of them explained. They are
#      legend entries now, so the canvas holds bars and nothing else.
#
# Type then had room to become legible: row labels 4.3 -> 5.4 pt, values 4.4 -> 5.0.
FS_ROW, FS_VAL, FS_TICK, FS_AXLAB, FS_LEG = 5.4, 5.0, 5.6, 6.2, 5.2
ROW_IN = 0.105          # vertical pitch per row
X_MAX = 368

# NO TIER GAPS EITHER. Two blank rows used to mark the tier boundaries. They said
# nothing the figure was not already saying: the bars are sorted by width, so the four
# full-orbit rows are self-evident at the top, and the floor rule marks the point-like
# boundary at the bottom. Tier names and counts stay in the caption.
#
# Value labels sit INSIDE their bar, right-aligned, in white. A label goes outside --
# ink, on the white ground -- for exactly two reasons, and both are tested, not listed:
# the bar is too short to hold it, or the floor rule would run through it. The second is
# what the seven 60 deg rows hit: right-aligned inside a 60 deg bar the text straddles
# 45. They move out past the bar end instead, which clears the rule and leaves every
# other row's label inside where it belongs.

# NO X GRID, AND NO TIER RULES. The figure carried two families of pale grey line at the
# same weight and colour: vertical grid at the x ticks, which means nothing, and
# horizontal tier separators, which mean everything and are named only in the caption.
# A reader cannot tell them apart, and the grid was competing with the three coloured
# reference rules that are the actual annotation. The grid is gone, and the tiers are
# separated by WHITESPACE -- the standard encoding for an ordinal group in a bar chart,
# and one that cannot be confused with a rule. It costs ~0.15 in of height.


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(ROWS)
    arcs = pd.read_csv(ARCS)

    # tiers, in the order they are drawn (widest at the top)
    d["tier"] = np.where(d.max_arc_bins >= FULL_BINS, 0,
                         np.where(d.max_arc_bins >= ARC_MIN, 1, 2))
    d = d.sort_values(["tier", "max_arc_deg", "graded_cells"],
                      ascending=[True, False, False]).reset_index(drop=True)

    med = d.groupby("kind").max_arc_deg.median().to_dict()
    tier_n = d.tier.value_counts().to_dict()

    # An ISOLATED cell is gate_map's `spike`: no graded neighbour in azimuth +-15 deg
    # NOR at the adjacent elevation (continuity_map.py head comment). A single-bin arc
    # in continuity_arcs.csv is NOT the same thing -- that cell may still have an
    # elevation neighbour, and counting those gives 19 rather than the map's 2.
    gate = pd.read_csv(GATE)
    graded = gate[gate.grade.isin(["strong", "moderate"])]
    iso = graded[graded.spike]
    n_iso = len(iso)
    iso_rows = sorted(iso.metric.unique())
    n_thin = int((arcs.bins == 1).sum())

    print(f"rows {len(d)}   tiers: azimuth-independent {tier_n.get(0, 0)} / "
          f"arc-limited {tier_n.get(1, 0)} / point-like {tier_n.get(2, 0)}")
    print("median widest arc by kind (deg):",
          {k: round(v) for k, v in med.items()})
    print("median widest arc by kind (bins):",
          {k: round(v / BIN_DEG, 1) for k, v in med.items()})
    print(f"rows holding an arc >= {ARC_MIN} bins: "
          f"{int((d.max_arc_bins >= ARC_MIN).sum())} of {len(d)}")
    print(f"isolated cells (gate_map.spike) {n_iso}: {iso_rows}")
    print(iso[["metric", "az", "el", "grade"]].to_string(index=False))
    print(f"  (for contrast, single-bin arcs at some elevation: {n_thin})")
    print("narrowest row:", d.iloc[-1].metric_id, int(d.iloc[-1].max_arc_deg), "deg")

    n = len(d)
    # SINGLE COLUMN (2026-08-01). At half width there is no room for a reserved label
    # column on the right, so the elevation each arc occurs at moved INTO the row label
    # -- the same device fig_graded_map.strip already uses -- and the tier names moved to
    # the caption. Nothing is drawn outside the axes and the figure is saved without a
    # tight bbox, so the authored width is the placed width.
    COL_W = BODY_W / 2
    # R leaves room for the "360" tick label: the axis now ENDS at 362, so at R=0.995
    # that label was half off the page
    L, R, TOP, BOT_IN = 0.358, 0.972, 0.995, 0.38

    span = n - 1
    y = span - np.arange(n)                  # first row of `d` at the top

    fig_h = ROW_IN * n + BOT_IN + 0.06
    fig, ax = plt.subplots(figsize=(COL_W, fig_h))
    fig.subplots_adjust(left=L, right=R, top=TOP, bottom=BOT_IN / fig_h)
    ax_in = COL_W * (R - L)

    # THE RULES GO IN FRONT (zorder 3, over the bars at 2). Behind the bars they showed
    # only in the gaps between rows, so a reader could not see which bars a rule cuts
    # and the whole point of drawing them was lost. Crossing the value labels is then
    # handled at the label, not by hiding the rule: each number carries a bbox in its
    # own bar's colour at zorder 4, so the rule visibly passes BEHIND the number and
    # nothing overprints.
    floor = ARC_MIN * BIN_DEG

    def deg_of(text):
        """width of `text` at FS_VAL, in data (degree) units"""
        return (len(text) * FS_VAL * 0.62 / 72.0) / ax_in * X_MAX

    ax.barh(y, d.max_arc_deg, 0.72,
            color=[KIND_C[k] for k in d.kind], edgecolor="white", lw=0.4, zorder=2)
    n_out = 0
    for yy, r in zip(y, d.itertuples(index=False)):
        lab = f"{int(r.max_arc_deg)}°"
        w = deg_of(lab)
        # 3 deg of inset and 6 of slack, not 5 and 10: the narrowest bar (30 deg) does
        # hold its own label, and the looser test pushed it outside and then past the
        # floor rule -- putting the number to the RIGHT of the line for the one row
        # whose entire point is that it falls short of it.
        inner_l, inner_r = r.max_arc_deg - 3 - w, r.max_arc_deg - 3
        if r.max_arc_deg >= w + 6 and not (inner_l - 2 <= floor <= inner_r + 2):
            ax.text(inner_r, yy, lab, va="center", ha="right",
                    fontsize=FS_VAL, color="white", zorder=4)
        else:
            x0 = r.max_arc_deg + 5
            if x0 - 2 <= floor <= x0 + w + 2:      # never let the rule strike it out
                x0 = floor + 5
            ax.text(x0, yy, lab, va="center", ha="left", fontsize=FS_VAL,
                    color=INK, zorder=4)
            n_out += 1
    print(f"value labels outside their bar: {n_out} of {n}")

    # The three rules, ON TOP OF EVERYTHING (zorder 5, over bars at 2 and numbers at 4).
    # They are annotation: they have to be readable across the whole chart, so nothing
    # masks them -- not the bars, not the value labels. Their colours (MED_C, FLOOR_C)
    # are chosen so they never collide with a bar colour, which is what makes putting
    # them on top actually work.
    handles = [Patch(facecolor=KIND_C[k], label=k)
               for k in KIND_C if k in set(d.kind)]
    ax.axvline(floor, color=FLOOR_C, lw=1.2, zorder=5)
    handles.append(Line2D([], [], color=FLOOR_C, lw=1.2,
                          label=f"{int(floor)}° aimability floor"))

    ax.set_yticks(y)
    ax.set_yticklabels([f"{short(m)[:26]}  (el {int(e)})"
                        for m, e in zip(d.metric_id, d.max_arc_el)],
                       fontsize=FS_ROW)
    ax.set_ylim(-0.85, span + 0.85)
    ax.set_xlim(0, X_MAX)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.tick_params(axis="x", labelsize=FS_TICK, length=2, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xlabel("widest contiguous azimuthal arc (deg)", fontsize=FS_AXLAB,
                  labelpad=1.5)
    # lifted clear of the point-like tier separator: at y=0 the separator ruled straight
    # through the legend entries
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.005, 0.010),
              frameon=False, fontsize=FS_LEG, ncol=1, handlelength=1.4,
              handleheight=0.8, labelspacing=0.30, borderpad=0.1)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 360)

    out = os.path.join(OUT, "fig_arc_width.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved -> {out}   {COL_W:.2f} x {fig_h:.2f} in")


if __name__ == "__main__":
    main()
