"""Body figure (the paper's roadmap): the ceiling, and the idealisations that lower it.

Every cell of the map is measured under conditions chosen to be favourable: the 2D pose
is a clean projection of the 3D truth, the temporal anchors are the ground-truth event
frames, the camera sits exactly on a swept grid point, and the calibration is fitted on
other pitchers. So the map is an UPPER BOUND, and the paper's structure is to state that
bound and then remove one idealisation at a time.

This figure is that structure, with the surviving evidence beside each rung. It exists
because the erosion chain is the one thing a reader has to hold in mind across four
result sections; a caption cannot carry it.

Every number is read at run time from the canonical CSVs and printed, so a re-render
cannot silently keep a stale value:

  gate_map.csv / layer_summary.csv   the ceiling: evaluated vs graded cells
  continuity_rows.csv                viewpoint slack: arc widths
  event_tolerance_map.csv            timing: grade held under an anchor shift
  fp_target_rows.csv                 anchor definition: fp_100 vs fp_10
  accuracy_bestcell_gt_clean.csv     purpose: pooled vs within-pitcher
  deploy_map_cells.csv               deployment: detected pose and events
  inference_trajectory.csv           the ceiling no viewpoint lifts

2026-08-03 revision, three changes plus the connector:

1. THE DEPLOYMENT RUNG NOW CARRIES ITS NUMBERS. It used to be a dashed placeholder
   holding the hard-coded string "TBD after fresh deployment sweep", written while the
   layer was blocked. The layer was unblocked and regenerated on 2026-07-31, so the
   placeholder was simply WRONG, and being hard-coded it could not be corrected by a
   re-render. It is now read from `deploy_map_cells.csv` like every other rung.

2. EACH RUNG CARRIES A RETENTION BAR. The figure was previously all prose: nothing was
   drawn to scale, so the one claim it exists to make -- the ceiling only ever falls --
   had to be read rather than seen. Each rung now shows what fraction of its own
   applicable set survives. The denominators genuinely differ from rung to rung, so
   each bar states its own beneath it; the anchor rung is a definition choice, not a
   retention step, and correctly shows no bar.

3. TYPE IS SET AT READABLE SIZE. Evidence text was 5.8 pt at a 7.16 in placement, about
   half the manuscript body size. It is now 8 pt, paid for by moving the scope lines out
   of the rungs (the trim the figure manifest already nominated) and shortening the rest.

   The connector is gone. It was a floating arrow at x = 0.055 aligned to nothing, then
   briefly a rail whose arrowhead was drawn in the rail's own colour and so could not be
   seen at all. The accent bar now simply continues through each gap, so the rungs are
   segments of one spine. A figure reads downward without being told to.

Run:  conda activate diamond
      cd src\\viz
      python fig_ladder.py
"""
import os, sys, textwrap
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", ".."):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
from fig_graded_map import INK, MUTE, BODY_W

V = config.OBP_VALIDATION_DIR
OUT = os.path.join(config.ROOT, "data", "outputs", "viz")

CEIL_C = "#0C447C"      # the ideal-condition ceiling
STEP_C = "#0FA3B1"      # an idealisation removed, quantified
WALL_C = "#A63A5E"      # the bound no viewpoint lifts
# WALL_C was #7C2D4B, whose greyscale luminance (72) sat too close to CEIL_C's (58) for
# the two to separate in a mono print; #A63A5E lands at 94, between CEIL and STEP.
TRACK_C = "#E3E9F1"     # unfilled part of a retention bar
GRADED = ("strong", "moderate")
N_EL, N_AZ = 7, 24
TOL_FRAMES = 3


def _ascii(s):
    """The Windows console is cp949 here, so the printed copy is transliterated.
    The FIGURE keeps the typographic characters; only stdout is sanitised."""
    for a, b in (("—", "-"), ("−", "-"), ("±", "+/-"), ("≥", ">="),
                 ("≤", "<="), ("°", " deg"), ("·", "|"),
                 ("→", "->"), ("×", "x"), ("²", "2")):
        s = s.replace(a, b)
    return s


def facts():
    """Read every number this figure prints.

    Returns (text, bars): `text` maps a rung key to its evidence block, `bars` maps the
    same key to (fraction, denominator label) or None where no retention is defined.
    """
    g = pd.read_csv(os.path.join(V, "gate_map.csv"))
    s = pd.read_csv(os.path.join(V, "layer_summary.csv"))
    reg = pd.read_csv(os.path.join(V, "paper_registry.csv"))
    cont = pd.read_csv(os.path.join(V, "continuity_rows.csv"))
    tol = pd.read_csv(os.path.join(V, "event_tolerance_map.csv"))
    fp = pd.read_csv(os.path.join(V, "fp_target_rows.csv"))
    fpc = pd.read_csv(os.path.join(V, "fp_target_cells.csv"))
    acc = pd.read_csv(os.path.join(V, "accuracy_bestcell_gt_clean.csv"))
    dep = pd.read_csv(os.path.join(V, "deploy_map_cells.csv"))
    inf = pd.read_csv(os.path.join(V, "inference_trajectory.csv"))

    f, b = {}, {}
    ret = s[s.map_cells > 0]
    n_ret, n_all = len(ret), len(s)
    n_eval = n_ret * N_EL * N_AZ
    n_strong = int((g.grade == "strong").sum())
    n_mod = int((g.grade == "moderate").sum())
    graded = n_strong + n_mod
    f["ceiling"] = (f"{graded:,} graded of {n_eval:,} evaluated cells\n"
                    f"{n_strong:,} strong + {n_mod:,} moderate;  "
                    f"{n_all} evaluated rows, {n_ret} retained")
    b["ceiling"] = (graded / n_eval, "cells")

    ARC_MIN = 3
    keep = int((cont.max_arc_bins >= ARC_MIN).sum())
    med = cont.groupby("kind").max_arc_deg.median().sort_values(ascending=False)
    f["view"] = (f"{keep} of {len(cont)} rows keep a contiguous arc ≥ {ARC_MIN} bins\n"
                 "median widest arc "
                 + "  ·  ".join(f"{int(v)}° {k.split('/')[0]}" for k, v in med.items()))
    b["view"] = (keep / len(cont), "rows")

    # The denominator is GRADED cells of the applicable rows, not every cell of them:
    # event_tolerance_map.csv carries the whole 168-cell field for each applicable row
    # plus the graded cells of the two rows that read no external anchor.
    ap = tol[tol.applicable & tol.grade0.isin(GRADED)]
    tot_c = len(ap)
    hold = int((ap.tol_map >= TOL_FRAMES).sum())
    die1 = int((ap.tol_map == 0).sum())
    n_na = int((~tol.applicable).sum())
    n_na_rows = tol[~tol.applicable].metric.nunique()
    assert tot_c + n_na == graded, (tot_c, n_na, graded)
    f["time"] = (f"{hold:,} of {tot_c:,} graded cells hold their grade at "
                 f"±{TOL_FRAMES} frames\n{die1} die at ±1;  {tot_c:,} = "
                 f"{graded:,} − {n_na} cells of {n_na_rows} event-free rows")
    b["time"] = (hold / tot_c, "cells")

    # 27 fp-dependent rows is a REGISTRY fact (anchor_type carries `fp` even when the
    # event key names the far end of a window); fp_target_rows.csv lists only the rows
    # that hold a cell under either anchor. This rung REPLACES one definition with
    # another rather than losing cells to it, so it carries no retention bar.
    n_fp_rows = int(reg.anchor_type.fillna("").str.contains("fp").sum())
    pref100, pref10 = int(fp.prefers_fp100.sum()), int(fp.prefers_fp10.sum())
    ties = int(fp.ties.sum())
    d100, d10 = int(fp.graded_fp100.sum()), int(fp.graded_fp10.sum())
    only = int((fp.graded_fp10 == 0).sum())
    f["anchor"] = (f"over {len(fpc):,} cells of {n_fp_rows} fp-dependent rows, "
                   f"fp_100 wins {pref100} to {pref10}\n{ties} ties;  "
                   f"{d100 - d10:+d} graded cells;  {only} rows exist only at fp_100")
    b["anchor"] = None

    # Arm Slot is excluded as the "good" exemplar: at az90 the 2D coronal definition IS
    # the 3D-direct definition, so its 1.00/1.00 is a synthetic identity, not skill.
    from fig_within_pitcher import IDENTITY
    a = acc.dropna(subset=["pooled_r2", "within_r2"]).copy()
    a["short"] = a.metric.str.replace(" [O]", "", regex=False)
    n_pre = len(a)
    a = a[~a["short"].isin(IDENTITY)]          # keeps the count off the identity row
    keep_w = int((a.within_r2 >= 0.60).sum())
    lo = a.loc[(a.pooled_r2 - a.within_r2).idxmax()]
    f["purpose"] = (f"{keep_w} of {len(a)} rows keep within-pitcher r² ≥ 0.60 at the "
                    f"same cell\nwidest drop {lo['short']} "
                    f"{lo.pooled_r2:.2f} → {lo.within_r2:.2f}  "
                    f"({len(a)} = {n_pre} − {', '.join(sorted(IDENTITY))})")
    # why Arm Slot is subtracted -- at az90 the 2D coronal definition IS the 3D-direct
    # one, so its 1.00/1.00 is a synthetic identity -- no longer fits on the rung at
    # 8 pt and belongs in the caption.
    b["purpose"] = (keep_w / len(a), "rows")

    # The deployment layer was regenerated from current code on the frozen population on
    # 2026-07-31. `grade_lopo` is the adopted routing (pitcher-blind LOPO CV); `rule` is
    # a single fixed release view and `oracle` is an unattainable per-cell choice, so the
    # two bracket it and are quoted as context, never as the result.
    d = dep[dep.onmap_gt]
    assert len(d) == graded, (len(d), graded)
    kept = d[d.grade_lopo.isin(GRADED)]
    n_dep, n_dep_s = len(kept), int((kept.grade_lopo == "strong").sum())
    n_rule = int(d.grade_rule.isin(GRADED).sum())
    n_orac = int(d.grade_oracle.isin(GRADED).sum())
    r_dep, r_gt = kept.metric.nunique(), d.metric.nunique()
    # `rule` (a single fixed release view) and `oracle` (an unattainable per-cell choice)
    # bracket the adopted routing. They are printed for the log but no longer fit on the
    # rung at 8 pt; they are section VIII-A's numbers, not the roadmap's. No value is
    # written here -- a number in a comment is the same stale-caption trap as one in a
    # caption, and a re-render cannot refresh it.
    f["deploy"] = (f"{n_dep} of {graded:,} graded cells survive detected pose and "
                   f"events\n{n_dep_s} strong + {n_dep - n_dep_s} moderate;  "
                   f"{r_dep} of {r_gt} rows keep a cell")
    f["_deploy_brackets"] = (f"fixed release-view rule {n_rule}  ·  "
                             f"unattainable per-cell oracle {n_orac}")
    b["deploy"] = (n_dep / graded, "cells")

    n_t = len(inf)
    pas = int((inf.best_r2 >= 0.60).sum())
    best = inf.loc[inf.best_r2.idxmax()]
    f["kinetic"] = (f"{pas} of {n_t} kinetic targets reach R² 0.60 from PERFECT 3D "
                    f"truth\nbest {best.target} {best.best_r2:.3f};  "
                    "not a projection limit")
    b["kinetic"] = (pas / n_t, "targets")

    for k, v in f.items():
        print(f"{k:9s} {_ascii(v)}".replace("\n", "\n          "))
    for k, v in b.items():
        print(f"  bar {k:9s} " + ("none" if v is None
                                  else f"{v[0] * 100:5.1f}%  {_ascii(v[1])}"))
    return f, b


# Layout is in DATA units so the line spacing of the evidence column is exact: one
# unit is one text line. Right-aligning a long line against the block edge is what made
# the first draft overprint its own titles, so the three columns are explicit and the
# evidence text is wrapped to the column width.
UNIT_IN = 0.152          # height of one text line: 10.9 pt leading on 8 pt type
GAP_U = 0.65             # blank units between rungs: room for the rail to be seen
PAD_U = 0.85             # blank units inside a rung, above and below the text block
X_LEFT, X_BODY, X_TEXT, X_BAR, X_R, X_END = 0.012, 0.300, 0.825, 0.860, 0.978, 0.988
ACCENT_W = 0.0055        # width of the left accent bar, in x units
# 8 pt over (X_TEXT - X_BODY) * BODY_W = 3.76 in holds ~65 characters; titles are kept
# under 25 characters so the left column never reaches X_BODY. Evidence strings carry
# their own newlines, so wrapping is a guard rail rather than the layout mechanism.
#
# EVERY RUNG IS TWO LINES. That is what keeps the figure to a third of a page: seven
# rungs at three lines cost 0.46 in more than the whole ceiling rung is worth. A third
# line does not "just wrap" -- it grows the float. Cut the sentence instead.
WRAP = 62
LINE_U = 1.0             # units advanced per evidence line
MIN_U = 2.3              # a rung is never shorter than tag + title, or than a bar
FS_TAG, FS_TITLE, FS_BODY, FS_PCT, FS_DEN = 6.5, 9.0, 8.0, 8.0, 6.2


def wrap(body):
    out = []
    for ln in body.split("\n"):
        out += textwrap.wrap(ln, width=WRAP) or [""]
    return out


def height(lines):
    return max(MIN_U, len(lines) * LINE_U + PAD_U)


def retention_bar(ax, yc, frac, unit, colour):
    """One rung's retention bar: "26 % of cells" over a track.

    The unit noun is on the bar because the DENOMINATOR CHANGES from rung to rung --
    cells here, rows there -- and two bars read side by side without it would look like
    one population. The full denominator is not repeated: the evidence line beside the
    bar already states it ("1,500 graded of 5,880 evaluated cells"), and printing it
    twice cost a line of height per rung for nothing.
    """
    # "26% of cells", closed up: IEEE house style, and what every other figure in this
    # repo already does (fig_event_tolerance, fig_parallax_resolution). The spaced form
    # is the SI/typographic convention and was inconsistent here.
    ax.text(X_R, yc - 0.42, f"{frac * 100:.0f}% of {unit}", fontsize=FS_PCT,
            color=colour, weight="bold", va="center", ha="right")
    ax.add_patch(plt.Rectangle((X_BAR, yc + 0.14), X_R - X_BAR, 0.38,
                               facecolor=TRACK_C, edgecolor="none", zorder=2))
    if frac > 0:
        ax.add_patch(plt.Rectangle((X_BAR, yc + 0.14), (X_R - X_BAR) * frac, 0.38,
                                   facecolor=colour, edgecolor="none", zorder=3))


def band(ax, y, colour, tag, title, lines, bar):
    """One rung. `y` is its TOP edge; the y axis is inverted. Returns its height."""
    h = height(lines)
    yc = y + h / 2.0
    ax.add_patch(plt.Rectangle((X_LEFT, y), X_END - X_LEFT, h, facecolor="#F4F7FB",
                               edgecolor="none", zorder=1))
    ax.add_patch(plt.Rectangle((X_LEFT, y), ACCENT_W, h, facecolor=colour,
                               edgecolor="none", zorder=2))
    # tag and title are centred as a pair, so the left column tracks the evidence
    # column instead of riding at the top of the taller rungs
    ax.text(X_LEFT + 0.016, yc - 0.52, tag, fontsize=FS_TAG, color=colour,
            weight="bold", va="center", ha="left")
    ax.text(X_LEFT + 0.016, yc + 0.50, title, fontsize=FS_TITLE, color=INK,
            weight="bold", va="center", ha="left")
    y0 = yc - (len(lines) - 1) * LINE_U / 2.0
    for i, ln in enumerate(lines):
        ax.text(X_BODY, y0 + i * LINE_U, ln, fontsize=FS_BODY, color=MUTE,
                va="center", ha="left")
    if bar is None:
        ax.text(X_R, yc, "definition choice", fontsize=FS_DEN, color=MUTE,
                style="italic", va="center", ha="right")
    else:
        retention_bar(ax, yc, bar[0], bar[1], colour)
    return h


def main():
    os.makedirs(OUT, exist_ok=True)
    f, b = facts()

    rungs = [
        (CEIL_C, "IDEAL CONDITIONS", "The graded map", f["ceiling"], b["ceiling"]),
        (STEP_C, "REMOVE  exact camera placement", "Azimuthal arcs",
         f["view"], b["view"]),
        (STEP_C, "REMOVE  exact event frames", "Temporal precision",
         f["time"], b["time"]),
        (STEP_C, "REMOVE  one event definition", "Anchor definition",
         f["anchor"], b["anchor"]),
        (STEP_C, "REMOVE  the pooled framing", "Within-pitcher tracking",
         f["purpose"], b["purpose"]),
        # The clean projection is NOT removed anywhere in the paper. The rung that
        # used to sit here reported 863 of 1,500 cells surviving detected pose and
        # events; those counts are retired (the detector set is not exhaustive over
        # viewpoints, so a count over the grid reports the dispatcher's coverage
        # rather than a property of the problem). Section VIII replaced them with
        # detector error in ms at three stations and makes no claim on the map.
        (WALL_C, "A BOUND NO VIEWPOINT LIFTS", "Joint kinetics",
         f["kinetic"], b["kinetic"]),
    ]
    wrapped = [wrap(r[3]) for r in rungs]
    heights = [height(w) for w in wrapped]
    RULE_U = 0.55                                   # the break before the last rung
    total = sum(heights) + GAP_U * (len(rungs) - 1) + RULE_U + 0.30

    fig, ax = plt.subplots(figsize=(BODY_W, total * UNIT_IN))
    # the axes fill the figure, so authored width == saved width == placed width
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(total, 0)                           # inverted: y grows downward
    ax.axis("off")

    # NO ARROWS. Two attempts failed: a floating arrow at x = 0.055 aligned to nothing,
    # then a rail with an arrowhead drawn in the RAIL'S OWN COLOUR, which simply
    # disappeared into the rail. The chain does not need one -- a figure reads downward
    # by default, the tags already say REMOVE in sequence, and the bars fall.
    #
    # Instead the accent bar CONTINUES through each gap in the next rung's colour, so
    # the rungs are segments of one spine rather than seven separate blocks. It is drawn
    # as a Rectangle at the same x and width as the accent bar, not as a line of matched
    # linewidth, so the join is exact at any dpi. The spine stops at the rule: the
    # kinetic bound is not the end of the chain, it is a different claim.
    y = 0.15
    for i, ((c, tag, title, _, bar), lines) in enumerate(zip(rungs, wrapped)):
        last = i == len(rungs) - 1
        if last:                                    # the kinetic bound is not a rung
            y += RULE_U / 2
            ax.plot([X_LEFT, X_END], [y - 0.10] * 2, color="#DCE3EC", lw=0.8)
            y += RULE_U / 2
        h = band(ax, y, c, tag, title, lines, bar)
        if i < len(rungs) - 2:
            ax.add_patch(plt.Rectangle((X_LEFT, y + h), ACCENT_W, GAP_U,
                                       facecolor=rungs[i + 1][0], edgecolor="none",
                                       zorder=2))
        y += h + GAP_U

    out = os.path.join(OUT, "fig_ladder.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved -> {out}   {BODY_W:.2f} x {total * UNIT_IN:.2f} in")


if __name__ == "__main__":
    main()
