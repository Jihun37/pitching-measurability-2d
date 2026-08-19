"""Body figure: most measurable cells are measurable AFTER a pitcher-blind correction.

The graded map answers "can this be measured here". It does not say whether the 2D
reading is already on the truth's scale. gate_map.csv's `verdict` column does, and it
splits the graded cells three ways (analysis/gate_map.py:152-161):

  DIRECT       raw (uncalibrated) CCC already >= the moderate floor 0.75
  CALIBRATE    the leave-one-pitcher-out model lifts CCC over the floor AND drops both
               MAE and the per-pitcher bias SD -- i.e. 2D reads the right signal with a
               geometric bias that generalises to an unseen pitcher
  PASS(weak)   clears the grade line, but the winning model does not shrink per-pitcher
               bias, so it is flagged: it will not carry individual tracking

NOTE the gate_map.py module docstring still says DIRECT = raw CCC >= 0.80; the code
uses `floor` (default 0.75). The code is the authority and this figure follows it.

TWO OUTPUTS. `fig_verdict.png` is the body figure at single-column width: the whole
graded set and the same split within strong and within moderate. `fig_verdict_rows.png`
is the appendix companion, the same split per row, which shows that DIRECT is not spread
evenly. They were one two-panel float until 2026-08-01; splitting them returned about
half a page of body to a paper that states the per-row facts in the text anyway.

Every count is read from gate_map.csv and printed on each run. Nothing is hard-coded.

Run:  conda activate diamond
      cd src\\viz
      python fig_verdict.py
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", ".."):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
from fig_graded_map import short, INK, MUTE, BODY_W

GATE = os.path.join(config.OBP_VALIDATION_DIR, "gate_map.csv")
SUMM = os.path.join(config.OBP_VALIDATION_DIR, "layer_summary.csv")
OUT = os.path.join(config.ROOT, "data", "outputs", "viz")

# a quality ladder, dark = needs nothing from us
ORDER = ["DIRECT", "CALIBRATE", "PASS(weak)"]
COL = {"DIRECT": "#0C447C", "CALIBRATE": "#0FA3B1", "PASS(weak)": "#9AA3B2"}
LABEL = {"DIRECT": "DIRECT  (raw CCC ≥ 0.75, no correction needed)",
         "CALIBRATE": "CALIBRATE  (pitcher-blind model lifts CCC and shrinks bias)",
         "PASS(weak)": "PASS(weak)  (grade held, per-pitcher bias not shrunk)"}


FS_LAB, FS_NUM, FS_LEG = 6.5, 6.8, 6.5


def stack(ax, y, counts, total, h=0.62, annotate=True, span=None):
    """one horizontal stacked bar of verdict counts, drawn as absolute cell counts.

    `span` is (axes width in inches, x range in data units). Given it, every segment is
    annotated: the label goes INSIDE when it fits and just above the bar when it does
    not. The old rule skipped any segment under 9 % of the grand total, which silently
    dropped moderate's 29 DIRECT cells -- tolerable while an x axis was there to read
    them off, wrong once the axis is gone.
    """
    left = 0.0
    for v in ORDER:
        w = counts.get(v, 0)
        if w == 0:
            continue
        ax.barh(y, w, h, left=left, color=COL[v], edgecolor="white", lw=0.6)
        if annotate:
            lab = f"{w}"
            if span is None:
                fits = w / max(total, 1) >= 0.09
            else:
                ax_in, x_range = span
                need_in = len(lab) * FS_NUM * 0.62 / 72.0 + 0.035
                fits = need_in / ax_in * x_range <= w
            if fits:
                ax.text(left + w / 2, y, lab, ha="center", va="center",
                        fontsize=FS_NUM, color="white", weight="bold")
            else:
                # BELOW its own bar, not centred in the gap: at the midpoint of the
                # inter-row gap the number reads as if it belonged to the bar above.
                ax.text(left + w / 2, y - h / 2 - 0.14, lab, ha="center",
                        va="center", fontsize=FS_NUM - 0.8, color=COL[v],
                        weight="bold")
        left += w
    return left


def main():
    os.makedirs(OUT, exist_ok=True)
    g = pd.read_csv(GATE)
    s = pd.read_csv(SUMM)
    gr = g[g.grade.isin(["strong", "moderate"])].copy()

    tot = len(gr)
    all_c = gr.verdict.value_counts().to_dict()
    str_c = gr[gr.grade == "strong"].verdict.value_counts().to_dict()
    mod_c = gr[gr.grade == "moderate"].verdict.value_counts().to_dict()

    print(f"graded cells {tot}  = " +
          " + ".join(f"{v} {all_c.get(v, 0)}" for v in ORDER))
    print(f"  strong   {sum(str_c.values())}: " +
          " ".join(f"{v} {str_c.get(v, 0)}" for v in ORDER))
    print(f"  moderate {sum(mod_c.values())}: " +
          " ".join(f"{v} {mod_c.get(v, 0)}" for v in ORDER))
    print("  calibration model over graded cells:",
          gr.model.value_counts().to_dict())

    # per-row composition, over the retained rows only
    rows = s[s.map_cells > 0][["metric", "source"]]
    per = []
    for r in rows.itertuples(index=False):
        sub = gr[(gr.metric == r.metric) & (gr.source == r.source)]
        c = sub.verdict.value_counts().to_dict()
        n = len(sub)
        per.append(dict(metric=r.metric, n=n,
                        direct=c.get("DIRECT", 0),
                        calibrate=c.get("CALIBRATE", 0),
                        weak=c.get("PASS(weak)", 0),
                        f_direct=c.get("DIRECT", 0) / n if n else np.nan))
    per = pd.DataFrame(per).sort_values(["f_direct", "n"], ascending=True)
    n_row = len(per)
    n_nodirect = int((per.direct == 0).sum())
    n_alldirect = int((per.f_direct == 1).sum())
    print(f"rows {n_row}:  {n_alldirect} DIRECT at every graded cell,  "
          f"{n_nodirect} with no DIRECT cell at all")

    # TWO FIGURES, not two panels of one. Panel (a) is the body claim and fits a single
    # column; the per-row composition is verification and goes to the appendix. Keeping
    # them in one float cost half a page of the body for material the text already states
    # row by row.
    COL_W = BODY_W / 2

    # ---- (a) body, single column -------------------------------------------
    # NO X AXIS. Every segment carries its own count and every row total is in its y
    # label, so the 0..1400 ruler and the "graded cells" title under it were restating
    # what the bars already said -- and "graded cells" is caption wording besides. The
    # bars still share one scale, so their lengths stay comparable. Dropping the axis
    # paid for readable type: 5.8 pt -> 6.5 pt, in a figure that also got shorter.
    L, R, TOP, BOT = 0.255, 0.99, 0.97, 0.20
    FIG_H = 1.30
    figa, axa = plt.subplots(figsize=(COL_W, FIG_H))
    figa.subplots_adjust(left=L, right=R, top=TOP, bottom=BOT)
    x_range = tot * 1.02
    span = (COL_W * (R - L), x_range)
    for i, c in enumerate([mod_c, str_c, all_c]):
        stack(axa, i, c, tot, span=span)
    axa.set_yticks(range(3))
    axa.set_yticklabels([f"moderate  ({sum(mod_c.values())})",
                         f"strong  ({sum(str_c.values())})",
                         f"all graded  ({tot})"], fontsize=FS_LAB)
    axa.set_xlim(0, x_range)
    axa.set_ylim(-0.78, 2.45)              # room for a below-bar count
    axa.set_xticks([])
    axa.tick_params(axis="y", length=0, pad=1.5)
    for sp in ("top", "right", "left", "bottom"):
        axa.spines[sp].set_visible(False)
    axa.legend(handles=[Patch(facecolor=COL[v], label=v) for v in ORDER],
               loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
               frameon=False, fontsize=FS_LEG, handlelength=1.1,
               handleheight=0.8, columnspacing=1.2)
    out_a = os.path.join(OUT, "fig_verdict.png")
    figa.savefig(out_a, dpi=300)
    plt.close(figa)
    print(f"saved -> {out_a}   {COL_W:.2f} x {FIG_H:.2f} in")

    # ---- (b) appendix, per row ---------------------------------------------
    ROW_IN = 0.104
    figb, axb = plt.subplots(figsize=(BODY_W, ROW_IN * n_row + 0.78))
    for i, r in enumerate(per.itertuples(index=False)):
        stack(axb, i, {"DIRECT": r.direct, "CALIBRATE": r.calibrate,
                       "PASS(weak)": r.weak}, r.n, h=0.74, annotate=False)
        axb.text(r.n + 1.5, i, f"{r.n}", va="center", ha="left",
                 fontsize=5.4, color=MUTE)
    axb.set_yticks(range(n_row))
    axb.set_yticklabels([short(m) for m in per.metric], fontsize=5.6)
    axb.set_ylim(-0.8, n_row - 0.2)
    axb.set_xlim(0, per.n.max() * 1.08)
    axb.set_xlabel("graded cells in that row   (grey number = row total)",
                   fontsize=6.8)
    axb.tick_params(axis="x", labelsize=6.0)
    axb.tick_params(axis="y", length=0, pad=1.5)
    axb.legend(handles=[Patch(facecolor=COL[v], label=v) for v in ORDER],
               loc="lower right", frameon=False, fontsize=6.0, ncol=3,
               handlelength=1.2, columnspacing=1.4)
    for sp in ("top", "right", "left"):
        axb.spines[sp].set_visible(False)
    figb.subplots_adjust(left=0.185, right=0.985, top=0.985, bottom=0.105)
    out_b = os.path.join(OUT, "fig_verdict_rows.png")
    figb.savefig(out_b, dpi=300)
    plt.close(figb)
    print("saved ->", out_b)

    print("CAPTION numbers: graded {} = DIRECT {} / CALIBRATE {} / PASS(weak) {}"
          .format(tot, all_c.get("DIRECT", 0), all_c.get("CALIBRATE", 0),
                  all_c.get("PASS(weak)", 0)))


if __name__ == "__main__":
    main()
