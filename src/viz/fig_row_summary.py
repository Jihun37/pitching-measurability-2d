"""F3 -- ONE per-row figure replacing two: azimuth tolerance beside verdict composition.

WHY THE MERGE (round 4 of the 2026-08-07 review). `fig_arc_width` and the per-row panel of
`fig_verdict` are both one-bar-per-retained-row charts over the SAME rows, so between them the
row labels were set twice and half a page went to repeating them. Sharing one row order
also puts the two readings side by side, which is where the interesting rows are: a wide
arc whose cells are all PASS(weak) is measurable from many angles and still will not track
an individual, and that is invisible when the panels live apart.

IT ALSO GIVES THE VERDICT ROWS A HOME. `fig_verdict` writes its per-row panel for an
appendix, and decision D4 (2026-08-07) removed the appendix along with the supplement.
Without this merge that panel has nowhere to go.

ROW ORDER is `fig_arc_width`'s: tier, then widest arc, then graded cells. The left panel is
therefore unchanged in reading order; only its width and the shared labels differ.

Every number is read from the canonical CSVs and printed on each run. Nothing is
hard-coded, so a regeneration cannot silently disagree with the text -- but the CAPTION
still has to be re-checked against the printed block, because a caption is not code.

Saved without a tight bbox at BODY_W, so the authored width is the placed width.

Run:  conda activate diamond; cd src\\viz; python fig_row_summary.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../analysis"):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
from fig_graded_map import short, INK, MUTE, BODY_W
from fig_verdict import COL, ORDER, stack
from fig_arc_width import FULL_BINS, ARC_MIN, BIN_DEG, KIND_C, FLOOR_C

OUT = os.path.join(config.ROOT, "data", "outputs", "viz")
V = config.OBP_VALIDATION_DIR
ROWS = os.path.join(V, "continuity_rows.csv")
ARCS = os.path.join(V, "continuity_arcs.csv")
GATE = os.path.join(V, "gate_map.csv")

ROW_IN = 0.105
FS_ROW, FS_VAL, FS_TICK, FS_AXLAB, FS_LEG = 5.4, 5.0, 5.6, 6.2, 5.2


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(ROWS)
    arcs = pd.read_csv(ARCS)
    gate = pd.read_csv(GATE)
    graded = gate[gate.grade.isin(["strong", "moderate"])]

    d["tier"] = np.where(d.max_arc_bins >= FULL_BINS, 0,
                         np.where(d.max_arc_bins >= ARC_MIN, 1, 2))
    d = d.sort_values(["tier", "max_arc_deg", "graded_cells"],
                      ascending=[True, False, False]).reset_index(drop=True)
    n = len(d)

    # verdict composition per row, on the SAME order
    vc = []
    for m in d.metric_id:
        sub = graded[graded.metric == m]
        c = sub.verdict.value_counts().to_dict()
        vc.append(dict(metric_id=m, n=len(sub), **{v: c.get(v, 0) for v in ORDER}))
    vc = pd.DataFrame(vc).set_index("metric_id").loc[d.metric_id].reset_index()
    assert (vc.n.to_numpy() == d.graded_cells.to_numpy()).all(), \
        "verdict counts disagree with continuity_rows.graded_cells"

    tier_n = d.tier.value_counts().to_dict()
    med = d.groupby("kind").max_arc_deg.median().to_dict()
    iso = graded[graded.spike]
    print(f"rows {n}   tiers: azimuth-independent {tier_n.get(0, 0)} / "
          f"arc-limited {tier_n.get(1, 0)} / narrower {tier_n.get(2, 0)}")
    print(f"rows holding an arc >= {ARC_MIN} bins: "
          f"{int((d.max_arc_bins >= ARC_MIN).sum())} of {n}")
    print("median widest arc by kind (deg):", {k: round(v) for k, v in med.items()})
    print("median widest arc by kind (bins):",
          {k: round(v / BIN_DEG, 1) for k, v in med.items()})
    print(f"contiguous arcs {len(arcs)}   mean {arcs.bins.mean():.2f} bins "
          f"({BIN_DEG * arcs.bins.mean():.0f} deg)   median {arcs.bins.median():.0f}   "
          f"single-bin {int((arcs.bins == 1).sum())}")
    print(f"isolated cells (gate_map.spike) {len(iso)}: {sorted(iso.metric.unique())}")
    tot = len(graded)
    allc = graded.verdict.value_counts().to_dict()
    print(f"graded cells {tot} = " +
          " + ".join(f"{v} {allc.get(v, 0)}" for v in ORDER))
    nod = int((vc[ORDER[0]] == 0).sum())
    alld = int((vc[ORDER[0]] == vc.n).sum())
    print(f"rows DIRECT at every graded cell {alld};  rows with no DIRECT cell {nod}")
    wide_weak = vc.assign(arc=d.max_arc_deg.to_numpy())
    wide_weak = wide_weak[(wide_weak.arc >= 90) &
                          (wide_weak["PASS(weak)"] > wide_weak.n / 2)]
    print(f"rows with an arc >= 90 deg whose graded cells are MOSTLY PASS(weak): "
          f"{len(wide_weak)}  {sorted(wide_weak.metric_id)}")

    # ---------------- figure -------------------------------------------------
    y = np.arange(n)[::-1]                       # first row of `d` at the top
    BOT_IN = 0.40
    fig_h = ROW_IN * n + BOT_IN + 0.06
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(BODY_W, fig_h), sharey=True,
        gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.045))
    fig.subplots_adjust(left=0.185, right=0.985, top=0.995,
                        bottom=BOT_IN / fig_h)

    # left: widest contiguous azimuthal arc
    # colour by KIND, as fig_arc_width does; the below-floor row is flagged instead
    for i, r in enumerate(d.itertuples(index=False)):
        c = FLOOR_C if r.max_arc_bins < ARC_MIN else KIND_C.get(r.kind, MUTE)
        axL.barh(y[i], r.max_arc_deg, height=0.74, color=c, zorder=3)
        axL.text(r.max_arc_deg + 4, y[i], f"{int(r.max_arc_deg)}",
                 va="center", ha="left", fontsize=FS_VAL, color=MUTE, zorder=4)
    axL.set_xlim(0, 362 + 26)
    axL.set_xticks([0, 90, 180, 270, 360])
    axL.set_xlabel("widest contiguous azimuthal arc (deg)", fontsize=FS_AXLAB)
    # panel letters: Sec V-C cites (b) and Sec VI-A cites (a), so the two sections can
    # share one float without either owning it
    axL.text(0.0, 1.004, "(a)", transform=axL.transAxes, fontsize=6.6,
             color=INK, weight="bold", ha="left", va="bottom")
    axL.set_yticks(y)
    axL.set_yticklabels([f"{short(m)}  el{int(e)}"
                         for m, e in zip(d.metric_id, d.max_arc_el)],
                        fontsize=FS_ROW)
    axL.legend(handles=[Patch(facecolor=KIND_C[k], label=k) for k in KIND_C]
                       + [Patch(facecolor=FLOOR_C,
                                label=f"below the {ARC_MIN}-bin floor")],
               loc="lower right", frameon=False, fontsize=FS_LEG,
               handlelength=1.2, columnspacing=1.2)

    # right: what the graded cells of that row needed
    # plain .iloc, not itertuples: "PASS(weak)" is not a valid attribute name and
    # itertuples silently renames it
    for i in range(len(vc)):
        r = vc.iloc[i]
        stack(axR, y[i], {v: int(r[v]) for v in ORDER}, int(r["n"]),
              h=0.74, annotate=False)
        axR.text(int(r["n"]) + 1.5, y[i], f"{int(r['n'])}", va="center",
                 ha="left", fontsize=FS_VAL, color=MUTE)
    axR.set_xlim(0, vc.n.max() * 1.10)
    axR.set_xlabel("graded cells in that row   (grey number = row total)",
                   fontsize=FS_AXLAB)
    axR.text(0.0, 1.004, "(b)", transform=axR.transAxes, fontsize=6.6,
             color=INK, weight="bold", ha="left", va="bottom")
    axR.legend(handles=[Patch(facecolor=COL[v], label=v) for v in ORDER],
               loc="lower right", frameon=False, fontsize=FS_LEG, ncol=1,
               handlelength=1.2, columnspacing=1.2)

    for ax in (axL, axR):
        ax.set_ylim(-0.8, n - 0.2)
        ax.tick_params(axis="x", labelsize=FS_TICK)
        ax.tick_params(axis="y", length=0, pad=1.5)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

    out = os.path.join(OUT, "fig_row_summary.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved -> {out}   {BODY_W:.2f} x {fig_h:.2f} in")


if __name__ == "__main__":
    main()
