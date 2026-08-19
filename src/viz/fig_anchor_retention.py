"""Act 3 §6.2 — what decides whether a row survives detection is its ANCHOR.

Retention of GT-map cells on the DETECTED-EVENT map, grouped by which event the row
is anchored to, under the ADOPTED FP-specific routing rule scored by pitcher-level
LOPO CV.

⚠ PROVENANCE, REBUILT 2026-07-30. The values live in `anchor_retention_summary.csv`,
which is now COMPUTED by `analysis/anchor_retention.py` from the deploy summary and
the code's own event keys. It used to be transcribed by hand out of
`docs/DEPLOYABLE_MAP_HANDOFF.md` §0b, and that transcription went stale silently
when the 2026-07-29 dedup moved the map from 40 rows to 35. This module reads the
computed table and asserts its totals against the canonical deploy CSV, so neither
file can drift alone. Nothing here is hard-coded; re-run the analysis script first
if the map changes.

⚠ TWO DENOMINATORS, NOT nested:
    any-grade retention = (detected strong + moderate) / (GT strong + moderate)
    strong retention    = (detected strong) / (GT strong)
A row can keep a cell while losing its grade, so strong retention is sometimes the
lower of the two and sometimes the higher (fp-only is the case where it is higher).
The figure draws them as bars vs markers so they cannot be read as a subset.

⚠ `pkh` IS ITS OWN CLASS since the rebuild. The pre-dedup table folded
`COG Velo @PKH [O]` into fp-only, which is what made fp-only read 0.404; with peak
knee height separated out, fp-only is lower. Do not quote 0.404 against the current
map, and do NOT use 35 % / 0.354 either -- that is the older `release_view` baseline.

Run:  conda activate diamond
      cd src\\viz
      python fig_anchor_retention.py
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", ".."):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
from fig_graded_map import INK, MUTE, C_STRONG, C_MOD

SUMM = os.path.join(config.OBP_VALIDATION_DIR, "deploy_map_summary.csv")
OUT = os.path.join(config.ROOT, "data", "outputs", "viz",
                   "fig_anchor_retention.png")

ANCHOR_CSV = os.path.join(config.OBP_VALIDATION_DIR,
                          "anchor_retention_summary.csv")
# only decoration, never a number: the two event-free rows named in the label
NO_EVENT_ROWS = "(Hip-Shoulder Sep, Wrist Speed)"


def load_anchor():
    """(label, rows, GT cells, any-grade retention, strong retention) from the small
    canonical table. Values are the frozen §0b figures; this module never edits them."""
    t = pd.read_csv(ANCHOR_CSV)
    out = []
    for r in t.itertuples(index=False):
        lab = str(r.label)
        if r.anchor == "none":
            lab += "\n" + NO_EVENT_ROWS
        out.append((lab, int(r.estimator_rows), int(r.gt_cells),
                    float(r.any_grade_retention), float(r.strong_retention)))
    return out


ANCHOR = load_anchor()


def main():
    d = pd.read_csv(SUMM)
    gt = int((d.gt_strong + d.gt_moderate).sum())
    kept = int(d.dep_total.sum())
    s, m = int(d.dep_strong.sum()), int(d.dep_moderate.sum())
    rows_sum = sum(a[1] for a in ANCHOR)
    cells_sum = sum(a[2] for a in ANCHOR)
    kept_sum = int(round(sum(a[2] * a[3] for a in ANCHOR)))
    # Cross-file consistency, not value pinning: the anchor table must partition
    # exactly the cells the deploy summary reports, whatever the map currently is.
    # Row/cell TOTALS are asserted; no individual count is written down here.
    onmap = int(((d.gt_strong + d.gt_moderate) > 0).sum())
    assert rows_sum == onmap, (rows_sum, onmap)
    assert cells_sum == gt, (cells_sum, gt)
    assert kept_sum == kept, (kept_sum, kept)
    print(f"CHECK rows {rows_sum}={onmap}  cells {cells_sum}={gt}  "
          f"kept {kept_sum}={kept} ({s} strong + {m} moderate)")

    labs = [a[0] for a in ANCHOR]
    keepf = np.array([a[3] for a in ANCHOR])
    strf = np.array([a[4] for a in ANCHOR])
    y = np.arange(len(ANCHOR))[::-1]

    # Authored at the IEEE two-column width so the type prints at its stated size.
    fig, ax = plt.subplots(figsize=(7.16, 2.95))
    # ⚠ The two series have DIFFERENT denominators (see DENOM_NOTE), so they are NOT
    # nested and must not be drawn as stacked or paired bars -- that would imply the
    # strong series is a subset of the any-grade series. It is why fp-only reads
    # 0.434 strong against 0.404 any-grade. Bars = any-grade, markers = strong.
    ax.barh(y, keepf, height=0.44, color=C_MOD,
            label="any-grade retention  (bars)")
    ax.plot(strf, y, "D", ms=5.5, color=C_STRONG, ls="none", zorder=5,
            label="strong retention  (markers, different denominator)")
    for i, a in enumerate(ANCHOR):
        yy = y[i]
        ax.text(a[3] + 0.014, yy + 0.16, f"{a[3]:.3f}", va="center",
                fontsize=6.4, weight="bold", color=INK)
        ax.text(a[4] + 0.014, yy - 0.18, f"{a[4]:.3f}", va="center",
                fontsize=5.8, color=C_STRONG)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}\n{a[1]} estimator rows · {a[2]} cells"
                        for l, a in zip(labs, ANCHOR)], fontsize=5.8)
    ax.set_xlim(0, 1.14)
    ax.set_xlabel("retention (see the two formulas below)", fontsize=6.6)
    ax.set_title("Among GT-measurable cells, detected-event retention varies "
                 "strongly\nby temporal anchor", fontsize=8.0, weight="bold",
                 color=INK, pad=10)
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(labelsize=5.8)
    fig.text(0.5, 0.022,
             "any-grade retention  =  (detected strong + moderate cells) / "
             "(GT strong + moderate cells)          "
             "strong retention  =  (detected strong cells) / (GT strong cells)\n"
             "The two denominators differ, so the series are not nested and strong "
             "retention can exceed any-grade retention.  "
             "One row per distinct quantity.\n"
             "FP-specific routing rule, pitcher-level LOPO cross-validation.  "
             f"{gt} GT-map cells -> {kept} retained ({s} strong + {m} moderate, "
             f"{kept/gt*100:.1f} %).  n = 394.",
             ha="center", va="bottom", fontsize=5.2, color=MUTE)
    # subplots_adjust reserves the footer band; tight_layout would override it
    fig.subplots_adjust(left=0.245, right=0.985, top=0.965, bottom=0.30)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
