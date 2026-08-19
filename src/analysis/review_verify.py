"""Review verification queue V1-V5 (2026-08-07). Read-only.

Re-derives, from the canonical CSVs, numbers the draft states in prose. Every one of
these is quoted in the manuscript; none is re-quoted here from a document.

  V1  the 20 association-without-agreement cells (audit item C2). The draft says their
      best out-of-fold CCC "runs from 0.734 to just below the floor", which cannot be a
      range -- 0.734 is already below the 0.75 floor. Recover the true min and max.
  V2  the foot-plant anchor rows that prefer fp_10 (Table III's four negative rows).
  V3  the kinetics figures: max, median, counts over/under thresholds.
      ⚠ the 0.597 here is the kinetics maximum. The 0.597 recorded in memory for the
      event-error `p2f` bug is a DIFFERENT quantity in a DIFFERENT file that happens to
      carry the same digits. Do not conflate them.
  V5  whether lead knee angle at release really is the ONLY row better within-pitcher
      than pooled.

Run:  conda activate diamond; cd src\\analysis; python review_verify.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", ".."):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config

VALID = config.OBP_VALIDATION_DIR
MODERATE, STRONG = 0.75, 0.80


def head(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def main():
    reg = pd.read_csv(os.path.join(VALID, "paper_registry.csv"))
    rows = set(reg.metric_id)
    g = pd.read_csv(os.path.join(VALID, "gate_map.csv"))
    g = g[g.metric.isin(rows)]

    # ------------------------------------------------------------------ V1
    head("V1  association without agreement (draft: 20 cells, 16 rows)")
    h = g[(g.r2 >= 0.60) & (g.ccc < MODERATE)]
    print(f"  cells {len(h)}   rows {h.metric.nunique()}")
    print(f"  raw CCC      min {h.raw_ccc.min():+.4f}  max {h.raw_ccc.max():+.4f}  "
          f"median {h.raw_ccc.median():+.4f}   below 0.20: {(h.raw_ccc < 0.20).sum()}")
    print(f"  best OOF CCC min {h.ccc.min():+.4f}  max {h.ccc.max():+.4f}  "
          f"median {h.ccc.median():+.4f}")
    print(f"  r2           min {h.r2.min():.4f}  max {h.r2.max():.4f}")
    print("\n  DRAFT SAYS  'best out-of-fold CCC runs from 0.734 to just below the floor'")
    print(f"  ACTUAL      min {h.ccc.min():.4f}, max {h.ccc.max():.4f} "
          f"-- the sentence must state the MAX, not a range from it")
    print(f"  draft also says raw CCC -0.168..0.738 med 0.280, 9 of 20 below 0.20, "
          f"r2 0.600..0.623")
    h.sort_values("ccc", ascending=False)[
        ["metric", "az", "el", "r2", "raw_ccc", "ccc", "model"]].to_csv(
        os.path.join(VALID, "review_verify_V1_hatch_cells.csv"), index=False)

    # the reverse direction: graded with r2 below the screen
    rev = g[(g.ccc >= MODERATE) & (g.ccc < STRONG) & (g.r2 < 0.60)]
    print(f"\n  reverse direction: moderate cells with r2 < 0.60 -> {len(rev)} cells, "
          f"{rev.metric.nunique()} rows, lowest r2 {rev.r2.min():.4f}   "
          f"(draft: 57 cells, 20 rows, lowest 0.571)")
    strong = g[g.ccc >= STRONG]
    print(f"  smallest r2 among strong cells: {strong.r2.min():.4f}  "
          f"(draft: 0.6463; CCC<=|r| guarantees >= 0.64)")

    # ------------------------------------------------------------------ V2
    head("V2  foot-plant anchor preference")
    fp = None
    for cand in ("fp_target_rows.csv", "fp_target_check.csv", "fp_target_summary.csv"):
        p = os.path.join(VALID, cand)
        if os.path.exists(p):
            fp = pd.read_csv(p); print(f"  source: {cand}  cols {list(fp.columns)}")
            break
    if fp is None:
        cands = [f for f in os.listdir(VALID) if f.startswith("fp_target")]
        print(f"  !! no fp_target_* summary found. present: {cands}")
    else:
        print(fp.to_string(index=False))

    # ------------------------------------------------------------------ V3
    head("V3  kinetics inference (draft: 0/34 at 0.60, max 0.597, median 0.226)")
    k = pd.read_csv(os.path.join(VALID, "inference_trajectory.csv"))
    print(f"  columns: {list(k.columns)}")
    sc = [c for c in k.columns if c.lower() in ("r2", "best_r2", "score", "cv_r2")]
    if sc:
        c = sc[0]
        print(f"  targets {len(k)}   using column '{c}'")
        print(f"  max {k[c].max():.4f}   median {k[c].median():.4f}   "
              f"min {k[c].min():.4f}")
        print(f"  >= 0.60 : {(k[c] >= 0.60).sum()}    > 0.50 : {(k[c] > 0.50).sum()}"
              f"    > 0.40 : {(k[c] > 0.40).sum()}")
        top = k.nlargest(5, c)
        print(top.to_string(index=False))
    else:
        print(k.head(8).to_string(index=False))

    # ------------------------------------------------------------------ V5
    head("V5  within- vs between-pitcher (draft: between > within in 33 of 34)")
    wp = None
    for cand in ("within_pitcher_agreement_gt.csv", "within_pitcher_agreement.csv",
                 "accuracy_bestcell_gt_clean.csv"):
        p = os.path.join(VALID, cand)
        if os.path.exists(p):
            wp = pd.read_csv(p); print(f"  source: {cand}"); break
    if wp is None:
        print("  !! no within-pitcher output found; run within_pitcher_agreement.py")
    else:
        print(f"  columns: {list(wp.columns)}")
        need = {"pooled_r2", "between_r2", "within_r2"}
        if need <= set(wp.columns):
            w = wp.dropna(subset=["between_r2", "within_r2"])
            better = w[w.within_r2 > w.between_r2]
            print(f"  rows {len(w)}   median pooled {w.pooled_r2.median():.4f}  "
                  f"between {w.between_r2.median():.4f}  within {w.within_r2.median():.4f}")
            print(f"  within >= 0.60 : {(w.within_r2 >= 0.60).sum()} of {len(w)}")
            print(f"  losing >= 0.20 vs pooled : "
                  f"{((w.pooled_r2 - w.within_r2) >= 0.20).sum()}")
            print(f"  WITHIN BEATS BETWEEN in {len(better)} row(s):")
            lab = [c for c in ("metric", "paper_name", "row") if c in w.columns]
            cols = lab[:1] + ["pooled_r2", "between_r2", "within_r2"]
            print(better[cols].to_string(index=False))
        else:
            print(f"  (columns do not include {need}; head follows)")
            print(wp.head(5).to_string(index=False))

    print("\ndone -- wrote review_verify_V1_hatch_cells.csv")


if __name__ == "__main__":
    main()
