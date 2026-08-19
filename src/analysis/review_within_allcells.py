"""A6 -- pooled / between / within-pitcher agreement over ALL graded cells (audit ST6).

WHY. Sec VI-E reads each row at the cell where its own CCC is highest ON THE SAME SAMPLE,
so those figures carry a selection optimism the manuscript already declares. The cheap way
to remove the objection rather than caveat it is to repeat the decomposition over every
graded cell of every retained row and report the two side by side. `accuracy_map` already
writes within_r2 / between_r2 / truth_icc per cell, so this needs no refitting.

Output: review_within_allcells.csv  (per retained row)
Run:  conda activate diamond; cd src\\analysis; python review_within_allcells.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", ".."):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config

V = config.OBP_VALIDATION_DIR


def main():
    reg = pd.read_csv(os.path.join(V, "paper_registry.csv"))
    gm = pd.read_csv(os.path.join(V, "gate_map.csv"))
    am = pd.read_csv(os.path.join(V, "accuracy_map_gt_clean.csv"))
    bc = pd.read_csv(os.path.join(V, "accuracy_bestcell_gt_clean.csv"))

    rows = set(reg.metric_id)
    g = gm[gm.metric.isin(rows) & gm.grade.isin(["strong", "moderate"])]
    d = g[["metric", "az", "el", "grade", "ccc"]].merge(
        am[["metric", "az", "el", "r2", "within_r2", "between_r2", "truth_icc"]],
        on=["metric", "az", "el"], how="left")
    assert d.within_r2.notna().all(), "a graded cell has no within_r2"
    print(f"graded cells {len(d)} over {d.metric.nunique()} retained rows")

    # the identity cell: the 2D coronal arm-slot definition coincides with its 3D truth
    ident = d[(d.r2 > 0.999) & (d.within_r2 > 0.999)]
    d = d.drop(ident.index)
    print(f"dropped {len(ident)} identity cells "
          f"({sorted(ident.metric.unique())})")

    print("\n=== all graded cells vs each row's best cell ===")
    bcs = bc.dropna(subset=["within_r2"])
    bcs = bcs[(bcs.pooled_r2 <= 0.999) | (bcs.within_r2 <= 0.999)]
    print(f"  {'':<26}{'all graded':>14}{'best cell':>14}")
    print(f"  {'median pooled r2':<26}{d.r2.median():>14.4f}{bcs.pooled_r2.median():>14.4f}")
    print(f"  {'median between r2':<26}{d.between_r2.median():>14.4f}"
          f"{bcs.between_r2.median():>14.4f}")
    print(f"  {'median within r2':<26}{d.within_r2.median():>14.4f}"
          f"{bcs.within_r2.median():>14.4f}")
    print(f"  {'share within >= 0.60':<26}{(d.within_r2 >= 0.60).mean():>14.3f}"
          f"{(bcs.within_r2 >= 0.60).mean():>14.3f}")
    print(f"  {'share between > within':<26}{(d.between_r2 > d.within_r2).mean():>14.3f}"
          f"{(bcs.between_r2 > bcs.within_r2).mean():>14.3f}")

    per = d.groupby("metric").agg(
        graded_cells=("within_r2", "size"),
        within_med=("within_r2", "median"),
        within_min=("within_r2", "min"),
        within_max=("within_r2", "max"),
        pooled_med=("r2", "median"),
        between_med=("between_r2", "median"),
        icc_med=("truth_icc", "median")).reset_index()
    per = per.merge(bcs[["metric", "within_r2", "pooled_r2"]].rename(
        columns={"within_r2": "within_bestcell", "pooled_r2": "pooled_bestcell"}),
        on="metric", how="left")
    per["best_minus_median"] = per.within_bestcell - per.within_med
    per = per.sort_values("best_minus_median", ascending=False)

    print("\n=== selection optimism per row: best-cell within r2 minus its median ===")
    bm = per.best_minus_median.dropna()
    print(f"  median {bm.median():+.4f}   mean {bm.mean():+.4f}   "
          f"p5 {bm.quantile(.05):+.4f}   p95 {bm.quantile(.95):+.4f}")
    print(f"  rows where the best cell OVERSTATES within r2: {int((bm > 0).sum())} of {len(bm)}")
    print(f"  rows where it understates:                     {int((bm < 0).sum())}")
    print("\n  largest overstatements:")
    print(per.head(6)[["metric", "graded_cells", "within_bestcell", "within_med",
                       "best_minus_median"]].to_string(index=False))
    print("\n  largest understatements:")
    print(per.tail(4)[["metric", "graded_cells", "within_bestcell", "within_med",
                       "best_minus_median"]].to_string(index=False))

    n60 = int((per.within_med >= 0.60).sum())
    print(f"\n  rows whose MEDIAN graded cell keeps within r2 >= 0.60: {n60} of {len(per)}"
          f"   (draft, at best cells: 17 of 34)")

    per.to_csv(os.path.join(V, "review_within_allcells.csv"), index=False)
    print("\nwrote review_within_allcells.csv")


if __name__ == "__main__":
    main()
