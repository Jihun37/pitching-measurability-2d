"""Review-round sensitivity analyses A1-A3 (2026-08-07).

Answers three points raised in the statistical audit, ALL by post-processing the frozen
`gate_map.csv`. Nothing here refits a correction, rescores a cell or rebuilds the map, so
the freeze is untouched and no canonical file is overwritten.

  A1  MODEL-SELECTION OPTIMISM (audit item ST1).
      `gate_map.score_cell` fits offset/ratio/linear leave-one-PITCHER-out, then picks the
      winner by the argmax of the out-of-fold CCC computed over ALL pitchers -- the
      held-out pitcher's own out-of-fold prediction included. Parameter fitting is
      pitcher-blind; MODEL SELECTION IS NOT.
      A PRESPECIFIED single model has zero selection by construction, so
          (selected map) - (best prespecified map)
      is an EMPIRICAL BOUND on the whole selection effect. No nesting required.
      Escalation rule fixed in docs/REVIEW_TRIAGE_20260807.md before this was run:
      escalate to nested validation only if the best prespecified map loses > 75 graded
      cells (5%) OR moves ANY row across the retained / non-retained line.

  A2  CONTOUR SENSITIVITY (audit item ST5). Retained rows and graded cells as the CCC
      contour sweeps 0.70 / 0.75 / 0.80 / 0.85 / 0.90. Published criteria disagree by
      about this much, so the map's dependence on the choice has to be shown.

  A3  FINITE-RETURN COVERAGE (audit item ST8). If an estimator declines to return a value
      the pair is dropped, so a cell could in principle be graded because its hard cases
      went missing. Reports the sample-size distribution AMONG GRADED CELLS specifically.

Row set: the canonical 47 of `paper_registry.csv`. NEVER re-derived from CANDS or
adopted_rows() -- those still carry rows the 2026-07-29 dedup removed.

Output: review_sensitivity_A1.csv / _A2.csv / _A3.csv  (new files, nothing overwritten)
Run:  conda activate diamond; cd src\\analysis; python review_sensitivity.py
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
STRONG, MODERATE = 0.80, 0.75
MODELS = ("offset", "ratio", "linear")
CONTOURS = (0.70, 0.75, 0.80, 0.85, 0.90)

# escalation rule, fixed before the run
ESCALATE_CELL_LOSS = 75


def grade_counts(df, ccc_col):
    """Graded / strong / moderate cells and retained rows under one CCC column."""
    c = df[ccc_col]
    strong = c >= STRONG
    moderate = (c >= MODERATE) & (c < STRONG)
    graded = c >= MODERATE
    retained = sorted(df.loc[graded, "metric"].unique())
    return dict(graded=int(graded.sum()), strong=int(strong.sum()),
                moderate=int(moderate.sum()), n_retained=len(retained)), set(retained)


def main():
    reg = pd.read_csv(os.path.join(VALID, "paper_registry.csv"))
    assert len(reg) == 42, f"registry is not 42 rows: {len(reg)}"
    rows = set(reg.metric_id)

    g = pd.read_csv(os.path.join(VALID, "gate_map.csv"))
    g = g[g.metric.isin(rows)].copy()
    assert set(g.metric) == rows, "gate_map does not cover every registry row"
    per = g.groupby("metric").size()
    assert (per == 168).all(), f"not 168 cells per row: {per[per != 168].to_dict()}"
    assert len(g) == 42 * 168 == 7056, f"expected 7056 cells, got {len(g)}"
    print(f"cells {len(g)}  rows {g.metric.nunique()}\n")

    # ---------------------------------------------------------------- A1
    sel, sel_rows = grade_counts(g, "ccc")
    print("A1  MODEL-SELECTION OPTIMISM")
    print(f"  {'protocol':<22}{'graded':>8}{'strong':>8}{'moder':>8}{'rows':>7}"
          f"{'d_graded':>10}{'d_rows':>8}")
    print(f"  {'selected (published)':<22}{sel['graded']:>8}{sel['strong']:>8}"
          f"{sel['moderate']:>8}{sel['n_retained']:>7}{'--':>10}{'--':>8}")

    recs = [dict(protocol="selected", **sel, d_graded=0, d_rows=0, rows_lost="")]
    best = None
    for mo in MODELS:
        st, st_rows = grade_counts(g, f"ccc_{mo}")
        lost = sorted(sel_rows - st_rows)
        gained = sorted(st_rows - sel_rows)
        recs.append(dict(protocol=f"prespecified {mo}", **st,
                         d_graded=st["graded"] - sel["graded"],
                         d_rows=st["n_retained"] - sel["n_retained"],
                         rows_lost="; ".join(lost), rows_gained="; ".join(gained)))
        print(f"  {'prespecified ' + mo:<22}{st['graded']:>8}{st['strong']:>8}"
              f"{st['moderate']:>8}{st['n_retained']:>7}"
              f"{st['graded'] - sel['graded']:>10}{st['n_retained'] - sel['n_retained']:>8}")
        if best is None or st["graded"] > best[1]["graded"]:
            best = (mo, st, st_rows)

    bmo, bst, brows = best
    loss = sel["graded"] - bst["graded"]
    moved = sorted(sel_rows ^ brows)
    print(f"\n  best prespecified model : {bmo}")
    print(f"  selection bound         : {loss} graded cells "
          f"({100.0 * loss / sel['graded']:.1f}% of {sel['graded']})")
    print(f"  rows crossing retained  : {len(moved)}  {moved if moved else ''}")
    escalate = loss > ESCALATE_CELL_LOSS or len(moved) > 0
    print(f"  RULE  > {ESCALATE_CELL_LOSS} cells or any row moved -> nested validation")
    print(f"  VERDICT: {'ESCALATE to nested' if escalate else 'BOUND HOLDS - freeze stands'}\n")

    # how often does the selected winner actually beat the others?
    spread = (g[[f"ccc_{m}" for m in MODELS]].max(axis=1)
              - g[[f"ccc_{m}" for m in MODELS]].min(axis=1))
    graded_sel = g[g.ccc >= MODERATE]
    spread_g = spread[graded_sel.index]
    print(f"  across-model CCC spread, graded cells: median {spread_g.median():.4f}, "
          f"p90 {spread_g.quantile(0.90):.4f}, max {spread_g.max():.4f}")
    print(f"  winning model, graded cells: "
          f"{graded_sel.model.value_counts().to_dict()}\n")
    pd.DataFrame(recs).to_csv(os.path.join(VALID, "review_sensitivity_A1.csv"),
                              index=False)

    # ---------------------------------------------------------------- A2
    print("A2  CONTOUR SENSITIVITY")
    print(f"  {'contour':>9}{'cells':>8}{'rows':>7}{'d_cells':>10}{'d_rows':>8}")
    a2 = []
    for t in CONTOURS:
        m = g.ccc >= t
        nrow = g.loc[m, "metric"].nunique()
        a2.append(dict(contour=t, graded_cells=int(m.sum()), retained_rows=int(nrow)))
        print(f"  {t:>9.2f}{int(m.sum()):>8}{nrow:>7}"
              f"{int(m.sum()) - sel['graded']:>10}{nrow - sel['n_retained']:>8}")
    print(f"  (published map = two contours: {sel['graded']} graded at >= {MODERATE}, "
          f"{sel['strong']} strong at >= {STRONG})\n")
    pd.DataFrame(a2).to_csv(os.path.join(VALID, "review_sensitivity_A2.csv"), index=False)

    # ---------------------------------------------------------------- A3
    print("A3  FINITE-RETURN COVERAGE")
    gr = g[g.ccc >= MODERATE]
    full = int((gr.n == 394).sum())
    print(f"  graded cells                : {len(gr)}")
    print(f"  at full n=394               : {full} ({100.0 * full / len(gr):.1f}%)")
    print(f"  n range                     : {int(gr.n.min())} .. {int(gr.n.max())}")
    print(f"  n percentiles p1/p5/p50     : {gr.n.quantile(0.01):.0f} / "
          f"{gr.n.quantile(0.05):.0f} / {gr.n.quantile(0.50):.0f}")
    print(f"  pitchers range              : {int(gr.n_pitcher.min())} .. "
          f"{int(gr.n_pitcher.max())}")
    r_nc = np.corrcoef(gr.n, gr.ccc)[0, 1]
    print(f"  corr(n, CCC) over graded    : {r_nc:+.4f}  "
          f"(positive => missingness does NOT inflate agreement)")
    sus = gr[gr.n < 380].sort_values("n")
    print(f"  graded cells with n < 380   : {len(sus)}")
    if len(sus):
        print(sus[["metric", "az", "el", "n", "n_pitcher", "ccc", "grade"]]
              .head(15).to_string(index=False))
    # the same view over ALL cells, for contrast
    print(f"\n  all evaluated cells: n range {int(g.n.min())} .. {int(g.n.max())}, "
          f"{100.0 * (g.n == 394).mean():.1f}% at full n")
    gr.assign(low_n=gr.n < 380).to_csv(
        os.path.join(VALID, "review_sensitivity_A3.csv"), index=False)

    print("\nwrote review_sensitivity_A1.csv / _A2.csv / _A3.csv")


if __name__ == "__main__":
    main()
