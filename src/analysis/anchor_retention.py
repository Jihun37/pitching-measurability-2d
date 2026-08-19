"""Retention of GT-map cells on the DETECTED-event map, grouped by TEMPORAL ANCHOR.

WHY THIS FILE EXISTS. The four retention figures used to live in a hand-transcribed
table (`anchor_retention_summary.csv`, copied out of `DEPLOYABLE_MAP_HANDOFF.md`
section 0b), and `viz/fig_anchor_retention.py` read them without re-deriving. That
survived exactly as long as the row set did: the 2026-07-29 dedup moved the map from
40 rows to 35 and the transcribed numbers silently became history. This script
derives the classification FROM THE CODE, so the same drift cannot happen again.

THE CLASSIFICATION RULE, stated once. Every map row is placed by which external
event anchors the value it reports:

    none      reads no external event anchor at all
    rel_mer   reads release, or the MER proxy which is an offset off release
    fp_rel    reads BOTH foot plant and release (window observables live here:
              a window maximum over [fp, rel] moves when either end moves)
    fp_only   reads foot plant and not release

Sources, both mechanical, neither a transcription:
    screened rows   `rejected_gt_full_sweep.CANDS[c]` gives the event key, and
                    `fp_target_check.fp_dependent_rows()` gives fp-dependence
                    including the window observables whose event key cannot
                    express that they read the fp end too.
    adopted rows    `adopted_anchor_classes.csv`'s `anchor_type`, written by
                    `analysis/adopted_event_tolerance_sweep.py` from the estimator
                    context each row actually reads.

ONE JUDGEMENT CALL, made explicit. `COG Velo @PKH [O]` is anchored to peak knee
height, not to fp or release, and pkh is a THIRD anchor. It is reported as its own
`pkh` class rather than being folded into one of the four, because folding it into
`fp_only` (which `fp_dependent_rows()` would do, since the estimator falls back to
ctx['fp'] when pkh is absent) is wrong under the GT-event convention where pkh is
never absent -- that fallback never fires, which is why its measured fp-sensitivity
is exactly 0.0000. Read the `pkh` bar as a single-row class.

TWO DENOMINATORS, NOT NESTED:
    any-grade retention = (detected strong + moderate) / (GT strong + moderate)
    strong retention    = (detected strong) / (GT strong)
A row can hold its cells while losing their grade, so strong retention is
sometimes the LOWER of the two and sometimes the higher. They are different
questions and must never be drawn as a subset relationship.

Input:  deploy_map_summary.csv (analysis/deploy_map.py), adopted_anchor_classes.csv
Output: anchor_retention_summary.csv  (the table viz/fig_anchor_retention.py reads)
Run:  conda activate diamond; cd src\\analysis; python anchor_retention.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import pandas as pd
import config
from rejected_gt_full_sweep import CANDS
from fp_target_check import fp_dependent_rows

V = config.OBP_VALIDATION_DIR

ORDER = ["none", "rel_mer", "fp_rel", "fp_only", "pkh"]
LABEL = {"none": "no event", "rel_mer": "release / MER-proxy",
         "fp_rel": "foot plant + release", "fp_only": "foot plant only",
         "pkh": "peak knee height"}


def classify(summary):
    """metric -> one of ORDER, from the code rather than from a transcription."""
    uses_fp = fp_dependent_rows()
    adopted = {}
    p = os.path.join(V, "adopted_anchor_classes.csv")
    if os.path.exists(p):
        a = pd.read_csv(p)
        adopted = dict(zip(a.metric, a.anchor_type.fillna("none")))

    out = {}
    for m, src in zip(summary.metric, summary.source):
        if src == "adopted":
            t = str(adopted.get(m, "")).strip()
            if t == "pkh":
                out[m] = "pkh"
            elif t in ("", "none", "nan"):
                out[m] = "none"
            elif t == "release+fp":
                out[m] = "fp_rel"
            elif t == "fp":
                out[m] = "fp_only"
            else:                                   # "release"
                out[m] = "rel_mer"
            continue
        # screened: event key + code-derived fp dependence
        ev = CANDS.get(m, (None, None))[1]
        fp = m in uses_fp
        if ev in ("rel", "mer"):
            # a window observable carries event key 'rel' but reads the fp end too
            out[m] = "fp_rel" if fp else "rel_mer"
        elif ev == "fp":
            out[m] = "fp_only"
        else:
            out[m] = "none"
    return out


def main():
    d = pd.read_csv(os.path.join(V, "deploy_map_summary.csv"))
    d = d[(d.gt_strong + d.gt_moderate) > 0].copy()      # map rows only
    d["anchor"] = d.metric.map(classify(d))

    rows = []
    for a in ORDER:
        g = d[d.anchor == a]
        if g.empty:
            continue
        gt_s, gt_m = int(g.gt_strong.sum()), int(g.gt_moderate.sum())
        dp_s, dp_m = int(g.dep_strong.sum()), int(g.dep_moderate.sum())
        gt_tot, dp_tot = gt_s + gt_m, dp_s + dp_m
        rows.append({
            "anchor": a, "label": LABEL[a],
            "estimator_rows": len(g), "gt_cells": gt_tot,
            "gt_strong": gt_s, "dep_cells": dp_tot, "dep_strong": dp_s,
            "any_grade_retention": round(dp_tot / gt_tot, 4) if gt_tot else 0.0,
            "strong_retention": round(dp_s / gt_s, 4) if gt_s else 0.0,
            "source": "analysis/anchor_retention.py (re-derived)",
        })
    out = pd.DataFrame(rows)

    tot_rows, tot_cells = int(out.estimator_rows.sum()), int(out.gt_cells.sum())
    print(f"map rows {tot_rows}   GT cells {tot_cells}   "
          f"deployed {int(out.dep_cells.sum())}")
    print(out[["anchor", "estimator_rows", "gt_cells", "dep_cells",
               "any_grade_retention", "strong_retention"]].to_string(index=False))
    print("\nrows per class:")
    for a in ORDER:
        ms = sorted(d.metric[d.anchor == a])
        if ms:
            print(f"  {a:<8} {len(ms):>2}  {', '.join(ms)}")

    p = os.path.join(V, "anchor_retention_summary.csv")
    out.to_csv(p, index=False)
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
