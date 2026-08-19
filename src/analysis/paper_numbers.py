"""Every number the manuscript states, re-derived from the canonical CSVs, printed beside
the value in the pre-nested draft so the prose can be diffed section by section.

Built 2026-08-08 for the switch to NESTED correction-model selection. The `was` column is
the value in `paper/first_draft.tex` as written against the pre-nested map; it is a
REFERENCE FOR DIFFING ONLY and is never an input to anything.

Run:  conda activate diamond; cd src\\analysis; python paper_numbers.py
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
STRONG, MODERATE = 0.80, 0.75
rows_out = []


def R(sec, what, now, was):
    now_s = f"{now}"
    was_s = f"{was}"
    flag = "" if now_s == was_s else "  <-- CHANGED"
    rows_out.append(dict(section=sec, quantity=what, now=now_s, draft=was_s,
                         changed=now_s != was_s))
    print(f"  {what:<52}{now_s:>14}{was_s:>14}{flag}")


def head(t):
    print("\n" + "=" * 96)
    print(f"{t}\n" + "-" * 96)
    print(f"  {'quantity':<52}{'NOW':>14}{'draft':>14}")


def main():
    reg = pd.read_csv(os.path.join(V, "paper_registry.csv"))
    g = pd.read_csv(os.path.join(V, "gate_map.csv"))
    g = g[g.metric.isin(set(reg.metric_id))]
    gr = g[g.grade.isin(["strong", "moderate"])]

    head("SEC V-A  the graded map")
    R("V-A", "graded cells", len(gr), 1500)
    R("V-A", "strong cells", int((g.grade == "strong").sum()), 1100)
    R("V-A", "moderate cells", int((g.grade == "moderate").sum()), 400)
    R("V-A", "retained rows", gr.metric.nunique(), 35)
    R("V-A", "non-retained rows", len(reg) - gr.metric.nunique(), 12)
    R("V-A", "evaluated cells over retained rows", gr.metric.nunique() * 168, 5880)
    sc = reg[reg.retained.astype(bool)]
    R("V-A", "strong-capable rows", int((sc.metric_grade == "strong-capable").sum()), 29)
    R("V-A", "moderate-only rows", int((sc.metric_grade == "moderate-only").sum()), 6)
    per = gr.groupby("metric").size()
    R("V-A", "per-row coverage min", int(per.min()), 4)
    R("V-A", "per-row coverage max", int(per.max()), 148)
    R("V-A", "widest row", per.idxmax(), "Stride Angle [O]")
    R("V-A", "narrowest rows", "; ".join(sorted(per[per == per.min()].index)),
      "pelvis_lateral_tilt_fp; max_elbow_flexion")
    ground = gr[gr.el == 0].metric.nunique()
    R("V-A", "retained rows with a graded cell at el 0", ground, 23)
    R("V-A", "retained rows with none at el 0", gr.metric.nunique() - ground, 12)
    print("  graded / strong by elevation:")
    for el in sorted(g.el.unique()):
        sub = g[g.el == el]
        gg = int(sub.grade.isin(["strong", "moderate"]).sum())
        ss = int((sub.grade == "strong").sum())
        print(f"     el {el:>2}: graded {gg:>4}   strong {ss:>4}/{gg}")

    head("SEC V-B  association vs agreement")
    h = g[(g.r2 >= 0.60) & (g.ccc < MODERATE)]
    R("V-B", "association-without-agreement cells", len(h), 20)
    R("V-B", "  spread over rows", h.metric.nunique(), 16)
    R("V-B", "  raw CCC min", f"{h.raw_ccc.min():.4f}", "-0.168")
    R("V-B", "  raw CCC max", f"{h.raw_ccc.max():.4f}", "0.738")
    R("V-B", "  raw CCC median", f"{h.raw_ccc.median():.4f}", "0.280")
    R("V-B", "  of them below raw CCC 0.20", int((h.raw_ccc < 0.20).sum()), 9)
    R("V-B", "  best OOF CCC min", f"{h.ccc.min():.4f}", "0.734")
    R("V-B", "  best OOF CCC max", f"{h.ccc.max():.4f}", "~0.75")
    R("V-B", "  r2 min", f"{h.r2.min():.4f}", "0.600")
    R("V-B", "  r2 max", f"{h.r2.max():.4f}", "0.623")
    rev = g[(g.grade == "moderate") & (g.r2 < 0.60)]
    R("V-B", "moderate cells with r2 below screen", len(rev), 57)
    R("V-B", "  as % of moderate", f"{100.0 * len(rev) / (g.grade == 'moderate').sum():.1f}", "14.2")
    R("V-B", "  spread over rows", rev.metric.nunique(), 20)
    R("V-B", "  lowest r2", f"{rev.r2.min():.4f}", "0.571")
    R("V-B", "smallest r2 among strong cells", f"{g[g.grade == 'strong'].r2.min():.4f}",
      "0.6463")

    head("SEC V-C  verdicts")
    for v, was in (("CALIBRATE", 853), ("PASS(weak)", 336), ("DIRECT", 311)):
        R("V-C", f"{v} cells", int((gr.verdict == v).sum()), was)
    R("V-C", "DIRECT share %", f"{100.0 * (gr.verdict == 'DIRECT').mean():.1f}", "20.7")
    st = g[g.grade == "strong"]
    md = g[g.grade == "moderate"]
    for v, w1, w2 in (("DIRECT", 282, 29), ("CALIBRATE", 625, 228), ("PASS(weak)", 193, 143)):
        R("V-C", f"  strong / {v}", int((st.verdict == v).sum()), w1)
        R("V-C", f"  moderate / {v}", int((md.verdict == v).sum()), w2)
    print("\n  ** MODEL COLUMN CHANGED MEANING under nested: the model is chosen PER FOLD,")
    print("     so `model` is now the MODAL fold choice. The draft's claims about a cell")
    print("     being 'won by' one model need rewording. Modal counts on graded cells:")
    print(f"     {gr.model.value_counts().to_dict()}   (draft: linear 589 + ratio 264 CALIBRATE,")
    print(f"      318 of 336 PASS(weak) won by offset)")
    cal = gr[gr.verdict == "CALIBRATE"]
    pw = gr[gr.verdict == "PASS(weak)"]
    print(f"     CALIBRATE modal: {cal.model.value_counts().to_dict()}")
    print(f"     PASS(weak) modal: {pw.model.value_counts().to_dict()}")
    sa = gr[gr.metric == "Stride Angle [O]"]
    R("V-C", "stride angle graded cells", len(sa), 148)
    R("V-C", "  all CALIBRATE?", str((sa.verdict == "CALIBRATE").all()), "True")
    R("V-C", "  uncorrected MAE median (deg)", f"{sa.raw_mae.median():.1f}", "86.7")
    R("V-C", "  uncorrected CCC min", f"{sa.raw_ccc.min():.3f}", "-0.979")
    R("V-C", "  r2 max", f"{sa.r2.max():.3f}", "0.981")
    R("V-C", "  corrected MAE median (deg)", f"{sa.mae.median():.2f}", "0.61")
    nod = sorted(set(gr.metric) - set(gr[gr.verdict == "DIRECT"].metric))
    R("V-C", "rows with no DIRECT cell anywhere", len(nod), 10)

    head("SEC V-D  the twelve non-retained rows")
    nr = g[~g.metric.isin(gr.metric.unique())]
    R("V-D", "non-retained rows", nr.metric.nunique(), 12)
    R("V-D", "their evaluated cells", len(nr), 2016)
    R("V-D", "highest OOF CCC among them", f"{nr.ccc.max():.4f}", "0.6802")
    R("V-D", "highest r2 among them", f"{nr.r2.max():.4f}", "0.4709")
    b = nr.loc[nr.ccc.idxmax()]
    R("V-D", "  at row / az / el", f"{b.metric} {int(b.az)}/{int(b.el)}",
      "max_torso_rotational_velo 30/75")
    best_per = nr.groupby("metric").ccc.max()
    R("V-D", "median of their best CCC", f"{best_per.median():.3f}", "0.335")
    R("V-D", "lowest of their best CCC", f"{best_per.min():.3f}", "0.154")

    head("SEC VI-A  azimuthal arcs")
    arcs = pd.read_csv(os.path.join(V, "continuity_arcs.csv"))
    crow = pd.read_csv(os.path.join(V, "continuity_rows.csv"))
    R("VI-A", "contiguous arcs", len(arcs), 242)
    bincol = "bins" if "bins" in arcs.columns else arcs.columns[-1]
    R("VI-A", "mean arc length (bins)", f"{arcs[bincol].mean():.2f}", "6.2")
    R("VI-A", "mean arc length (deg)", f"{15 * arcs[bincol].mean():.0f}", "93")
    R("VI-A", "median arc length (bins)", f"{arcs[bincol].median():.0f}", "5")
    R("VI-A", "single-bin arcs", int((arcs[bincol] == 1).sum()), 19)
    R("VI-A", "isolated cells (spike)", int(gr.spike.sum()), 2)
    wcol = "max_arc_bins" if "max_arc_bins" in crow.columns else None
    if wcol:
        R("VI-A", "rows with widest arc >= 3 bins", int((crow[wcol] >= 3).sum()), 34)
        R("VI-A", "rows azimuth-independent (24 bins)", int((crow[wcol] == 24).sum()), 4)
        R("VI-A", "median widest arc, all retained (bins)",
          f"{crow[wcol].median():.0f}", "-")

    head("SEC VI-C  temporal tolerance")
    et = pd.read_csv(os.path.join(V, "event_tolerance_map.csv"))
    print(f"  columns: {list(et.columns)}")
    print(et.head(3).to_string(index=False))

    head("SEC VI-D  foot-plant anchor (Table III)")
    fp = pd.read_csv(os.path.join(V, "fp_target_rows.csv"))
    R("VI-D", "rows in the table", len(fp), 17)
    R("VI-D", "cells graded under either anchor", int(fp.cells.sum()), 629)
    R("VI-D", "graded under fp_100", int(fp.graded_fp100.sum()), 577)
    R("VI-D", "graded under fp_10", int(fp.graded_fp10.sum()), 471)
    R("VI-D", "prefers fp_100", int(fp.prefers_fp100.sum()), 474)
    R("VI-D", "prefers fp_10", int(fp.prefers_fp10.sum()), 155)
    R("VI-D", "ties", int(fp.ties.sum()), 0)
    R("VI-D", "net delta", int(fp.delta.sum()), 106)
    R("VI-D", "rows preferring fp_10 (delta<0)", int((fp.delta < 0).sum()), 4)
    print("\n  TABLE III, new values:")
    print(fp.sort_values("delta", ascending=False).to_string(index=False))

    head("SEC VI-E  pooled vs within-pitcher")
    bc = pd.read_csv(os.path.join(V, "accuracy_bestcell_gt_clean.csv"))
    w = bc.dropna(subset=["between_r2", "within_r2"])
    ident = w[(w.pooled_r2 > 0.999) & (w.within_r2 > 0.999)]
    nw = w[~w.index.isin(ident.index)]
    R("VI-E", "non-identity rows", len(nw), 34)
    R("VI-E", "median pooled r2", f"{nw.pooled_r2.median():.4f}", "0.792")
    R("VI-E", "median between r2", f"{nw.between_r2.median():.4f}", "0.805")
    R("VI-E", "median within r2", f"{nw.within_r2.median():.4f}", "0.585")
    R("VI-E", "rows with within r2 >= 0.60", int((nw.within_r2 >= 0.60).sum()), 17)
    R("VI-E", "rows losing >= 0.20 vs pooled",
      int(((nw.pooled_r2 - nw.within_r2) >= 0.20).sum()), 13)
    R("VI-E", "rows where between > within", int((nw.between_r2 > nw.within_r2).sum()), 33)

    out = pd.DataFrame(rows_out)
    out.to_csv(os.path.join(V, "paper_numbers.csv"), index=False)
    ch = out[out.changed]
    print("\n" + "=" * 96)
    print(f"CHANGED vs the draft: {len(ch)} of {len(out)} tracked quantities")
    print("=" * 96)
    for _, r in ch.iterrows():
        print(f"  [{r.section}] {r.quantity:<50} {r.draft:>14} -> {r.now}")
    print("\nwrote paper_numbers.csv")


if __name__ == "__main__":
    main()
