"""Before/after audit for ONE intended estimator change: the search window of
`lead_knee_extension_angular_velo_max`, [fp, rel] -> [fp, rel+12f@360Hz]
(rejected_gt_full_sweep.WINDOW_END_OFFSET_F360, adopted 2026-07-27 by user
decision; evidence in docs/legacy_pre_dedup/EVENT_SYSTEM_HANDOFF_2026-07-27.md 6b).

Same shape as setup_anchor_regression.py: an intended change is only safe once the
UNintended part of the map is proved untouched, cell by cell, so the audit is a
deliverable and not a spot check.

Reports, old vs new:
  1  every OTHER row bit-identical (CCC / r2 / grade / verdict / model / spike)
  2  strong + moderate cell counts for the changed row
  3  best CCC and the cell it sits in
  4  DIRECT / CALIBRATE / PASS(weak) composition
  5  contiguous valid zones per elevation, and isolated (spike) cells
  6  the deployment layer: the same row under DETECTED release + foot plant

Inputs: gate_map.csv (new) and backup_pre_kneewindow_20260727/gate_map.csv (old);
        gate_map_detected.csv + its backup, when the deployment run has been made.

Run:  conda activate diamond; cd src\\analysis; python knee_window_regression.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config
from layer_report import arcs

V = config.OBP_VALIDATION_DIR
BACKUP = os.path.join(V, "backup_pre_kneewindow_20260727")
TARGET = "lead_knee_extension_angular_velo_max"
KEY = ["metric", "source", "az", "el"]
NUM = ["ccc", "r2", "raw_ccc", "raw_mae", "raw_pbsd", "mae", "pbsd",
       "ccc_offset", "ccc_ratio", "ccc_linear"]
CAT = ["grade", "verdict", "model", "gate_pass", "spike", "hatch"]


def zones(g, col="gate_pass"):
    """Contiguous azimuth arcs of passing cells, per elevation."""
    out = []
    for el, sub in g[g[col]].groupby("el"):
        for lo, hi in arcs(sorted(sub.az.astype(int))):
            n = (hi - lo) // 15 + 1
            out.append((int(el), lo % 360, hi % 360, n))
    return sorted(out, key=lambda t: (-t[3], t[0]))


def summarise(g, tag):
    ok = g[g.gate_pass]
    strong = g[g.grade == "strong"]
    cnt = ok.verdict.value_counts()
    best = ok.loc[ok.ccc.idxmax()] if len(ok) else (
        g.loc[g.ccc.idxmax()] if g.ccc.notna().any() else None)
    return dict(
        tag=tag, strong=len(strong), moderate=len(ok) - len(strong),
        gate=len(ok), r2=int((g.r2 >= 0.60).sum()),
        best_ccc=float(best.ccc) if best is not None else np.nan,
        best_cell=f"{int(best.az)}/{int(best.el)}" if best is not None else "-",
        best_r2=float(g.r2.max()),
        best_r2_cell=f"{int(g.loc[g.r2.idxmax()].az)}/{int(g.loc[g.r2.idxmax()].el)}",
        D=int(cnt.get("DIRECT", 0)), C=int(cnt.get("CALIBRATE", 0)),
        W=int(cnt.get("PASS(weak)", 0)), spikes=int(g.spike.sum()))


def compare(new_p, old_p, title):
    if not (os.path.exists(new_p) and os.path.exists(old_p)):
        print(f"\n[{title}] missing input -- skipped\n  new {new_p}\n  old {old_p}")
        return None
    new, old = pd.read_csv(new_p), pd.read_csv(old_p)
    print("\n" + "=" * 100)
    print(f"{title}   old {os.path.basename(old_p)} -> new {os.path.basename(new_p)}")
    print("=" * 100)

    m = old.merge(new, on=KEY, suffixes=("_o", "_n"), how="outer", indicator=True)
    if (m._merge != "both").any():
        print(f"  !! cell sets differ: {m._merge.value_counts().to_dict()}")
    oth = m[(m.metric != TARGET) & (m._merge == "both")]
    print(f"\n[1] UNCHANGED ROWS  ({oth.metric.nunique()} metrics, {len(oth)} cells)")
    worst = 0.0
    for c in NUM:
        if f"{c}_o" not in oth:
            continue
        d = np.nanmax((oth[f"{c}_o"] - oth[f"{c}_n"]).abs())
        worst = max(worst, 0.0 if not np.isfinite(d) else d)
    ndiff = sum(int((oth[f"{c}_o"].astype(str) != oth[f"{c}_n"].astype(str)).sum())
                for c in CAT if f"{c}_o" in oth)
    print(f"    max |delta| over {len(NUM)} numeric columns : {worst:.10f}")
    print(f"    categorical cells differing ({'/'.join(CAT)}) : {ndiff}")
    print("    -> " + ("CLEAN: the change is confined to the target row"
                       if worst == 0.0 and ndiff == 0 else
                       "!! COLLATERAL CHANGE -- investigate before freezing"))

    o = old[old.metric == TARGET]
    n = new[new.metric == TARGET]
    so, sn = summarise(o, "old [fp,rel]"), summarise(n, "new [fp,rel+12]")
    print(f"\n[2-4] {TARGET}")
    hdr = (f"{'':<18}{'strong':>8}{'moder':>7}{'gate':>6}{'r2':>5}"
           f"{'best CCC':>10}{'cell':>9}{'best r2':>9}{'cell':>9}{'D/C/W':>12}"
           f"{'spike':>7}")
    print(hdr); print("-" * len(hdr))
    for s in (so, sn):
        dcw = "{}/{}/{}".format(s["D"], s["C"], s["W"])
        print(f"{s['tag']:<18}{s['strong']:>8}{s['moderate']:>7}{s['gate']:>6}"
              f"{s['r2']:>5}{s['best_ccc']:>10.3f}{s['best_cell']:>9}"
              f"{s['best_r2']:>9.3f}{s['best_r2_cell']:>9}{dcw:>12}"
              f"{s['spikes']:>7}")
    print(f"{'delta':<18}{sn['strong']-so['strong']:>+8}"
          f"{sn['moderate']-so['moderate']:>+7}{sn['gate']-so['gate']:>+6}"
          f"{sn['r2']-so['r2']:>+5}{sn['best_ccc']-so['best_ccc']:>+10.3f}")

    print(f"\n[5] CONTIGUOUS ZONES (gate_pass), az arcs per elevation")
    for lbl, g in (("old", o), ("new", n)):
        z = zones(g)
        txt = "  ".join(f"el{el}: az {lo}-{hi} ({c})" for el, lo, hi, c in z[:8])
        print(f"    {lbl:<4} {len(z)} arc(s)   {txt if z else '(none)'}")
        iso = [f"{int(r.az)}/{int(r.el)}" for r in g[g.spike].itertuples()]
        print(f"         isolated cells: {len(iso)}"
              + (f"  [{', '.join(iso)}]" if iso else ""))
    zs_o, zs_n = zones(o.assign(gate_pass=o.grade == "strong")), \
                 zones(n.assign(gate_pass=n.grade == "strong"))
    for lbl, z in (("old", zs_o), ("new", zs_n)):
        txt = "  ".join(f"el{el}: az {lo}-{hi} ({c})" for el, lo, hi, c in z[:8])
        print(f"    {lbl} STRONG arcs: {len(z)}   {txt if z else '(none)'}")
    return so, sn


def main():
    compare(os.path.join(V, "gate_map.csv"),
            os.path.join(BACKUP, "gate_map.csv"),
            "[GT EVENTS]  the paper layer (OBP landmarks)")
    # The deployment layer had no pre-change table (the --detected mode did not
    # exist), so its "old" side is generated on purpose with --no-window-ext, which
    # reproduces the [fp, rel] window exactly. Both sides are scored with
    # --pop-frozen so they sit on the same 394 pitches as the paper layer.
    compare(os.path.join(V, "gate_map_detected.csv"),
            os.path.join(V, "gate_map_detected_oldwin.csv"),
            "[6] [DETECTED EVENTS]  deployment layer -- our release + foot plant, "
            "per-view")


if __name__ == "__main__":
    main()
