"""Two-layer measurability report: the r2 SCREENING layer and the LOCO-CCC GATE
layer, on the same population, cell for cell.

Both layers are rebuilt on the gt_clean population (394 pitches). That matters:
the old screen in rejected_gt_full_sweep.csv ran on all 403 pitches including the
8 broken-landmark ones, and for line-orientation observables those 2 % of pitches
decide the verdict -- pelvis_rotation_fp screened 0.573 / 0 cells on 403 and
0.680 / 66 cells on the paper population. Layer 1 numbers here supersede that CSV.

Outputs (OBP_VALIDATION_DIR):
  screening_map_gt_clean.csv   metric x az x el -> r2, tier            (layer 1)
  layer_summary.csv            one row per metric, both layers joined

Rules (standing instruction, 2026-07-27): redundancy and prior adoption are NEVER
pass/fail criteria. They are carried as annotation columns (`adopted`,
`colin_with`, `colin`) so they can be read and argued about, never to exclude a
cell. The final measurable-viewpoint claim is the GATE cells, not the r2 cells.

Run:  conda activate diamond; cd src\\analysis; python layer_report.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config
from angle_map_2d import adopted_rows, gt_only_rows

GATE = os.path.join(config.OBP_VALIDATION_DIR, "gate_map.csv")
OLD_SCREEN = os.path.join(config.OBP_VALIDATION_DIR, "rejected_gt_full_sweep.csv")
OFFICIAL = os.path.join(config.OBP_VALIDATION_DIR, "angle_zone_sweep_gt_clean.csv")
SCREEN_OUT = os.path.join(config.OBP_VALIDATION_DIR, "screening_map_gt_clean.csv")
ADOPTED_PAIRS = os.path.join(config.OBP_VALIDATION_DIR, "angle_zone_pairs_gt.csv.gz")
SCREEN_PAIRS = os.path.join(config.OBP_VALIDATION_DIR, "rejected_gt_pairs.csv.gz")
SUMM_OUT = os.path.join(config.OBP_VALIDATION_DIR, "layer_summary.csv")
FLOOR_R2, RELIABLE_R2 = 0.60, 0.80
AZ_STEP = 15


def arcs(azs):
    """Contiguous azimuth runs (wrapping at 360) from a set of azimuths."""
    if not azs:
        return []
    s = sorted(azs)
    runs = [[s[0], s[0]]]
    for a in s[1:]:
        if a - runs[-1][1] == AZ_STEP:
            runs[-1][1] = a
        else:
            runs.append([a, a])
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == 360 - AZ_STEP:
        runs[0][0] = runs[-1][0] - 360      # wrap the last run into the first
        runs.pop()
    return runs


def zone_text(g):
    """Largest contiguous pass zone, as 'az a-b @ el e (n cells)'."""
    best = None
    for el, sub in g[g.gate_pass].groupby("el"):
        for lo, hi in arcs(sorted(sub.az.astype(int))):
            n = (hi - lo) // AZ_STEP + 1
            if best is None or n > best[0]:
                best = (n, lo % 360, hi % 360, int(el))
    if best is None:
        return "", 0
    n, lo, hi, el = best
    return (f"az {lo}-{hi} @ el {el}" if n > 1 else f"az {lo} @ el {el}"), n


def main():
    if not os.path.exists(GATE):
        sys.exit("run gate_map.py first")
    g = pd.read_csv(GATE)

    # ---- annotations (never filters) -------------------------------------
    ad_truth = {t for _, _, t in adopted_rows() + gt_only_rows() if isinstance(t, str)}
    ad_label = {l.strip() for l, _, _ in adopted_rows() + gt_only_rows()}
    # Some screened columns target the SAME truth as an adopted label. Two very
    # different things look alike in the summary, so the kind is decided by
    # comparing the actual per-cell estimates, not by assuming:
    #   same-estimator  the two code paths compute the same number -> a pipeline
    #                   cross-check, and one quantity counted twice
    #   alt-estimator   same truth, a genuinely different 2D observable -> two real
    #                   measurements of one quantity, with different quality
    # All rows stay in the map either way; only the label differs.
    same_as = {t: l.strip() for l, _, t in adopted_rows() + gt_only_rows()
               if isinstance(t, str)}
    dup_kind = {}
    if os.path.exists(ADOPTED_PAIRS) and os.path.exists(SCREEN_PAIRS):
        pa = pd.read_csv(ADOPTED_PAIRS); pb = pd.read_csv(SCREEN_PAIRS)
        for col, lab in same_as.items():
            x, y = pa[pa.metric == lab], pb[pb.metric == col]
            if x.empty or y.empty:
                continue
            m = x.merge(y, on=["az", "el", "session_pitch"], suffixes=("_a", "_b"))
            d = (m.est_a - m.est_b).abs()
            d = d[np.isfinite(d)]
            sd = float(np.nanstd(m.est_a))
            if not len(d) or sd <= 0:
                continue
            # the dumps are written at %.6g, so equality means "within rounding"
            dup_kind[col] = ("same-estimator" if d.max() / sd < 1e-4
                             else "alt-estimator")
    colin = {}
    if os.path.exists(OLD_SCREEN):
        o = pd.read_csv(OLD_SCREEN).set_index("column")
        colin = {c: (float(o.loc[c, "adopted_corr"]), str(o.loc[c, "colinear_with"]),
                     int(o.loc[c, "cells_usable"])) for c in o.index}

    # ---- layer 1: screening map on gt_clean ------------------------------
    scr = g[["metric", "source", "az", "el", "r2"]].copy()
    scr["tier"] = np.where(scr.r2 >= RELIABLE_R2, "reliable",
                           np.where(scr.r2 >= FLOOR_R2, "usable", "invalid"))
    scr.to_csv(SCREEN_OUT, index=False, float_format="%.6g")
    print(f"layer 1 saved -> {SCREEN_OUT}  ({len(scr):,} cells)")

    if os.path.exists(OFFICIAL):
        off = pd.read_csv(OFFICIAL)
        chk = g[g.source == "adopted"].merge(off, on=["metric", "az", "el"],
                                             suffixes=("_new", "_off"))
        d = (chk.r2_new - chk.r2_off).abs()
        print(f"  sanity vs angle_zone_sweep_gt_clean.csv: {len(chk)} adopted cells, "
              f"max |diff| = {d.max():.9f}")

    # ---- per-metric summary ---------------------------------------------
    rows = []
    for (metric, src), sub in g.groupby(["metric", "source"]):
        ok = sub[sub.gate_pass]
        br = sub.loc[sub.r2.idxmax()] if sub.r2.notna().any() else None
        bc = ok.loc[ok.ccc.idxmax()] if len(ok) else (
            sub.loc[sub.ccc.idxmax()] if sub.ccc.notna().any() else None)
        zt, zn = zone_text(sub)
        strong = sub[sub.grade == "strong"]
        zs, _ = zone_text(sub.assign(gate_pass=sub.grade == "strong"))
        cnt = ok.verdict.value_counts()
        # metric grade: strong-capable if any cell reaches 0.80, moderate-only if it
        # only ever reaches the 0.75 band, else off the map.
        mgrade = ("strong-capable" if len(strong) else
                  ("moderate-only" if len(ok) else "below"))
        cr, cw, old = colin.get(metric, (np.nan, "", -1))
        rows.append(dict(
            metric=metric, source=src,
            r2_cells=int((sub.r2 >= FLOOR_R2).sum()),
            r2_reliable=int((sub.r2 >= RELIABLE_R2).sum()),
            metric_grade=mgrade,
            map_cells=len(ok), strong_cells=len(strong),
            moderate_cells=len(ok) - len(strong),
            strong_zone=zs, hatch_cells=int(sub.hatch.sum()),
            gate_cells=len(ok),
            best_r2=None if br is None else round(float(br.r2), 4),
            best_r2_view="" if br is None else f"{int(br.az)}/{int(br.el)}",
            best_ccc=None if bc is None else round(float(bc.ccc), 4),
            best_ccc_view="" if bc is None else f"{int(bc.az)}/{int(bc.el)}",
            direct=int(cnt.get("DIRECT", 0)), calibrate=int(cnt.get("CALIBRATE", 0)),
            pass_weak=int(cnt.get("PASS(weak)", 0)),
            zone=zt, zone_cells=zn, spikes=int(sub.spike.sum()),
            adopted=(metric in ad_truth or metric.strip() in ad_label),
            pairs_with=(same_as.get(metric, "") if src == "screened" else ""),
            pair_kind=(dup_kind.get(metric, "") if src == "screened" else ""),
            colin=None if not np.isfinite(cr) else round(cr, 3), colin_with=cw,
            old403_cells=old))
    s = pd.DataFrame(rows).sort_values(["gate_cells", "r2_cells"], ascending=False)
    s.to_csv(SUMM_OUT, index=False)
    print(f"layer summary saved -> {SUMM_OUT}\n")

    W = 100
    print("=" * W)
    print("PER-METRIC: layer 1 (r2 >= 0.60, gt_clean)  ->  layer 2 (LOCO CCC >= 0.80)")
    print("=" * W)
    print(f"{'metric':38}{'grade':>15}{'S':>4}{'M':>4}{'best CCC':>13}"
          f"{'D/C/W':>10}  strong zone")
    print("-" * W)
    for _, r in s[s.map_cells > 0].iterrows():
        bc = f"{r.best_ccc:.3f}@{r.best_ccc_view}" if r.best_ccc is not None else "-"
        mark = "*" if r.adopted else " "
        print(f"{mark}{r.metric[:37]:37}{r.metric_grade:>15}{r.strong_cells:>4}"
              f"{r.moderate_cells:>4}{bc:>13}"
              f"{f'{r.direct}/{r.calibrate}/{r.pass_weak}':>10}  "
              f"{r.strong_zone or '(moderate only) ' + r.zone}"
              f"{'  [spike]' if r.spikes else ''}")

    passed = s[s.gate_cells > 0]
    failed = s[s.gate_cells == 0]
    same = passed[passed.pair_kind == "same-estimator"]
    alt = passed[passed.pair_kind == "alt-estimator"]
    print("\n" + "=" * W)
    print(f"COUNTING: {len(passed)} rows pass the gate.")
    print(f"  {len(same)} are the SAME estimator run through a second pipeline "
          f"(cross-check, not a new measurement):")
    for _, r in same.iterrows():
        print(f"      {r.metric:<40} == {r.pairs_with}")
    print(f"  {len(alt)} measure a truth already in the list with a DIFFERENT 2D "
          f"observable (a real second method, different quality):")
    for _, r in alt.iterrows():
        ref = passed[passed.metric == r.pairs_with]
        rc = int(ref.gate_cells.iloc[0]) if len(ref) else -1
        print(f"      {r.metric:<40} vs {r.pairs_with}   "
              f"{int(r.gate_cells)} cells vs {rc}")
    print(f"  -> rows {len(passed)}, distinct QUANTITIES "
          f"{len(passed) - len(same) - len(alt)}, distinct METHODS "
          f"{len(passed) - len(same)}")
    print("=" * W)
    print(f"PASS (>=1 gate cell): {len(passed)} metrics")
    print("=" * W)
    print(", ".join(sorted(passed.metric)))
    print("\n" + "=" * W)
    print(f"FAIL (0 gate cells anywhere): {len(failed)} metrics")
    print("=" * W)
    for _, r in failed.iterrows():
        why = ("no cell clears r2 either" if r.r2_cells == 0 else
               f"{r.r2_cells} r2 cells but CCC tops out at {r.best_ccc}")
        print(f"  {r.metric:44}{why}")

    print("\n" + "=" * W)
    print("VERDICT CHANGED vs the old 403-pitch screen (layer 1 population fix)")
    print("=" * W)
    ch = s[(s.old403_cells >= 0) &
           (((s.old403_cells == 0) & (s.r2_cells > 0)) |
            ((s.old403_cells > 0) & (s.r2_cells == 0)) |
            ((s.r2_cells - s.old403_cells).abs() >= 5))]
    for _, r in ch.sort_values("r2_cells", ascending=False).iterrows():
        print(f"  {r.metric:44}{r.old403_cells:>4} -> {r.r2_cells:>4} r2 cells"
              f"   (gate {r.gate_cells})")

    print("\n" + "=" * W)
    print("ADOPTED BUT FAILS THE GATE EVERYWHERE  (paper impact)")
    print("=" * W)
    for _, r in s[(s.adopted) & (s.gate_cells == 0)].iterrows():
        print(f"  {r.metric:44}r2 cells {r.r2_cells}, best CCC {r.best_ccc} "
              f"@ {r.best_ccc_view}")


if __name__ == "__main__":
    main()
