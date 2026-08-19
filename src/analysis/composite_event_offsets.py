"""
Diamond - composite temporal references: shift the two ends INDEPENDENTLY.

Step 5 of the event-axis rebuild. Every offset sweep we have moved the anchors in
LOCKSTEP: rejected_gt_full_sweep.py:291 reads `obs(fp + k, rel + k)`, so the search
interval keeps its length and only slides. Real detectors do not do that -- foot
plant and release are found by different signals, at different views, with different
errors, so the interval also STRETCHES and SHRINKS. Nothing has ever measured that.

This file sweeps delta_fp x delta_rel over {-3..+3} independently (49 combinations)
for every row whose value depends on both ends:

  lead_knee_extension_from_fp_to_br     kang[rel] - kang[fp]   -- a DIFFERENCE of
                                        two instants, the purest case
  max_pelvis_rotational_velo            |max| over [fp, rel]
  lead_knee_extension_angular_velo_max  max over [fp, rel]
  max_elbow_flexion                     max over [fp, rel]
  Release Ext [O]                       CONTROL: rel is the anchor, fp only ends the
                                        quiet-window search, so its delta_fp row
                                        should be flat. If it is not, the harness is
                                        wrong.

Reported against the lockstep diagonal, because that is the number the existing
tolerance table (event_tolerance.csv) is built on:
  diag   delta_fp == delta_rel   (what we have been measuring)
  indep  the worst cell of the whole box at that radius (what deployment gets)
  len    the two pure interval-length corners (-d, +d) and (+d, -d)

Definitions come from rejected_gt_full_sweep.observables and metrics; scoring from
gate_map.score_cell. Population = the frozen map's 394. Cells = each row's own gate
cells (100 distinct).

Outputs: composite_event_offsets.csv
Run:  cd src\\analysis
      python composite_event_offsets.py --limit 20
      python composite_event_offsets.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)

import config
import metrics as M
import obp_project as O
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from obp_gt_events import load_gt_events
from gate_map import score_cell
from mer_proxy_map import map_population
from rejected_gt_full_sweep import observables
from angle_map_2d import t3_release_ext

OFFSETS = list(range(-3, 4))
ROWS = {  # metric -> (kind, truth column or None for 3D-direct)
    "lead_knee_extension_from_fp_to_br":   ("knee_diff", "lead_knee_extension_from_fp_to_br"),
    "max_pelvis_rotational_velo":          ("hip_velo_max", "max_pelvis_rotational_velo"),
    "lead_knee_extension_angular_velo_max": ("knee_velo_max", "lead_knee_extension_angular_velo_max"),
    "max_elbow_flexion":                   ("elbow_flex_max", "max_elbow_flexion"),
    "Release Ext [O]":                     ("release_ext", None),
}


def cells_for(metric, gate):
    s = gate[gate.metric == metric]
    return [(int(a), int(e)) for a, e in zip(s.az, s.el)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--floor", type=float, default=0.75)
    ap.add_argument("--strong", type=float, default=0.80)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    root = os.path.join(config.OBP_DATA_DIR, "c3d")
    gt = load_gt_events()
    pop = map_population()
    gate = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR, "gate_map.csv"))
    gate = gate[gate.gate_pass]
    CELLS = {m: cells_for(m, gate) for m in ROWS}
    ALL = sorted({c for v in CELLS.values() for c in v})
    print(f"{len(ROWS)} rows, {len(ALL)} distinct cells, "
          f"{len(OFFSETS)**2} offset combinations")

    est = {(m, c, i, j): [] for m in ROWS for c in CELLS[m]
           for i in OFFSETS for j in OFFSETS}
    tru = {m: [] for m in ROWS}
    users = []
    done = fail = 0
    for n, r in enumerate(md.itertuples(index=False)):
        if a.limit and n >= a.limit:
            break
        sp = r.session_pitch
        if sp not in pop or sp not in poi.index:
            continue
        g = gt.get(sp)
        if not g or not {"fp", "rel"} <= set(g):
            fail += 1; continue
        path = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = load_feet(path)
            arm = O.detect_throwing_arm(joints, fps)
        except Exception:
            fail += 1; continue
        lead = "left" if arm == "right" else "right"
        fp, rel = int(g["fp"]), int(g["rel"])
        h = float(r.session_height_m)
        users.append(r.user)
        for m, (_, col) in ROWS.items():
            tru[m].append(float(poi.loc[sp, col]) if col else
                          t3_release_ext(joints, {"arm": arm, "fp": fp, "rel": rel,
                                                  "fps": fps}))

        for c in ALL:
            try:
                df = project_cam(joints, c[0], c[1])
                o, _ = observables(df, fps)
                # the lead-knee series rejected_gt_full_sweep builds inline
                # (rejected_gt_full_sweep.py, "lead knee ext velo needs ..."):
                def kxy(k):
                    return (df[f"{k}_x"].to_numpy(float),
                            df[f"{k}_y"].to_numpy(float))
                hx, hy = kxy(f"{lead}_hip"); kx, ky = kxy(f"{lead}_knee")
                axx, ay = kxy(f"{lead}_ankle")
                kang = M._angle(hx, hy, kx, ky, axx, ay)
                kvel = np.gradient(kang) * fps
                ok = True
            except Exception:
                ok = False
            for m, (kind, _) in ROWS.items():
                if c not in CELLS[m]:
                    continue
                for i in OFFSETS:
                    for j in OFFSETS:
                        v = np.nan
                        if ok:
                            lo, hi = fp + i, rel + j
                            try:
                                if kind == "knee_diff":
                                    v = (float(kang[hi] - kang[max(0, lo)])
                                         if 0 <= lo < len(kang) and 0 <= hi < len(kang)
                                         else np.nan)
                                elif kind == "knee_velo_max":
                                    v = (float(np.nanmax(kvel[max(0, lo):hi + 1]))
                                         if hi > lo else np.nan)
                                elif kind == "hip_velo_max":
                                    v = o["hip_line_velo_max"](lo, hi)
                                elif kind == "elbow_flex_max":
                                    v = o["elbow_flex_max"](lo, hi)
                                elif kind == "release_ext":
                                    v = M.release_extension(df, arm, hi, max(0, lo),
                                                            fps, M.JOINTS, h)
                            except Exception:
                                v = np.nan
                        est[(m, c, i, j)].append(v)
        done += 1
        if done % 25 == 0:
            print(f"  ...{done} processed")
    print(f"processed {done} / failed {fail}\n")

    codes = pd.Series(users).astype("category").cat.codes.to_numpy()
    rows = []
    for m in ROWS:
        t = np.asarray(tru[m], float)
        for c in CELLS[m]:
            for i in OFFSETS:
                for j in OFFSETS:
                    e = np.asarray(est[(m, c, i, j)], float)
                    s = score_cell(e, t, codes, a.floor, a.strong)
                    if s is None:
                        continue
                    rows.append(dict(metric=m, az=c[0], el=c[1], d_fp=i, d_rel=j,
                                     ccc=s["ccc"], r2=s["r2"], grade=s["grade"]))
    out = pd.DataFrame(rows)
    p = os.path.join(config.OBP_VALIDATION_DIR, "composite_event_offsets.csv")
    out.to_csv(p, index=False, float_format="%.6g")

    pd.set_option("display.width", 200)
    print("=" * 100)
    print(f"COMPOSITE OFFSETS -- {len(set(users))} pitchers, {len(users)} pitches")
    print("=" * 100)

    print("\n[1] CONTROL: Release Ext should be flat in d_fp (fp only ends a search)")
    ctl = out[out.metric == "Release Ext [O]"]
    if len(ctl):
        piv = ctl.pivot_table(index="d_fp", columns="d_rel", values="ccc",
                              aggfunc="median")
        print(piv.round(4).to_string())
        # spread must be taken WITHIN a cell and a fixed d_rel; pooling cells would
        # measure the map's own cell-to-cell range instead.
        rng = ctl.groupby(["az", "el", "d_rel"]).ccc.apply(lambda s: s.max() - s.min())
        print(f"  spread across d_fp, within cell & fixed d_rel: median "
              f"{rng.median():.5f}  p90 {rng.quantile(.9):.5f}  max {rng.max():.5f}"
              f"   -> {'FLAT, harness OK' if rng.median() < 0.01 else 'NOT FLAT'}")

    print("\n[2] lockstep (the number event_tolerance.csv is built on) vs independent")
    hdr = (f"{'metric':<38}{'cells':>6}{'d':>3}{'diag min':>10}{'box min':>9}"
           f"{'len -d,+d':>11}{'len +d,-d':>11}")
    print(hdr); print("-" * len(hdr))
    for m in ROWS:
        s = out[out.metric == m]
        if not len(s):
            continue
        base = s[(s.d_fp == 0) & (s.d_rel == 0)].set_index(["az", "el"]).ccc
        for d in (1, 2, 3):
            box = s[(s.d_fp.abs() <= d) & (s.d_rel.abs() <= d)]
            diag = s[(s.d_fp == s.d_rel) & (s.d_fp.abs() <= d)]
            bmin = box.groupby(["az", "el"]).ccc.min()
            dmin = diag.groupby(["az", "el"]).ccc.min()
            l1 = s[(s.d_fp == -d) & (s.d_rel == d)].set_index(["az", "el"]).ccc
            l2 = s[(s.d_fp == d) & (s.d_rel == -d)].set_index(["az", "el"]).ccc
            print(f"{(m if d==1 else ''):<38}{len(base):>6}{d:>3}"
                  f"{(dmin - base).median():>10.4f}{(bmin - base).median():>9.4f}"
                  f"{(l1 - base).median():>11.4f}{(l2 - base).median():>11.4f}")
        print()
    print("  values are median dCCC vs (0,0). 'diag min' = worst lockstep shift within")
    print("  +-d; 'box min' = worst of all 49 independent combinations within +-d.")

    print("\n[3] tolerance, honestly: largest d whose WHOLE box keeps the grade")
    hdr2 = (f"{'metric':<38}{'cells':>6}{'tol diag':>10}{'tol box':>9}{'lost':>7}")
    print(hdr2); print("-" * len(hdr2))
    for m in ROWS:
        s = out[out.metric == m]
        if not len(s):
            continue
        tdi, tbo = [], []
        for (az, el), g2 in s.groupby(["az", "el"]):
            g0 = g2[(g2.d_fp == 0) & (g2.d_rel == 0)]
            if not len(g0) or g0.grade.iloc[0] == "limited":
                continue
            # "keeps" must mean DOES NOT DEGRADE, not "is identical": most of these
            # cells sit within 0.02 of a grade line, so an equality test scores an
            # improvement (moderate -> strong) as a tolerance failure and the whole
            # table then reads as fragility. Compare grade RANK instead.
            RANK = {"limited": 0, "moderate": 1, "strong": 2}
            want = RANK[g0.grade.iloc[0]]
            def keeps(sub):
                return len(sub) and (sub.grade.map(RANK) >= want).all()
            td = max([0] + [d for d in (1, 2, 3)
                            if keeps(g2[(g2.d_fp == g2.d_rel) & (g2.d_fp.abs() <= d)])])
            tb = max([0] + [d for d in (1, 2, 3)
                            if keeps(g2[(g2.d_fp.abs() <= d) & (g2.d_rel.abs() <= d)])])
            tdi.append(td); tbo.append(tb)
        if tdi:
            print(f"{m:<38}{len(tdi):>6}{np.median(tdi):>10.1f}{np.median(tbo):>9.1f}"
                  f"{int(np.sum(np.array(tbo) < np.array(tdi))):>7}")
    print("  tol = frames @360Hz; 'lost' = cells whose tolerance drops once the two")
    print("  ends are allowed to move independently.")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
