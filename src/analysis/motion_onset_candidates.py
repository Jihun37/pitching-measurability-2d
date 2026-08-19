"""
Diamond - Motion onset / setup quiet-window: candidate comparison (2026-07-27).

Step 3 of the event-axis rebuild. `Stride (anchor)` and `Release Ext` both measure
FROM where the pitcher was standing, which metrics.trail_anchor_x recovers as the
median of the trail ankle over its pre-motion quiet window. That window ends at a
single hard-coded crossing -- the first frame the trail |vx| exceeds 20 % of its own
peak -- with no persistence guard, no GT and no per-view number. One jittery frame
can end it early.

The plateau result (stride_plateau_2d.py) made this the priority: reading stride at
the WRONG INSTANT costs nothing (|dCCC| <= 0.006 over 27 criteria), while using the
wrong ORIGIN costs 0.148 CCC and all 20 strong cells. The accuracy lives here.

Candidates: threshold in {10, 20, 30} % of the trail ankle's own peak speed, times a
persistence guard in {1, 5, 7, 10} frames @360 Hz (guard 1 = today's behaviour).

Selection criteria -- deliberately NOT onset-frame accuracy (user, 2026-07-27):
  1. quiet-window success  share of pitches whose window reaches min_quiet_s = 0.10 s
                           (the gate release_extension already defers on)
  2. anchor position error vs a GT anchor built from 3D DISPLACEMENT, not from any
                           velocity threshold, so the reference cannot favour a
                           member of the family being tested
  3. final performance     Stride (anchor) over its 30 gate cells and Release Ext
                           over its 49, scored with gate_map.score_cell

If the candidates come out equal on all three, motion onset is not an event: it is a
setup quiet-window ANCHOR, and the inventory records it as such.

GT anchor: t0 = the last frame before the 3D trail ankle has moved 1 cm from its
start (capped at foot plant); GT anchor in a view = median of the PROJECTED trail x
over [0, t0]. Threshold-free with respect to speed.

Outputs: motion_onset_candidates.csv (per variant x cell), motion_onset_pitch.csv
Run:  cd src\\analysis
      python motion_onset_candidates.py --limit 20
      python motion_onset_candidates.py
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
from gt_landmark_outlier_effect import outlier_pitches
from mer_proxy_map import map_population
from gate_map import score_cell

STRIDE_CELLS = [(0, 0), (0, 15), (0, 30), (0, 75), (0, 85), (15, 0), (15, 15),
                (150, 0), (150, 15), (150, 30), (150, 45), (165, 0), (165, 15),
                (165, 30), (165, 45), (180, 0), (180, 15), (180, 30), (195, 0),
                (195, 15), (330, 0), (330, 15), (330, 30), (330, 75), (330, 85),
                (345, 0), (345, 15), (345, 30), (345, 75), (345, 85)]
RELEXT_CELLS = [(0, 0), (0, 15), (0, 30), (0, 75), (0, 85), (15, 0), (15, 15),
                (15, 30), (15, 45), (15, 60), (15, 75), (15, 85), (30, 0),
                (30, 15), (30, 30), (30, 75), (30, 85), (150, 0), (150, 15),
                (150, 30), (150, 45), (165, 0), (165, 15), (165, 30), (165, 45),
                (180, 0), (180, 15), (180, 30), (180, 45), (180, 75), (180, 85),
                (195, 0), (195, 15), (195, 30), (195, 75), (195, 85), (210, 0),
                (210, 15), (210, 85), (330, 0), (330, 15), (330, 30), (330, 75),
                (330, 85), (345, 0), (345, 15), (345, 30), (345, 75), (345, 85)]
CELLS = sorted(set(STRIDE_CELLS) | set(RELEXT_CELLS))
THR = [0.10, 0.20, 0.30]
GUARD = [1, 5, 7, 10]
MIN_QUIET_S = 0.10
SETTLE_S = 0.08
GT_DISP_M = 0.01          # 1 cm: the trail ankle has "moved"


def quiet_end(trail_x, end, fps, thr, guard):
    """Index where the pre-motion quiet window ends: the first frame whose speed
    exceeds thr * peak AND stays above it for `guard` frames. guard=1 reproduces
    metrics.trail_anchor_x exactly."""
    seg = trail_x[:int(end) + 1]
    if len(seg) < 5 or np.all(np.isnan(seg)):
        return 3
    v = np.abs(np.diff(seg)) * fps
    vpk = np.nanmax(v)
    if not np.isfinite(vpk) or vpk <= 0:
        return len(seg) - 1
    over = v > thr * vpk
    if guard <= 1:
        idx = np.where(over)[0]
        return max(3, int(idx[0]) if len(idx) else len(seg) - 1)
    run = 0
    for i, o in enumerate(over):
        run = run + 1 if o else 0
        if run >= guard:
            return max(3, i - guard + 1)
    return max(3, len(seg) - 1)


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
    # the frozen map's 394 ids, not gt_clean alone: angle_zone_sweep also
    # requires pkh and rel > fp+1, which drops one more pitch. Official
    # numbers must be on the same population as gate_map.csv.
    pop = map_population()
    VS = [(t, g) for t in THR for g in GUARD]

    stride = {(v, c): [] for v in VS for c in STRIDE_CELLS}
    relext = {(v, c): [] for v in VS for c in RELEXT_CELLS}
    aerr = {(v, c): [] for v in VS for c in CELLS}
    qlen = {v: [] for v in VS}
    t_str, t_rel, users, gt_qlen = [], [], [], []
    done = fail = 0
    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
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
        lan = "l_an" if lead == "left" else "r_an"
        tan = "r_an" if lead == "left" else "l_an"
        wkey = "r_wr" if arm == "right" else "l_wr"
        fp, rel = int(g["fp"]), int(g["rel"])
        h = float(r.session_height_m)

        # ---- GT quiet window from 3D displacement (no speed threshold) --------
        X3 = joints["right_ankle" if lead == "left" else "left_ankle"][0].astype(float)
        d = np.abs(X3[:fp + 1] - np.nanmedian(X3[:5]))
        moved = np.where(d > GT_DISP_M)[0]
        t0 = int(moved[0]) if len(moved) else fp
        t0 = max(3, t0)
        gt_qlen.append(t0 / fps)

        t_str.append(float(poi.loc[sp, "stride_length"]))
        # Release Ext truth is 3D-direct (t3_release_ext); read it the same way
        W3 = joints[("right_wrist" if arm == "right" else "left_wrist")][0].astype(float)
        t_rel.append(abs(float(W3[rel]) - float(np.nanmedian(X3[:t0 + 1]))))
        users.append(r.user)

        for c in CELLS:
            try:
                df = project_cam(joints, c[0], c[1])
                xt = df[f"{M.JOINTS[tan]}_x"].to_numpy(float)
                xl = df[f"{M.JOINTS[lan]}_x"].to_numpy(float)
                xw = df[f"{M.JOINTS[wkey]}_x"].to_numpy(float)
                stat = M.pixel_stature(df, M.JOINTS)
                gt_anc = float(np.nanmedian(xt[:t0 + 1]))
                w = max(1, int(round(SETTLE_S * fps)))
                x_set = float(np.nanmedian(xl[max(0, rel - w):rel + 1]))
                for v in VS:
                    m = quiet_end(xt, fp, fps, v[0], v[1])
                    anc = float(np.nanmedian(xt[:m]))
                    ql = m / fps
                    if c == CELLS[0]:
                        qlen[v].append(ql)
                    aerr[(v, c)].append((anc - gt_anc) / stat)
                    if (v, c) in stride:
                        stride[(v, c)].append(abs(x_set - anc) / stat)
                    if (v, c) in relext:
                        relext[(v, c)].append(
                            abs(xw[rel] - anc) / stat * h
                            if ql >= MIN_QUIET_S else np.nan)
            except Exception:
                for v in VS:
                    aerr[(v, c)].append(np.nan)
                    if (v, c) in stride:
                        stride[(v, c)].append(np.nan)
                    if (v, c) in relext:
                        relext[(v, c)].append(np.nan)
        done += 1
        if done % 50 == 0:
            print(f"  ...{done} processed")
    print(f"processed {done} / failed {fail}\n")

    codes = pd.Series(users).astype("category").cat.codes.to_numpy()
    ts = np.asarray(t_str, float); tr = np.asarray(t_rel, float)
    rows = []
    for v in VS:
        ql = np.asarray(qlen[v], float)
        for c in CELLS:
            ae = np.asarray(aerr[(v, c)], float)
            ae = ae[np.isfinite(ae)]
            rec = dict(thr=v[0], guard=v[1], az=c[0], el=c[1],
                       quiet_ok=float(np.mean(ql >= MIN_QUIET_S)) if ql.size else np.nan,
                       quiet_med_s=float(np.median(ql)) if ql.size else np.nan,
                       anc_mae_stat=float(np.mean(np.abs(ae))) if ae.size else np.nan,
                       anc_med_stat=float(np.median(ae)) if ae.size else np.nan)
            if (v, c) in stride:
                s = score_cell(np.asarray(stride[(v, c)], float), ts, codes,
                               a.floor, a.strong)
                rec.update({f"stride_{k}": s[k] for k in ("ccc", "r2", "grade")}
                           if s else {})
            if (v, c) in relext:
                s = score_cell(np.asarray(relext[(v, c)], float), tr, codes,
                               a.floor, a.strong)
                rec.update({f"relext_{k}": s[k] for k in ("ccc", "r2", "grade")}
                           if s else {})
            rows.append(rec)
    out = pd.DataFrame(rows)
    p = os.path.join(config.OBP_VALIDATION_DIR, "motion_onset_candidates.csv")
    out.to_csv(p, index=False, float_format="%.6g")

    pd.set_option("display.width", 200)
    gq = np.asarray(gt_qlen, float)
    print("=" * 100)
    print(f"MOTION ONSET CANDIDATES -- n={len(ts)} pitches, {len(set(users))} "
          f"pitchers, {len(CELLS)} cells")
    print("=" * 100)
    print(f"GT quiet window (3D, 1 cm displacement): median {np.median(gq)*1000:.0f} ms,"
          f"  p5 {np.percentile(gq,5)*1000:.0f} ms,  share >= 100 ms "
          f"{np.mean(gq>=MIN_QUIET_S):.1%}\n")
    hdr = (f"{'thr':>5}{'guard':>7}{'quiet_ok':>10}{'quiet_med_ms':>14}"
           f"{'anchor MAE':>12}{'stride CCC':>12}{'stride str':>11}"
           f"{'relext CCC':>12}{'relext str':>11}")
    print(hdr); print("-" * len(hdr))
    for v in VS:
        s = out[(out.thr == v[0]) & (out.guard == v[1])]
        sc = s.stride_ccc.dropna(); rc = s.relext_ccc.dropna()
        print(f"{v[0]:>5.2f}{v[1]:>7d}{s.quiet_ok.iloc[0]:>10.3f}"
              f"{s.quiet_med_s.iloc[0]*1000:>14.0f}{s.anc_mae_stat.median():>12.4f}"
              f"{sc.median():>12.3f}{int((s.stride_grade=='strong').sum()):>11d}"
              f"{rc.median():>12.3f}{int((s.relext_grade=='strong').sum()):>11d}")
    print("\n  anchor MAE = |variant anchor - GT anchor| in statures, median over cells")
    print("  quiet_ok   = share of pitches whose window reaches 100 ms "
          "(release_extension's defer gate)")

    base = out[(out.thr == 0.20) & (out.guard == 1)].set_index(["az", "el"])
    print("\nvs the CURRENT rule (thr 0.20, guard 1), median over cells:")
    for v in VS:
        if v == (0.20, 1):
            continue
        s = out[(out.thr == v[0]) & (out.guard == v[1])].set_index(["az", "el"])
        ds = (s.stride_ccc - base.stride_ccc).median()
        dr = (s.relext_ccc - base.relext_ccc).median()
        print(f"  thr {v[0]:.2f} guard {v[1]:>2}:  stride {ds:+.4f}   relext {dr:+.4f}"
              f"   anchor MAE {(s.anc_mae_stat - base.anc_mae_stat).median():+.4f}")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
