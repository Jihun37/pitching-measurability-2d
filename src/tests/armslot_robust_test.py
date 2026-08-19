"""
Diamond - Arm slot robustness validation (paper-grade)
armslot_robust_test.py

Why: the existing armslot_validate result (r2=1.00 @90deg, clean projection) is
near-tautological -- at azimuth 90 the projection IS the frontal plane the 3D
truth is defined on. The paper-grade number is r2 UNDER NOISE, where the
identity breaks and real measurement performance shows.

Protocol (same r2 language as all other metrics):
  1) 3D truth (once per pitch): angle of (hand-shoulder) Y-Z components vs
     vertical -- identical to armslot_validate.truth_3d_frontal.
  2) 2D estimate: project at azimuth in {60,75,90} with Gaussian keypoint
     noise in {0,2,4,6}px, seeds {0,1,2} -> shoulder->wrist vs vertical.
  3) Report per (azimuth x noise), seed-averaged:
       - r2  (vs 3D truth)
       - MAE (deg)  -- more intuitive for an angle metric
       - slot-category accuracy (overhand <=45 / three-quarter 45-70 /
         sidearm >70), the value scouts actually use.
  4) Sanity anchor: corr(our 3D truth, OBP forearm arm_slot column).
     Different definitions, but both describe "slot" -> moderate positive
     correlation expected; this externally anchors our constructed truth
     (same role as the knee cross-check r2=0.846).

Release frame is detected ONCE on the clean side view (consistent with all
prior validation) so the table isolates ANGLE-measurement robustness from
event-detection robustness.

Usage:
    python armslot_robust_test.py --limit 50
    python armslot_robust_test.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(_HERE, "..", "stage3"))

import config
import obp_project as O
import metrics as M

AZIMUTHS = [60, 75, 90]
NOISE_PX = [0, 2, 4, 6]          # ~0 / 1.4 / 2.8 / 4.2 % of body scale (142px)
SEEDS = [0, 1, 2]
BODY_SCALE_REF = 142.0

# slot categories (deg from vertical): overhand <=45, three-quarter 45-70, sidearm >70
CAT_EDGES = (45.0, 70.0)


def truth_3d_frontal(joints, arm, rel):
    """3D frontal-plane arm slot truth (same as armslot_validate)."""
    S = joints[f"{arm}_shoulder"][:, rel]
    W = joints[f"{arm}_wrist"][:, rel]
    vec = W - S                      # (X,Y,Z); X=pitch dir, Y=lateral, Z=up
    run = abs(vec[1])
    rise = vec[2]
    return float(np.degrees(np.arctan2(run, rise)))


def estimate_2d(df, arm, rel):
    sx = df[f"{arm}_shoulder_x"].iloc[rel]; sy = df[f"{arm}_shoulder_y"].iloc[rel]
    wx = df[f"{arm}_wrist_x"].iloc[rel];   wy = df[f"{arm}_wrist_y"].iloc[rel]
    return float(np.degrees(np.arctan2(abs(wx - sx), (sy - wy))))


def category(a):
    if a <= CAT_EDGES[0]:
        return 0   # overhand
    if a <= CAT_EDGES[1]:
        return 1   # three-quarter
    return 2       # sidearm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    obp_slot = poi.set_index("session_pitch")["arm_slot"].to_dict() \
        if "arm_slot" in poi.columns else {}
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    truths, obp_vals = [], []
    est = {(az, n, s): [] for az in AZIMUTHS for n in NOISE_PX for s in SEEDS}
    done = fail = 0

    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = O.load_c3d_joints(path)
            arm = O.detect_throwing_arm(joints, fps)
            df0 = O.project_view(joints, azimuth_deg=0.0)
            rel = M.release_frame(df0, arm, fps, M.JOINTS)
            t = truth_3d_frontal(joints, arm, rel)
            truths.append(t)
            obp_vals.append(obp_slot.get(r.session_pitch, np.nan))
            for az in AZIMUTHS:
                for n in NOISE_PX:
                    for s in SEEDS:
                        df = O.project_view(joints, azimuth_deg=az,
                                            noise_px=n, seed=s)
                        est[(az, n, s)].append(estimate_2d(df, arm, rel))
            done += 1
        except Exception:
            fail += 1
        if done and done % 100 == 0:
            print(f"  ...{done} processed")
    print(f"processed {done} / failed {fail}\n")

    t = np.asarray(truths, float)
    tcat = np.array([category(v) for v in t])

    def cell(az, n):
        """Seed-averaged (r2, MAE, category accuracy) for one condition."""
        r2s, maes, accs = [], [], []
        for s in SEEDS:
            e = np.asarray(est[(az, n, s)], float)
            m = np.isfinite(e) & np.isfinite(t)
            if m.sum() <= 2:
                continue
            r = np.corrcoef(e[m], t[m])[0, 1]
            r2s.append(r * r)
            maes.append(np.mean(np.abs(e[m] - t[m])))
            ecat = np.array([category(v) for v in e[m]])
            accs.append(np.mean(ecat == tcat[m]))
        return (np.mean(r2s) if r2s else np.nan,
                np.mean(maes) if maes else np.nan,
                np.mean(accs) if accs else np.nan)

    results = {(az, n): cell(az, n) for az in AZIMUTHS for n in NOISE_PX}

    def table(idx, title, fmt):
        print("=" * 64)
        print(title)
        print("=" * 64)
        hdr = "azimuth".ljust(10) + "".join(
            f"{n}px(~{n/BODY_SCALE_REF*100:.1f}%)".rjust(13) for n in NOISE_PX)
        print(hdr); print("-" * len(hdr))
        for az in AZIMUTHS:
            line = f"{az}deg".ljust(10)
            for n in NOISE_PX:
                line += fmt(results[(az, n)][idx]).rjust(13)
            print(line)
        print()

    table(0, "[arm slot r2 vs 3D frontal truth]  (seed-avg)",
          lambda v: f"{v:.3f}")
    table(1, "[arm slot MAE, deg]  (seed-avg)",
          lambda v: f"{v:.1f}")
    table(2, "[slot category accuracy: OH<=45 / TQ 45-70 / SA>70]  (seed-avg)",
          lambda v: f"{v*100:.1f}%")

    # truth distribution + category counts (for context)
    cnt = [int((tcat == k).sum()) for k in (0, 1, 2)]
    print(f"truth distribution: mean {t.mean():.1f} / std {t.std():.1f}  "
          f"| categories OH {cnt[0]} / TQ {cnt[1]} / SA {cnt[2]}")

    # sanity anchor: our truth vs OBP forearm slot (different definition!)
    o = np.asarray(obp_vals, float)
    m = np.isfinite(o) & np.isfinite(t)
    if m.sum() > 2:
        r = np.corrcoef(t[m], o[m])[0, 1]
        print(f"\n[sanity] our 3D truth vs OBP forearm arm_slot column: "
              f"r = {r:+.3f}  (different definitions; moderate correlation "
              f"expected -> external anchor for our constructed truth)")

    # save
    rows = []
    for (az, n), (r2v, mae, acc) in results.items():
        rows.append({"azimuth": az, "noise_px": n,
                     "pct_body": round(n / BODY_SCALE_REF * 100, 1),
                     "r2": r2v, "mae_deg": mae, "cat_acc": acc})
    out = os.path.join(config.OBP_VALIDATION_DIR, "armslot_robust_results.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()