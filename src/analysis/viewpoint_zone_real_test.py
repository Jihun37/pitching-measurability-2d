"""
Diamond - REAL-VIDEO spot check of the viewpoint-bin -> zone-lookup classifier.

Predicts the (az, el) bin of actual smartphone clips (RTMPose coords) with the
kNN trained on the frozen OBP feature bank (viewpoint_zone_features.csv), then
shows the per-metric zone verdict that the auto-measurement pipeline would act
on. Domain gap: OBP is clean orthographic projection; the clips have
perspective, lens, and real detector error.

IMPORTANT (user, 2026-07-10): the clips were filmed casually - "lateral" is
NOT guaranteed az=0/el=0. Evaluation is therefore REGION plausibility, not
exact-bin match:
  lateral   : el <= 30 and az within +-45 of 0 or 180
  frontier  : el <= 30 and az within +-45 of 90 or 270
  overhead  : el >= 75 (az free)
Predicted bins inside the region are OK even if far from the nominal angle.

Reported per clip: top-3 voted bins (spread = uncertainty), region verdict,
out-of-distribution feature flags (vs OBP 2-98% range), and the claimed-zone
metric list at the top bin (reliable / usable / point / out).

Run:  cd src\\analysis
      python viewpoint_zone_real_test.py            (rtmp backbone)
      python viewpoint_zone_real_test.py --backbone mediapipe
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import config
import metrics as M
from station_classify import features, FEATS
from viewpoint_zone_classify import zone_masks, ZONE_METRICS, FLOOR
from real_station_test import detect_arm_2d

VIDEOS = [
    ("pitching_lateral_02",  "lateral"),
    ("pitching_frontier_03", "frontier"),
    ("pitching_overhead_01", "overhead"),
    ("pitching_overhead_02", "overhead"),
    ("pitching_overhead_03", "overhead"),
]
# point recommendations for the velocity metrics (§11.2), same rules as
# deploy/measure_auto.py: wrist = side ground (az 0 +-1 bin, el=0);
# pelvis = overhead el=85, azimuth-FREE (overhead is az-invariant).
POINTS = {"Wrist Speed [O]": (0, 0), "Pelvis Rot Velo [O]": (None, 85)}


def in_region(az, el, region):
    if region == "overhead":
        return el >= 75
    centers = (0, 180) if region == "lateral" else (90, 270)
    d = min(abs((az - c + 180) % 360 - 180) for c in centers)
    return el <= 30 and d <= 45


def load_clip(name, suffix):
    p = os.path.join(config.OUTPUT_DIR.replace(config.VIDEO_NAME, name),
                     f"{name}_smoothed{suffix}.csv")
    df = pd.read_csv(p)
    if "nose_x" in df.columns and "head_x" not in df.columns:
        df = df.rename(columns={"nose_x": "head_x", "nose_y": "head_y",
                                "nose_v": "head_v"})
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="rtmp", choices=["rtmp", "mediapipe"])
    ap.add_argument("--k", type=int, default=25)
    a = ap.parse_args()
    suffix = "_rtmp" if a.backbone == "rtmp" else ""

    bank = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR,
                                    "viewpoint_zone_features.csv")
                       ).dropna(subset=FEATS).reset_index(drop=True)
    print(f"[viewpoint-zone real test]  backbone={a.backbone}  "
          f"bank={len(bank)} rows (clean+aug, all OBP)\n")
    Xtr = bank[FEATS].to_numpy(float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    tree = cKDTree((Xtr - mu) / sd)
    baz = bank["az"].to_numpy(int); bel = bank["el"].to_numpy(int)
    lo = np.nanpercentile(Xtr, 2, axis=0)
    hi = np.nanpercentile(Xtr, 98, axis=0)
    masks = zone_masks()

    for name, region in VIDEOS:
        df = load_clip(name, suffix)
        arm = detect_arm_2d(df)
        f = features(df, arm)
        xv = np.array([f[k] for k in FEATS], float)
        _, idx = tree.query((xv - mu) / sd, k=a.k)
        pairs, cnts = np.unique(list(zip(baz[idx], bel[idx])),
                                axis=0, return_counts=True)
        top = np.argsort(cnts)[::-1][:3]
        paz, pel = int(pairs[top[0]][0]), int(pairs[top[0]][1])
        ok = "OK" if in_region(paz, pel, region) else "OUT-OF-REGION"
        ood = [FEATS[j] for j in range(len(FEATS))
               if not (lo[j] <= xv[j] <= hi[j])]

        print("=" * 66)
        print(f"{name}   rough label: {region}   arm={arm}")
        votes = "  ".join(f"(az={p[0]},el={p[1]}) {c / a.k:.2f}"
                          for p, c in zip(pairs[top], cnts[top]))
        print(f"  top bins : {votes}")
        print(f"  verdict  : {ok}" + (f"   OOD feats: {ood}" if ood else ""))
        print(f"  zone verdict at (az={paz}, el={pel}):")
        for m in ZONE_METRICS:
            print(f"    {m:24s} {'USABLE' if masks[m][(paz, pel)] else 'out'}")
        for m, (p_az, p_el) in POINTS.items():
            # elevation must match the point row; az free (None) or +-1 bin
            hit = (pel == p_el) and (
                p_az is None or abs((paz - p_az + 180) % 360 - 180) <= 15)
            print(f"    {m:24s} {'POINT HIT' if hit else 'out (point-only)'}")
        print()


if __name__ == "__main__":
    main()
