"""
Diamond - DOMAIN-SHIFT CALIBRATION transfer probe (read-only).

viewpoint_az_probe.py showed the folded-bin residual is a systematic
real-vs-OBP feature shift (comparable to the adjacent-bin signal itself,
pulling votes toward fold 0). This probe tests the only in-budget fix:
subtract a GLOBAL offset vector from the real-video query before the kNN.

Overfit hygiene: the offset is estimated on the DEV-15 set (independent
clips, nominal-az GT) and scored on the HOLDOUT-60. One global vector
(6 numbers), no per-az fitting, no holdout contact during estimation.

  offset = mean over dev15 of (x_clip - bank_shelf(fold(nominal_az), el=0))
  query' = x_clip - offset

Scored on folded-bin accuracy (the bits/unfold stage is orthogonal and
already solved: family correct 54/60).

Run:  cd src\tests
      python viewpoint_calib_probe.py
"""
import os, sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("..", "../stage2", "../analysis", "../deploy"):
    sys.path.insert(0, os.path.join(_HERE, p))

import config
from station_classify import features, FEATS
from real_station_test import detect_arm_2d
from measure_auto import load_clip
from scipy.spatial import cKDTree

K = 25


def fold(az):
    a = az % 180
    return int(min(a, 180 - a))


def main():
    bank = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR,
                                    "viewpoint_zone_features.csv")
                       ).dropna(subset=FEATS).reset_index(drop=True)
    bank["azf"] = bank.az.map(fold)
    X = bank[FEATS].to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    tree = cKDTree((X - mu) / sd)
    naz_b = bank.az.to_numpy(int)
    nel_b = bank.el.to_numpy(int)
    shelf0 = bank[bank.el == 0].groupby("azf")[FEATS].mean()

    def clip_feats(name):
        df, _ = load_clip(name, "_rtmp")
        return features(df, detect_arm_2d(df))

    def vote(xv):
        _, idx = tree.query((xv - mu) / sd, k=K)
        fpairs, fcnts = np.unique([(fold(a), e) for a, e in
                                   zip(naz_b[idx], nel_b[idx])],
                                  axis=0, return_counts=True)
        i = np.argmax(fcnts)
        return int(fpairs[i][0]), int(fpairs[i][1])

    # ---- estimate the offset on DEV-15 --------------------------------
    gt = pd.read_csv(os.path.join(config.ROOT, "data", "outputs",
                                  "release_gt_real15.csv"))
    deltas = []
    for r in gt.itertuples():
        f = clip_feats(r.clip)
        # dev nominal az is free-form; snap the folded value to the bank grid
        azf = min(shelf0.index, key=lambda b: abs(b - fold(int(r.nominal_az))))
        ref = shelf0.loc[azf]
        deltas.append([f[k] - ref[k] for k in FEATS])
    off = np.mean(deltas, axis=0)
    off_sd = np.std(deltas, axis=0)
    print("dev-15 offset vector (raw units | global-sd units | spread sd):")
    for j, k in enumerate(FEATS):
        print(f"  {k:8s} {off[j]:+8.3f}   {off[j] / sd[j]:+6.2f} sd"
              f"   (clip spread {off_sd[j] / sd[j]:.2f} sd)")

    # ---- score on HOLDOUT-60 -------------------------------------------
    clips = [(f"angle{i:02d}_{p:02d}", (30 * i) % 360)
             for i in range(12) for p in range(5)]
    res = []
    for name, az in clips:
        f = clip_feats(name)
        xv = np.array([f[k] for k in FEATS], float)
        fp_b, el_b = vote(xv)                 # baseline (wired behavior)
        fp_c, el_c = vote(xv - off)           # calibrated
        res.append({"clip": name, "foldT": fold(az),
                    "fp_base": fp_b, "fp_cal": fp_c,
                    "el_base": el_b, "el_cal": el_c})
    d = pd.DataFrame(res)
    for tag, col in (("baseline ", "fp_base"), ("calibrated", "fp_cal")):
        e = (d[col] - d.foldT).to_numpy(int)
        print(f"[{tag}] fold exact {(e == 0).sum()}/60"
              f"  +-15 {(np.abs(e) <= 15).sum()}/60"
              f"  +-30 {(np.abs(e) <= 30).sum()}/60"
              f"  mean signed {e.mean():+.1f}")
    for tag, col in (("baseline ", "el_base"), ("calibrated", "el_cal")):
        print(f"[{tag}] el>=15 votes: {(d[col] >= 15).sum()}/60"
              f"  el>=30: {(d[col] >= 30).sum()}/60")

    print("\nper true-az family (exact base -> cal):")
    for ft in sorted(d.foldT.unique()):
        s = d[d.foldT == ft]
        print(f"  fold{ft:>3} (n={len(s):2d}): "
              f"{(s.fp_base == ft).sum()} -> {(s.fp_cal == ft).sum()}")

    out = os.path.join(config.ROOT, "data", "outputs",
                       "viewpoint_calib_probe.csv")
    d.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
