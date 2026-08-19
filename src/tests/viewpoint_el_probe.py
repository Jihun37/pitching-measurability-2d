"""
Diamond - ELEVATION-BIAS diagnostic probe (read-only, no wiring changes).

Problem (holdout 60, true el~0 everywhere): 23/60 clips vote el>=15 in the
wired fold+bits chain. Wrong el>=15 wrongly DENIES stride & release height
(CONFOUNDED at el>0) on ground-level clips.

This probe asks WHICH feature pushes real ground clips toward elevated OBP
bank neighbors, and whether the push is a systematic real-vs-OBP domain
shift (fixable by a prior / feature correction) or per-clip noise.

Method per clip (features are event-free; no release anchor needed):
  1. kNN el vote (same bank / K / standardization as measure_auto; the el
     of the top folded (az,el) joint bin == the wired chain's el, because
     unfolding remaps az within its own fold family).
  2. For each feature: z-score of the clip value against the bank rows of
     the TRUE az fold family at el=0, in units of the bank's GLOBAL sd
     (the kNN metric), signed toward the el=15 family mean:
        z0 = (x - m0) / sd_global
        resp = (x - m0) / (m15 - m0)   # 1.0 => sits AT the el15 level
     resp is the responsibility score: which feature has moved a ground
     clip to the elevated shelf of the bank.

Run:  cd src\tests
      python viewpoint_el_probe.py
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
EL_FEATS = ["torso", "headv", "bbox"]        # documented el direction
def fold(az):
    a = az % 180
    return int(min(a, 180 - a))


def main():
    bank = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR,
                                    "viewpoint_zone_features.csv")
                       ).dropna(subset=FEATS).reset_index(drop=True)
    Xtr = bank[FEATS].to_numpy(float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    tree = cKDTree((Xtr - mu) / sd)
    naz_bank = bank["az"].to_numpy(int)
    nel_bank = bank["el"].to_numpy(int)

    clips = [(f"angle{i:02d}_{p:02d}", (30 * i) % 360)
             for i in range(12) for p in range(5)]

    rows = []
    print(f"{'clip':14s}{'az':>4}{'elP':>4}{'p15+':>6}"
          + "".join(f"{f+'_z':>9}" for f in FEATS))
    for name, true_az in clips:
        df, _ = load_clip(name, "_rtmp")
        arm = detect_arm_2d(df)
        feat = features(df, arm)
        xv = np.array([feat[k] for k in FEATS], float)
        _, idx = tree.query((xv - mu) / sd, k=K)
        nel = nel_bank[idx]
        naz = naz_bank[idx]

        # wired el = el of the top folded joint bin
        fpairs, fcnts = np.unique([(fold(a), e) for a, e in zip(naz, nel)],
                                  axis=0, return_counts=True)
        el_pred = int(fpairs[np.argmax(fcnts)][1])
        p_hi = float((nel >= 15).sum()) / K

        # bank shelves of the TRUE az fold family
        azf = fold(true_az)
        fam = {azf % 360, (180 - azf) % 360, (180 + azf) % 360,
               (360 - azf) % 360}
        b0 = bank[(bank.az.isin(fam)) & (bank.el == 0)]
        b15 = bank[(bank.az.isin(fam)) & (bank.el == 15)]
        rec = {"clip": name, "az": true_az, "el_pred": el_pred,
               "p_el15plus": p_hi}
        zs = []
        for j, f in enumerate(FEATS):
            m0, m15 = b0[f].mean(), b15[f].mean()
            z0 = (feat[f] - m0) / sd[j]
            rec[f + "_z0"] = z0
            rec[f + "_resp"] = (feat[f] - m0) / (m15 - m0) \
                if abs(m15 - m0) > 1e-9 else np.nan
            zs.append(z0)
        rows.append(rec)
        print(f"{name:14s}{true_az:>4}{el_pred:>4}{p_hi:>6.2f}"
              + "".join(f"{z:>9.2f}" for z in zs))

    d = pd.DataFrame(rows)
    hi = d.el_pred >= 15
    print(f"\nvoted el>=15: {int(hi.sum())}/60   "
          f"(el15 {int((d.el_pred == 15).sum())}, "
          f"el30 {int((d.el_pred == 30).sum())}, "
          f"el45+ {int((d.el_pred >= 45).sum())})")

    print("\nmean z0 vs bank(true-az family, el=0)  [global-sd units]")
    print(f"{'feature':10s}{'all60':>8}{'el-wrong':>10}{'el-right':>10}"
          f"{'el15-el0 gap':>14}")
    for j, f in enumerate(FEATS):
        # bank shelf gap in the same units, averaged over the 12 families
        gaps = []
        for azf in sorted({fold(a) for _, a in clips}):
            fam = {azf % 360, (180 - azf) % 360, (180 + azf) % 360,
                   (360 - azf) % 360}
            b0 = bank[(bank.az.isin(fam)) & (bank.el == 0)][f].mean()
            b15 = bank[(bank.az.isin(fam)) & (bank.el == 15)][f].mean()
            gaps.append((b15 - b0) / sd[j])
        print(f"{f:10s}{d[f + '_z0'].mean():>8.2f}"
              f"{d.loc[hi, f + '_z0'].mean():>10.2f}"
              f"{d.loc[~hi, f + '_z0'].mean():>10.2f}"
              f"{np.mean(gaps):>14.2f}")

    print("\nmean responsibility (x-m0)/(m15-m0), el-sensitive feats "
          "(1.0 = at the el15 shelf)")
    for f in EL_FEATS:
        print(f"{f:10s} all {d[f + '_resp'].mean():>6.2f}   "
              f"el-wrong {d.loc[hi, f + '_resp'].mean():>6.2f}   "
              f"el-right {d.loc[~hi, f + '_resp'].mean():>6.2f}")

    print("\nel-wrong count by true az:")
    g = d[hi].groupby("az").size()
    print("  " + "  ".join(f"az{a}:{n}" for a, n in g.items()))

    out = os.path.join(config.ROOT, "data", "outputs",
                       "viewpoint_el_probe.csv")
    d.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
