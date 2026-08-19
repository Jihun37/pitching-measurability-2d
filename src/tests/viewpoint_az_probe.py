"""
Diamond - FOLDED-AZIMUTH residual diagnostic probe (read-only).

After the fold+bits fix the symmetry is solved (family correct 54/60) but
the folded [0,90] bin itself is wrong on 42/60 (exact 18/60, mostly +-1-2
bins). The el probe (viewpoint_el_probe.py) found a systematic real-vs-OBP
feature domain shift (-0.4..-1.5 sd); this probe asks whether that shift
explains the folded-bin residual, and which feature is the culprit.

Read-only questions:
  1. RESOLUTION: bank gap between ADJACENT folded 15-deg bins (el=0
     shelves, global-sd units) - how big is the az signal vs the shift?
  2. DIRECTION: signed folded-bin error per clip / per true-az family -
     does the shift push consistently (e.g. toward fold 0 or fold 90)?
  3. CULPRIT: leave-one-feature-out kNN refit - which single feature,
     when dropped, recovers the most folded bins?

Run:  cd src\tests
      python viewpoint_az_probe.py
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


def top_fold_bin(naz, nel):
    fpairs, fcnts = np.unique([(fold(a), e) for a, e in zip(naz, nel)],
                              axis=0, return_counts=True)
    i = np.argmax(fcnts)
    return int(fpairs[i][0]), float(fcnts[i]) / len(naz)


def main():
    bank = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR,
                                    "viewpoint_zone_features.csv")
                       ).dropna(subset=FEATS).reset_index(drop=True)
    bank["azf"] = bank.az.map(fold)
    clips = [(f"angle{i:02d}_{p:02d}", (30 * i) % 360)
             for i in range(12) for p in range(5)]

    # clip features once
    feats = {}
    for name, _ in clips:
        df, _r = load_clip(name, "_rtmp")
        arm = detect_arm_2d(df)
        feats[name] = features(df, arm)

    # ---- 1. bank adjacent-fold-bin resolution at el=0 ----------------
    sd_g = bank[FEATS].std() + 1e-9
    shelf = bank[bank.el == 0].groupby("azf")[FEATS].mean()
    print("bank el=0 shelves: gap to NEXT fold bin (global-sd units)")
    bins = sorted(shelf.index)
    print(f"{'bin pair':>12}" + "".join(f"{f:>9}" for f in FEATS) + f"{'|L2|':>7}")
    for a, b in zip(bins[:-1], bins[1:]):
        g = (shelf.loc[b] - shelf.loc[a]) / sd_g
        print(f"{a:>5} ->{b:>4}" + "".join(f"{g[f]:>9.2f}" for f in FEATS)
              + f"{np.linalg.norm(g):>7.2f}")

    # ---- kNN helper over a feature subset ----------------------------
    def vote_all(sub):
        X = bank[sub].to_numpy(float)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        tree = cKDTree((X - mu) / sd)
        naz_b = bank.az.to_numpy(int)
        nel_b = bank.el.to_numpy(int)
        out = {}
        for name, _ in clips:
            xv = np.array([feats[name][k] for k in sub], float)
            _, idx = tree.query((xv - mu) / sd, k=K)
            out[name] = top_fold_bin(naz_b[idx], nel_b[idx])
        return out

    base = vote_all(FEATS)

    # ---- 2. per-clip signed folded error + shelf geometry ------------
    print("\nper clip: true fold, voted fold (share), signed err,"
          " nearest el0 shelf")
    print(f"{'clip':14s}{'az':>4}{'foldT':>6}{'foldP':>6}{'share':>7}"
          f"{'err':>5}{'nearest':>8}")
    rows = []
    Xs = ((shelf - bank[FEATS].mean()) / sd_g)     # standardized shelves
    for name, az in clips:
        ft, (fp, share) = fold(az), base[name]
        xv = np.array([(feats[name][k] - bank[k].mean()) / sd_g[k]
                       for k in FEATS])
        dist = ((Xs - xv) ** 2).sum(axis=1) ** 0.5
        near = int(dist.idxmin())
        err = fp - ft
        rows.append({"clip": name, "az": az, "foldT": ft, "foldP": fp,
                     "share": share, "err": err, "near0": near})
        print(f"{name:14s}{az:>4}{ft:>6}{fp:>6}{share:>7.2f}{err:>+5d}"
              f"{near:>8}")
    d = pd.DataFrame(rows)

    print("\nsigned folded-bin error by true az family:")
    for ft in sorted(d.foldT.unique()):
        sub = d[d.foldT == ft]
        print(f"  fold{ft:>3} (n={len(sub):2d}): mean {sub.err.mean():+6.1f}"
              f"  median {sub.err.median():+5.0f}"
              f"  exact {(sub.err == 0).sum()}/{len(sub)}"
              f"  errs {sorted(sub.err)}")
    print(f"\noverall: exact {(d.err == 0).sum()}/60"
          f"  +-15 {(d.err.abs() <= 15).sum()}/60"
          f"  mean signed {d.err.mean():+.1f}"
          f"  (toward 90: {(d.err > 0).sum()}, toward 0: {(d.err < 0).sum()})")
    print(f"nearest-el0-shelf agrees with vote: "
          f"{(d.near0 == d.foldP).sum()}/60; nearest exact-correct: "
          f"{(d.near0 == d.foldT).sum()}/60")

    # ---- 3. leave-one-feature-out ------------------------------------
    print("\nleave-one-feature-out (folded bin vs GT):")
    print(f"{'dropped':>10}{'exact':>8}{'+-1bin':>8}")
    ex = (d.err == 0).sum()
    e1 = (d.err.abs() <= 15).sum()
    print(f"{'-none-':>10}{ex:>5}/60{e1:>5}/60")
    for f in FEATS:
        sub = [k for k in FEATS if k != f]
        v = vote_all(sub)
        err = np.array([v[n][0] - fold(a) for n, a in clips])
        print(f"{f:>10}{(err == 0).sum():>5}/60{(np.abs(err) <= 15).sum():>5}/60")

    out = os.path.join(config.ROOT, "data", "outputs",
                       "viewpoint_az_probe.csv")
    d.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
