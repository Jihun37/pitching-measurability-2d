"""
Diamond - 30-DEG FOLDED-CLASS classifier probe (read-only).

User-specified experiment (2026-07-12): keep the dev-15 global domain-shift
calibration, but make the kNN vote compete over the FOUR folded classes
{0, 30, 60, 90} from the start (not a 15-deg bin relabeled at report time).
Mid-grid neighbors (fold 15/45/75) are handled by PRE-DEFINED rules only
(no GT-tuned thresholds):

  split : 0.5 / 0.5 to the two adjacent 30-deg classes (a fold-15 neighbor
          is angularly equidistant from 0 and 30 - the only tuning-free
          "distance-based" split on this grid)
  drop  : mid-grid neighbors do not vote
  idw   : split rule, neighbor weight = 1 / (feature-space distance)

Frozen: sign bits / unfold (deploy.viewpoint_fold), FEATS, k=25, bank,
coordinate conventions, release detectors (frames come from the evidence
ledger, not re-run).

Configs compared end-to-end (az -> release-method map -> release frames):
  A baseline   wired fold+bits (15-deg joint bin, no calibration)
  B cal15      calibration only, same 15-deg mechanics
  C cal30-*    calibration + 30-deg folded-class vote (3 mid-bin rules)

Run:  cd src\tests
      python viewpoint_fold30_probe.py
"""
import os, sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("..", "../stage2", "../analysis", "../deploy"):
    sys.path.insert(0, os.path.join(_HERE, p))

import config
import metrics as M
from station_classify import features, FEATS
from real_station_test import detect_arm_2d
from measure_auto import load_clip
from viewpoint_fold import fold, unfold, sign_bits
from viewpoint_fold_probe import video_fps
from scipy.spatial import cKDTree

K = 25
OUT = os.path.join(config.ROOT, "data", "outputs")
METHODS = ["side", "side_off", "frontal", "deploy", "ballsep"]


def circ(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def arc30(az):
    """Snap an azimuth to the release-map 30-deg arc (predefined
    half-up rule for 15-deg midpoints, applied to every config)."""
    return int((az + 15) // 30 * 30) % 360


def class_vote(folds, w, rule):
    """Accumulate folded-neighbor weights into classes {0,30,60,90}."""
    acc = {0: 0.0, 30: 0.0, 60: 0.0, 90: 0.0}
    for f, wi in zip(folds, w):
        if f % 30 == 0:
            acc[f] += wi
        elif rule in ("split", "idw"):
            acc[f - 15] += wi / 2
            acc[f + 15] += wi / 2
        # rule == "drop": mid-grid neighbor ignored
    return max(sorted(acc), key=lambda c: acc[c])


def main():
    bank = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR,
                                    "viewpoint_zone_features.csv")
                       ).dropna(subset=FEATS).reset_index(drop=True)
    X = bank[FEATS].to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    tree = cKDTree((X - mu) / sd)
    naz_b = bank.az.to_numpy(int)
    nel_b = bank.el.to_numpy(int)
    shelf0 = bank.assign(azf=bank.az.map(fold))
    shelf0 = shelf0[shelf0.el == 0].groupby("azf")[FEATS].mean()

    # dev-15 global calibration vector (identical to viewpoint_calib_probe)
    gt15 = pd.read_csv(os.path.join(OUT, "release_gt_real15.csv"))
    deltas = []
    for r in gt15.itertuples():
        df, _ = load_clip(r.clip, "_rtmp")
        f = features(df, detect_arm_2d(df))
        azf = min(shelf0.index, key=lambda b: abs(b - fold(int(r.nominal_az))))
        deltas.append([f[k] - shelf0.loc[azf][k] for k in FEATS])
    off = np.mean(deltas, axis=0)

    # release-map routing assets (frames from the FROZEN evidence ledger)
    mp = pd.read_csv(os.path.join(OUT, "release_method_map.csv")
                     ).set_index("az")
    ev = pd.read_csv(os.path.join(OUT, "release_method_evidence.csv")
                     ).set_index("clip")

    clips = [(f"angle{i:02d}_{p:02d}", (30 * i) % 360)
             for i in range(12) for p in range(5)]

    # per-clip: features, bits, kNN neighborhood (query once per config set)
    per = {}
    for name, az in clips:
        df, raw = load_clip(name, "_rtmp")
        arm = detect_arm_2d(df)
        fps = video_fps(name)
        dx, sho = sign_bits(df, raw, arm, fps)
        xv = np.array([features(df, arm)[k] for k in FEATS], float)
        per[name] = {"az": az, "xv": xv, "dx": dx, "sho": sho}

    def bin15_az(xq, dx, sho):
        """Wired mechanics: per-neighbor unfold -> top joint (az,el) bin."""
        _, idx = tree.query((xq - mu) / sd, k=K)
        naz = np.array([unfold(a, dx, sho) for a in naz_b[idx]])
        pairs, cnts = np.unique(list(zip(naz, nel_b[idx])), axis=0,
                                return_counts=True)
        return int(pairs[np.argmax(cnts)][0])

    def cls30_az(xq, dx, sho, rule):
        d, idx = tree.query((xq - mu) / sd, k=K)
        folds = [fold(a) for a in naz_b[idx]]
        w = 1.0 / (d + 1e-9) if rule == "idw" else np.ones(K)
        c = class_vote(folds, w, rule)
        return int(unfold(c, dx, sho))

    preds = {}
    for name, _az in clips:
        p = per[name]
        preds.setdefault("A_baseline", {})[name] = \
            bin15_az(p["xv"], p["dx"], p["sho"])
        xc = p["xv"] - off
        preds.setdefault("B_cal15", {})[name] = \
            bin15_az(xc, p["dx"], p["sho"])
        for rule in ("split", "drop", "idw"):
            preds.setdefault(f"C_cal30_{rule}", {})[name] = \
                cls30_az(xc, p["dx"], p["sho"], rule)

    # ---- scoring -------------------------------------------------------
    print(f"{'config':16s}{'exact':>6}{'15off':>6}{'30off':>6}{'>=60':>6}"
          f"{'med':>6}{'mean':>7}{'90/270':>8}{'mapAgr':>8}"
          f"{'rel<=1f':>8}{'rel<=2f':>8}{'n':>4}")
    rows = []
    for cfg, pv in preds.items():
        errs, e9070, agree, r1, r2, n = [], [], 0, 0, 0, 0
        for name, az in clips:
            pa = pv[name]
            e = circ(pa, az)
            errs.append(e)
            if az in (90, 270):
                e9070.append(e)
            m_pred = mp.loc[arc30(pa)].method
            m_gt = mp.loc[az].method
            agree += m_pred == m_gt
            fr = ev.loc[name][m_pred]
            if pd.isna(fr):
                fr = ev.loc[name]["deploy"]
            if not pd.isna(fr):
                tr = ev.loc[name]["true_release"]
                n += 1
                r1 += abs(int(fr) - tr) <= 1
                r2 += abs(int(fr) - tr) <= 2
        errs = np.array(errs)
        e9070 = np.array(e9070)
        row = {"config": cfg,
               "exact": int((errs == 0).sum()),
               "off15": int(((errs > 0) & (errs <= 15)).sum()),
               "off30": int(((errs > 15) & (errs <= 30)).sum()),
               "ge60": int((errs >= 60).sum()),
               "med": float(np.median(errs)),
               "mean": float(errs.mean()),
               "x9070": int((e9070 == 0).sum()),
               "mapAgree": agree, "rel1": r1, "rel2": r2, "n": n}
        rows.append(row)
        print(f"{cfg:16s}{row['exact']:>6}{row['off15']:>6}{row['off30']:>6}"
              f"{row['ge60']:>6}{row['med']:>6.0f}{row['mean']:>7.1f}"
              f"{row['x9070']:>7}/10{agree:>7}/60"
              f"{r1:>8}{r2:>8}{n:>4}")

    # per-clip dump of the best-looking C config for inspection
    dump = pd.DataFrame(
        [{"clip": name, "az_true": az,
          **{cfg: pv[name] for cfg, pv in preds.items()}}
         for name, az in clips])
    out = os.path.join(OUT, "viewpoint_fold30_probe.csv")
    dump.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
