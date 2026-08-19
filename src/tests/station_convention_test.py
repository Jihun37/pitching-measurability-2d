"""
Diamond - Station classifier: landmark-convention robustness probe.

Finding that motivated this (2026-07-04): with RTMPose coords,
pitching_lateral_02 flips from side(0.68) to reject(0.76). Neighbor analysis
showed the flip is driven by ONE feature, shhip (shoulder/hip width ratio,
-0.50 sigma) — a systematic landmark-convention difference between backbones
(MediaPipe vs Halpe26 vs the OBP mocap markers the classifier is trained on),
not a detection error. The clip sits at the side/reject elevation boundary,
so the convention offset flips the kNN vote.

Experiments (both evaluated in the original 5-fold grouped-CV frame):
  1. AUGMENT: add convention-perturbed copies of each training projection
     (shoulder/hip chord widths scaled U(0.85,1.15) about their midpoints,
     head vertical offset U(-0.15,0.15) of torso length) so the kNN boundary
     becomes insensitive to backbone conventions. Test folds stay CLEAN.
  2. ABLATE: drop shhip entirely (6 features).

Success criteria: OBP CV accuracy/safety preserved AND lateral_02 (rtmp)
returns to side while frontier_03 stays front.

Run:  cd src\tests
      python station_convention_test.py [--limit 250] [--aug-k 2]
"""
import os, sys, argparse, zlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(HERE, "..", "stage3"))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import config
import obp_project as O
import metrics as M
from hss_elevation_test import project_cam
from station_classify import (features, station, FEATS, AZ, EL, gkf, knn,
                              perturb_convention)

USABLE = {"side", "front", "overhead"}
REAL = [("pitching_lateral_02", "side"), ("pitching_frontier_03", "front")]

# NOTE (2026-07-04): this experiment led to adoption — perturb_convention and
# the shhip removal now live in station_classify itself (imported above).
# The FEATS list here therefore no longer contains shhip; the "ablate"
# variants below reproduce the adopted config, the "7f" baselines require
# temporarily re-adding shhip and are kept for the historical record.
FEATS7 = ["shhip"] + list(FEATS)


def build(limit, aug_k):
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    rows, done = [], 0
    for r in md.itertuples(index=False):
        if limit and done >= limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            continue
        try:
            joints, fps = O.load_c3d_joints(path)
            arm = O.detect_throwing_arm(joints, fps)
        except Exception:
            continue
        for az in AZ:
            for el in EL:
                try:
                    df = project_cam(joints, az, el)
                    base = {"session_pitch": r.session_pitch,
                            "y": station(az, el)}
                    f = features(df, arm)
                    rows.append({**f, **base, "aug": 0})
                    for a_i in range(aug_k):
                        # crc32 for cross-process reproducibility (see
                        # station_classify.build)
                        rng = np.random.default_rng(zlib.crc32(
                            f"{r.session_pitch}|{az}|{el}|{a_i}".encode()))
                        f = features(perturb_convention(df, rng), arm)
                        rows.append({**f, **base, "aug": 1})
                except Exception:
                    pass
        done += 1
        if done % 50 == 0:
            print(f"  ...{done} pitches", flush=True)
    print(f"processed {done} pitches -> {len(rows)} rows")
    return pd.DataFrame(rows)


def real_vec(name, feats):
    p = os.path.join(config.ROOT, "data", "outputs", name)
    out = {}
    for sfx, tag in (("", "mediapipe"), ("_rtmp", "rtmp")):
        df = pd.read_csv(os.path.join(p, f"{name}_smoothed{sfx}.csv"))
        if "nose_x" in df.columns and "head_x" not in df.columns:
            df = df.rename(columns={"nose_x": "head_x", "nose_y": "head_y",
                                    "nose_v": "head_v"})
        f = features(df, "right")   # both clips: arm=right on both backbones
        out[tag] = np.array([f[k] for k in feats], float)
    return out


def evaluate(data, feats, use_aug, label):
    d = data.dropna(subset=feats).reset_index(drop=True)
    clean = d[d.aug == 0].reset_index(drop=True)
    pool = d if use_aug else clean

    # grouped CV: test on clean rows only; train pool from other groups
    Xc = clean[feats].to_numpy(float)
    yc = clean["y"].to_numpy()
    gc = clean["session_pitch"].to_numpy()
    pred = np.empty(len(clean), dtype=yc.dtype)
    for tr_idx, te_idx in gkf(gc, 5):
        te_groups = set(gc[te_idx].tolist())
        trp = pool[~pool.session_pitch.isin(te_groups)]
        pred[te_idx], _ = knn(trp[feats].to_numpy(float),
                              trp["y"].to_numpy(), Xc[te_idx])
    acc = float((pred == yc).mean())
    rej = yc == "reject"
    fa = float(np.isin(pred[rej], list(USABLE)).mean())
    us = np.isin(yc, list(USABLE))
    oc = float((pred[us] == "reject").mean())

    # real-video predictions with the full training pool
    Xtr = pool[feats].to_numpy(float)
    ytr = pool["y"].to_numpy()
    reals = []
    for name, expected in REAL:
        vs = real_vec(name, feats)
        for tag in ("mediapipe", "rtmp"):
            lab, shr = knn(Xtr, ytr, vs[tag][None, :])
            ok = "OK" if lab[0] == expected else "MISS"
            reals.append(f"{name.split('_')[1][:7]}/{tag[:4]}={lab[0]}({shr[0]:.2f}){ok}")
    print(f"{label:24s} acc={acc:.3f}  false-accept={fa:.3f}  "
          f"over-caution={oc:.3f}")
    print(f"{'':24s} " + "  ".join(reals))
    return acc, fa, oc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--aug-k", type=int, default=2)
    a = ap.parse_args()

    data = build(a.limit, a.aug_k)
    feats6 = [f for f in FEATS7 if f != "shhip"]
    print(f"\n{'variant':24s} OBP 5-fold CV (test folds always clean)")
    print("-" * 100)
    evaluate(data, FEATS7, False, "baseline 7f")
    evaluate(data, feats6, False, "ablate shhip 6f")
    evaluate(data, FEATS7, True, "augment 7f")
    evaluate(data, feats6, True, "augment+ablate 6f")


if __name__ == "__main__":
    main()
