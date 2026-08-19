"""
Diamond - STAGE-1 viewpoint-fix probe (read-only): folded-symmetry kNN +
sign-signal unfolding.

Diagnosis (2026-07-11, 60-clip holdout, clean track poses): the plain kNN
viewpoint vote is right within 30deg on only 24/60 clips, BUT 54/60 are
correct up to the 4-fold pose symmetry {az, -az, az+180, 180-az} - the
event-free ratio features are symmetry-BLIND by construction, and detector
noise kills the residual asymmetries that separated the families on clean
OBP projections.

Stage-1 design (no new tuned constants):
  1. FOLD: each kNN neighbor votes with its folded azimuth
     fold(az) = min(az%180, 180-az%180) in [0,90] x el -> top bin az_f.
  2. UNFOLD: the 4 candidates {az_f, 180-az_f, 180+az_f, 360-az_f} carry
     the 4 sign combinations of (cos az, sin az). Two real-video signals
     decide the signs:
       - sin sign (front/rear hemisphere): shoulder-line CHIRALITY -
         image-x sign of (right_shoulder - left_shoulder) at the release
         candidate. RTMPose labels anatomical left/right from appearance,
         so this sign carries the true chirality the dimensionless ratio
         features throw away; ~ -sin(az) on this setup (SHO handedness).
         Verified 49/50 sign-correct on the holdout (only miss: a 3 px
         zero-margin read; sin=0 side views excluded - the bit is unused
         there). The earlier face-visibility bit failed at the barely-rear
         wedge az195-225 (frozen 0.95 gate was validated on az240-300
         only); kept in the report columns for reference.
       - cos sign (which side line): stride direction in image x - at foot
         plant the lead ankle is toward home of the trail ankle; the image
         sign of that world direction is proportional to cos(az) times a
         fixed camera handedness (HAND, a coordinate convention, not a
         tuned parameter).
     Candidate score = |sin c| * (+-1 face agree) + |cos c| * (+-1 stride
     agree); the |.| weights make the degenerate orbits (az_f = 0 or 90,
     2 candidates) decide by the informative bit automatically.

Scored against the camera-position GT az = 30n (official convention 2026-07-15) (60-clip
holdout) and nominal_az (15-clip dev set, transfer check).

Run:  cd src\tests
      python viewpoint_fold_probe.py [--dev15]
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("..", "../stage2", "../analysis", "../deploy"):
    sys.path.insert(0, os.path.join(_HERE, p))

import config
import metrics as M
from station_classify import features, FEATS
from real_station_test import detect_arm_2d
from measure_auto import load_clip
from release_offset import _face_hidden
from scipy.spatial import cKDTree

K = 25
# Camera handedness: image-x sign of the world home direction at cos(az)>0
# views. Fixed coordinate convention (one global sign for the entire
# projection/recording setup), determined once from the az180 side clips
# (angle00: stride toward home reads image dx>0 while cos(180)=-1 -> -1).
# 2026-07-15 az-convention fix: deploy/viewpoint_fold.py uses HAND=+1, SHO=-1
# after the projection was mirrored to the official convention. This probe's
# HAND was left at the pre-fix -1, which flips the cos (stride) bit and makes
# unfold pick the wrong symmetry member. Aligned to the deployed constants.
HAND = 1.0
SHO = -1.0


def fold(az):
    a = az % 180
    return int(min(a, 180 - a))


def video_fps(name):
    for e in (".mov", ".MOV", ".mp4"):
        p = os.path.join(config.ROOT, "data", "videos", name + e)
        if os.path.exists(p):
            cap = cv2.VideoCapture(p)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps and fps > 0:
                return fps
    return config.FPS_DEFAULT


def stride_dx(df, arm, fps, rel):
    """Image-x displacement lead ankle - trail ankle over foot plant ->
    release (median). Sign ~ HAND * cos(az)."""
    lead = "left" if arm == "right" else "right"
    trail = "right" if lead == "left" else "left"
    fp = M.foot_plant_frame(df, lead, fps, M.JOINTS, rel)
    lo, hi = int(fp), max(int(fp) + 1, int(rel))
    d = (df[f"{lead}_ankle_x"].to_numpy(float)[lo:hi + 1]
         - df[f"{trail}_ankle_x"].to_numpy(float)[lo:hi + 1])
    return float(np.nanmedian(d))


def classify(df, raw, arm, fps, tree, naz_bank, nel_bank):
    """Returns (az_pred, el_pred, az_f, face_vis, dx, plain_az, plain_el)."""
    xv = np.array([features(df, arm)[k] for k in FEATS], float)
    _, idx = tree[0].query((xv - tree[1]) / tree[2], k=K)
    naz, nel = naz_bank[idx], nel_bank[idx]

    # plain top bin (deployed behavior, for the side-by-side)
    pairs, cnts = np.unique(list(zip(naz, nel)), axis=0, return_counts=True)
    plain_az, plain_el = map(int, pairs[np.argmax(cnts)])

    # folded vote
    fpairs, fcnts = np.unique([(fold(a), e) for a, e in zip(naz, nel)],
                              axis=0, return_counts=True)
    az_f, el_pred = map(int, fpairs[np.argmax(fcnts)])

    # sign bits
    rel_side = M.release_frame(df, arm, fps, M.JOINTS, view="side",
                               raw_df=raw)
    face_vis = not _face_hidden(df, rel_side)      # reported, not used
    dx = stride_dx(df, arm, fps, rel_side)
    lo, hi = max(0, rel_side - 2), min(len(df), rel_side + 3)
    sho = float(np.nanmedian(
        df["right_shoulder_x"].to_numpy(float)[lo:hi]
        - df["left_shoulder_x"].to_numpy(float)[lo:hi]))

    cands = sorted({az_f % 360, (180 - az_f) % 360,
                    (180 + az_f) % 360, (360 - az_f) % 360})
    best, best_s = None, -np.inf
    for c in cands:
        r = np.radians(c)
        s = (abs(np.sin(r)) * (1 if (np.sin(r) * SHO > 0) == (sho > 0)
                               else -1)
             + abs(np.cos(r)) * (1 if (np.cos(r) * HAND > 0) == (dx > 0)
                                 else -1))
        if s > best_s:
            best, best_s = c, s
    return best, el_pred, az_f, face_vis, dx, sho, plain_az, plain_el


def circ(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def run(clips, label):
    bank = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR,
                                    "viewpoint_zone_features.csv")
                       ).dropna(subset=FEATS).reset_index(drop=True)
    Xtr = bank[FEATS].to_numpy(float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    tree = (cKDTree((Xtr - mu) / sd), mu, sd)
    naz_bank = bank["az"].to_numpy(int)
    nel_bank = bank["el"].to_numpy(int)

    print(f"\n=== {label} ===")
    print(f"{'clip':22s}{'true':>5}{'plain':>7}{'foldF':>6}{'face':>6}"
          f"{'sho':>7}{'dx':>8}{'pred':>6}{'err':>5}")
    e_plain, e_new, fold_ok = [], [], 0
    for name, true_az in clips:
        df, raw = load_clip(name, "_rtmp")
        arm = detect_arm_2d(df)
        fps = video_fps(name)
        pred, el, az_f, face, dx, sho, paz, pel = classify(
            df, raw, arm, fps, tree, naz_bank, nel_bank)
        ok_f = fold(true_az) == az_f
        fold_ok += ok_f
        e_plain.append(circ(paz, true_az))
        e_new.append(circ(pred, true_az))
        print(f"{name:22s}{true_az:>5}{paz:>7}{az_f:>5}{'*' if ok_f else '!':1s}"
              f"{'Y' if face else 'n':>5}{sho:>7.0f}{dx:>8.0f}{pred:>6}"
              f"{e_new[-1]:>5}")
    e_plain, e_new = np.array(e_plain), np.array(e_new)
    n = len(clips)
    for tag, e in (("plain", e_plain), ("fold+bits", e_new)):
        print(f"[{tag:9s}] exact {int((e == 0).sum())}/{n}"
              f"  <=15 {int((e <= 15).sum())}/{n}"
              f"  <=30 {int((e <= 30).sum())}/{n}"
              f"  median {np.median(e):.0f}deg"
              f"  >=90 {int((e >= 90).sum())}")
    print(f"folded-az bin correct: {fold_ok}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev15", action="store_true",
                    help="also run the 15-clip dev set (nominal_az GT)")
    a = ap.parse_args()

    # Exclude the 4 sidearm-delivery clips (one each at the cardinal angles
    # 0/90/180/270); their pro-form prior is broken, out of scope (user).
    SIDEARM = {"angle00_02", "angle03_00", "angle06_01", "angle09_02"}
    clips = [(f"angle{i:02d}_{p:02d}", (30 * i) % 360)
             for i in range(12) for p in range(5)
             if f"angle{i:02d}_{p:02d}" not in SIDEARM]
    run(clips, "HOLDOUT 56 (sidearm excluded)")

    if a.dev15:
        gt = pd.read_csv(os.path.join(config.ROOT, "data", "outputs",
                                      "release_gt_real15.csv"))
        dev = [(r.clip, int(r.nominal_az)) for r in gt.itertuples()]
        run(dev, "DEV 15 (nominal_az)")


if __name__ == "__main__":
    main()
