"""
Diamond - Real-video test of the station classifier.

Train the 4-class station classifier on OBP projections, then predict on ACTUAL
smartphone pitch videos (MediaPipe 2D coords). This is the real domain-gap test:
OBP is clean orthographic projection of 3D markers; the phone videos have
perspective, lens, and real MediaPipe detector error.

Expectation:
  pitching_lateral_02  -> side   (filmed from the side, ground level)
  pitching_frontier_03 -> front  (filmed from the front, ground level)

Features are the same event-free dimensionless ratios used in training, so the
meter(OBP) vs pixel(video) scale difference cancels.

Run:  cd src\analysis
      python real_station_test.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import config
import metrics as M
from station_classify import build, features, knn, FEATS

SUFFIX = ""   # set from --backbone in main()

TRAIN_LIMIT = 250          # OBP pitches for the training set (kNN needs no more)
VIDEOS = [
    ("pitching_lateral_02",  "side"),
    ("pitching_frontier_03", "front"),
]


def detect_arm_2d(df):
    """Throwing arm from the LEAD-LEG knee lift (throwing arm = opposite of
    the lifted leg). The knee lift is the pitch's biggest, slowest,
    best-tracked motion, so it survives views where the wrist race breaks.

    Replaces the wrist-speed p95 race (2026-07-11): on the 60-clip holdout
    the wrist rule mirror-flipped 8 rear clips (a teleporting glove-wrist
    track wins the race on garbage); the knee rule is correct on 73/75
    real clips (60 holdout + 15 dev) and its 2 misses are broken-pose clips
    that the wrist-glitch gate defers anyway. Percentiles only - no tuned
    constants. Falls back to the wrist race if knee/hip data is unusable."""
    try:
        hy = (df["left_hip_y"].to_numpy(float)
              + df["right_hip_y"].to_numpy(float)) / 2.0

        def lift(side):
            h = hy - df[f"{side}_knee_y"].to_numpy(float)   # up = positive
            h = h[np.isfinite(h)]
            if not len(h):
                raise ValueError("no knee data")
            return np.percentile(h, 98) - np.percentile(h, 50)

        # lead leg lifts highest; throwing arm is the opposite side
        return "right" if lift("left") >= lift("right") else "left"
    except (KeyError, ValueError):
        def pk(j):
            x = df[f"{j}_x"].to_numpy(float)
            y = df[f"{j}_y"].to_numpy(float)
            s = np.hypot(np.diff(x), np.diff(y))
            s = s[np.isfinite(s)]
            return np.percentile(s, 95) if len(s) else 0.0
        return "right" if pk("right_wrist") >= pk("left_wrist") else "left"


def load_video(name):
    p = os.path.join(config.OUTPUT_DIR.replace(config.VIDEO_NAME, name),
                     f"{name}_smoothed{SUFFIX}.csv")
    df = pd.read_csv(p)
    if "nose_x" in df.columns and "head_x" not in df.columns:
        df = df.rename(columns={"nose_x": "head_x", "nose_y": "head_y",
                                "nose_v": "head_v"})
    return df


def main():
    global SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="rtmp", choices=["rtmp", "mediapipe"],
                    help="which extraction's smoothed CSVs to load")
    a = ap.parse_args()
    SUFFIX = "_rtmp" if a.backbone == "rtmp" else ""
    print(f"[real-video station test]  backbone={a.backbone}  "
          f"training on {TRAIN_LIMIT} OBP pitches\n")
    train = build(TRAIN_LIMIT, 0.0).dropna(subset=FEATS).reset_index(drop=True)
    Xtr = train[FEATS].to_numpy(float)
    ytr = train["y"].to_numpy()

    # per-feature OBP training range (to spot out-of-distribution real values)
    lo = np.nanpercentile(Xtr, 2, axis=0)
    hi = np.nanpercentile(Xtr, 98, axis=0)

    print(f"{'video':22s}{'arm':6s}{'pred':10s}{'share':7s}{'expected':10s}  result")
    print("-" * 70)
    rows = []
    for name, expected in VIDEOS:
        df = load_video(name)
        arm = detect_arm_2d(df)
        f = features(df, arm)
        xv = np.array([f[k] for k in FEATS], float)
        lab, shr = knn(Xtr, ytr, xv[None, :], k=25)
        pred, share = lab[0], shr[0]
        ok = "OK" if pred == expected else "MISS"
        print(f"{name:22s}{arm:6s}{pred:10s}{share:>5.2f}  {expected:10s}  {ok}")
        rows.append((name, expected, pred, share, f, xv))

    # feature diagnostics: is each real value inside the OBP training range?
    print("\n[feature check]  real value vs OBP 2-98% range (out-of-range = domain gap)")
    hdr = f"{'feature':10s}" + "".join(f"{n.split('_')[0][:8]:>12s}" for n, *_ in rows)
    print(hdr + f"{'OBP range':>22s}")
    for j, feat in enumerate(FEATS):
        line = f"{feat:10s}"
        for (_n, _e, _p, _s, _f, xv) in rows:
            flag = "" if lo[j] <= xv[j] <= hi[j] else "*"
            line += f"{xv[j]:>11.2f}{flag:1s}"
        line += f"   [{lo[j]:6.2f},{hi[j]:6.2f}]"
        print(line)
    print("  (* = outside OBP training range)")

    # show the metrics each predicted station would output
    print("\n[metrics at predicted station]")
    for (name, expected, pred, share, f, xv) in rows:
        df = load_video(name)
        arm = detect_arm_2d(df)
        view = "frontal" if pred == "front" else "side"
        cand = M.compute_candidates(df, fps=config.FPS_DEFAULT, arm=arm, view=view)
        vals = {k: v for k, (v, _) in cand.items()}
        if pred == "front":
            keys = ["arm_slot"]
        elif pred == "side":
            keys = ["lead_knee_angle", "stride_pct_height", "trunk_anterior_tilt",
                    "knee_ext_velo_br", "wrist_speed"]
        else:
            keys = []
        print(f"\n  {name}  -> {pred}  (arm={arm})")
        for k in keys:
            print(f"    {k:22s} {vals.get(k, float('nan')):8.2f}")


if __name__ == "__main__":
    main()
