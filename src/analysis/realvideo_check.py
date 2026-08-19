"""
Domain check on a real recording.

Runs the coordinate CSV that stage1 and stage2 produced from a phone clip
through metrics, and asks whether the reliable quantities land inside the
distribution the OBP projections produce (candidate_features_obp.csv). Landing
inside it means the recording is in the same domain the validation was done in,
and the quantities can be trusted on it.

Usage:
    python realvideo_check.py --coords ../../data/outputs/test_03/test_03_smoothed.csv --fps 30
"""
import os, sys, argparse
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
import metrics as M
from smoother import smooth_coordinates

REF = os.path.join(config.OBP_VALIDATION_DIR, "candidate_features_obp.csv")
ARMSLOT_REF = os.path.join(config.OBP_VALIDATION_DIR, "armslot_ref.csv")
# The five validated side-view (azimuth 0) quantities -> their reference column
# in the OBP projected distribution.
# Note: the reference CSV (candidate_features_obp.csv) has to have been
#       regenerated under the current metrics definitions, by rerunning
#       obp_project.py in batch. knee_ext_velo_br and wrist_speed are absent
#       from older copies and show as having no reference until it is.
CORE = {
    "lead_knee_angle":     "lead_knee_at_release",   # absolute angle; the reference uses the alias
    "stride_pct_height":   "stride_pct_height",
    "trunk_anterior_tilt": "lateral_trunk_tilt",     # same value; the reference uses the alias column
    "knee_ext_velo_br":    "knee_ext_velo_br",
    "wrist_speed":         "wrist_speed",
}


def detect_arm(df, fps):
    """Detect the throwing arm, robustly against noise.

    The single fastest frame can be one raw-coordinate spike, so the 95th
    percentile is used instead. That is what fixed a left-handed clip where a
    glove-hand spike named the wrong arm.
    """
    def pk(j):
        x = df[f"{j}_x"].to_numpy(float); y = df[f"{j}_y"].to_numpy(float)
        s = np.hypot(np.diff(x), np.diff(y))
        s = s[np.isfinite(s)]
        return np.percentile(s, 95) if len(s) else 0.0
    return "right" if pk("right_wrist") >= pk("left_wrist") else "left"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", required=True, help="coordinate CSV from stage1/stage2")
    ap.add_argument("--fps", type=float, default=120.0)
    ap.add_argument("--arm", default=None, help="right/left; detected when omitted")
    ap.add_argument("--smooth", type=int, default=0,
                    help="SG window in frames; 0 for an already smoothed csv")
    ap.add_argument("--view", default="side",
                    help="side = trunk, knee and stride; frontal = arm slot")
    ap.add_argument("--release", type=int, default=None,
                    help="set the release frame by hand, for a dry-form clip where detection has no ball to find")
    a = ap.parse_args()

    df = pd.read_csv(a.coords)
    if "_coords" in os.path.basename(a.coords) and not a.smooth:
        print("[warning] using raw coordinates (_coords.csv). A noise spike can\n"              "          mislead the arm and release detection. Prefer _smoothed.csv, or --smooth 7.\n")
    # Our pose_extractor stores the head as 'nose'; metrics expects 'head'.
    if "nose_x" in df.columns and "head_x" not in df.columns:
        df = df.rename(columns={"nose_x": "head_x", "nose_y": "head_y", "nose_v": "head_v"})
    if a.smooth and a.smooth >= 5:
        df = smooth_coordinates(df, window=a.smooth)

    arm = a.arm or detect_arm(df, a.fps)
    if not a.arm:
        print(f"[note] throwing arm detected as {arm}. Override with --arm left/right.")
    # Where the raw coordinates sit beside the smoothed ones, they refine the
    # release frame (see release_frame).
    raw_path = a.coords.replace("_smoothed", "_coords")
    raw_df = None
    if raw_path != a.coords and os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        if "nose_x" in raw_df.columns and "head_x" not in raw_df.columns:
            raw_df = raw_df.rename(columns={"nose_x": "head_x", "nose_y": "head_y"})
    cand = M.compute_candidates(df, fps=a.fps, arm=arm, view=a.view, raw_df=raw_df)
    vals = {k: v for k, (v, _) in cand.items()}

    print(f"throwing arm: {arm} / fps: {a.fps:.0f} / frames: {len(df)} / view: {a.view}\n")

    if a.view == "frontal":
        # frontal: arm slot, shoulder->hand against the vertical, against the
        # OBP 3D coronal truth distribution
        ours = vals.get("arm_slot", np.nan)
        if a.release is not None:                 # release set by hand
            rel = int(a.release)
            sx, sy = df[f"{arm}_shoulder_x"].iloc[rel], df[f"{arm}_shoulder_y"].iloc[rel]
            wx, wy = df[f"{arm}_wrist_x"].iloc[rel],   df[f"{arm}_wrist_y"].iloc[rel]
            ours = float(np.degrees(np.arctan2(abs(wx-sx), (sy-wy))))
            print(f"[release set by hand] frame{rel}  arm_slot recomputed = {ours:.1f}\n")
        print("[domain check, frontal]  arm slot  vs  the OBP 3D coronal truth distribution")
        print(f"{'quantity':14s}{'ours':>9s}{'OBP mean':>9s}{'OBP std':>9s}{'z':>7s}  verdict")
        print("-" * 56)
        if os.path.exists(ARMSLOT_REF):
            ref = pd.read_csv(ARMSLOT_REF)["arm_slot_truth"]
            mu, sd = ref.mean(), ref.std()
            z = (ours - mu) / sd if sd > 0 else np.nan
            ok = "in range" if abs(z) <= 2 else "outside (check the shot or the detection)"
            print(f"{'arm_slot':14s}{ours:>9.2f}{mu:>9.2f}{sd:>9.2f}{z:>7.2f}  {ok}")
        else:
            print(f"{'arm_slot':14s}{ours:>9.2f}   (no armslot_ref.csv -> run armslot_validate.py --batch first)")
        return

    # side view, the default: the five validated quantities
    ref = pd.read_csv(REF) if os.path.exists(REF) else None
    print("[domain check]  our clip  vs  the OBP projected distribution")
    print(f"{'quantity':22s}{'ours':>10s}{'OBP mean':>10s}{'OBP std':>9s}{'z':>7s}  verdict")
    print("-" * 70)
    for ours, refcol in CORE.items():
        v = vals.get(ours, np.nan)
        if ref is not None and refcol in ref.columns:
            mu, sd = ref[refcol].mean(), ref[refcol].std()
            z = (v - mu) / sd if sd > 0 else np.nan
            ok = "in range" if abs(z) <= 2 else "outside (check the shot or the detection)"
            print(f"{ours:22s}{v:>10.2f}{mu:>10.2f}{sd:>9.2f}{z:>7.2f}  {ok}")
        else:
            tag = "(reference column missing - rerun obp_project)" if ref is not None else "(no reference csv)"
            print(f"{ours:22s}{v:>10.2f}   {tag}")

    print("\n[reference] every candidate quantity:")
    for k, v in vals.items():
        print(f"  {k:24s} {v:8.2f}")


if __name__ == "__main__":
    main()