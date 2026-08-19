"""
Diamond - Soft-recovery ceiling for "posture-type" un-measurable columns
posture_inference_ceiling.py

Question
  The measurability survey excludes rotation / lateral-tilt / MER-timepoint
  angles because a single 2D camera cannot image them DIRECTLY (geometry).
  But can we INFER them statistically from the sagittal kinematics we CAN
  measure? (e.g. does stride + arm slot + trunk tilt jointly predict torso
  rotation at release?)

Design (mirrors phase1_velocity_model.py)
  - Ceiling estimate: inputs are OBP 3D-direct columns for the sagittal
    quantities we can measure -> the perfect-input upper bound. If the ceiling
    is low, 2D inference is dead before we add pose noise.
  - Pitcher leakage blocked with GroupKFold(by user).
  - For every target report anthro-only baseline vs anthro+2D, so we see the
    INCREMENTAL R2 that the mechanics inputs add beyond body size / age.
  - Targets = posture-type columns only: rotation angles, lateral tilts, MER
    timepoint angles, external rotation, horizontal abduction. No velocities
    (those are "output", already shown dead), no kinetics.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config

ROOT = config.OBP_DATA_DIR
META = os.path.join(ROOT, "metadata.csv")
POI = os.path.join(ROOT, "poi", "poi_metrics.csv")

# Inputs: 2D-measurable sagittal kinematics (same pool as phase1 TWO_D).
ANTHRO = ["session_height_m", "session_mass_kg", "age_yrs"]
INPUT_2D = [
    "stride_length", "stride_angle", "arm_slot",
    "torso_anterior_tilt_fp", "torso_anterior_tilt_br",
    "elbow_flexion_fp", "max_elbow_flexion",
    "lead_knee_extension_angular_velo_fp",
    "lead_knee_extension_angular_velo_br",
    "lead_knee_extension_angular_velo_max",
    "lead_knee_extension_from_fp_to_br",
]

# Targets: posture-type un-measurable columns (angles only; no velo, no kinetics).
TARGETS = [
    # transverse rotation angles
    "max_rotation_hip_shoulder_separation",
    "rotation_hip_shoulder_separation_fp",
    "torso_rotation_fp", "torso_rotation_br", "torso_rotation_mer",
    "torso_rotation_min", "pelvis_rotation_fp",
    # shoulder external rotation / abduction (coronal / transverse)
    "max_shoulder_external_rotation", "shoulder_external_rotation_fp",
    "glove_shoulder_external_rotation_fp",
    "shoulder_horizontal_abduction_fp", "max_shoulder_horizontal_abduction",
    "glove_shoulder_horizontal_abduction_fp",
    "shoulder_abduction_fp", "glove_shoulder_abduction_fp",
    "elbow_pronation_fp",
    # lateral tilt (coronal)
    "torso_lateral_tilt_fp", "torso_lateral_tilt_mer", "torso_lateral_tilt_br",
    "pelvis_lateral_tilt_fp", "pelvis_anterior_tilt_fp",
    # MER timepoint (need a rotation-defined event; test inferability anyway)
    "glove_shoulder_abduction_mer", "elbow_flexion_mer", "torso_anterior_tilt_mer",
]


def load():
    md = pd.read_csv(META)
    poi = pd.read_csv(POI)
    df = poi.merge(md[["session_pitch", "user"] + ANTHRO],
                   on="session_pitch", how="inner")
    df = df.dropna(subset=["user"])
    return df


def cv_r2(df, feats, target, model="gbm"):
    """Pitcher-grouped GroupKFold CV R2 for one target."""
    feats = [f for f in feats if f in df.columns]
    sub = df.dropna(subset=[target])
    if len(sub) < 50 or sub["user"].nunique() < 6:
        return None, len(sub)
    X = sub[feats].to_numpy(float)
    y = sub[target].to_numpy(float)
    groups = sub["user"].to_numpy()
    if model == "linear":
        est = make_pipeline(SimpleImputer(strategy="median"),
                            StandardScaler(), LinearRegression())
    else:
        est = make_pipeline(SimpleImputer(strategy="median"),
                            GradientBoostingRegressor(random_state=0,
                                n_estimators=300, max_depth=3, learning_rate=0.03))
    cv = GroupKFold(n_splits=5)
    pred = cross_val_predict(est, X, y, groups=groups, cv=cv)
    return r2_score(y, pred), len(sub)


def main():
    df = load()
    print(f"pitches {len(df)} / pitchers {df['user'].nunique()}")
    print(f"inputs: anthro({len(ANTHRO)}) + 2D-sagittal({len(INPUT_2D)})\n")

    header = f"{'target':40s} {'n':>4s} {'anthro':>8s} {'+2D(gbm)':>9s} {'+2D(lin)':>9s} {'incr':>7s}"
    print(header)
    print("-" * len(header))

    rows = []
    for t in TARGETS:
        if t not in df.columns:
            print(f"{t:40s}  (column not found)")
            continue
        r2_base, _ = cv_r2(df, ANTHRO, t, model="gbm")
        r2_gbm, n = cv_r2(df, ANTHRO + INPUT_2D, t, model="gbm")
        r2_lin, _ = cv_r2(df, ANTHRO + INPUT_2D, t, model="linear")
        if r2_gbm is None:
            print(f"{t:40s} {n:>4d}  (too few samples)")
            continue
        best_full = max(r2_gbm, r2_lin)
        incr = best_full - (r2_base if r2_base is not None else 0.0)
        rows.append((t, n, r2_base, r2_gbm, r2_lin, incr))
        print(f"{t:40s} {n:>4d} {r2_base:8.3f} {r2_gbm:9.3f} {r2_lin:9.3f} {incr:7.3f}")

    print("\n[ranked by best full-model CV R2]")
    for t, n, rb, rg, rl, incr in sorted(rows, key=lambda x: -max(x[3], x[4])):
        best = max(rg, rl)
        flag = "ADOPT?" if best >= 0.50 and incr >= 0.10 else ""
        print(f"  {t:40s} R2={best:6.3f}  incr(over anthro)={incr:+.3f}  {flag}")


if __name__ == "__main__":
    main()
