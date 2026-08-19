"""
Phase 1: the ceiling on predicting ball speed from 3D quantities.

Purpose
  Before any computer vision enters, measure how much of pitch_speed_mph the
  release's own gold-standard 3D quantities explain. That is the ceiling on the
  whole pipeline.
  The features are split into four groups so the R-squared reachable using only
  what a sagittal 2D view could measure can be read separately.

Two things to watch
  A pitcher contributes several pitches, so the cross-validation is grouped by
  user. Without that the R-squared is inflated by memorising the pitcher rather
  than learning the mechanics.
  max_*_velo, the arm and torso angular velocities, are an OUTPUT of throwing
  hard and predict ball speed almost tautologically, so they are kept in their
  own group.
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
from sklearn.metrics import r2_score, mean_squared_error
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config

ROOT = config.OBP_DATA_DIR
META = os.path.join(ROOT, "metadata.csv")
POI  = os.path.join(ROOT, "poi", "poi_metrics.csv")

# Feature groups
ANTHRO = ["session_height_m", "session_mass_kg", "age_yrs"]

# Kinematics with some prospect of being recovered from a sagittal 2D view,
# which is the realistic target.
TWO_D = [
    "stride_length", "stride_angle", "arm_slot",
    "torso_anterior_tilt_fp", "torso_anterior_tilt_br",
    "elbow_flexion_fp", "max_elbow_flexion",
    "lead_knee_extension_angular_velo_fp",
    "lead_knee_extension_angular_velo_br",
    "lead_knee_extension_angular_velo_max",
    "lead_knee_extension_from_fp_to_br",
]
# Transverse-plane rotations: not readable from the side either, included to
# measure what excluding them costs.
ROTATION = [
    "rotation_hip_shoulder_separation_fp", "max_rotation_hip_shoulder_separation",
    "torso_rotation_fp", "torso_rotation_br", "pelvis_rotation_fp",
]
# Output angular velocity: nearly tautological with ball speed, kept only to
# mark the ceiling.
OUTPUT_VELO = [
    "max_shoulder_internal_rotational_velo", "max_elbow_extension_velo",
    "max_torso_rotational_velo", "max_pelvis_rotational_velo",
]
TARGET = "pitch_speed_mph"


def load():
    md = pd.read_csv(META)
    poi = pd.read_csv(POI)
    df = poi.merge(md[["session_pitch", "user"] + ANTHRO],
                   on="session_pitch", how="inner")
    if TARGET + "_x" in df:  # tidy up a merge duplicate
        df[TARGET] = df[TARGET + "_x"]
    df = df.dropna(subset=[TARGET, "user"])
    return df


def evaluate(df, feats, label, model="gbm"):
    """Cross-validated R2 and RMSE, GroupKFold by pitcher."""
    feats = [f for f in feats if f in df.columns]
    X = df[feats].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    groups = df["user"].to_numpy()
    if model == "linear":
        est = make_pipeline(SimpleImputer(strategy="median"),
                            StandardScaler(), LinearRegression())
    else:
        est = make_pipeline(SimpleImputer(strategy="median"),
                            GradientBoostingRegressor(random_state=0,
                                n_estimators=300, max_depth=3, learning_rate=0.03))
    cv = GroupKFold(n_splits=5)
    pred = cross_val_predict(est, X, y, groups=groups, cv=cv)
    r2 = r2_score(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    print(f"  {label:36s} n_feat={len(feats):2d}  CV R²={r2:5.3f}  RMSE={rmse:4.2f} mph")
    return r2, rmse


def single_feature_corr(df):
    feats = ANTHRO + TWO_D + ROTATION
    rows = []
    for f in feats:
        if f in df.columns:
            rows.append((f, df[f].corr(df[TARGET])))
    print("\n[single feature against ball speed, by absolute correlation]")
    for f, r in sorted(rows, key=lambda t: -abs(t[1])):
        tag = ("anthro" if f in ANTHRO else "2D" if f in TWO_D else "rotation")
        print(f"  {f:42s} r={r:+.3f}   [{tag}]")


def gbm_importance(df, feats, label):
    feats = [f for f in feats if f in df.columns]
    X = df[feats].to_numpy(float); y = df[TARGET].to_numpy(float)
    est = make_pipeline(SimpleImputer(strategy="median"),
                        GradientBoostingRegressor(random_state=0,
                            n_estimators=300, max_depth=3, learning_rate=0.03))
    est.fit(X, y)
    imp = est[-1].feature_importances_
    print(f"\n[{label} - GBM feature importance, top]")
    for f, w in sorted(zip(feats, imp), key=lambda t: -t[1])[:8]:
        print(f"  {f:42s} {w:.3f}")


def main():
    df = load()
    print(f"{len(df)} pitches / {df['user'].nunique()} pitchers / speed "
          f"{df[TARGET].mean():.1f}±{df[TARGET].std():.1f} mph "
          f"({df[TARGET].min():.0f}~{df[TARGET].max():.0f})")

    print("\n[cross-validated R2 / RMSE]  (GroupKFold by pitcher)")
    evaluate(df, ANTHRO, "M0 stature, mass and age only (baseline)")
    evaluate(df, ANTHRO + TWO_D, "M1 + sagittal 2D kinematics (realistic target)")
    evaluate(df, ANTHRO + TWO_D + ROTATION, "M2 + rotations (not readable from the side)")
    evaluate(df, ANTHRO + TWO_D + ROTATION + OUTPUT_VELO, "M3 + output angular velocity (tautological ceiling)")
    evaluate(df, ANTHRO + TWO_D, "M1 (linear regression)", model="linear")

    single_feature_corr(df)
    gbm_importance(df, ANTHRO + TWO_D, "M1 realistic target")


if __name__ == "__main__":
    main()