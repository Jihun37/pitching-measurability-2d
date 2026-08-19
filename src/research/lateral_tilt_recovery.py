"""
Diamond - Clean re-confirmation of lateral trunk tilt soft-recovery
lateral_tilt_recovery.py

The posture scan flagged torso_lateral_tilt (coronal, not directly imageable
from a side camera) as the one un-measurable posture column that is partially
inferable from sagittal kinematics: br R2=0.40, mer 0.54.

This script cleans that up:
  (1) Driver: which single input carries it? Hypothesis = arm_slot (arm slot and
      contralateral trunk tilt are the same "tilt-and-reach" mechanic).
  (2) Deployability: does R2 survive when inputs are restricted to the ADOPTED
      2D-measurable set only (arm_slot, stride_length, torso_anterior_tilt_br,
      knee_velo_br)? The full sagittal pool includes REJECTED quantities we
      cannot actually measure in 2D (stride_angle, elbow_flexion_fp, ...).
  (3) Event: report at release (br) - the deployable event we already detect -
      not mer (a rotation-defined event we cannot locate from 2D).

Caveat printed at end: arm_slot here is the OBP forearm-based column, while our
measured arm slot is shoulder-to-wrist (3D direct). So this is a coupling-exists
ceiling, not the final deployed number.
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

ROOT = r"D:\project\diamond\data\datasets\OBP\openbiomechanics\baseball_pitching\data"
META = os.path.join(ROOT, "metadata.csv")
POI = os.path.join(ROOT, "poi", "poi_metrics.csv")

ANTHRO = ["session_height_m", "session_mass_kg", "age_yrs"]

# Full sagittal pool (includes rejected/marginal 2D quantities).
INPUT_FULL = [
    "stride_length", "stride_angle", "arm_slot",
    "torso_anterior_tilt_fp", "torso_anterior_tilt_br",
    "elbow_flexion_fp", "max_elbow_flexion",
    "lead_knee_extension_angular_velo_fp",
    "lead_knee_extension_angular_velo_br",
    "lead_knee_extension_angular_velo_max",
    "lead_knee_extension_from_fp_to_br",
]
# Only the quantities we actually adopted as 2D-measurable.
INPUT_ADOPTED = [
    "arm_slot", "stride_length", "torso_anterior_tilt_br",
    "lead_knee_extension_angular_velo_br",
]

TARGETS = ["torso_lateral_tilt_br", "torso_lateral_tilt_fp"]


def load():
    md = pd.read_csv(META)
    poi = pd.read_csv(POI)
    df = poi.merge(md[["session_pitch", "user"] + ANTHRO],
                   on="session_pitch", how="inner")
    return df.dropna(subset=["user"])


def cv_r2(df, feats, target, model="linear"):
    feats = [f for f in feats if f in df.columns]
    sub = df.dropna(subset=[target])
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
    pred = cross_val_predict(est, X, y, groups=groups, cv=GroupKFold(n_splits=5))
    return r2_score(y, pred), len(sub)


def analyze(df, target):
    print(f"\n{'='*70}\nTARGET: {target}")
    sub = df.dropna(subset=[target])
    print(f"n={len(sub)}  mean={sub[target].mean():.1f}  sd={sub[target].std():.1f} deg")

    r2_full_lin, _ = cv_r2(df, ANTHRO + INPUT_FULL, target, "linear")
    r2_full_gbm, _ = cv_r2(df, ANTHRO + INPUT_FULL, target, "gbm")
    r2_adopt_lin, _ = cv_r2(df, ANTHRO + INPUT_ADOPTED, target, "linear")
    r2_adopt_gbm, _ = cv_r2(df, ANTHRO + INPUT_ADOPTED, target, "gbm")
    print(f"  full pool (anthro+11)   CV R2  lin={r2_full_lin:.3f}  gbm={r2_full_gbm:.3f}")
    print(f"  adopted-only (anthro+4) CV R2  lin={r2_adopt_lin:.3f}  gbm={r2_adopt_gbm:.3f}")

    # single-feature drivers (linear CV R2 + Pearson r)
    print("  single-feature (linear CV R2 | Pearson r vs target):")
    rows = []
    for f in ANTHRO + INPUT_FULL:
        if f not in df.columns:
            continue
        r2_f, _ = cv_r2(df, [f], target, "linear")
        r = sub[f].corr(sub[target])
        rows.append((f, r2_f, r))
    for f, r2_f, r in sorted(rows, key=lambda x: -x[1]):
        tag = "adopted" if f in INPUT_ADOPTED else "anthro" if f in ANTHRO else "full-only"
        print(f"    {f:38s} R2={r2_f:6.3f}  r={r:+.3f}  [{tag}]")

    # ablation: drop arm_slot from adopted-only model
    no_slot = [f for f in INPUT_ADOPTED if f != "arm_slot"]
    r2_noslot, _ = cv_r2(df, ANTHRO + no_slot, target, "linear")
    print(f"  ablation: adopted-only WITHOUT arm_slot  CV R2 lin={r2_noslot:.3f} "
          f"(vs {r2_adopt_lin:.3f} with)")


def main():
    df = load()
    print(f"pitches {len(df)} / pitchers {df['user'].nunique()}")
    for t in TARGETS:
        analyze(df, t)
    print(f"\n{'='*70}")
    print("CAVEAT: arm_slot input = OBP forearm-based column; our deployed arm")
    print("slot is shoulder-to-wrist (3D direct). This is a coupling-exists")
    print("ceiling, not the final 2D-pipeline number.")


if __name__ == "__main__":
    main()
