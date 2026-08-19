"""Soft-recovery RETRY with the inputs we have since unlocked (+ kinetics).

`posture_inference_ceiling.py` (2026-06) concluded the rotation family is DEAD by
inference, with the stated cause: "the information lives in the transverse /
kinetic axes 2D cannot see." Two things have changed since, which justify one
retry rather than a re-derivation:

1. We subsequently PROVED 2D can see two transverse quantities -- from OVERHEAD:
   max_pelvis_rotational_velo (r2 0.80 @el85) and
   max_rotation_hip_shoulder_separation (r2 0.63 @el85). The original probe had
   NEITHER as an input. So the question becomes well-posed and new: measure 2 of
   the 21 transverse columns from overhead, predict the other 19.
2. The COG pair (max_cog_velo_x, cog_velo_pkh) was adopted 2026-07-18, after
   that probe ran, and carries whole-body momentum information nothing in the
   old input set had.

Also, that probe explicitly excluded kinetics ("no kinetics"). So the INJURY axis
-- elbow varus moment above all, the single most-cited injury variable in the
literature -- has never been tested for inferability at all. Literature puts the
ceiling from full 3D kinematics near R2 0.40 (11 predictors), i.e. probably below
our 0.50 floor, but that is a different variable set and worth settling directly.

Design is inherited deliberately: OBP 3D columns as inputs = PERFECT-INPUT
ceiling (if the ceiling is low, 2D inference is dead before pose noise is added);
GroupKFold by pitcher blocks leakage; every target reports the anthro-only
baseline alongside anthro+mechanics, so what matters is the INCREMENTAL R2 over
body size, not the raw number (body mass alone predicts kinetics quite well and
that is not mechanics).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402

META = os.path.join(config.OBP_DATA_DIR, "metadata.csv")
POI = os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")

ANTHRO = ["session_height_m", "session_mass_kg", "age_yrs"]

# The old sagittal-only pool (for the A/B that isolates what the new inputs add)
INPUT_OLD = [
    "stride_length", "stride_angle", "arm_slot",
    "torso_anterior_tilt_fp", "torso_anterior_tilt_br",
    "elbow_flexion_fp", "max_elbow_flexion",
    "lead_knee_extension_angular_velo_fp",
    "lead_knee_extension_angular_velo_br",
    "lead_knee_extension_angular_velo_max",
    "lead_knee_extension_from_fp_to_br",
]
# What we can measure NOW but could not then: two transverse columns from
# overhead + the whole-body COM pair.
INPUT_NEW = [
    "max_pelvis_rotational_velo",              # overhead, r2 0.80
    "max_rotation_hip_shoulder_separation",    # overhead, r2 0.63
    "max_cog_velo_x", "cog_velo_pkh",          # adopted 2026-07-18
]

TARGETS_ROT = [
    "torso_rotation_fp", "torso_rotation_br", "torso_rotation_mer",
    "torso_rotation_min", "pelvis_rotation_fp",
    "rotation_hip_shoulder_separation_fp",
    "max_torso_rotational_velo", "max_shoulder_internal_rotational_velo",
    "max_shoulder_external_rotation", "shoulder_external_rotation_fp",
    "shoulder_horizontal_abduction_fp", "max_shoulder_horizontal_abduction",
    "elbow_pronation_fp",
    "torso_lateral_tilt_fp", "torso_lateral_tilt_mer", "torso_lateral_tilt_br",
    "pelvis_lateral_tilt_fp", "pelvis_anterior_tilt_fp",
    "timing_peak_torso_to_peak_pelvis_rot_velo",
]
TARGETS_KIN = [
    "elbow_varus_moment", "shoulder_internal_rotation_moment",
    "rear_grf_mag_max", "lead_grf_mag_max", "peak_rfd_rear", "peak_rfd_lead",
    "shoulder_transfer_fp_br", "elbow_transfer_fp_br",
    "lead_hip_absorption_fp_br", "pelvis_lumbar_transfer_fp_br",
    "thorax_distal_transfer_fp_br",
]
FLOOR = 0.50


def cv_r2(df, feats, target, model="gbm", n_splits=5):
    sub = df.dropna(subset=[target]).copy()
    if len(sub) < 60:
        return None, 0
    X = sub[feats].to_numpy(float)
    y = sub[target].to_numpy(float)
    g = sub["user"].to_numpy()
    est = (GradientBoostingRegressor(random_state=0) if model == "gbm"
           else LinearRegression())
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
    pred = cross_val_predict(pipe, X, y, groups=g,
                             cv=GroupKFold(n_splits=n_splits))
    return float(r2_score(y, pred)), len(sub)


def run(df, targets, title):
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)
    print(f"  {'target':<44}{'anthro':>9}{'+old':>9}{'+NEW':>9}"
          f"{'gain':>9}{'n':>7}")
    print("  " + "-" * 96)
    rows = []
    for t in targets:
        if t not in df.columns:
            print(f"  {t:<44}  (column not found)")
            continue
        r_base, _ = cv_r2(df, ANTHRO, t)
        r_old, _ = cv_r2(df, ANTHRO + INPUT_OLD, t)
        r_new, n = cv_r2(df, ANTHRO + INPUT_OLD + INPUT_NEW, t)
        if r_new is None:
            continue
        gain = r_new - max(r_old, r_base)
        flag = "  <-- CLEARS FLOOR" if r_new >= FLOOR else ""
        print(f"  {t:<44}{r_base:>9.3f}{r_old:>9.3f}{r_new:>9.3f}"
              f"{gain:>+9.3f}{n:>7d}{flag}")
        rows.append(dict(target=t, anthro=r_base, old=r_old, new=r_new,
                         gain=gain, n=n))
    return rows


def main():
    md = pd.read_csv(META)
    poi = pd.read_csv(POI)
    df = poi.merge(md[["session_pitch", "user"] + ANTHRO], on="session_pitch")
    print(f"n = {len(df)} pitches, {df.user.nunique()} pitchers")
    print(f"inputs: anthro({len(ANTHRO)}) + old-sagittal({len(INPUT_OLD)})"
          f" + NEW({len(INPUT_NEW)})")
    print("  NEW = overhead-measurable pelvis rot velo & HSS, plus the COG pair")
    print(f"  adoption floor R2 = {FLOOR};  'gain' = +NEW over the better of"
          " anthro / +old")

    r1 = run(df, TARGETS_ROT,
             "[A] ROTATION / TILT FAMILY - does the overhead unlock rescue"
             " inference?")
    r2 = run(df, TARGETS_KIN,
             "[B] KINETICS - never attempted before (injury axis)")

    out = pd.DataFrame(r1 + r2)
    dst = os.path.join(config.ROOT, "data", "outputs", "obp_validation",
                       "inference_retry_enriched.csv")
    out.to_csv(dst, index=False)
    n_ok = int((out.new >= FLOOR).sum())
    print(f"\n  targets clearing the {FLOOR} floor: {n_ok} / {len(out)}")
    if n_ok:
        print(out[out.new >= FLOOR].to_string(index=False))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
