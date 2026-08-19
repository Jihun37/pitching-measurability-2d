"""
Diamond - Lateral tilt recovery with OUR 3D-direct arm slot as the driver input
lateral_tilt_armslot3d.py

Follow-up to lateral_tilt_recovery.py. There the driver of torso_lateral_tilt_br
recovery was arm_slot, but the arm_slot used was the OBP forearm-based column,
which is NOT our deployed definition. Our arm slot is shoulder-to-wrist vs
vertical (3D-direct, Escamilla & Fleisig), the one adopted at the front (90) view.

Since arm_slot IS the driver, the deployable number depends on which arm-slot
definition we feed. This script computes OUR 3D-direct arm slot per pitch from
the c3d (via obp_project.truth is reproduced here) and re-runs the release-event
lateral tilt recovery, side by side with the OBP-column version.

Ceiling logic unchanged: 3D-direct inputs = perfect-input upper bound; pitcher
held out with GroupKFold; linear is the honest model (single dominant feature).
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import config
import obp_project as O
import metrics as M

ANTHRO = ["session_height_m", "session_mass_kg", "age_yrs"]
# Adopted-measurable inputs; arm_slot swapped for our 3D-direct version below.
OTHER_ADOPTED = ["stride_length", "torso_anterior_tilt_br",
                 "lead_knee_extension_angular_velo_br"]
TARGET = "torso_lateral_tilt_br"


def arm_slot_3d_frontal(joints, arm, rel):
    """Our arm slot: (wrist-shoulder) frontal-plane [Y horiz, Z vert] vs vertical."""
    S = joints[f"{arm}_shoulder"][:, rel]
    W = joints[f"{arm}_wrist"][:, rel]
    vec = W - S
    return float(np.degrees(np.arctan2(abs(vec[1]), vec[2])))


def compute_our_armslot():
    """Return {session_pitch: our 3D-direct arm slot} over all OBP c3d."""
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    out = {}
    n = fail = 0
    for r in md.itertuples(index=False):
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1
            continue
        try:
            joints, fps = O.load_c3d_joints(path)
            arm = O.detect_throwing_arm(joints, fps)
            df0 = O.project_view(joints, azimuth_deg=0.0)
            rel = M.release_frame(df0, arm, fps, M.JOINTS)
            out[r.session_pitch] = arm_slot_3d_frontal(joints, arm, rel)
            n += 1
        except Exception:
            fail += 1
    print(f"our arm slot computed: {n} ok / {fail} fail")
    return out


def cv_r2(df, feats, target):
    feats = [f for f in feats if f in df.columns]
    sub = df.dropna(subset=[target] + feats)
    X = sub[feats].to_numpy(float)
    y = sub[target].to_numpy(float)
    groups = sub["user"].to_numpy()
    est = make_pipeline(SimpleImputer(strategy="median"),
                        StandardScaler(), LinearRegression())
    pred = cross_val_predict(est, X, y, groups=groups, cv=GroupKFold(n_splits=5))
    return r2_score(y, pred), len(sub)


def main():
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    df = poi.merge(md[["session_pitch", "user"] + ANTHRO],
                   on="session_pitch", how="inner").dropna(subset=["user"])

    our = compute_our_armslot()
    df["arm_slot_3d"] = df["session_pitch"].map(our)

    both = df.dropna(subset=["arm_slot", "arm_slot_3d", TARGET])
    print(f"\nn with both arm slots + target = {len(both)}")
    print(f"OBP arm_slot     mean={both['arm_slot'].mean():.1f} sd={both['arm_slot'].std():.1f}")
    print(f"our arm_slot_3d  mean={both['arm_slot_3d'].mean():.1f} sd={both['arm_slot_3d'].std():.1f}")
    print(f"corr(OBP arm_slot, our arm_slot_3d) = {both['arm_slot'].corr(both['arm_slot_3d']):+.3f}")
    print(f"corr(our arm_slot_3d, {TARGET})    = {both['arm_slot_3d'].corr(both[TARGET]):+.3f}")
    print(f"corr(OBP arm_slot,   {TARGET})    = {both['arm_slot'].corr(both[TARGET]):+.3f}")

    print(f"\n[TARGET: {TARGET}]  linear, GroupKFold(by user)")
    # single-feature drivers
    r2_our1, n = cv_r2(df, ["arm_slot_3d"], TARGET)
    r2_obp1, _ = cv_r2(df, ["arm_slot"], TARGET)
    print(f"  single feature  our arm_slot_3d : R2={r2_our1:.3f}  (n={n})")
    print(f"  single feature  OBP arm_slot    : R2={r2_obp1:.3f}")
    # adopted-only models
    r2_our_full, _ = cv_r2(df, ANTHRO + ["arm_slot_3d"] + OTHER_ADOPTED, TARGET)
    r2_obp_full, _ = cv_r2(df, ANTHRO + ["arm_slot"] + OTHER_ADOPTED, TARGET)
    print(f"  adopted model   with our arm_slot_3d : R2={r2_our_full:.3f}")
    print(f"  adopted model   with OBP arm_slot    : R2={r2_obp_full:.3f}")
    # ablation: drop arm slot entirely
    r2_noslot, _ = cv_r2(df, ANTHRO + OTHER_ADOPTED, TARGET)
    print(f"  ablation        NO arm slot          : R2={r2_noslot:.3f}")


if __name__ == "__main__":
    main()
