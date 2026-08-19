"""
Diamond - Extended metrics feasibility test with view sweep
extended_metrics_test.py

Question:
Can we extract NEW candidate metrics from 2D projections at different camera
azimuths, and do they recover OBP 3D ground truth?

New candidates:
  1) knee_ext_velo_max / fp / br
  2) elbow_flexion_fp, max_elbow_flexion
  3) hip_drive_velo_fp
  4) time_fp_to_br_ms

Usage:
    python extended_metrics_test.py
    python extended_metrics_test.py --limit 50
"""

import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))

import config
import obp_project as O
import metrics as M

AZIMUTHS = [0, 15, 30, 45, 60, 75, 90]

TRUTH = {
    "knee_ext_velo_max":  ("lead_knee_extension_angular_velo_max", "direct"),
    "knee_ext_velo_fp":   ("lead_knee_extension_angular_velo_fp",  "direct"),
    "knee_ext_velo_br":   ("lead_knee_extension_angular_velo_br",  "direct"),
    "elbow_flexion_fp":   ("elbow_flexion_fp",                     "direct"),
    "max_elbow_flexion":  ("max_elbow_flexion",                    "direct"),
    "elbow_ext_velo_max": ("max_elbow_extension_velo",             "direct"),
    "elbow_ext_velo_br":  ("max_elbow_extension_velo",             "direct"),
    # --- survey batch 1: event-ready candidates (fp / no-event) ---
    "torso_ant_tilt_fp":           ("torso_anterior_tilt_fp",       "direct"),
    "stride_angle":                ("stride_angle",                 "direct"),
    "cog_velo_x_max":              ("max_cog_velo_x",               "direct"),
    "shoulder_abduction_fp":       ("shoulder_abduction_fp",        "direct"),
    "glove_shoulder_abduction_fp": ("glove_shoulder_abduction_fp",  "direct"),
    "hip_drive_velo_fp":  (None, "Level-B only"),
    "time_fp_to_br_ms":   (None, "Level-B only"),
}


def _xy(df, joint):
    return df[f"{joint}_x"].to_numpy(float), df[f"{joint}_y"].to_numpy(float)


def compute_extended(df, fps, arm, events=None):
    """events: optional {"rel": frame, "fp": frame} to use INSTEAD of detecting.
    Passing the OBP landmark events separates the projection question (is this
    quantity recoverable from this view at all) from our detector error -- most of
    these candidates are anchored at foot plant, our noisiest event."""
    J = M.JOINTS
    lead = "left" if arm == "right" else "right"

    if events is not None:
        rel, fp = int(events["rel"]), int(events["fp"])
    else:
        rel = M.release_frame(df, arm, fps, J)
        fp = M.foot_plant_frame(df, lead, fps, J, rel)

    out = {}
    if rel <= fp + 1:
        return out

    hx, hy = _xy(df, f"{lead}_hip")
    kx, ky = _xy(df, f"{lead}_knee")
    ax_, ay = _xy(df, f"{lead}_ankle")

    knee = M._angle(hx, hy, kx, ky, ax_, ay)
    kvel = np.gradient(knee) * fps
    seg = kvel[fp:rel + 1]

    out["knee_ext_velo_max"] = float(np.nanmax(seg))
    out["knee_ext_velo_fp"] = float(kvel[fp])
    out["knee_ext_velo_br"] = float(kvel[rel])

    sx, sy = _xy(df, f"{arm}_shoulder")
    ex, ey = _xy(df, f"{arm}_elbow")
    wx, wy = _xy(df, f"{arm}_wrist")

    elbow = M._angle(sx, sy, ex, ey, wx, wy)   # included angle, 180 = full extension
    flex = 180.0 - elbow
    out["elbow_flexion_fp"] = float(flex[fp])
    out["max_elbow_flexion"] = float(np.nanmax(flex[fp:rel + 1]))

    # elbow extension angular velocity (deg/s, + = extending). Port of the
    # knee_ext_velo logic to the elbow joint. Truth = OBP max_elbow_extension_velo.
    evel = np.gradient(elbow) * fps
    out["elbow_ext_velo_max"] = float(np.nanmax(evel[fp:rel + 1]))
    out["elbow_ext_velo_br"] = float(evel[rel])

    rhx = _xy(df, "right_hip")[0]
    midhx = (hx + rhx) / 2
    stat = M.pixel_stature(df, J)
    hipv = np.abs(np.gradient(midhx)) * fps / stat
    out["hip_drive_velo_fp"] = float(hipv[fp])

    out["time_fp_to_br_ms"] = float((rel - fp) / fps * 1000.0)

    # ── survey batch 1 ──────────────────────────────────────────────
    glove = lead  # glove (non-throwing) arm is on the lead-leg side

    # torso anterior tilt at foot plant (sagittal) — same definition as the
    # adopted *_br metric, evaluated at fp; pitch-direction sign normalised.
    mhx = (_xy(df, "left_hip")[0] + _xy(df, "right_hip")[0]) / 2
    mhy = (_xy(df, "left_hip")[1] + _xy(df, "right_hip")[1]) / 2
    msx = (_xy(df, "left_shoulder")[0] + _xy(df, "right_shoulder")[0]) / 2
    msy = (_xy(df, "left_shoulder")[1] + _xy(df, "right_shoulder")[1]) / 2
    trunk = np.degrees(np.arctan2(msx - mhx, -(msy - mhy)))
    _d = ax_[fp] - np.nanmedian(ax_[:max(3, fp // 4)])
    pdir = 1.0 if _d >= 0 else -1.0
    out["torso_ant_tilt_fp"] = float(trunk[fp] * pdir)

    # stride angle proxy: image-plane angle of the lead->trail ankle line at fp
    tax, tay = _xy(df, f"{arm}_ankle")          # trail (pivot) = throwing-arm side
    out["stride_angle"] = float(np.degrees(np.arctan2(ay[fp] - tay[fp],
                                                      ax_[fp] - tax[fp])))

    # max COG forward velocity proxy: hip-midpoint speed, /stature (unit-free,
    # r² invariant), max over the delivery up to release. In-plane at 0°.
    cvel = np.abs(np.gradient(mhx)) * fps / stat
    out["cog_velo_x_max"] = float(np.nanmax(cvel[:rel + 1]))

    # shoulder abduction at fp (arm-to-trunk angle): vertex=shoulder, rays to
    # elbow and same-side hip. Coronal-plane -> expected best near the front (90°).
    def _abd(side):
        ssx, ssy = _xy(df, f"{side}_shoulder")
        sex, sey = _xy(df, f"{side}_elbow")
        shx, shy = _xy(df, f"{side}_hip")
        return float(M._angle(sex, sey, ssx, ssy, shx, shy)[fp])
    out["shoulder_abduction_fp"] = _abd(arm)
    out["glove_shoulder_abduction_fp"] = _abd(glove)

    return out


def corr_r2(df, x, y):
    d = df[[x, y]].dropna()
    if len(d) <= 2:
        return np.nan, np.nan
    r = d[x].corr(d[y])
    return r, r * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    rows, done, fail = [], 0, 0

    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break

        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1
            continue

        try:
            joints, fps = O.load_c3d_joints(path)
            arm = O.detect_throwing_arm(joints, fps)

            for az in AZIMUTHS:
                df2d = O.project_view(joints, azimuth_deg=az)
                feats = compute_extended(df2d, fps, arm)

                if not feats:
                    continue

                rows.append({
                    "session_pitch": r.session_pitch,
                    "azimuth": az,
                    **feats
                })

            done += 1

        except Exception:
            fail += 1

        if done and done % 100 == 0:
            print(f"  ...{done} processed")

    print(f"processed {done} / failed {fail}\n")

    feat = pd.DataFrame(rows)
    df = feat.merge(
        poi,
        on="session_pitch",
        how="inner",
        suffixes=("_our", "")
    )

    print(f"matched rows: {len(df)}")
    print(f"matched pitches: {df['session_pitch'].nunique()}\n")

    print("=" * 90)
    print("[Level-A] new 2D candidates vs OBP 3D truth by camera azimuth")
    print("=" * 90)

    header = f"{'our metric':22s}" + "".join(f"{az:>8d}°" for az in AZIMUTHS) + "   best"
    print(header)
    print("-" * len(header))

    summary_rows = []

    for ours, (truth, note) in TRUTH.items():
        if truth is None:
            continue

        line = f"{ours:22s}"
        best_az = None
        best_r2 = -1

        for az in AZIMUTHS:
            sub = df[df["azimuth"] == az]
            oc = ours + "_our" if ours + "_our" in sub.columns else ours

            if truth not in sub.columns or oc not in sub.columns:
                r2 = np.nan
            else:
                _, r2 = corr_r2(sub, oc, truth)

            if pd.notna(r2) and r2 > best_r2:
                best_r2 = r2
                best_az = az

            line += f"{r2:8.2f}"

            summary_rows.append({
                "metric": ours,
                "azimuth": az,
                "truth": truth,
                "r2": r2
            })

        line += f"   {best_az}° ({best_r2:.2f})"
        print(line)

    print("\n" + "=" * 90)
    print("[Level-B] new 2D candidates vs pitch_speed_mph by camera azimuth")
    print("=" * 90)

    sp = "pitch_speed_mph"
    header = f"{'our metric':22s}" + "".join(f"{az:>8d}°" for az in AZIMUTHS)
    print(header)
    print("-" * len(header))

    for ours in TRUTH:
        line = f"{ours:22s}"

        for az in AZIMUTHS:
            sub = df[df["azimuth"] == az]
            oc = ours + "_our" if ours + "_our" in sub.columns else ours

            if sp in sub.columns and oc in sub.columns:
                r, _ = corr_r2(sub, oc, sp)
            else:
                r = np.nan

            line += f"{r:8.2f}"

        print(line)

    out_csv = os.path.join(
        config.OBP_VALIDATION_DIR,
        "extended_metrics_view_sweep.csv"
    )
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)

    feat_csv = os.path.join(
        config.OBP_VALIDATION_DIR,
        "extended_metrics_view_sweep_features.csv"
    )
    feat.to_csv(feat_csv, index=False)

    print(f"\nsaved summary -> {out_csv}")
    print(f"saved features -> {feat_csv}")


if __name__ == "__main__":
    main()