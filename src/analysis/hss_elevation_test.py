"""
Diamond - Keystone feasibility test:
Does hip-shoulder separation (HSS), degenerate from any GROUND-level camera
(r2~0, patent 구성 I), become RECOVERABLE from ELEVATED / overhead views?

HSS is a transverse-plane (axial rotation) quantity. At elevation=0 the pelvis
and shoulder lines are seen edge-on -> their in-plane angle is unrecoverable.
As the camera rises, the transverse plane rotates into the image plane, so the
angle between the two lines should become measurable. This script sweeps
elevation and correlates a 2D HSS angle against OBP 3D ground truth.

EXPERIMENTAL metric definition (lives here for now; promote to metrics.py only
if adopted): signed angle between the shoulder-line vector and the hip-line
vector in the image plane. Scale/zoom-invariant by construction (it is an angle).

  sh  = r_shoulder - l_shoulder      (image x,y)
  hip = r_hip - l_hip
  sep = degrees(atan2(cross(sh,hip), dot(sh,hip)))     # per frame, signed

Truth columns (OBP poi):
  max_rotation_hip_shoulder_separation      <- peak |sep| over pitch (event-free)
  rotation_hip_shoulder_separation_fp       <- sep at foot-plant frame

Event isolation note: to measure "is the info visible from this view" without
confounding by event-detection breakdown at high elevation, the foot-plant frame
is detected ONCE on the el=0 side projection and reused across (az,el) for the
same pitch. This deviates from the per-angle re-detection rule ON PURPOSE - it
isolates metric visibility. Full-pipeline validation must re-detect per angle.

Run:
  cd src\analysis
  python hss_elevation_test.py                 # all pitches
  python hss_elevation_test.py --limit 120
"""
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import config
import obp_project as O
import metrics as M

AZIMUTHS = [0, 45, 90]                    # side / diagonal / front
ELEVATIONS = [0, 15, 30, 45, 60, 75, 85, 90]   # sweep all the way to bird's-eye


def project_cam(joints, az_deg, el_deg):
    """Local orthographic projection robust up to el=90 (bird's-eye).
    The shared engine (obp_project.project_view) gimbal-locks at el=90 because
    its up-reference is world_up; here we swap the reference when the view is
    near-vertical. Angle metrics are scale-free so ppm is irrelevant -> ppm=1."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    # 2026-07-15 official az convention (matches obp_project.project_view):
    # az increases 3B -> home; az90 = front/catcher.
    vd = np.array([-np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    ref = np.array([0, 0, 1.0]) if abs(vd[2]) < 0.999 else np.array([0, 1.0, 0])
    right = np.cross(vd, ref); right /= (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, vd); up /= (np.linalg.norm(up) + 1e-9)
    cols = {}
    for name, j in joints.items():
        P = j.T
        cols[f"{name}_x"] = P @ right
        cols[f"{name}_y"] = -(P @ up)      # image y grows downward
    return pd.DataFrame(cols)


# Promoted to metrics.py (2026-07-04, adopted after OBP validation).
# Re-exported here so existing importers keep working.
hss_sep_series = M.hss_sep_series


def r2(a, b):
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < 3:
        return np.nan
    r = d["a"].corr(d["b"])
    return r * r if pd.notna(r) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    J = M.JOINTS

    # acc[(az,el)] = list of {session_pitch, hss_max, hss_fp}
    acc = {(az, el): [] for az in AZIMUTHS for el in ELEVATIONS}
    done = fail = 0
    for r in md.itertuples(index=False):
        if a.limit and done >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = O.load_c3d_joints(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
            # detect foot-plant ONCE on clean side view, reuse across (az,el)
            df0 = O.project_view(joints, azimuth_deg=0, elevation_deg=0)
            rel0 = M.release_frame(df0, arm, fps, J, view="side")
            fp0 = M.foot_plant_frame(df0, lead, fps, J, rel0)
        except Exception:
            fail += 1; continue

        for az in AZIMUTHS:
            for el in ELEVATIONS:
                try:
                    df = project_cam(joints, az, el)
                    sep = hss_sep_series(df, J)
                    hss_max = float(np.nanmax(np.abs(sep)))
                    hss_fp = float(abs(sep[fp0]))
                    acc[(az, el)].append({"session_pitch": r.session_pitch,
                                          "hss_max": hss_max, "hss_fp": hss_fp})
                except Exception:
                    pass
        done += 1
        if done % 100 == 0:
            print(f"  ...{done} pitches")
    print(f"processed {done} / {fail} missing\n")

    # r2 tables: rows=elevation, cols=azimuth
    def table(our_col, truth_col):
        tab = {}
        for az in AZIMUTHS:
            for el in ELEVATIONS:
                feat = pd.DataFrame(acc[(az, el)])
                if feat.empty:
                    tab[(az, el)] = np.nan; continue
                m = feat.merge(poi[["session_pitch", truth_col]],
                               on="session_pitch", how="inner")
                tab[(az, el)] = r2(m[our_col], m[truth_col])
        return tab

    for our_col, truth_col, title in [
        ("hss_max", "max_rotation_hip_shoulder_separation",
         "PEAK |sep|  vs  max_rotation_hip_shoulder_separation  (event-free)"),
        ("hss_fp", "rotation_hip_shoulder_separation_fp",
         "sep@footplant  vs  rotation_hip_shoulder_separation_fp  (frame reused)"),
    ]:
        tab = table(our_col, truth_col)
        print("=" * 60)
        print(f"[{title}]")
        print("r2   " + "".join(f"  az={az:>2d}" for az in AZIMUTHS))
        print("-" * 60)
        for el in ELEVATIONS:
            line = f"el={el:>2d}  " + "".join(
                f"  {tab[(az, el)]:>5.2f}" if pd.notna(tab[(az, el)]) else "    -"
                for az in AZIMUTHS)
            print(line)
        print()

    # save raw
    rows = []
    for (az, el), lst in acc.items():
        for d in lst:
            rows.append({"az": az, "el": el, **d})
    out = os.path.join(config.OBP_VALIDATION_DIR, "hss_elevation_features.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"features saved -> {out}")


if __name__ == "__main__":
    main()
