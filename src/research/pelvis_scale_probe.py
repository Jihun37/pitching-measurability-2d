"""
Diamond - Where does the real-video pelvis rot-velo ~2x scale offset come from?
pelvis_scale_probe.py

Real overhead clips read ~2-2.8x the OBP 3D pelvis rotational velocity. The OBP
audit (orthographic, el=85) read ~1x, so the offset is real-camera specific.
Candidates: (1) camera elevation != 85 (lower el amplifies the projected angle
sweep near edge-on), (2) perspective (real lens vs orthographic audit).

For N OBP pitches, take the TRUE 3D pelvis omega_ax peak as reference, then measure
the projected hip-line yaw-rate peak / true, across elevation and perspective. The
condition whose ratio ~2 is the cause -> gives the deployable calibration.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import config
import obp_project as O
import metrics as M
import truth_redefine_gate as T

ELS = [89.5, 85, 75, 65, 55, 45]
PERSP = [("ortho", False, None), ("persp d=8", True, 8.0), ("persp d=4", True, 4.0)]
N = 80


def yaw_peak(df, lo, hi, fps):
    ang = np.unwrap(np.arctan2(
        df["right_hip_y"].to_numpy() - df["left_hip_y"].to_numpy(),
        df["right_hip_x"].to_numpy() - df["left_hip_x"].to_numpy()))
    win = min(max(5, int(round(0.05 * fps)) | 1), len(ang) - (len(ang) % 2 == 0))
    vel = np.abs(np.degrees(savgol_filter(ang, win, 3, deriv=1, delta=1.0/fps, mode="interp")))
    return float(np.nanmax(vel[lo:hi + 1]))


def main():
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    acc = {(pl[0], el): [] for pl in PERSP for el in ELS}
    n = fail = 0
    for r in md.itertuples(index=False):
        if n >= N:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            mk, fps = T.load_markers(path, T.PEL_MK)
            joints, _ = O.load_c3d_joints(path)
            arm = O.detect_throwing_arm(joints, fps)
            rel = M.release_frame(O.project_view(joints, azimuth_deg=0.0), arm, fps, M.JOINTS)
            nf = joints["left_hip"].shape[1]
            lo, hi = max(0, rel - int(0.40*fps)), min(nf-1, rel + int(0.05*fps))
            # true 3D pelvis rotational velocity peak
            Rp, sp = T.pelvis_frame(mk)
            w = np.degrees(np.sum(T.omega_world(Rp, fps) * sp, axis=0))
            win = min(max(5, int(round(0.05*fps)) | 1), nf - (nf % 2 == 0))
            w = savgol_filter(w, win, 3, mode="interp")
            true_pk = float(np.nanmax(np.abs(w[lo:hi+1])))
            if true_pk < 50:
                continue
            for label, pf, cd in PERSP:
                for el in ELS:
                    kw = dict(azimuth_deg=0.0, elevation_deg=el)
                    if pf:
                        kw.update(perspective=True, cam_dist=cd)
                    df = O.project_view(joints, **kw)
                    acc[(label, el)].append(yaw_peak(df, lo, hi, fps) / true_pk)
            n += 1
        except Exception:
            fail += 1

    print(f"OBP n={n} (fail {fail}). Ratio = projected hip-line yaw-rate peak / true 3D pelvis rot-velo peak")
    print(f"(ratio 1.0 = correct absolute scale; the real clips sit ~2-2.8)\n")
    print(f"{'condition':>10} " + " ".join(f"el{el:>5}" for el in ELS))
    print("-" * (11 + 7 * len(ELS)))
    for label, _, _ in PERSP:
        row = " ".join(f"{np.median(acc[(label, el)]):6.2f}" for el in ELS)
        print(f"{label:>10} {row}")


if __name__ == "__main__":
    main()
