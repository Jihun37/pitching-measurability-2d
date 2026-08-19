"""
Diamond - Pelvis overhead projection-loss audit
audit_pelvis_projection.py

Why does the 3D-ideal hip-line pelvis magnitude gate (r~0.89) fall to r~0.55 in
the 2D overhead deploy path? Ladder of 5 signals isolates the loss stage. All
scored vs OBP max_pelvis_rotational_velo (magnitude gate r), plus peak-time error
vs the omega_ax pelvis truth and the peak-value scale (deg/s).

  A ideal_xy       : 3D hip-line azimuth rate in the global horizontal XY plane
                     (Z dropped). No projection. This is the control (~0.89).
  B projected_flat : hip endpoints flattened to a common Z (tilt removed), then
                     project_view(az,el) -> 2D angle derivative.
  C projected_raw  : real 3D hip endpoints -> project_view(az,el) -> 2D angle deriv.
  D deploy_proxy   : the exact current deploy path (collapsed joints df).
  E clean_proj     : real hip endpoints projected with a GLOBAL-consistent basis
                     (u=P.right, v=P.up, NO per-joint vv.max() offset).

Diagnosis:
  A high, B low   -> project_view basis / v-normalization code problem
  B high, C low   -> real pelvis tilt / projection mixing loss
  C high, D low   -> deploy proxy implementation problem
  C & D low, E high -> project_view's per-joint v-offset is the culprit (fixable)
  C & D & E low   -> genuine 2D overhead deployment limit

Basis norm (=cos(el)) printed to rule out the el->90 gimbal singularity.
"""
import os, sys, argparse
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

AZIMUTHS = [0, 90, 180, 240, 270, 300]
ELEVATIONS = [75, 85, 88, 89, 89.5]
FPS_STRIDE = 1            # 360 fps only; projection loss is fps-independent
SIGNALS = ["A_ideal", "B_flat", "C_raw", "D_deploy", "E_clean"]


def sg_deriv(sig, fps):
    n = len(sig)
    win = max(5, int(round(0.05 * fps)))
    win += (win % 2 == 0)
    win = min(win, n - (n % 2 == 0))
    if win <= 3:
        return np.gradient(sig) * fps
    return savgol_filter(sig, win, 3, deriv=1, delta=1.0 / fps, mode="interp")


def peak_win(vel, lo, hi):
    seg = np.abs(vel[lo:hi + 1])
    if seg.size == 0 or np.all(np.isnan(seg)):
        return None, np.nan
    k = lo + int(np.nanargmax(seg))
    return k, float(np.abs(vel[k]))


def ang_rate_2d(rx, ry, lx, ly, fps):
    return np.degrees(sg_deriv(np.unwrap(np.arctan2(ry - ly, rx - lx)), fps))


def clean_basis(az, el):
    a, e = np.radians(az), np.radians(el)
    view = np.array([np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)])
    right = np.array([np.cos(a), -np.sin(a), 0.0])
    up = np.cross(right, view)
    return right, up / (np.linalg.norm(up) + 1e-12), np.cos(e)  # cos(e)=|right_raw|


def process(path):
    mk, fps = T.load_markers(path, T.PEL_MK)
    if any(m not in mk for m in T.PEL_MK):
        return None
    joints, _ = O.load_c3d_joints(path)
    arm = O.detect_throwing_arm(joints, fps)
    rel = M.release_frame(O.project_view(joints, azimuth_deg=0.0), arm, fps, M.JOINTS)
    n = joints["left_hip"].shape[1]
    lo, hi = max(0, rel - int(0.40 * fps)), min(n - 1, rel + int(0.05 * fps))

    # omega_ax pelvis truth peak (for timing error)
    Rp, sp = T.pelvis_frame(mk)
    truth = np.degrees(np.sum(T.omega_world(Rp, fps) * sp, axis=0))
    kt, _ = peak_win(_smooth(truth, fps), lo, hi)
    if kt is None:
        return None

    lhip, rhip = joints["left_hip"], joints["right_hip"]   # (3,n) each

    out = {}
    # A: ideal 3D XY
    a = ang_rate_2d(rhip[0], rhip[1], lhip[0], lhip[1], fps)
    out["A_ideal"] = {("*", "*"): peak_win(a, lo, hi)}

    # flattened joints (hips share mean Z) for B
    jflat = {k: v.copy() for k, v in joints.items()}
    mz = (lhip[2] + rhip[2]) / 2
    jflat["left_hip"] = np.vstack([lhip[0], lhip[1], mz])
    jflat["right_hip"] = np.vstack([rhip[0], rhip[1], mz])

    for s in ["B_flat", "C_raw", "D_deploy", "E_clean"]:
        out[s] = {}
    for el in ELEVATIONS:
        for az in AZIMUTHS:
            dfB = O.project_view(jflat, azimuth_deg=az, elevation_deg=el)
            dfC = O.project_view(joints, azimuth_deg=az, elevation_deg=el)
            bB = ang_rate_2d(dfB["right_hip_x"].to_numpy(), dfB["right_hip_y"].to_numpy(),
                             dfB["left_hip_x"].to_numpy(), dfB["left_hip_y"].to_numpy(), fps)
            bC = ang_rate_2d(dfC["right_hip_x"].to_numpy(), dfC["right_hip_y"].to_numpy(),
                             dfC["left_hip_x"].to_numpy(), dfC["left_hip_y"].to_numpy(), fps)
            out["B_flat"][(el, az)] = peak_win(bB, lo, hi)
            out["C_raw"][(el, az)] = peak_win(bC, lo, hi)
            out["D_deploy"][(el, az)] = peak_win(bC, lo, hi)  # same path as C by design
            # E: clean global-basis projection (no per-joint v offset)
            right, up, _ = clean_basis(az, el)
            uR, vR = rhip.T @ right, rhip.T @ up
            uL, vL = lhip.T @ right, lhip.T @ up
            bE = ang_rate_2d(uR, vR, uL, vL, fps)
            out["E_clean"][(el, az)] = peak_win(bE, lo, hi)
    return out, kt, fps


def _smooth(sig, fps):
    n = len(sig)
    win = max(5, int(round(0.05 * fps)))
    win += (win % 2 == 0)
    win = min(win, n - (n % 2 == 0))
    return savgol_filter(sig, win, 3, mode="interp") if win > 3 else sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    opel = poi.set_index("session_pitch")["max_pelvis_rotational_velo"].to_dict()

    rows = []
    n = fail = 0
    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1
            continue
        try:
            res = process(path)
            if res is None:
                fail += 1
                continue
            out, kt, fps = res
            ob = opel.get(r.session_pitch, np.nan)
            for s in SIGNALS:
                for key, (k, v) in out[s].items():
                    el, az = key
                    rows.append(dict(sp=r.session_pitch, sig=s, el=el, az=az,
                                     peak_frame=k, peak_val=v,
                                     err_ms=(abs(k - kt) * 1000.0 / fps) if k is not None else np.nan,
                                     obp=ob))
            n += 1
        except Exception:
            fail += 1
    print(f"processed {n} / fail {fail}\n")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(config.OBP_VALIDATION_DIR, "audit_pelvis_projection.csv"), index=False)

    # gimbal diagnostic
    print("[basis norm |right_raw| = cos(el)]  (>>1e-9 = safe)")
    for el in ELEVATIONS:
        print(f"  el={el:>5}  cos(el)={np.cos(np.radians(el)):.6f}")
    print()

    def gate(sub):
        d = sub.dropna(subset=["peak_val", "obp"])
        if len(d) < 15 or d["peak_val"].std() < 1e-9:
            return np.nan
        return float(np.corrcoef(d["peak_val"], d["obp"])[0, 1])

    # A is az/el-independent
    A = df[df.sig == "A_ideal"]
    print(f"A_ideal (3D XY control): mag_r={gate(A):.3f}  "
          f"err={A.err_ms.median():.1f}ms  scale={A.peak_val.median():.0f} deg/s\n")

    for el in ELEVATIONS:
        print(f"--- elevation {el} ---")
        print(f"{'signal':>9} {'mag_r':>7} {'err_ms':>8} {'scale':>8}   (best az in 240-300)")
        for s in ["B_flat", "C_raw", "D_deploy", "E_clean"]:
            best = None
            for az in AZIMUTHS:
                sub = df[(df.sig == s) & (df.el == el) & (df.az == az)]
                g = gate(sub)
                if best is None or (np.nan_to_num(g) > np.nan_to_num(best[1])):
                    best = (az, g, sub.err_ms.median(), sub.peak_val.median())
            print(f"{s:>9} {best[1]:>7.3f} {best[2]:>7.1f}m {best[3]:>8.0f}   az={best[0]}")
        print()


if __name__ == "__main__":
    main()
