"""
Diamond - Deployable overhead recovery of rotation magnitude + timing (step 3)
sequencing_overhead_deploy.py

Truth is now the decoded OBP quantity: omega_ax (rigid-body angular velocity about
the segment superior axis), which reproduces OBP max_*_rotational_velo at r=0.92
(pelvis) / 0.99 (torso). The DEPLOYABLE 2D signal is only the line that 2D pose can
see: hip line (pelvis) and shoulder line (torso), via each line's 2D image-angle
derivative (the `p2_angle` proxy, the overhead winner).

Per (fps, elevation, azimuth) we ask the two product questions:
  MAGNITUDE  : does the 2D line peak |dangle/dt| correlate with OBP
               max_pelvis/torso_rotational_velo?  (the app's "peak deg/s")
  TIMING     : per-segment peak-frame error vs the omega_ax truth (ms), and the
               pelvis->torso interval dt_r2 vs OBP timing column.
Shown per elevation (best over azimuth) so the overhead effect is explicit.

Reuses the rigid-body truth builders from truth_redefine_gate.
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
import truth_redefine_gate as T   # load_markers, pelvis_frame, torso_frame, omega_world, unit

AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]
ELEVATIONS = [0, 30, 45, 60, 75, 85]
DECIMS = [(1, 360.0), (3, 120.0)]


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


def omega_ax_signal(R, sup, fps):
    w = T.omega_world(R, fps)
    return np.degrees(np.sum(w * sup, axis=0))


def line_angle_2d(df, r_name, l_name):
    rx, ry = df[f"{r_name}_x"].to_numpy(float), df[f"{r_name}_y"].to_numpy(float)
    lx, ly = df[f"{l_name}_x"].to_numpy(float), df[f"{l_name}_y"].to_numpy(float)
    return np.unwrap(np.arctan2(ry - ly, rx - lx))


def line_yaw_3d(a, b, fps):
    """True horizontal-plane azimuth rate of the 3D line a->b (deg/s), Z dropped."""
    ang = np.unwrap(np.arctan2(b[1] - a[1], b[0] - a[0]))
    return np.degrees(sg_deriv(ang, fps))


def process(path):
    mk_full, fps0 = T.load_markers(path, T.PEL_MK + T.TOR_MK + ["LSHO", "RSHO"])
    if any(m not in mk_full for m in T.PEL_MK + T.TOR_MK + ["LSHO", "RSHO"]):
        return []
    joints_full, _ = O.load_c3d_joints(path)
    arm = O.detect_throwing_arm(joints_full, fps0)

    rows = []
    for stride, fps in DECIMS:
        mk = {k: v[:, ::stride] for k, v in mk_full.items()}
        joints = {k: v[:, ::stride] for k, v in joints_full.items()}
        df0 = O.project_view(joints, azimuth_deg=0.0)
        try:
            rel = M.release_frame(df0, arm, fps, M.JOINTS)
        except Exception:
            continue
        n = mk["LASI"].shape[1]
        lo, hi = max(0, rel - int(0.40 * fps)), min(n - 1, rel + int(0.05 * fps))

        # 3D truth (omega_ax)
        Rp, sp = T.pelvis_frame(mk)
        Rt, st = T.torso_frame(mk)
        pel_t = sg_smooth(omega_ax_signal(Rp, sp, fps), fps)
        tor_t = sg_smooth(omega_ax_signal(Rt, st, fps), fps)
        kpt, _ = peak_win(pel_t, lo, hi)
        ktt, _ = peak_win(tor_t, lo, hi)
        if kpt is None or ktt is None:
            continue
        dt_truth = (ktt - kpt) / fps

        # CONTROL: ideal 3D horizontal line yaw (no projection) — isolates the
        # projection loss from the line-vs-omega_ax loss.
        lhip = (mk["LASI"] + mk["LPSI"]) / 2
        rhip = (mk["RASI"] + mk["RPSI"]) / 2
        pel3d = line_yaw_3d(lhip, rhip, fps)
        tor3d = line_yaw_3d(mk["LSHO"], mk["RSHO"], fps)
        kp3, vp3 = peak_win(pel3d, lo, hi)
        kt3, vt3 = peak_win(tor3d, lo, hi)
        rows.append(dict(
            session_pitch=None, fps=int(fps), elev=-1, az=-1,   # el=-1 marks control
            pel_err_ms=abs(kp3 - kpt) * 1000.0 / fps,
            tor_err_ms=abs(kt3 - ktt) * 1000.0 / fps,
            pel_mag=vp3, tor_mag=vt3,
            dt_2d=(kt3 - kp3) / fps, dt_truth=dt_truth))

        for el in ELEVATIONS:
            for az in AZIMUTHS:
                df = O.project_view(joints, azimuth_deg=az, elevation_deg=el)
                pel_v = sg_deriv(line_angle_2d(df, "right_hip", "left_hip"), fps)
                tor_v = sg_deriv(line_angle_2d(df, "right_shoulder", "left_shoulder"), fps)
                pel_v = np.degrees(pel_v)
                tor_v = np.degrees(tor_v)
                kp2, vp2 = peak_win(pel_v, lo, hi)
                kt2, vt2 = peak_win(tor_v, lo, hi)
                if kp2 is None or kt2 is None:
                    continue
                rows.append(dict(
                    session_pitch=None, fps=int(fps), elev=el, az=az,
                    pel_err_ms=abs(kp2 - kpt) * 1000.0 / fps,
                    tor_err_ms=abs(kt2 - ktt) * 1000.0 / fps,
                    pel_mag=vp2, tor_mag=vt2,
                    dt_2d=(kt2 - kp2) / fps, dt_truth=dt_truth))
    return rows


def sg_smooth(sig, fps):
    n = len(sig)
    win = max(5, int(round(0.05 * fps)))
    win += (win % 2 == 0)
    win = min(win, n - (n % 2 == 0))
    if win <= 3:
        return sig
    return savgol_filter(sig, win, 3, mode="interp")


def r2_of(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 15 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1] ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    all_rows = []
    n = fail = 0
    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1
            continue
        try:
            rows = process(path)
            for row in rows:
                row["session_pitch"] = r.session_pitch
            all_rows.extend(rows)
            n += 1 if rows else 0
        except Exception:
            fail += 1
    print(f"processed {n} / fail {fail}")

    df = pd.DataFrame(all_rows).merge(
        poi[["session_pitch", "max_pelvis_rotational_velo",
             "max_torso_rotational_velo",
             "timing_peak_torso_to_peak_pelvis_rot_velo"]],
        on="session_pitch", how="left")
    out = os.path.join(config.OBP_VALIDATION_DIR, "sequencing_overhead_deploy.csv")
    df.to_csv(out, index=False)
    print(f"saved -> {out}\n")

    # control: ideal 3D horizontal line (el=-1)
    print("[CONTROL: ideal 3D horizontal line yaw, no projection]")
    for fps in [360, 120]:
        c = df[(df.fps == fps) & (df.elev == -1)]
        if len(c) < 15:
            continue
        pm = np.corrcoef(c.pel_mag, c.max_pelvis_rotational_velo)[0, 1]
        tm = np.corrcoef(c.tor_mag, c.max_torso_rotational_velo)[0, 1]
        print(f"  {fps}fps  PELmag_r={pm:.3f} TORmag_r={tm:.3f} "
              f"pel_err={c.pel_err_ms.median():.1f}ms tor_err={c.tor_err_ms.median():.1f}ms "
              f"dtR2_OBP={r2_of(c.dt_2d, c.timing_peak_torso_to_peak_pelvis_rot_velo):.3f}")
    print()

    for fps in [360, 120]:
        s = df[(df.fps == fps) & (df.elev >= 0)]
        print(f"================= {fps} fps =================")
        print(f"{'el':>3} {'PELmag_r':>9} {'TORmag_r':>9} {'pel_err':>8} "
              f"{'tor_err':>8} {'dtR2_OBP':>9} {'dtR2_tru':>9}   (best over az)")
        for el in ELEVATIONS:
            best = None
            for az in AZIMUTHS:
                g = s[(s.elev == el) & (s.az == az)]
                if len(g) < 15:
                    continue
                pm = np.corrcoef(g.pel_mag, g.max_pelvis_rotational_velo)[0, 1] \
                    if g.max_pelvis_rotational_velo.notna().sum() > 15 else np.nan
                tm = np.corrcoef(g.tor_mag, g.max_torso_rotational_velo)[0, 1] \
                    if g.max_torso_rotational_velo.notna().sum() > 15 else np.nan
                row = dict(az=az, pm=pm, tm=tm,
                           pe=g.pel_err_ms.median(), te=g.tor_err_ms.median(),
                           d_obp=r2_of(g.dt_2d, g.timing_peak_torso_to_peak_pelvis_rot_velo),
                           d_tru=r2_of(g.dt_2d, g.dt_truth))
                # rank elevation's best azimuth by pelvis magnitude gate
                if best is None or (np.nan_to_num(row["pm"]) > np.nan_to_num(best["pm"])):
                    best = row
            if best:
                print(f"{el:>3} {best['pm']:>9.3f} {best['tm']:>9.3f} "
                      f"{best['pe']:>7.1f}m {best['te']:>7.1f}m "
                      f"{best['d_obp']:>9.3f} {best['d_tru']:>9.3f}   az={best['az']}")
        print()


if __name__ == "__main__":
    main()
