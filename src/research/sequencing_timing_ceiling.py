"""
Diamond - Ceiling scan: is kinematic-sequencing TIMING recoverable from 2D?
sequencing_timing_ceiling.py

Hypothesis: rotational-velocity MAGNITUDE dies in 2D (survey), but the TIMING of
the pelvis/torso rotational-velocity peaks (the proximal-to-distal sequence the
O Seung-hwan app shows) may survive projection.

3D truth (per pitch, from c3d):
  pelvis axial angle = atan2 of the hip line (L_hip->R_hip) in the horizontal X-Y
  plane; torso axial angle = same for the shoulder line. SG-smoothed d/dt = the
  rotational velocity; its |peak| frame in a window around release = truth peak.
  dt_truth = (t_torso_peak - t_pelvis_peak). Cross-checked against OBP column
  timing_peak_torso_to_peak_pelvis_rot_velo before the rest is trusted.

2D proxies (per azimuth x elevation), each yields a pelvis signal (hip line) and a
torso signal (shoulder line); we take the SG d/dt |peak| frame of each:
  p1 width  : |R_x - L_x|            (projected segment width)
  p2 angle  : atan2(R_y-L_y, R_x-L_x) (2D image-plane line angle; needs elevation)
  p3 xvel   : R_x                     (single landmark x; confounded by translation)

Reported per (fps, elevation, azimuth, proxy):
  pelvis_err_ms / torso_err_ms  : median |t_2d - t_3d|
  dt_r2                         : r^2(dt_2d, dt_truth) across pitches
  order_acc vs order_base       : sign(dt) match rate vs majority-class baseline
Grading (recalibrated to the data: gap is 12+-20 ms, base sign rate ~0.70):
  A: peak err <=8 ms, dt_r2>=0.5, order_acc clears base by >=0.10
  360 fps is the real ceiling; 120 fps is the deployment-realistic limit.
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

# Full horizontal circle (incl. behind the pitcher, az 90-270) and the full
# elevation range up to near-overhead. Axial (Z-axis) rotation is seen FACE-ON
# from above, so overhead (high elev) is the physically-motivated regime for
# rotation-timing recovery - same reason HSS only recovered from overhead.
# el=90 is a gimbal singularity in project_view (view_dir || world_up); cap 85.
AZIMUTHS = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
ELEVATIONS = [0, 15, 30, 45, 60, 75, 85]
DECIMS = [(1, 360.0), (3, 120.0)]     # (stride, effective fps)
PROXIES = ["p1_width", "p2_angle", "p3_xvel"]


def sg_deriv(sig, fps):
    """SG first derivative; window ~50 ms, robust |peak| picking."""
    n = len(sig)
    win = max(5, int(round(0.05 * fps)))
    if win % 2 == 0:
        win += 1
    win = min(win, n if n % 2 == 1 else n - 1)
    if win <= 3:
        return np.gradient(sig) * fps
    return savgol_filter(sig, win, 3, deriv=1, delta=1.0 / fps, mode="interp")


def peak_in_window(vel, lo, hi):
    seg = np.abs(vel[lo:hi + 1])
    if seg.size == 0 or np.all(np.isnan(seg)):
        return None
    return lo + int(np.nanargmax(seg))


def line_angle(px_r, py_r, px_l, py_l):
    return np.unwrap(np.arctan2(py_r - py_l, px_r - px_l))


def truth_peaks(joints, fps, rel):
    """3D pelvis/torso rotational-velocity |peak| frames in a window at release."""
    Lh, Rh = joints["left_hip"], joints["right_hip"]
    Ls, Rs = joints["left_shoulder"], joints["right_shoulder"]
    th_p = np.unwrap(np.arctan2(Rh[1] - Lh[1], Rh[0] - Lh[0]))   # X-Y horizontal
    th_t = np.unwrap(np.arctan2(Rs[1] - Ls[1], Rs[0] - Ls[0]))
    wp, wt = sg_deriv(th_p, fps), sg_deriv(th_t, fps)
    n = len(th_p)
    lo, hi = max(0, rel - int(0.40 * fps)), min(n - 1, rel + int(0.05 * fps))
    return peak_in_window(wp, lo, hi), peak_in_window(wt, lo, hi), (lo, hi)


def proxy_peaks(df, arm, fps, window, proxy):
    lo, hi = window

    def col(name, ax):
        return df[f"{name}_{ax}"].to_numpy(float)

    if proxy == "p1_width":
        pel = np.abs(col("right_hip", "x") - col("left_hip", "x"))
        tor = np.abs(col("right_shoulder", "x") - col("left_shoulder", "x"))
    elif proxy == "p2_angle":
        pel = line_angle(col("right_hip", "x"), col("right_hip", "y"),
                         col("left_hip", "x"), col("left_hip", "y"))
        tor = line_angle(col("right_shoulder", "x"), col("right_shoulder", "y"),
                         col("left_shoulder", "x"), col("left_shoulder", "y"))
    else:  # p3_xvel : single landmark x (translation-confounded)
        pel = col("right_hip", "x")
        tor = col(f"{arm}_shoulder", "x")
    wp, wt = sg_deriv(pel, fps), sg_deriv(tor, fps)
    return peak_in_window(wp, lo, hi), peak_in_window(wt, lo, hi)


def process_pitch(joints_full, session_pitch):
    rows = []
    dt_truth_360 = None
    arm = O.detect_throwing_arm(joints_full, 360.0)
    for stride, fps in DECIMS:
        joints = {k: v[:, ::stride] for k, v in joints_full.items()}
        df0 = O.project_view(joints, azimuth_deg=0.0)     # side view for release
        try:
            rel = M.release_frame(df0, arm, fps, M.JOINTS)
        except Exception:
            continue
        tp3, tt3, window = truth_peaks(joints, fps, rel)
        if tp3 is None or tt3 is None:
            continue
        dt_truth = (tt3 - tp3) / fps
        if stride == 1:
            dt_truth_360 = dt_truth
        for el in ELEVATIONS:
            for az in AZIMUTHS:
                df = O.project_view(joints, azimuth_deg=az, elevation_deg=el)
                for proxy in PROXIES:
                    tp2, tt2 = proxy_peaks(df, arm, fps, window, proxy)
                    if tp2 is None or tt2 is None:
                        continue
                    dt_2d = (tt2 - tp2) / fps
                    rows.append(dict(
                        session_pitch=session_pitch, fps=int(fps), elev=el, az=az,
                        proxy=proxy,
                        pel_err_ms=abs(tp2 - tp3) * 1000.0 / fps,
                        tor_err_ms=abs(tt2 - tt3) * 1000.0 / fps,
                        dt_2d=dt_2d, dt_truth=dt_truth))
    return rows, dt_truth_360


def r2_of(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
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
    truth_by_sp = {}
    n = fail = 0
    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1
            continue
        try:
            joints, _ = O.load_c3d_joints(path)
            rows, dt360 = process_pitch(joints, r.session_pitch)
            all_rows.extend(rows)
            if dt360 is not None:
                truth_by_sp[r.session_pitch] = dt360
            n += 1
        except Exception:
            fail += 1
    print(f"processed {n} / fail {fail}")

    df = pd.DataFrame(all_rows)
    # attach OBP's official interval so we can score 2D vs the REAL metric,
    # not only vs our (line-based) 3D truth.
    obp_col = "timing_peak_torso_to_peak_pelvis_rot_velo"
    df = df.merge(poi[["session_pitch", obp_col]].rename(columns={obp_col: "obp_dt"}),
                  on="session_pitch", how="left")
    out_csv = os.path.join(config.OBP_VALIDATION_DIR, "sequencing_timing_proxy_survey.csv")
    df.to_csv(out_csv, index=False)
    print(f"saved rows -> {out_csv}\n")

    # ---- GATE: my 3D dt_truth vs OBP column ----
    tv = pd.DataFrame({"session_pitch": list(truth_by_sp),
                       "dt_mine": list(truth_by_sp.values())})
    tv = tv.merge(poi[["session_pitch", "timing_peak_torso_to_peak_pelvis_rot_velo"]],
                  on="session_pitch", how="inner").dropna()
    obp = tv["timing_peak_torso_to_peak_pelvis_rot_velo"].to_numpy()
    mine = tv["dt_mine"].to_numpy()
    rr = np.corrcoef(mine, obp)[0, 1] if len(tv) > 10 else np.nan
    print(f"[GATE] 3D dt_truth vs OBP column: n={len(tv)}  Pearson r={rr:+.3f}  "
          f"(mine mean={mine.mean()*1000:+.1f}ms, OBP mean={obp.mean()*1000:+.1f}ms)")
    print("  -> if |r| is low, the 3D truth axis/window is wrong; treat table with care.\n")

    # ---- baselines ----
    for fps in [360, 120]:
        sub = df[df.fps == fps]
        base = (sub.groupby("session_pitch").dt_truth.first())
        pos = float((base > 0).mean())
        print(f"[{fps}fps] order majority-class baseline = {max(pos, 1-pos):.3f} "
              f"(pos frac {pos:.3f})")
    print()

    # ---- aggregate every cell (to CSV summary + in-memory) ----
    g = df.groupby(["fps", "elev", "az", "proxy"])
    summary = []
    for (fps, el, az, proxy), grp in g:
        summary.append(dict(
            fps=fps, elev=el, az=az, proxy=proxy,
            pel_err_ms=grp.pel_err_ms.median(),
            tor_err_ms=grp.tor_err_ms.median(),
            dt_r2=r2_of(grp.dt_2d, grp.dt_truth),        # vs our line-based truth
            dt_r2_obp=r2_of(grp.dt_2d, grp.obp_dt),      # vs OBP official metric
            order_acc=float((np.sign(grp.dt_2d.replace(0, 1)) ==
                             np.sign(grp.dt_truth.replace(0, 1))).mean())))
    sm = pd.DataFrame(summary)
    sm_csv = os.path.join(config.OBP_VALIDATION_DIR, "sequencing_timing_summary.csv")
    sm.to_csv(sm_csv, index=False)
    print(f"saved cell summary -> {sm_csv}\n")

    # ---- overhead effect: per elevation, best over azimuth (360fps) ----
    s = sm[sm.fps == 360]
    print("[does OVERHEAD help? per-elevation best over azimuth, 360fps]")
    print(f"{'proxy':>9} {'elev':>5} {'dt_r2(mine)':>12} {'dt_r2(OBP)':>11} "
          f"{'min_pel_err':>12} {'min_tor_err':>12}")
    for proxy in PROXIES:
        for el in ELEVATIONS:
            c = s[(s.proxy == proxy) & (s.elev == el)]
            if c.empty:
                continue
            print(f"{proxy:>9} {el:>5} {c.dt_r2.max():>12.3f} "
                  f"{c.dt_r2_obp.max():>11.3f} "
                  f"{c.pel_err_ms.min():>11.1f}m {c.tor_err_ms.min():>11.1f}m")
        print()

    # ---- best cells overall (360fps) ----
    print("[top 12 cells by dt_r2 vs OBP official metric, 360fps]")
    for _, x in s.dropna(subset=["dt_r2_obp"]).nlargest(12, "dt_r2_obp").iterrows():
        print(f"  el={int(x.elev):>2} az={int(x.az):>3} {x.proxy:>9}  "
              f"dt_r2_OBP={x.dt_r2_obp:.3f}  dt_r2_mine={x.dt_r2:.3f}  "
              f"pel_err={x.pel_err_ms:.1f}ms tor_err={x.tor_err_ms:.1f}ms")
    print("\n[top 8 cells by lowest pelvis-peak error, 360fps]")
    for _, x in s.nsmallest(8, "pel_err_ms").iterrows():
        print(f"  el={int(x.elev):>2} az={int(x.az):>3} {x.proxy:>9}  "
              f"pel_err={x.pel_err_ms:.1f}ms tor_err={x.tor_err_ms:.1f}ms "
              f"dt_r2_OBP={x.dt_r2_obp:.3f}")


if __name__ == "__main__":
    main()
