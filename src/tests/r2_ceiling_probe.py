"""
Diamond - r2 loss decomposition probe: wrist_speed & knee_ext_velo_br
r2_ceiling_probe.py

Goal (user roadmap 2026-07-09): before sweeping variants, measure WHERE each
metric's r2 loss comes from. OBP projections have no detection error, so the
loss splits into (1) projection geometry, (2) normalization, (3) event/
derivative processing. Nothing is adopted here; metrics.py is NOT modified.

A) wrist_speed (truth = 3D peak wrist speed, m/s; baseline 0.598 @0):
     w_stat      baseline: 2D px peak / pixel_stature
     w_raw       2D px peak, UN-normalized. ppm is constant across pitches, so
                 this equals the in-plane 3D speed peak -> pure GEOMETRY
                 ceiling (not deployable -- diagnostic only).
     w_arm       / arm length (95th pct of shoulder->wrist px distance,
                 throwing arm). Speed = lever x angular velo, so arm length
                 may be the physically right scale, not stature.
     w_forearm   / forearm length (95th pct elbow->wrist px).
     w_subframe  parabolic sub-frame peak refinement, / stature
                 (measures frame-discretization loss).
     w_height    (px peak / pixel_stature) * session_height_m: reconstructs
                 ABSOLUTE m/s using the subject's known height. Deployable --
                 a smartphone user can enter their height once. Should
                 approach the w_raw geometry ceiling if normalization is the
                 bottleneck.
   Readout @0: geometry loss = 1 - r2(w_raw); normalization cost =
   r2(w_raw) - r2(w_stat); alt scales try to close that gap.

B) knee_ext_velo_br (truth = OBP column lead_knee_extension_angular_velo_br;
   baseline 0.65 @0). All 2D variants share the same knee-angle series
   (np.gradient * fps), only the sampling around release changes:
     k_base      baseline: gradient value at the release frame
     k_win1/2/3  mean of gradient over rel +/- 1/2/3 frames
                 (kills single-frame release-timing sensitivity)
     k_sg11      Savitzky-Golay(11, poly2) on the angle, then gradient @rel
   3D ceilings (no projection, computed once, azimuth-invariant, reported
   in the key block only):
     k3d_base    gradient of the 3D hip-knee-ankle angle @rel
     k3d_win2    same, mean over rel +/- 2
   If k3d_* is well below 1.0, our release timing / angle definition differs
   from how OBP computed the column -- that gap is unreachable from 2D.

Usage:
    python r2_ceiling_probe.py --limit 50
    python r2_ceiling_probe.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
import config
import obp_project as O
import metrics as M

AZIMUTHS = [0, 15, 30, 45, 60, 75, 90]
KNEE_TRUTH_COL = "lead_knee_extension_angular_velo_br"


def dist_pct(df, n1, n2, pct=95):
    """95th percentile of the px distance between two joints (robust 'segment
    length at full extension')."""
    d = np.hypot(df[f"{n1}_x"].to_numpy(float) - df[f"{n2}_x"].to_numpy(float),
                 df[f"{n1}_y"].to_numpy(float) - df[f"{n2}_y"].to_numpy(float))
    s = np.nanpercentile(d, pct)
    return float(s) if s > 1 else 1.0


def peak_subframe(v):
    """Peak of a series with parabolic sub-frame interpolation around argmax."""
    i = int(np.nanargmax(v))
    if i == 0 or i >= len(v) - 1:
        return float(v[i])
    y0, y1, y2 = v[i - 1], v[i], v[i + 1]
    denom = (y0 - 2 * y1 + y2)
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return float(y1)
    return float(y1 - (y0 - y2) ** 2 / (8 * denom))


def angle3d_series(joints, lead):
    """Per-frame 3D hip-knee-ankle angle (deg) of the lead leg."""
    h = joints[f"{lead}_hip"]; k = joints[f"{lead}_knee"]; a = joints[f"{lead}_ankle"]
    v1 = h - k; v2 = a - k
    dot = np.sum(v1 * v2, axis=0)
    n = np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0) + 1e-9
    return np.degrees(np.arccos(np.clip(dot / n, -1, 1)))


def knee2d_series(df, lead):
    hx = df[f"{lead}_hip_x"].to_numpy(float);  hy = df[f"{lead}_hip_y"].to_numpy(float)
    kx = df[f"{lead}_knee_x"].to_numpy(float); ky = df[f"{lead}_knee_y"].to_numpy(float)
    ax = df[f"{lead}_ankle_x"].to_numpy(float); ay = df[f"{lead}_ankle_y"].to_numpy(float)
    return M._angle(hx, hy, kx, ky, ax, ay)


def win_mean(v, i, k):
    lo, hi = max(0, i - k), min(len(v), i + k + 1)
    return float(np.nanmean(v[lo:hi]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    poi_knee = dict(zip(poi["session_pitch"], poi[KNEE_TRUTH_COL]))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    WRIST = ["w_stat", "w_raw", "w_arm", "w_forearm", "w_subframe", "w_height"]
    KNEE = ["k_base", "k_win1", "k_win2", "k_win3", "k_sg11"]
    T = {"wspeed": [], "knee": [], "k3d_base": [], "k3d_win2": []}
    E = {k: {az: [] for az in AZIMUTHS} for k in WRIST + KNEE}
    done = fail = 0

    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = O.load_c3d_joints(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
            df0 = O.project_view(joints, azimuth_deg=0.0)
            rel = M.release_frame(df0, arm, fps, M.JOINTS)
            fp = M.foot_plant_frame(df0, lead, fps, M.JOINTS, rel)
            if rel <= fp + 1 or fp < 3:
                fail += 1; continue

            # truths
            w = joints[f"{arm}_wrist"]
            T["wspeed"].append(float(np.nanmax(
                np.linalg.norm(np.diff(w, axis=1), axis=0) * fps)))
            T["knee"].append(poi_knee.get(r.session_pitch, np.nan))
            # 3D processing ceilings for knee velo (azimuth-invariant)
            kv3 = np.gradient(angle3d_series(joints, lead)) * fps
            T["k3d_base"].append(float(kv3[rel]))
            T["k3d_win2"].append(win_mean(kv3, rel, 2))

            wkey, skey = f"{arm}_wrist", f"{arm}_shoulder"
            ekey = f"{arm}_elbow"
            for az in AZIMUTHS:
                df = O.project_view(joints, azimuth_deg=az)
                stat = M.pixel_stature(df, M.JOINTS)
                spd = M._speed(df[f"{wkey}_x"].to_numpy(float),
                               df[f"{wkey}_y"].to_numpy(float), fps)
                pk = float(np.nanmax(spd))
                E["w_stat"][az].append(pk / stat)
                E["w_raw"][az].append(pk)
                E["w_arm"][az].append(pk / dist_pct(df, skey, wkey))
                E["w_forearm"][az].append(pk / dist_pct(df, ekey, wkey))
                E["w_subframe"][az].append(peak_subframe(spd) / stat)
                E["w_height"][az].append(pk / stat * float(r.session_height_m))

                knee = knee2d_series(df, lead)
                kv = np.gradient(knee) * fps
                E["k_base"][az].append(float(kv[rel]))
                E["k_win1"][az].append(win_mean(kv, rel, 1))
                E["k_win2"][az].append(win_mean(kv, rel, 2))
                E["k_win3"][az].append(win_mean(kv, rel, 3))
                ks = savgol_filter(knee, 11, 2) if len(knee) >= 11 else knee
                E["k_sg11"][az].append(float((np.gradient(ks) * fps)[rel]))
            done += 1
        except Exception:
            fail += 1
        if done and done % 100 == 0:
            print(f"  ...{done} processed", flush=True)
    print(f"processed {done} / failed {fail}\n")

    def r2(est_list, truth_list):
        e = np.asarray(est_list, float); t = np.asarray(truth_list, float)
        m = np.isfinite(e) & np.isfinite(t)
        if m.sum() <= 2:
            return np.nan
        r = np.corrcoef(e[m], t[m])[0, 1]
        return r * r

    PLAN = [
        ("wspd /stature (basel.)", "w_stat",     "wspeed"),
        ("wspd raw px (geom max)", "w_raw",      "wspeed"),
        ("wspd /arm length",       "w_arm",      "wspeed"),
        ("wspd /forearm length",   "w_forearm",  "wspeed"),
        ("wspd subframe /stature", "w_subframe", "wspeed"),
        ("wspd /stat*height_m",    "w_height",   "wspeed"),
        ("kvelo @rel (baseline)",  "k_base",     "knee"),
        ("kvelo win+-1",           "k_win1",     "knee"),
        ("kvelo win+-2",           "k_win2",     "knee"),
        ("kvelo win+-3",           "k_win3",     "knee"),
        ("kvelo SG(11,2) @rel",    "k_sg11",     "knee"),
    ]

    print("=" * 80)
    print("[r2 ceiling probe] loss decomposition, by azimuth (r2)")
    print("  wrist truth = 3D peak wrist speed; knee truth = OBP angular_velo_br")
    print("=" * 80)
    hdr = f"{'variant':24s}" + "".join(f"{az:>7d}" for az in AZIMUTHS) + "   best"
    print(hdr); print("-" * len(hdr))
    for label, ekey, tkey in PLAN:
        line = f"{label:24s}"; best_az, best = None, -1
        for az in AZIMUTHS:
            val = r2(E[ekey][az], T[tkey])
            if pd.notna(val) and val > best:
                best, best_az = val, az
            line += f"{val:7.2f}"
        line += f"   {best_az} ({best:.2f})"
        print(line)

    print("\nDecomposition @0deg:")
    g = r2(E["w_raw"][0], T["wspeed"]); s = r2(E["w_stat"][0], T["wspeed"])
    print(f"  wrist: geometry ceiling r2={g:.3f} (loss {1-g:.3f});"
          f" normalization cost = {g-s:.3f} (stature {s:.3f})")
    print(f"         alt scales: arm={r2(E['w_arm'][0], T['wspeed']):.3f}"
          f"  forearm={r2(E['w_forearm'][0], T['wspeed']):.3f}"
          f"  subframe={r2(E['w_subframe'][0], T['wspeed']):.3f}"
          f"  stat*height_m={r2(E['w_height'][0], T['wspeed']):.3f}")
    print(f"  knee:  3D processing ceilings vs OBP column:"
          f" @rel={r2(T['k3d_base'], T['knee']):.3f}"
          f"  win+-2={r2(T['k3d_win2'], T['knee']):.3f}")
    print(f"         2D @0: base={r2(E['k_base'][0], T['knee']):.3f}"
          f"  win1={r2(E['k_win1'][0], T['knee']):.3f}"
          f"  win2={r2(E['k_win2'][0], T['knee']):.3f}"
          f"  win3={r2(E['k_win3'][0], T['knee']):.3f}"
          f"  sg11={r2(E['k_sg11'][0], T['knee']):.3f}")


if __name__ == "__main__":
    main()
