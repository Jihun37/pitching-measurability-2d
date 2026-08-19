"""
Diamond - viewpoint QUADRANT disambiguation probe (open task (d), extended).

Motivation (2026-07-11 PM, after the peak-selection study): with the TRUE az,
the existing OBP offset grid + race rule puts 12/14 dev clips within +-2
frames - the release bottleneck is now viewpoint MIS-identification, not the
detector. kNN vote errors on the dev set are mirror-family members of the
true az: +180 flips (clips 10, 13) and u-flips 180-az (clips 02, 06, 08).

Two signals resolve the 4-fold family {v, 180-v, v+180, 360-v}:
  1. face bit: nose_v mean over [rel-60f, rel+12f]. Dev set: open half
     (az 30-210) 0.996-1.04 vs closed half 0.67-0.91 (clean gap at 0.95).
     No OBP analog (projections have no confidence) - real-video only.
  2. drive-direction bit: stature-normalized image-x drift of the hip
     midpoint from windup to release. Geometric, so its az curve is
     measured on OBP projections and applies to real video.

Probe outputs:
  --obp   : median drift dx(az, el) curve on OBP -> viewpoint_drift_curve.csv
  --real  : per clip - kNN-voted az, mirror-family candidates, disambiguated
            az (drift + face consistency), vs TRUE az; then the full release
            chain rescored with the corrected az (race rule + gated offset).

Read-only study; nothing wired into deploy/.

Run:  cd src\tests
      python viewpoint_disambig_probe.py --obp [--limit 40]
      python viewpoint_disambig_probe.py --real
"""
import os, sys, argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
for p in ("..", "../stage2", "../stage3", "../analysis", "../deploy"):
    sys.path.insert(0, os.path.join(_HERE, p))

import config
import obp_project as O
import metrics as M

AZ = list(range(0, 360, 15))
EL = [0, 15, 30]
CURVE_CSV = os.path.join(config.OBP_VALIDATION_DIR, "viewpoint_drift_curve.csv")
NOSE_OPEN_MIN = 0.95          # dev-set gap: open half >=0.996, closed <=0.908


def hip_drift(df, rel, J):
    """Stature-normalized image-x drift of the hip midpoint, windup->release.
    Start = median over the first 20% of frames before foot plant motion
    (robust to early jitter); end = release frame."""
    lx = df[f"{J['l_hip']}_x"].to_numpy(float)
    rx = df[f"{J['r_hip']}_x"].to_numpy(float)
    mx = (lx + rx) / 2.0
    stat = M.pixel_stature(df, J)
    if not np.isfinite(stat) or stat <= 0:
        return np.nan
    n0 = max(3, int(0.2 * rel))
    x0 = np.nanmedian(mx[:n0])
    x1 = mx[int(rel)]
    return float((x1 - x0) / stat)


def run_obp(limit):
    from master_angle_table import load_feet
    from hss_elevation_test import project_cam

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    acc = {(az, el): [] for az in AZ for el in EL}
    done = fail = 0
    for r in md.itertuples(index=False):
        if limit and done >= limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = load_feet(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
            df0 = O.project_view(joints, azimuth_deg=0.0)
            rel = M.release_frame(df0, arm, fps, M.JOINTS)
            fp = M.foot_plant_frame(df0, lead, fps, M.JOINTS, rel)
            if rel <= fp + 1 or fp < 3:
                fail += 1; continue
            for az in AZ:
                for el in EL:
                    df = project_cam(joints, az, el)
                    acc[(az, el)].append(hip_drift(df, rel, M.JOINTS))
            done += 1
        except Exception:
            fail += 1
        if done and done % 20 == 0:
            print(f"  ...{done} pitches")
    print(f"processed {done} / failed {fail}")

    rows = [{"az": az, "el": el,
             "dx_med": float(np.nanmedian(acc[(az, el)])),
             "dx_iqr": float(np.nanpercentile(acc[(az, el)], 75)
                             - np.nanpercentile(acc[(az, el)], 25))}
            for az in AZ for el in EL]
    t = pd.DataFrame(rows)
    t.to_csv(CURVE_CSV, index=False)
    print(f"curve -> {CURVE_CSV}\n")
    for el in EL:
        s = t[t.el == el]
        print(f"el={el:>2}  " + " ".join(f"{v:+.2f}" for v in s.dx_med))
    print("az:    " + " ".join(f"{az:>5d}" for az in AZ))


def family(az):
    """4-fold mirror family of a voted azimuth, snapped to the 15deg grid."""
    return sorted({int(round((a % 360) / 15)) * 15 % 360
                   for a in (az, 180 - az, az + 180, 360 - az)})


def nose_bit(name, rel):
    p = os.path.join(config.ROOT, "data", "outputs", name,
                     f"{name}_coords_rtmp.csv")
    v = pd.read_csv(p)["nose_v"].to_numpy(float)
    lo, hi = max(0, int(rel) - 60), min(len(v), int(rel) + 12)
    return float(np.nanmean(v[lo:hi]))


def in_open_half(az):
    """Open half = az 15-210: the dev-set face-visible arc (az15 clips read
    nose_v ~1.0, az0 reads 0.908, az225+ reads <=0.91). Dev-derived boundary."""
    return 15 <= (az % 360) <= 210


def disambiguate(voted_az, el, dx, nose, curve):
    """Pick the mirror-family member most consistent with the drift value and
    the face bit. Score = |dx - dx_med(cand)| + penalty if the face bit
    contradicts the candidate's half."""
    cands = family(voted_az)
    face_open = nose >= NOSE_OPEN_MIN
    best = None
    for c in cands:
        row = curve[(curve.az == c) & (curve.el == el)]
        if row.empty:
            continue
        s = abs(dx - float(row.dx_med.iloc[0]))
        if in_open_half(c) != face_open:
            s += 10.0                      # face bit is near-binary: veto
        if best is None or s < best[0]:
            best = (s, c)
    return (best[1], best[0]) if best else (voted_az, np.inf)


def run_real(el0=False, gt_csv=None):
    from measure_auto import load_clip
    from real_station_test import detect_arm_2d
    from release_offset import offset_ms, bin_iqr_ms, wrist_glitch_near, \
        MIN_VOTE_SHARE, MAX_BIN_IQR_MS, CROSS_STRATEGY_MAX_MS
    from angle_zone_sweep import release_view
    from peak_select_probe import peak_candidates, wrist_ext
    import cv2

    curve = pd.read_csv(CURVE_CSV)
    # Axis-convention calibration: project_cam's u axis and the real-video
    # pixel x axis point opposite ways - ALL 14 measurable dev clips show the
    # OBP-predicted drift with the sign flipped (2^-14 by chance). One global
    # sign, not a per-clip fit.
    curve["dx_med"] = -curve["dx_med"]
    gt = pd.read_csv(gt_csv or os.path.join(config.ROOT, "data", "outputs",
                                            "release_gt_real15.csv"))

    print(f"{'clip':<22}{'true':>5}{'voted':>6}{'disamb':>7}   "
          f"{'dx':>6} {'resid':>5} {'nose':>5}  {'strat':>7} "
          f"{'off_s':>7} {'err_s':>6} {'off_r':>7} {'err_r':>6}")
    errs = {}
    for r in gt.itertuples(index=False):
        name = r.clip
        df, raw = load_clip(name, "_rtmp")
        arm = detect_arm_2d(df)
        vp = next((os.path.join(config.ROOT, "data", "videos", name + ext)
                   for ext in (".mp4", ".MOV", ".mov")
                   if os.path.exists(os.path.join(config.ROOT, "data",
                                                  "videos", name + ext))), None)
        cap = cv2.VideoCapture(vp); fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # provisional release (argmax side) anchors drift + nose windows
        rel0 = M.release_frame(df, arm, fps, M.JOINTS, view="side", raw_df=raw)
        dx = hip_drift(df, rel0, M.JOINTS)
        nv = nose_bit(name, rel0)
        vaz, vel = int(r.voted_az), int(r.voted_el)
        if el0:
            # user-DECLARED camera elevation (ground-level shoot). Camera
            # height is user-controlled and knowable, unlike azimuth noise;
            # tests the deployment mode where the app asks for it.
            vel = 0
        el = min(EL, key=lambda e: abs(e - vel))
        caz, resid = disambiguate(vaz, el, dx, nv, curve)

        # --- release chain with the corrected az ---
        view = release_view(caz, vel)
        wx, wy, ext = wrist_ext(df, arm, M.JOINTS)
        spd = M._speed(wx, wy, fps)
        pk, h = peak_candidates(spd, fps)
        gm = int(pk[np.argmax(h)]) if pk.size else rel0
        rel = gm
        if view == "frontal":
            rel_f = M.release_frame(df, arm, fps, M.JOINTS, view="frontal",
                                    raw_df=raw)
            if abs(rel_f - gm) / fps * 1000.0 <= CROSS_STRATEGY_MAX_MS:
                rel, view = rel_f, "frontal"
            else:
                view = "side"
        if view == "side":
            # race rule, gated on the face bit (rear = face hidden)
            sel = pk[h >= 0.5 * h.max()] if pk.size else np.array([], int)
            if (nv < NOSE_OPEN_MIN and sel.size
                    and (gm - int(sel[0])) / fps * 1000.0 >= 80.0):
                rel = int(sel[0])
            # raw refine (mirrors metrics.release_frame two-stage)
            if raw is not None:
                rwx, rwy, _ = wrist_ext(raw, arm, M.JOINTS)
                rspd = M._speed(rwx, rwy, fps)
                lo, hi = max(0, rel - 3), min(len(rspd) - 1, rel + 3)
                if hi >= lo and not np.all(np.isnan(rspd[lo:hi + 1])):
                    rel = lo + int(np.nanargmax(rspd[lo:hi + 1]))
        if wrist_glitch_near(df, arm, fps, M.JOINTS, rel, raw_df=raw):
            print(f"{name:<22}{int(r.nominal_az):>5}{vaz:>6}{caz:>7}   "
                  f"{dx:>+6.2f} {nv:>5.2f}  {'DEFER':>7}")
            continue
        # offset, two gate variants sharing the disambiguated az + IQR gate:
        #   strict: kNN top-bin share >= 0.20 (the shipped gate)
        #   relax : share OR a tight drift-curve match (resid <= 0.15) vouches
        #           for the bin - tests whether the disambiguator can replace
        #           vote confidence as the offset's trust signal
        off_s = off_r = 0.0
        if view == "side" and bin_iqr_ms(caz, vel) <= MAX_BIN_IQR_MS:
            share = float(r.vote_share)
            o = offset_ms(caz, vel)
            if share >= MIN_VOTE_SHARE:
                off_s = o
            if share >= MIN_VOTE_SHARE or resid <= 0.15:
                off_r = o

        def apply(off):
            return int(np.clip(int(round(rel - off * fps / 1000.0)),
                               0, len(df) - 1)) - int(r.true_release)
        e_s, e_r = apply(off_s), apply(off_r)
        errs[name] = (e_s, e_r)
        print(f"{name:<22}{int(r.nominal_az):>5}{vaz:>6}{caz:>7}   "
              f"{dx:>+6.2f} {resid:>5.2f} {nv:>5.2f}  {view:>7} "
              f"{off_s:>+7.1f} {e_s:>+6d} {off_r:>+7.1f} {e_r:>+6d}")

    for i, lab in ((0, "strict"), (1, "relax ")):
        ae = np.abs([v[i] for v in errs.values()])
        print(f"\n[{lab}] measured {len(ae)}/{len(gt)}"
              f"   <=1f {int((ae <= 1).sum())}/{len(ae)}"
              f"   <=2f {int((ae <= 2).sum())}/{len(ae)}"
              f"   <=3f {int((ae <= 3).sum())}/{len(ae)}"
              f"   <=5f {int((ae <= 5).sum())}/{len(ae)}"
              f"   >10f {int((ae > 10).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obp", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--el0", action="store_true",
                    help="user-declared ground-level camera (el=0)")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--gt", default=None,
                    help="alternate GT csv (release_gt_real15 schema)")
    a = ap.parse_args()
    if a.obp:
        run_obp(a.limit)
    if a.real:
        run_real(el0=a.el0, gt_csv=a.gt)
