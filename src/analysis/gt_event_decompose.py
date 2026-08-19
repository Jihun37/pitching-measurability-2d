"""Why do three metrics LOSE r2 when handed the OBP ground-truth events?

Context (2026-07-24). angle_zone_sweep.py --gt-events rebuilds the angle map on the
OBP landmark events instead of our detectors, which is what the map's question --
"is this metric recoverable from this viewpoint" -- actually asks. Nine of thirteen
metrics move by <=0.007 and COG Velo @PKH gains, but three fall hard:

    Stride (anchor)   0.839 -> 0.546      Knee Ext Velo BR  0.671 -> 0.427
    Stride Angle      0.968 -> 0.930  (at el0/az90: 0.91 -> 0.49)

Alignment is NOT the explanation: our peak-knee-height detector matches the OBP
landmark at median 0 frames, SD 0.5, 100% within 3 frames (obp_gt_events.py), so the
landmark grid and the c3d frame index are the same grid.

Three competing explanations, one test each:

  A. FIXED-OFFSET DEFINITION. Our release sits 4 frames before the OBP BR landmark
     with SD 0.9 -- a constant, not noise. If the metric is simply defined at a
     slightly different instant, then GT_event + that fixed offset recovers the r2,
     and the honest statement is a definition ("read 11 ms before release"), not a
     detector advantage. Scanned for both events.

  B. THE EVENT IS NOT NEEDED. Stride is a distance that PLATEAUS once the lead foot
     is down, so any frame inside the plateau gives the same answer. If a plateau
     median beats both event choices, the metric never needed a precise foot plant
     and its event sensitivity was self-inflicted.

  C. BRANCH-CUT ARTIFACT. stride_angle_2d returns atan2 in (-180, 180]. Pearson r2
     across a wrapped angle is meaningless. At el0 az and az+180 are exact mirrors,
     so their r2 MUST match -- the GT map gives az90 0.49 vs az270 0.87, which is
     only possible if one of them is wrapping. Re-scored on a branch-consistent
     representation (values recentred on their own circular mean).

Run:  conda activate diamond; cd src\\analysis; python gt_event_decompose.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config, metrics as M, obp_project as O
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from obp_gt_events import load_gt_events

LIMIT = 411
REL_OFF = list(range(-14, 7, 2))      # frames from the GT BR landmark
FP_OFF = list(range(-30, 31, 5))      # frames from the GT foot-plant landmark


def r2(e, t):
    e, t = np.asarray(e, float), np.asarray(t, float)
    m = np.isfinite(e) & np.isfinite(t)
    return np.corrcoef(e[m], t[m])[0, 1] ** 2 if m.sum() > 4 else np.nan


def knee_velo(df, lead, fps, f):
    hx, hy = df[f"{lead}_hip_x"].to_numpy(float), df[f"{lead}_hip_y"].to_numpy(float)
    kx, ky = df[f"{lead}_knee_x"].to_numpy(float), df[f"{lead}_knee_y"].to_numpy(float)
    ax, ay = df[f"{lead}_ankle_x"].to_numpy(float), df[f"{lead}_ankle_y"].to_numpy(float)
    k = M._angle(hx, hy, kx, ky, ax, ay)
    return (np.gradient(k) * fps)[f]


def recentre(deg):
    """Put a circular sample on one branch: rotate to its own circular mean, so a
    cluster straddling +-180 stops being split. Linear stats are only meaningful
    after this; unchanged for samples that never touch the cut."""
    a = np.asarray(deg, float)
    m = np.isfinite(a)
    if m.sum() < 2:
        return a
    mu = np.degrees(np.arctan2(np.nanmean(np.sin(np.radians(a[m]))),
                               np.nanmean(np.cos(np.radians(a[m])))))
    return (a - mu + 180.0) % 360.0 - 180.0


def main():
    gt = load_gt_events()
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    root = os.path.join(config.OBP_DATA_DIR, "c3d")

    A = {"det": [], "gt": [], **{f"o{o}": [] for o in REL_OFF}, "t": []}
    B = {"det": [], "gt": [], **{f"o{o}": [] for o in FP_OFF},
         "plateau_fp": [], "plateau_rel05": [], "plateau_rel08": [], "t": []}
    C = {f"{k}_{az}": [] for az in (90, 270) for k in ("det", "gt")}
    C["t"] = []
    frames = []
    done = 0

    for r in md.itertuples(index=False):
        if done >= LIMIT:
            break
        sp = r.session_pitch
        g = gt.get(sp)
        if sp not in poi.index or not g or not {"rel", "fp", "pkh"} <= set(g):
            continue
        path = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            continue
        try:
            j, fps = load_feet(path)
            arm = O.detect_throwing_arm(j, fps)
            lead = "left" if arm == "right" else "right"
            trail = "right" if lead == "left" else "left"
            # detect-once convention: events from the az0/el0 side view, reused
            df0 = O.project_view(j, azimuth_deg=0.0)
            rel_d = int(M.release_frame(df0, arm, fps, M.JOINTS))
            fp_d = int(M.foot_plant_frame(df0, lead, fps, M.JOINTS, rel_d))
            if rel_d <= fp_d + 1 or fp_d < 3:
                continue
            rel_g, fp_g = int(g["rel"]), int(g["fp"])
            d90 = project_cam(j, 90, 0)
            d180 = project_cam(j, 180, 0)
            d270 = project_cam(j, 270, 0)
            n = len(df0)
        except Exception:
            continue

        def ok(f):
            return 1 <= int(f) < n - 1

        # --- A: knee extension velocity at release, az0 (its own best view) ---
        tA = poi.loc[sp, "lead_knee_extension_angular_velo_br"]
        if np.isfinite(tA):
            A["t"].append(tA)
            A["det"].append(knee_velo(df0, lead, fps, rel_d) if ok(rel_d) else np.nan)
            A["gt"].append(knee_velo(df0, lead, fps, rel_g) if ok(rel_g) else np.nan)
            for o in REL_OFF:
                f = rel_g + o
                A[f"o{o}"].append(knee_velo(df0, lead, fps, f) if ok(f) else np.nan)

        # --- B: stride length at foot plant, az180 (its own best view) ---
        tB = poi.loc[sp, "stride_length"]
        la = d180[f"{lead}_ankle_x"].to_numpy(float)
        ta = d180[f"{trail}_ankle_x"].to_numpy(float)
        stat = M.pixel_stature(d180, M.JOINTS)
        if np.isfinite(tB) and np.isfinite(stat) and stat > 0:
            B["t"].append(tB)

            def stride_at(f):
                f = int(f)
                if not ok(f):
                    return np.nan
                return abs(la[f] - M.trail_anchor_x(ta, f, fps)) / stat

            B["det"].append(stride_at(fp_d))
            B["gt"].append(stride_at(fp_g))
            for o in FP_OFF:
                B[f"o{o}"].append(stride_at(fp_g + o))
            anc_r = M.trail_anchor_x(ta, rel_g, fps)
            # plateau variants: read the SETTLED lead-ankle position instead of
            # the position at one detected frame
            B["plateau_fp"].append(abs(float(np.nanmedian(la[fp_g:rel_g + 1])) - anc_r) / stat)
            w5 = max(1, int(0.05 * fps)); w8 = max(1, int(0.08 * fps))
            B["plateau_rel05"].append(
                abs(float(np.nanmedian(la[max(0, rel_g - w5):rel_g + 1])) - anc_r) / stat)
            B["plateau_rel08"].append(
                abs(float(np.nanmedian(la[max(0, rel_g - w8):rel_g + 1])) - anc_r) / stat)

        # --- C: stride angle at foot plant, az90 vs its az270 mirror ---
        tC = poi.loc[sp, "stride_angle"]
        if np.isfinite(tC):
            C["t"].append(tC)
            for az, dfv in ((90, d90), (270, d270)):
                C[f"det_{az}"].append(M.stride_angle_2d(dfv, arm, fp_d, M.JOINTS)
                                      if ok(fp_d) else np.nan)
                C[f"gt_{az}"].append(M.stride_angle_2d(dfv, arm, fp_g, M.JOINTS)
                                     if ok(fp_g) else np.nan)

        frames.append(dict(sp=sp, rel_d=rel_d, rel_g=rel_g, fp_d=fp_d, fp_g=fp_g,
                           d_rel=rel_d - rel_g, d_fp=fp_d - fp_g,
                           span=rel_g - int(g["pkh"])))
        done += 1

    fr = pd.DataFrame(frames)
    print(f"pitches {done}\n")
    print(f"event offsets (ours - GT), frames @360fps:")
    for c in ("d_rel", "d_fp"):
        v = fr[c].to_numpy(float)
        print(f"  {c:>6}  median {np.median(v):>+6.1f}  SD {v.std():>6.1f}  "
              f"IQR {np.percentile(v,25):>+.0f}..{np.percentile(v,75):>+.0f}  "
              f"|.|<=3f {np.mean(np.abs(v)<=3):>6.1%}")
    print(f"  pkh->release span: median {np.median(fr.span):.0f} frames "
          f"({np.median(fr.span)/360*1000:.0f} ms)")

    print("\n" + "=" * 72)
    print("[A] Knee Ext Velo BR @az0  -- is it a fixed-offset definition?")
    print("=" * 72)
    print(f"  our detected release      r2 {r2(A['det'], A['t']):.3f}")
    print(f"  GT BR landmark            r2 {r2(A['gt'], A['t']):.3f}")
    print(f"  {'offset from GT BR':>19} {'ms':>6} {'r2':>8}")
    for o in REL_OFF:
        print(f"  {o:>19} {o/360*1000:>6.0f} {r2(A[f'o{o}'], A['t']):>8.3f}")

    print("\n" + "=" * 72)
    print("[B] Stride (anchor) @az180 -- does it need the event at all?")
    print("=" * 72)
    print(f"  our detected foot plant   r2 {r2(B['det'], B['t']):.3f}")
    print(f"  GT foot plant (fp_100)    r2 {r2(B['gt'], B['t']):.3f}")
    print(f"  {'offset from GT fp':>19} {'ms':>6} {'r2':>8}")
    for o in FP_OFF:
        print(f"  {o:>19} {o/360*1000:>6.0f} {r2(B[f'o{o}'], B['t']):>8.3f}")
    print("  event-free plateau reads (lead ankle settled position):")
    for k, lbl in (("plateau_fp", "median over [fp_gt, rel]"),
                   ("plateau_rel05", "median over [rel-50ms, rel]"),
                   ("plateau_rel08", "median over [rel-80ms, rel]")):
        print(f"    {lbl:<28} r2 {r2(B[k], B['t']):.3f}")

    print("\n" + "=" * 72)
    print("[C] Stride Angle @el0 -- is the az90/az270 asymmetry a branch cut?")
    print("=" * 72)
    print(f"  {'':>10} {'raw r2':>9} {'recentred r2':>14} {'spread(deg)':>13} "
          f"{'near +-180':>11}")
    for k in ("det_90", "gt_90", "det_270", "gt_270"):
        v = np.asarray(C[k], float)
        fin = v[np.isfinite(v)]
        near = np.mean(np.abs(np.abs(fin) - 180) < 30) if fin.size else np.nan
        spread = (np.nanpercentile(fin, 97.5) - np.nanpercentile(fin, 2.5)
                  if fin.size else np.nan)
        print(f"  {k:>10} {r2(v, C['t']):>9.3f} {r2(recentre(v), C['t']):>14.3f} "
              f"{spread:>13.0f} {near:>11.1%}")
    print("\n  el0 mirrors must agree: az90 and az270 are the same plane u-flipped,")
    print("  so any raw r2 difference between them is a wrapping artifact.")


if __name__ == "__main__":
    main()
