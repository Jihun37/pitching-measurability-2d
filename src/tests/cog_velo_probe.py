"""
Diamond - COG forward-velocity DEFINITION probe (whole-body centroid).

ROADMAP #1 batch-1 flagged max_cog_velo_x as MARGINAL: r2 ~0.41 @0deg with the
COG approximated by the HIP MIDPOINT (crude), and noted "a full-body centroid
may push it over 0.5 - refine before verdict." This probe does that refinement,
plus applies the wrist-speed lesson (unit-free -> absolute m/s scaling can lift
r2 on its own).

Two independent axes tested against the SAME GT (poi max_cog_velo_x, in m/s,
mean 3.06 / sd 0.255):
  (1) CENTROID definition:
        hipmid          - hip-midpoint x (current proxy, baseline)
        shomid_hipmid   - mean(shoulder-mid, hip-mid) (upper-body 2-point)
        joints_mean     - unweighted mean of all 13 joints
        segw_com        - Winter segment-mass-weighted whole-body COM
  (2) SCALING of the peak forward speed:
        unit-free (px/frame*fps / pixel_stature) - reproduces ROADMAP's 0.41
        m/s       (unit-free * session_height_m) - matches GT units, adoptable

Fixed like the rest of the abs-accuracy thread: side view az0/el0 (forward
velocity is the pitching-direction axis, best seen from the side); events
detected ONCE on project_view(az=0) el=0; speed = max |d(COM_x)/dt| over
[0, release]; LOCO folds = OBP subject `user`; models from loco_calibration.

The official cog_velo_x_max (extended_metrics_test.py / metrics) is UNTOUCHED;
adopt a refined centroid only if it CLEARLY clears the ~0.5 r2 floor AND wins
raw+LOCO absolute stats over the hip-midpoint baseline.

Run:  cd src\tests
      python cog_velo_probe.py
Output: cog_velo_probe.csv
"""
import os, sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3", "../analysis"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)

import config
import obp_project as O
import metrics as M
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from loco_calibration import fit_apply, score, MODELS

AZ, EL = 0, 0
GT_COL = "max_cog_velo_x"

# Winter segment mass fractions + COM location (fraction from proximal joint).
# hand folded into forearm ("forearm+hand"), foot COM approximated at the ankle
# (no toe marker). Weights sum to 1.0.
def _xy(df, j):
    return df[f"{j}_x"].to_numpy(float), df[f"{j}_y"].to_numpy(float)


def segw_com_x(df):
    """Winter segment-weighted whole-body COM, image-x series (px)."""
    def P(j):  # proximal/distal point series as (x,y)
        return np.array(_xy(df, j))
    Ls, Rs = P("left_shoulder"), P("right_shoulder")
    Lh, Rh = P("left_hip"), P("right_hip")
    sho_mid = (Ls + Rs) / 2
    hip_mid = (Lh + Rh) / 2
    head = P("head")
    segs = []
    segs.append((0.081, head))                                  # head+neck
    segs.append((0.497, sho_mid + 0.50 * (hip_mid - sho_mid)))  # trunk
    for side in ("left", "right"):
        sh, el = P(f"{side}_shoulder"), P(f"{side}_elbow")
        wr = P(f"{side}_wrist")
        hp, kn = P(f"{side}_hip"), P(f"{side}_knee")
        an = P(f"{side}_ankle")
        segs.append((0.028, sh + 0.436 * (el - sh)))            # upper arm
        segs.append((0.022, el + 0.682 * (wr - el)))            # forearm+hand
        segs.append((0.100, hp + 0.433 * (kn - hp)))            # thigh
        segs.append((0.0465, kn + 0.433 * (an - kn)))           # shank
        segs.append((0.0145, an))                               # foot ~ ankle
    com = sum(w * pt for w, pt in segs)                          # weights sum 1.0
    return com[0]                                                # x series


def centroid_x(df, name):
    if name == "hipmid":
        return (_xy(df, "left_hip")[0] + _xy(df, "right_hip")[0]) / 2
    if name == "shomid_hipmid":
        hip = (_xy(df, "left_hip")[0] + _xy(df, "right_hip")[0]) / 2
        sho = (_xy(df, "left_shoulder")[0] + _xy(df, "right_shoulder")[0]) / 2
        return (hip + sho) / 2
    if name == "joints_mean":
        js = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
              "left_wrist", "right_wrist", "left_hip", "right_hip",
              "left_knee", "right_knee", "left_ankle", "right_ankle", "head"]
        return np.mean([_xy(df, j)[0] for j in js], axis=0)
    if name == "segw_com":
        return segw_com_x(df)
    raise ValueError(name)


CENTROIDS = ["hipmid", "shomid_hipmid", "joints_mean", "segw_com"]


def peak_fwd_speed(cx, fps, rel, stat):
    """max |d(cx)/dt| over [0, rel], in statures/s (unit-free)."""
    v = np.abs(np.gradient(cx)) * fps / stat
    seg = v[:rel + 1]
    return float(np.nanmax(seg)) if seg.size and not np.all(np.isnan(seg)) else np.nan


def main():
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    rows = []
    done = fail = 0
    for r in md.itertuples(index=False):
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
            sp = r.session_pitch
            if sp not in poi.index or GT_COL not in poi.columns:
                fail += 1; continue
            df = project_cam(joints, AZ, EL)
            stat = M.pixel_stature(df, M.JOINTS)
            h = float(r.session_height_m)
            rec = {"truth": float(poi.loc[sp, GT_COL]), "user": int(r.user)}
            for name in CENTROIDS:
                uf = peak_fwd_speed(centroid_x(df, name), fps, rel, stat)
                rec[f"{name}__unitfree"] = uf
                rec[f"{name}__ms"] = uf * h
            rows.append(rec)
            done += 1
        except Exception:
            fail += 1
    print(f"processed {done} / failed {fail}\n")

    dat = pd.DataFrame(rows)
    t_all = dat.truth.to_numpy(float)
    users_all = dat.user.to_numpy()

    def r2_of(col):
        e = dat[col].to_numpy(float)
        m = np.isfinite(e) & np.isfinite(t_all)
        return float(np.corrcoef(e[m], t_all[m])[0, 1] ** 2) if m.sum() > 2 else np.nan

    print("=" * 92)
    print(f"[COG-VELO CENTROID PROBE]  @az={AZ}/el={EL}, GT={GT_COL} "
          f"(m/s, mean 3.06/sd 0.255), n={done}")
    print("=" * 92)
    print("r2 cross-check (scale-sensitive per-pitch, so unitfree != ms):")
    print(f"  {'centroid':<16}{'unitfree r2':>13}{'ms r2':>9}")
    for name in CENTROIDS:
        tag = "  <-- ROADMAP baseline (~0.41)" if name == "hipmid" else ""
        print(f"  {name:<16}{r2_of(f'{name}__unitfree'):>13.3f}"
              f"{r2_of(f'{name}__ms'):>9.3f}{tag}")

    out = []
    print("\n[m/s scaling]  raw + leave-one-pitcher-out (adoptable units)")
    print(f"{'centroid':<16}{'model':<8}{'bias':>9}{'MAE':>8}{'CCC':>8}"
          f"{'r2':>7}{'pitcher_bias_sd':>17}")
    print("-" * 73)
    for name in CENTROIDS:
        e_all = dat[f"{name}__ms"].to_numpy(float)
        m = np.isfinite(e_all) & np.isfinite(t_all)
        e, t, users = e_all[m], t_all[m], users_all[m]
        preds = {mo: np.full(len(e), np.nan) for mo in MODELS}
        for u in np.unique(users):
            te = users == u; tr = ~te
            for mo in MODELS:
                preds[mo][te] = fit_apply(mo, e[tr], t[tr], e[te])
        stat = {"raw": score(e, t, users)}
        for mo in MODELS:
            stat[mo] = score(preds[mo], t, users)
        for nm, s in stat.items():
            out.append({"centroid": name, "scaling": "ms", "model": nm,
                        "n": int(m.sum()), **s})
        tag = "  <-- BASELINE" if name == "hipmid" else ""
        for nm in ("raw", "offset", "linear"):
            s = stat[nm]
            print(f"{name:<16}{nm:<8}{s['bias']:>+9.3f}{s['mae']:>8.3f}"
                  f"{s['ccc']:>8.3f}{s['r2']:>7.3f}{s['pitcher_bias_sd']:>17.3f}"
                  f"{tag if nm == 'raw' else ''}")
        print()

    outp = os.path.join(config.OBP_VALIDATION_DIR, "cog_velo_probe.csv")
    pd.DataFrame(out).round(4).to_csv(outp, index=False)
    print(f"saved -> {outp}")
    print("\nAdopt a refined centroid only if it CLEARLY clears r2~0.5 AND beats "
          "the hip-midpoint baseline on raw+LOCO MAE/CCC/bias.")


if __name__ == "__main__":
    main()
