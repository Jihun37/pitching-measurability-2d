"""
Diamond - trunk anterior-tilt DEFINITION probe (torso-axis variants).

Tier-1 gave trunk a viewpoint win (el0 -> el15). This probe asks the ORTHOGONAL
question the user actually raised: keeping the viewpoint fixed at az0/el15 and
the release frame fixed, does a DIFFERENT torso-axis definition (different 2D
landmarks) beat the current hip-mid -> shoulder-mid segment against the SAME
OBP ground truth (torso_anterior_tilt_br)?

The official estimator (master_angle_table.est_trunk / metrics.py) is UNTOUCHED;
this is a read-only exploration. A candidate is only worth adopting if it CLEARLY
beats the baseline on BOTH raw and leave-one-pitcher-out (LOCO) calibrated
MAE / CCC / bias / pitcher_bias_sd - not just raw, and not marginally.

Available 2D torso landmarks (OBP MARKER_MAP): left/right shoulder, left/right
hip (each already ASI+PSI-averaged -> pelvis sagittal axis is NOT recoverable,
documented joint-model limit), head (4 head markers averaged). Candidates are
built only from these.

Convention matched to absacc/loco EXACTLY: measurement projection
project_cam(joints, 0, 15); events detected ONCE on project_view(az=0) el=0
(paper detect-once convention); GT = poi torso_anterior_tilt_br; LOCO folds =
OBP subject id `user`; models ratio/offset/linear from loco_calibration.

Run:  cd src\tests
      python trunk_axis_probe.py
Output: trunk_axis_probe.csv (metric=candidate, model, raw/LOCO stats).
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

AZ, EL = 0, 15
GT_COL = "torso_anterior_tilt_br"


def _xy(df, j):
    return df[f"{j}_x"].to_numpy(float), df[f"{j}_y"].to_numpy(float)


def _ang(tx, ty, bx, by):
    """Lean of the bottom->top segment from image vertical, per frame (deg).
    Matches est_trunk: arctan2(dx, up) where up = by - ty (>0, since top is
    higher = smaller y). Positive = leaning toward +x."""
    return np.degrees(np.arctan2(tx - bx, by - ty))


def candidates(df, ctx):
    """All torso-axis angles at the release frame. Same sign convention as the
    official estimator so numbers are directly comparable."""
    r = ctx["rel"]; a = ctx["arm"]; lead = ctx["lead"]
    lsx, lsy = _xy(df, "left_shoulder"); rsx, rsy = _xy(df, "right_shoulder")
    lhx, lhy = _xy(df, "left_hip");      rhx, rhy = _xy(df, "right_hip")
    sho_mx, sho_my = (lsx + rsx) / 2, (lsy + rsy) / 2
    hip_mx, hip_my = (lhx + rhx) / 2, (lhy + rhy) / 2
    has_head = "head_x" in df.columns
    hdx, hdy = _xy(df, "head") if has_head else (None, None)

    out = {}
    # B0: official definition (hip-mid -> shoulder-mid)
    out["B0_hipmid_shomid"] = _ang(sho_mx, sho_my, hip_mx, hip_my)[r]
    # ipsilateral (throwing-side) hip -> shoulder
    out["V_ipsi_side"] = _ang(*_xy(df, f"{a}_shoulder"), *_xy(df, f"{a}_hip"))[r]
    # contralateral (lead/glove-side) hip -> shoulder
    out["V_contra_side"] = _ang(*_xy(df, f"{lead}_shoulder"),
                                *_xy(df, f"{lead}_hip"))[r]
    if has_head:
        neck_x, neck_y = (sho_mx + hdx) / 2, (sho_my + hdy) / 2
        # hip-mid -> head (longest lever, includes cervical)
        out["V_hipmid_head"] = _ang(hdx, hdy, hip_mx, hip_my)[r]
        # hip-mid -> neck (C7-ish proxy = shoulder/head midpoint)
        out["V_hipmid_neck"] = _ang(neck_x, neck_y, hip_mx, hip_my)[r]
        # upper trunk only (shoulder-mid -> head)
        out["V_shomid_head"] = _ang(hdx, hdy, sho_mx, sho_my)[r]
        # PCA axis through the 3 torso points (hip-mid, shoulder-mid, head)
        pts = np.array([[hip_mx[r], hip_my[r]],
                        [sho_mx[r], sho_my[r]],
                        [hdx[r], hdy[r]]], float)
        if np.isfinite(pts).all():
            c = pts - pts.mean(0)
            _, _, vt = np.linalg.svd(c, full_matrices=False)
            d = vt[0]
            if d[1] > 0:            # orient upward (image up = -y)
                d = -d
            out["V_pca3_torso"] = float(np.degrees(np.arctan2(d[0], -d[1])))
        else:
            out["V_pca3_torso"] = np.nan
    # B0 with a 3-frame median around release (temporal-robustness variant)
    ser = _ang(sho_mx, sho_my, hip_mx, hip_my)
    lo, hi = max(0, r - 1), min(len(ser), r + 2)
    out["V_med3_hipmid_shomid"] = float(np.nanmedian(ser[lo:hi]))
    return out


def main():
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    rows = []          # per pitch: {cand: val, ..., truth, user}
    done = fail = 0
    for r in md.itertuples(index=False):
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = load_feet(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
            trail = "right" if lead == "left" else "left"
            df0 = O.project_view(joints, azimuth_deg=0.0)
            rel = M.release_frame(df0, arm, fps, M.JOINTS)
            fp = M.foot_plant_frame(df0, lead, fps, M.JOINTS, rel)
            if rel <= fp + 1 or fp < 3:
                fail += 1; continue
            sp = r.session_pitch
            if sp not in poi.index or GT_COL not in poi.columns:
                fail += 1; continue
            tval = poi.loc[sp, GT_COL]
            ctx = {"arm": arm, "lead": lead, "trail": trail,
                   "rel": rel, "fp": fp, "fps": fps}
            df = project_cam(joints, AZ, EL)
            rec = candidates(df, ctx)
            rec["truth"] = float(tval); rec["user"] = int(r.user)
            rows.append(rec)
            done += 1
        except Exception:
            fail += 1
    print(f"processed {done} / failed {fail}\n")

    dat = pd.DataFrame(rows)
    cand_cols = [c for c in dat.columns if c not in ("truth", "user")]
    t_all = dat.truth.to_numpy(float)
    users_all = dat.user.to_numpy()

    out = []
    print("=" * 96)
    print(f"[TRUNK-AXIS DEFINITION PROBE]  @az={AZ}/el={EL}, release, "
          f"GT={GT_COL}, n={done}")
    print("=" * 96)
    print(f"{'candidate':<24}{'model':<8}{'bias':>9}{'MAE':>8}{'CCC':>8}"
          f"{'r2':>7}{'pitcher_bias_sd':>17}")
    print("-" * 81)
    for cand in cand_cols:
        e_all = dat[cand].to_numpy(float)
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
        for name, s in stat.items():
            out.append({"candidate": cand, "model": name, "n": int(m.sum()), **s})
        tag = "  <-- BASELINE" if cand == "B0_hipmid_shomid" else ""
        for name in ("raw", "offset", "linear"):
            s = stat[name]
            print(f"{cand:<24}{name:<8}{s['bias']:>+9.3f}{s['mae']:>8.3f}"
                  f"{s['ccc']:>8.3f}{s['r2']:>7.3f}"
                  f"{s['pitcher_bias_sd']:>17.3f}"
                  f"{tag if name == 'raw' else ''}")
        print()

    outp = os.path.join(config.OBP_VALIDATION_DIR, "trunk_axis_probe.csv")
    pd.DataFrame(out).round(4).to_csv(outp, index=False)
    print(f"saved -> {outp}")
    print("\nAdopt a candidate only if it CLEARLY beats B0 on BOTH raw and LOCO "
          "MAE/CCC/bias AND does not raise pitcher_bias_sd.")


if __name__ == "__main__":
    main()
