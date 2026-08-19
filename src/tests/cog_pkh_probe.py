"""
Diamond - COG velocity @ peak-knee-height (cog_velo_pkh) DEFINITION probe.

Follow-up to the adopted COG Fwd Velo (max over the pitch). GT here is
poi cog_velo_pkh = "center of gravity velocity towards home plate at peak knee
height (m/s)" (OBP README) - an INSTANTANEOUS forward COM velocity at a single
EARLY event (top of the leg lift / balance point), mean 0.32 / sd 0.10 m/s,
distinct from max_cog_velo_x (peak over the whole delivery, mean 3.06).

Needs a NEW 2D event: peak knee height = the lead knee at its highest (min image
y) during the lift, BEFORE foot plant -> argmin(lead_knee_y) over [0, fp]. Then
sample the forward COM velocity there. Reuses the adopted metrics.body_com
(Winter whole-body COM). Sign: at az0 image-x increases toward home (project_view
docstring), so signed d(COM_x)/dt > 0 = toward home, matching the GT's signed
"towards home plate".

Variants tested (az0/el0, raw+LOCO vs GT cog_velo_pkh):
  centroid  : hipmid (old proxy) vs segw_com (adopted Winter COM)
  velocity  : signed (toward home) vs abs
Adopt as an 11th metric only if it CLEARLY clears r2~0.5 AND wins raw+LOCO.

HISTORICAL NOTE (2026-07-18): this probe uses a 2-frame np.gradient and that is
what its printed numbers (r2 ~0.800, raw CCC ~0.853) reflect. It PASSED and the
metric was adopted — but the ADOPTED definition then switched to a Savitzky-Golay
derivative (metrics.cog_velo_at_pkh), which reads r2 0.888 / raw CCC 0.905 and is
what every table and doc quotes. Re-running this probe reproduces the OLD values
by design; do not treat them as the current numbers.

Run:  cd src\tests
      python cog_pkh_probe.py
Output: cog_pkh_probe.csv
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
GT_COL = "cog_velo_pkh"


def _xy(df, j):
    return df[f"{j}_x"].to_numpy(float), df[f"{j}_y"].to_numpy(float)


def pkh_frame(df, lead, fp):
    """Peak knee height = lead knee highest (min image y) before foot plant."""
    ky = _xy(df, f"{lead}_knee")[1]
    hi = max(3, min(fp, len(ky)))
    return int(np.nanargmin(ky[:hi]))


def centroid_x(df, name):
    if name == "hipmid":
        return (_xy(df, "left_hip")[0] + _xy(df, "right_hip")[0]) / 2
    if name == "segw_com":
        return M.body_com(df, M.JOINTS)[0]
    raise ValueError(name)


CENTROIDS = ["hipmid", "segw_com"]
SIGNS = ["signed", "abs"]


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
            pkh = pkh_frame(df, lead, fp)
            rec = {"truth": float(poi.loc[sp, GT_COL]), "user": int(r.user)}
            for name in CENTROIDS:
                v = np.gradient(centroid_x(df, name)) * fps / stat * h
                rec[f"{name}__signed"] = float(v[pkh])
                rec[f"{name}__abs"] = float(abs(v[pkh]))
            rows.append(rec)
            done += 1
        except Exception:
            fail += 1
    print(f"processed {done} / failed {fail}\n")

    dat = pd.DataFrame(rows)
    t_all = dat.truth.to_numpy(float)
    users_all = dat.user.to_numpy()

    out = []
    print("=" * 92)
    print(f"[COG-VELO @ PKH PROBE]  @az={AZ}/el={EL}, GT={GT_COL} "
          f"(m/s, mean {np.nanmean(t_all):.3f}/sd {np.nanstd(t_all):.3f}), n={done}")
    print("=" * 92)
    print(f"{'candidate':<22}{'model':<8}{'bias':>9}{'MAE':>8}{'CCC':>8}"
          f"{'r2':>7}{'pitcher_bias_sd':>17}")
    print("-" * 79)
    for name in CENTROIDS:
        for sg in SIGNS:
            col = f"{name}__{sg}"
            e_all = dat[col].to_numpy(float)
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
                out.append({"candidate": col, "model": nm, "n": int(m.sum()), **s})
            for nm in ("raw", "offset", "linear"):
                s = stat[nm]
                print(f"{col:<22}{nm:<8}{s['bias']:>+9.4f}{s['mae']:>8.4f}"
                      f"{s['ccc']:>8.3f}{s['r2']:>7.3f}{s['pitcher_bias_sd']:>17.4f}")
            print()

    outp = os.path.join(config.OBP_VALIDATION_DIR, "cog_pkh_probe.csv")
    pd.DataFrame(out).round(5).to_csv(outp, index=False)
    print(f"saved -> {outp}")
    print("\nAdopt as an 11th metric only if it CLEARLY clears r2~0.5 AND wins "
          "raw+LOCO. cog_velo_pkh is a small early-event velocity (~0.3 m/s), so "
          "the pkh detection has to be solid.")


if __name__ == "__main__":
    main()
