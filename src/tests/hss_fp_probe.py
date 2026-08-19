"""
Diamond - HSS definition probe: windowed peak vs value at foot plant.

The adopted overhead HSS (metrics.hss_peak_overhead, windowed coil peak vs
max_rotation_hip_shoulder_separation) ceilings at r2~0.63. OBP also ships a
foot-plant-anchored truth column (rotation_hip_shoulder_separation_fp). This
probe asks whether the FP-anchored definition validates better in 2D:

  A  adopted peak recipe            vs max_rotation_hip_shoulder_separation  (control)
  B  sep at FP, same preprocessing  vs rotation_hip_shoulder_separation_fp
  B' sep at FP, raw (no medfilt)    vs rotation_hip_shoulder_separation_fp
  +  cross combos and truth-vs-truth r2 as diagnostics (definition overlap)

Estimator B reuses the adopted pipeline verbatim (hss_sep_series -> chord gate
-> 90 ms medfilt) and reads |sep| at the foot-plant frame instead of running
the transition anchor + window. FP is detected once on the el=0/az=0 side view
(paper-table convention). Overhead = project_cam el=85; az 0 and 90 both
measured to confirm azimuth invariance.

Run:  cd src\\tests
      python hss_fp_probe.py --limit 40    (smoke test)
      python hss_fp_probe.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
for p in ("..", "../stage2", "../stage3", "../analysis"):
    sys.path.insert(0, os.path.join(_HERE, p))

import config
import obp_project as O
import metrics as M
from master_angle_table import load_feet
from hss_elevation_test import project_cam

EL = 85
AZS = [0, 90]
COL_MAX = "max_rotation_hip_shoulder_separation"
COL_FP = "rotation_hip_shoulder_separation_fp"


def sep_at_fp(df, fps, fp, filtered=True):
    """|HSS| at the foot-plant frame, using the adopted preprocessing
    (chord-validity gate + 90 ms medfilt) from metrics; no anchor/window."""
    sep = M.hss_sep_series(df, M.JOINTS)
    if not filtered:
        v = sep[fp] if 0 <= fp < len(sep) else np.nan
        return float(abs(v))
    valid = M.hss_chord_valid(df, M.JOINTS)
    k = max(3, int(0.09 * fps) // 2 * 2 + 1)
    sep_f = M._medfilt(np.where(valid, np.nan_to_num(sep, nan=0.0), 0.0),
                       kernel_size=k)
    v = sep_f[fp] if 0 <= fp < len(sep_f) else np.nan
    return float(abs(v))


def r2(e, t):
    e = np.asarray(e, float); t = np.asarray(t, float)
    m = np.isfinite(e) & np.isfinite(t)
    return np.corrcoef(e[m], t[m])[0, 1] ** 2 if m.sum() > 2 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    est = {(name, az): [] for name in ("peak", "fp", "fp_raw") for az in AZS}
    tru = {"max": [], "fp": []}
    done = fail = 0

    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
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
            sp = r.session_pitch
            tru["max"].append(poi.loc[sp, COL_MAX] if sp in poi.index else np.nan)
            tru["fp"].append(poi.loc[sp, COL_FP] if sp in poi.index else np.nan)

            for az in AZS:
                df = project_cam(joints, az, EL)
                res = M.hss_peak_overhead(df, fps, M.JOINTS)
                est[("peak", az)].append(res["hss"] if res else np.nan)
                est[("fp", az)].append(sep_at_fp(df, fps, fp, filtered=True))
                est[("fp_raw", az)].append(sep_at_fp(df, fps, fp, filtered=False))
            done += 1
        except Exception:
            fail += 1
        if done and done % 100 == 0:
            print(f"  ...{done} processed")
    print(f"processed {done} / failed {fail}\n")

    print("=" * 66)
    print(f"[HSS DEFINITION PROBE]  overhead el={EL}, n={done}")
    print("=" * 66)
    print(f"{'estimator':16s}{'truth':10s}" +
          "".join(f"{'r2@az' + str(az):>10s}" for az in AZS))
    print("-" * 56)
    for name, label in (("peak", "peak(adopted)"), ("fp", "sep@FP"),
                        ("fp_raw", "sep@FP raw")):
        for tkey, tlabel in (("max", "MAX col"), ("fp", "FP col")):
            vals = "".join(f"{r2(est[(name, az)], tru[tkey]):>10.3f}" for az in AZS)
            print(f"{label:16s}{tlabel:10s}{vals}")
    print("-" * 56)
    print(f"{'truth vs truth':16s}{'':10s}{r2(tru['max'], tru['fp']):>10.3f}")


if __name__ == "__main__":
    main()
