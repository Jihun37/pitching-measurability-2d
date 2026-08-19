"""
Diamond - Can we raise HSS r2 above the ~0.61 baseline? Estimator comparison.

Baseline (hss_elevation_test) used global max|sep| over the whole clip. Here we
try better estimators at the recovery regime (el=60/75/90), all validated
against OBP 3D truth. Goal: (a) higher r2, and (b) clear r2>=0.6 at a LOWER
elevation (= easier camera position to deploy).

Estimators (peak-type -> vs max_rotation_hip_shoulder_separation):
  raw_absmax   : max|sep| over whole clip                       (baseline)
  win_absmax   : max|sep| within [foot_plant, release] window
  win_sg       : SG-smooth sep, then max|sep| within window
  win_signed   : sign-normalized by throwing arm, max within window
Frame-type -> vs rotation_hip_shoulder_separation_fp:
  fp_raw       : |sep| at foot-plant frame
  fp_sg        : |sep| at foot-plant frame after SG smoothing

Foot-plant / release frames detected once on the el=0 side view and reused
(isolates metric visibility; time index is projection-independent).

Run:
  cd src\analysis
  python hss_estimator_sweep.py --limit 200
  python hss_estimator_sweep.py
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
from hss_elevation_test import project_cam, hss_sep_series  # reuse projector + metric

AZ = 0                       # elevation regime is azimuth-invariant (shown)
ELEVATIONS = [60, 75, 90]


def sg(a):
    """Savitzky-Golay on a 1D array with NaN-safe fallback."""
    x = np.asarray(a, float)
    if np.isnan(x).any():
        idx = np.arange(len(x))
        good = ~np.isnan(x)
        if good.sum() < 5:
            return x
        x = np.interp(idx, idx[good], x[good])
    win = min(11, len(x) if len(x) % 2 else len(x) - 1)
    if win < 5:
        return x
    return savgol_filter(x, win, 2)


def r2(a, b):
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < 3:
        return np.nan
    r = d["a"].corr(d["b"])
    return r * r if pd.notna(r) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    J = M.JOINTS

    EST = ["raw_absmax", "win_absmax", "win_sg", "win_signed", "fp_raw", "fp_sg"]
    acc = {el: [] for el in ELEVATIONS}
    done = fail = 0
    for r in md.itertuples(index=False):
        if a.limit and done >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = O.load_c3d_joints(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
            df0 = O.project_view(joints, azimuth_deg=0, elevation_deg=0)
            rel0 = M.release_frame(df0, arm, fps, J, view="side")
            fp0 = M.foot_plant_frame(df0, lead, fps, J, rel0)
        except Exception:
            fail += 1; continue
        lo, hi = min(fp0, rel0), max(fp0, rel0) + 1
        armsign = 1.0 if arm == "right" else -1.0

        for el in ELEVATIONS:
            try:
                df = project_cam(joints, AZ, el)
                sep = hss_sep_series(df, J)
                sep_s = sg(sep)
                win = sep[lo:hi]; win_s = sep_s[lo:hi]
                row = {
                    "session_pitch": r.session_pitch,
                    "raw_absmax": float(np.nanmax(np.abs(sep))),
                    "win_absmax": float(np.nanmax(np.abs(win))) if len(win) else np.nan,
                    "win_sg":     float(np.nanmax(np.abs(win_s))) if len(win_s) else np.nan,
                    "win_signed": float(np.nanmax(armsign * win_s)) if len(win_s) else np.nan,
                    "fp_raw":     float(abs(sep[fp0])),
                    "fp_sg":      float(abs(sep_s[fp0])),
                }
                acc[el].append(row)
            except Exception:
                pass
        done += 1
        if done % 100 == 0:
            print(f"  ...{done} pitches")
    print(f"processed {done} / {fail} missing\n")

    TRUTH = {"raw_absmax": "max_rotation_hip_shoulder_separation",
             "win_absmax": "max_rotation_hip_shoulder_separation",
             "win_sg":     "max_rotation_hip_shoulder_separation",
             "win_signed": "max_rotation_hip_shoulder_separation",
             "fp_raw":     "rotation_hip_shoulder_separation_fp",
             "fp_sg":      "rotation_hip_shoulder_separation_fp"}

    print("=" * 66)
    print("HSS r2 by estimator x elevation   (baseline = raw_absmax)")
    print("estimator      " + "".join(f"  el={el:>2d}" for el in ELEVATIONS) + "   truth")
    print("-" * 66)
    for e in EST:
        feat = {el: pd.DataFrame(acc[el]) for el in ELEVATIONS}
        vals = []
        for el in ELEVATIONS:
            f = feat[el]
            if f.empty:
                vals.append(np.nan); continue
            m = f.merge(poi[["session_pitch", TRUTH[e]]], on="session_pitch", how="inner")
            vals.append(r2(m[e], m[TRUTH[e]]))
        tname = "max_rot" if TRUTH[e].startswith("max") else "fp"
        line = f"{e:<14}" + "".join(
            f"  {v:>5.2f}" if pd.notna(v) else "     -" for v in vals) + f"   {tname}"
        print(line)
    print("=" * 66)


if __name__ == "__main__":
    main()
