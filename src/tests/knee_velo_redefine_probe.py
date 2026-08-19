"""
Diamond - knee_ext_velo REDEFINITION probe (3D, definition/event search).

Motivation: even the 3D included-angle recipe (no projection error) tops out at
r2 ~= 0.67 vs the OBP column lead_knee_extension_angular_velo_br
(r2_ceiling_probe, 2026-07-09). A ceiling that low IN 3D means the DEFINITION,
not pose/projection, is the loss. This probe keeps the adopted metric untouched
and searches for a better definition, testing two hypotheses:

  (1) EVENT mismatch: our detected release != OBP's BR event. Sweep the sample
      frame and correlate against _br; also correlate our @rel value against all
      three OBP columns (br / fp / max).
  (2) DEFINITION mismatch: the 3-point included angle (hip-knee-ankle) conflates
      ab/adduction + internal rotation with flexion. OBP's model measures knee
      flexion about the anatomical ML axis. Candidate fix computable from the
      same 3 keypoints: the SAGITTAL-plane flexion angle (thigh/shank projected
      into the plane perpendicular to the hip line), plus the fp->release PEAK
      (matches the _max column concept).

Read-only; 3D-direct (azimuth-invariant) so this is the information ceiling.
Nothing in metrics.py changes. n=408.

Run (diamond env):
  conda activate diamond
  cd src/tests
  python knee_velo_redefine_probe.py           # full n=408
  python knee_velo_redefine_probe.py --limit 80
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import config
import metrics as M
import obp_project as O

TRUTH_COLS = ["lead_knee_extension_angular_velo_br",
              "lead_knee_extension_angular_velo_fp",
              "lead_knee_extension_angular_velo_max",
              "lead_knee_extension_from_fp_to_br"]   # a RANGE, not a velocity
SWEEP = range(-5, 6)   # frame offsets around detected release


def _unit(v):
    return v / (np.linalg.norm(v, axis=0) + 1e-9)


def incl_angle_3d(h, k, a):
    """3D hip-knee-ankle included angle (deg). Baseline definition."""
    v1, v2 = h - k, a - k
    dot = np.sum(v1 * v2, axis=0)
    n = np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0) + 1e-9
    return np.degrees(np.arccos(np.clip(dot / n, -1, 1)))


def sagittal_flex_3d(h, k, a, e_ml):
    """Knee flexion in the body-sagittal plane: thigh/shank projected to the
    plane perpendicular to the hip (ML) axis, then the included angle. Removes
    the ab/adduction + long-axis-rotation the raw included angle mixes in."""
    thigh, shank = h - k, a - k
    thigh_s = thigh - np.sum(thigh * e_ml, axis=0) * e_ml
    shank_s = shank - np.sum(shank * e_ml, axis=0) * e_ml
    dot = np.sum(thigh_s * shank_s, axis=0)
    n = np.linalg.norm(thigh_s, axis=0) * np.linalg.norm(shank_s, axis=0) + 1e-9
    return np.degrees(np.arccos(np.clip(dot / n, -1, 1)))


def knee2d_series(df, lead):
    """2D projected hip-knee-ankle included angle (deg)."""
    hx = df[f"{lead}_hip_x"].to_numpy(float);  hy = df[f"{lead}_hip_y"].to_numpy(float)
    kx = df[f"{lead}_knee_x"].to_numpy(float); ky = df[f"{lead}_knee_y"].to_numpy(float)
    ax = df[f"{lead}_ankle_x"].to_numpy(float); ay = df[f"{lead}_ankle_y"].to_numpy(float)
    return M._angle(hx, hy, kx, ky, ax, ay)


def r2(e, t):
    e = np.asarray(e, float); t = np.asarray(t, float)
    m = np.isfinite(e) & np.isfinite(t)
    if m.sum() <= 2:
        return np.nan
    r = np.corrcoef(e[m], t[m])[0, 1]
    return r * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    truth = {c: dict(zip(poi["session_pitch"], poi[c])) for c in TRUTH_COLS}
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    cand = {k: [] for k in
            ["incl@rel", "incl@fp", "incl_peak", "incl_abspeak",
             "sag@rel", "sag@fp", "sag_peak", "sag_abspeak",
             "incl_range", "incl_avgvel",     # range = angle change fp->br
             "range_2d@0", "range_2d@45"]}    # deployable (projected side/oblique)
    sweep = {off: [] for off in SWEEP}          # incl velocity @ rel+off (vs _br)
    sweep_sag = {off: [] for off in SWEEP}      # sagittal velocity @ rel+off
    T = {c: [] for c in TRUTH_COLS}
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
            spid = r.session_pitch
            if any(not np.isfinite(truth[c].get(spid, np.nan)) for c in TRUTH_COLS):
                fail += 1; continue

            h = joints[f"{lead}_hip"]; k = joints[f"{lead}_knee"]; a_ = joints[f"{lead}_ankle"]
            e_ml = _unit(joints["right_hip"] - joints["left_hip"])

            incl = incl_angle_3d(h, k, a_)
            sag = sagittal_flex_3d(h, k, a_, e_ml)
            iv = np.gradient(incl) * fps
            sv = np.gradient(sag) * fps
            segsl = slice(fp, rel + 1)

            cand["incl@rel"].append(float(iv[rel]))
            cand["incl@fp"].append(float(iv[fp]))
            cand["incl_peak"].append(float(np.nanmax(iv[segsl])))
            cand["incl_abspeak"].append(float(iv[segsl][np.nanargmax(np.abs(iv[segsl]))]))
            cand["sag@rel"].append(float(sv[rel]))
            cand["sag@fp"].append(float(sv[fp]))
            cand["sag_peak"].append(float(np.nanmax(sv[segsl])))
            cand["sag_abspeak"].append(float(sv[segsl][np.nanargmax(np.abs(sv[segsl]))]))
            rng = float(incl[rel] - incl[fp])                       # extension range (deg)
            cand["incl_range"].append(rng)
            cand["incl_avgvel"].append(rng / max((rel - fp) / fps, 1e-6))  # avg deg/s
            # deployable 2D range: included angle change fp->br in a projected view
            k2 = knee2d_series(df0, lead)                           # az=0 (side)
            cand["range_2d@0"].append(float(k2[rel] - k2[fp]))
            df45 = O.project_view(joints, azimuth_deg=45.0)
            k45 = knee2d_series(df45, lead)
            cand["range_2d@45"].append(float(k45[rel] - k45[fp]))

            for off in SWEEP:
                j = rel + off
                sweep[off].append(float(iv[j]) if 0 <= j < len(iv) else np.nan)
                sweep_sag[off].append(float(sv[j]) if 0 <= j < len(sv) else np.nan)

            for c in TRUTH_COLS:
                T[c].append(float(truth[c][spid]))
            done += 1
        except Exception:
            fail += 1
        if done and done % 100 == 0:
            print(f"  ...{done} processed", flush=True)

    print(f"\nprocessed {done} / failed {fail}   (3D-direct, no projection)\n")

    print("=" * 78)
    print("[knee-velo definition search]  r2 of each candidate vs each OBP column")
    print("  (3D ceiling; baseline incl@rel vs _br was 0.67)")
    print("=" * 78)
    print(f"{'candidate':16s}{'vs _br':>10s}{'vs _fp':>10s}{'vs _max':>10s}{'vs _range':>11s}")
    print("-" * 57)
    short = {"lead_knee_extension_angular_velo_br": "br",
             "lead_knee_extension_angular_velo_fp": "fp",
             "lead_knee_extension_angular_velo_max": "max",
             "lead_knee_extension_from_fp_to_br": "range"}
    best = (None, None, -1.0)
    for name, vals in cand.items():
        cells = []
        for c in TRUTH_COLS:
            v = r2(vals, T[c])
            cells.append(v)
            if np.isfinite(v) and v > best[2]:
                best = (name, short[c], v)
        print(f"{name:16s}" + "".join(f"{v:10.3f}" for v in cells))

    print("\n[event sweep] incl & sagittal velocity @ rel+offset  vs _br")
    print(f"{'offset':>7s}{'incl':>9s}{'sagittal':>10s}")
    for off in SWEEP:
        print(f"{off:>7d}{r2(sweep[off], T[TRUTH_COLS[0]]):9.3f}"
              f"{r2(sweep_sag[off], T[TRUTH_COLS[0]]):10.3f}")

    print(f"\nBEST: {best[0]} vs _{best[1]}  r2={best[2]:.3f}")
    print("(baseline incl@rel vs _br ~ 0.67; a materially higher cell = the")
    print(" adopted definition is measuring the wrong thing, and this is better.)")


if __name__ == "__main__":
    main()
