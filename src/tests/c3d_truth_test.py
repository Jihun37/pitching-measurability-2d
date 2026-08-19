"""
Diamond - c3d direct-truth validation (H1, H3b, release height)
c3d_truth_test.py

These metrics have NO direct poi_metrics column, so -- exactly like
armslot_validate.py -- we build the ground truth from the 3D c3d coordinates
and compare our 2D estimate (per azimuth) against it.

Metrics:
  H1  knee_at_br : lead knee ABSOLUTE angle at release (hip-knee-ankle).
        3D truth = same angle from 3D vectors. Question: does absolute angle
        beat the fp->br DIFFERENCE (0.622)? (difference stacks two frames'
        detection error; absolute is single-instant)
      also knee_at_fp for completeness.

  H3b wrist_speed : throwing-wrist peak speed.
        3D truth = peak 3D linear speed of the wrist marker (m/s).
        Earlier we wrongly compared vs max_elbow_extension_velo; here we
        compare like-for-like against the wrist's own 3D speed.
        Two normalizations of our 2D estimate: /body_scale and /stature.

  relh  release_height : throwing-wrist height above ground at release.
        3D truth = wrist Z minus min foot Z at release (m).
        Our 2D uses ankle-ground vs heel-ground.

Truth is computed ONCE per pitch in 3D (azimuth-invariant); only our 2D
estimate varies with azimuth. r2 is corr^2 between estimate and 3D truth
across all pitches. metrics.py is NOT modified.

Usage:
    python c3d_truth_test.py --limit 50
    python c3d_truth_test.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
import config
import obp_project as O
import metrics as M
import ezc3d

AZIMUTHS = [0, 15, 30, 45, 60, 75, 90]

HEEL_MARKERS = {"left_heel": ["LHEE"], "right_heel": ["RHEE"]}


def load_with_heels(path):
    c = ezc3d.c3d(path)
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    fps = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    pts = c["data"]["points"][:3]
    idx = {l: i for i, l in enumerate(labels)}
    joints = {}
    for name, mks in {**O.MARKER_MAP, **HEEL_MARKERS}.items():
        mk = [m for m in mks if m in idx]
        if not mk:
            if name in HEEL_MARKERS:
                continue
            raise KeyError(f"missing marker: {name}")
        joints[name] = np.nanmean([pts[:, idx[m], :] for m in mk], axis=0)
    return joints, fps


def angle_3d(a, b, c):
    """Angle at vertex b formed by a-b-c, from 3D points (each shape (3,))."""
    v1 = a - b; v2 = c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


# ---------- 3D ground truths (computed once per pitch) ----------
def truth_knee(joints, lead, frame):
    h = joints[f"{lead}_hip"][:, frame]
    k = joints[f"{lead}_knee"][:, frame]
    a = joints[f"{lead}_ankle"][:, frame]
    return angle_3d(h, k, a)


def truth_wrist_speed(joints, arm, fps):
    w = joints[f"{arm}_wrist"]                 # (3, n)
    v = np.linalg.norm(np.diff(w, axis=1), axis=0) * fps  # m/s
    return float(np.nanmax(v))


def truth_release_height(joints, arm, frame):
    wz = joints[f"{arm}_wrist"][2, frame]
    foot_z = []
    for f in ("left_ankle", "right_ankle", "left_heel", "right_heel"):
        if f in joints:
            foot_z.append(joints[f][2, frame])
    ground = np.nanmin(foot_z)
    return float(wz - ground)


# ---------- 2D estimates (vary with azimuth) ----------
def est_knee(df, lead, rel, fp, which):
    hx = df[f"{lead}_hip_x"].to_numpy(float);  hy = df[f"{lead}_hip_y"].to_numpy(float)
    kx = df[f"{lead}_knee_x"].to_numpy(float); ky = df[f"{lead}_knee_y"].to_numpy(float)
    ax = df[f"{lead}_ankle_x"].to_numpy(float); ay = df[f"{lead}_ankle_y"].to_numpy(float)
    knee = M._angle(hx, hy, kx, ky, ax, ay)
    return float(knee[rel] if which == "br" else knee[fp])


def est_wrist_speed(df, arm, fps, norm, J):
    wx = df[f"{arm}_wrist_x"].to_numpy(float)
    wy = df[f"{arm}_wrist_y"].to_numpy(float)
    pk = float(np.nanmax(M._speed(wx, wy, fps)))
    denom = M.body_scale_px(df, J) if norm == "scale" else M.pixel_stature(df, J)
    return pk / denom


def est_release_height(df, arm, rel, ground_kind, J):
    wy = df[f"{arm}_wrist_y"].to_numpy(float)
    scale = M.body_scale_px(df, J)
    if ground_kind == "heel" and "left_heel_y" in df.columns:
        g = np.nanmax(np.concatenate([df["left_heel_y"].to_numpy(float),
                                      df["right_heel_y"].to_numpy(float)]))
    else:
        g = np.nanmax(np.concatenate([df["left_ankle_y"].to_numpy(float),
                                      df["right_ankle_y"].to_numpy(float)]))
    return float((g - wy[rel]) / scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    # collect: per pitch, one truth each + per-azimuth estimates
    T = {"knee_br": [], "knee_fp": [], "wspeed": [], "relh": []}
    E = {k: {az: [] for az in AZIMUTHS} for k in
         ["knee_br", "knee_fp", "wspeed_scale", "wspeed_stat",
          "relh_ankle", "relh_heel"]}
    done = fail = 0

    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = load_with_heels(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
            df0 = O.project_view(joints, azimuth_deg=0.0)
            rel = M.release_frame(df0, arm, fps, M.JOINTS)
            fp = M.foot_plant_frame(df0, lead, fps, M.JOINTS, rel)
            if rel <= fp + 1 or fp < 3:
                fail += 1; continue

            # 3D truths (once)
            T["knee_br"].append(truth_knee(joints, lead, rel))
            T["knee_fp"].append(truth_knee(joints, lead, fp))
            T["wspeed"].append(truth_wrist_speed(joints, arm, fps))
            T["relh"].append(truth_release_height(joints, arm, rel))

            # 2D estimates per azimuth
            for az in AZIMUTHS:
                df = O.project_view(joints, azimuth_deg=az)
                E["knee_br"][az].append(est_knee(df, lead, rel, fp, "br"))
                E["knee_fp"][az].append(est_knee(df, lead, rel, fp, "fp"))
                E["wspeed_scale"][az].append(est_wrist_speed(df, arm, fps, "scale", M.JOINTS))
                E["wspeed_stat"][az].append(est_wrist_speed(df, arm, fps, "stat", M.JOINTS))
                E["relh_ankle"][az].append(est_release_height(df, arm, rel, "ankle", M.JOINTS))
                E["relh_heel"][az].append(est_release_height(df, arm, rel, "heel", M.JOINTS))
            done += 1
        except Exception:
            fail += 1
        if done and done % 100 == 0:
            print(f"  ...{done} processed")
    print(f"processed {done} / failed {fail}\n")

    def r2(est_list, truth_list):
        e = np.asarray(est_list, float); t = np.asarray(truth_list, float)
        m = np.isfinite(e) & np.isfinite(t)
        if m.sum() <= 2:
            return np.nan
        r = np.corrcoef(e[m], t[m])[0, 1]
        return r * r

    # variant -> truth key
    PLAN = [
        ("knee_at_br (abs)",  "knee_br",      "knee_br"),
        ("knee_at_fp (abs)",  "knee_fp",      "knee_fp"),
        ("wrist_spd /scale",  "wspeed_scale", "wspeed"),
        ("wrist_spd /stature","wspeed_stat",  "wspeed"),
        ("relh ankle-ground", "relh_ankle",   "relh"),
        ("relh heel-ground",  "relh_heel",    "relh"),
    ]

    print("=" * 80)
    print("[c3d direct-truth] our 2D estimate vs 3D truth, by azimuth (r2)")
    print("=" * 80)
    hdr = f"{'variant':22s}" + "".join(f"{az:>7d}" for az in AZIMUTHS) + "   best"
    print(hdr); print("-" * len(hdr))
    for label, ekey, tkey in PLAN:
        line = f"{label:22s}"; best_az, best = None, -1
        for az in AZIMUTHS:
            val = r2(E[ekey][az], T[tkey])
            if pd.notna(val) and val > best:
                best, best_az = val, az
            line += f"{val:7.2f}"
        line += f"   {best_az} ({best:.2f})"
        print(line)

    print("\nKey comparisons @0deg:")
    print(f"  H1 knee:  diff(baseline)=0.622   "
          f"abs_br={r2(E['knee_br'][0], T['knee_br']):.3f}   "
          f"abs_fp={r2(E['knee_fp'][0], T['knee_fp']):.3f}")
    print(f"  H3b wrist: /scale={r2(E['wspeed_scale'][0], T['wspeed']):.3f}   "
          f"/stature={r2(E['wspeed_stat'][0], T['wspeed']):.3f}")
    print(f"  relh:     ankle={r2(E['relh_ankle'][0], T['relh']):.3f}   "
          f"heel={r2(E['relh_heel'][0], T['relh']):.3f}")


if __name__ == "__main__":
    main()