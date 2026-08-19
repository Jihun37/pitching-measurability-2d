"""
Diamond - Knee fair comparison (same 3D-truth basis)
knee_fair_test.py

Earlier: abs_br r2=0.944 (vs 3D truth) vs diff r2=0.622 (vs OBP column).
That is apples-to-oranges. Here ALL knee variants are scored against the
SAME 3D direct truth so "absolute beats difference" is a clean claim:

  3D truth, absolute : 3D hip-knee-ankle angle at release          -> est knee[rel]
  3D truth, absolute : 3D hip-knee-ankle angle at foot plant       -> est knee[fp]
  3D truth, difference: 3D angle(rel) - 3D angle(fp)               -> est knee[rel]-knee[fp]

If absolute@br on the same basis still clearly tops difference, the single-
instant definition is confirmed superior (fewer stacked detection errors).

Usage:
    python knee_fair_test.py --limit 50
    python knee_fair_test.py
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

AZIMUTHS = [0, 15, 30, 45, 60, 75, 90]


def angle_3d(a, b, c):
    v1 = a - b; v2 = c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def knee3d(joints, lead, frame):
    return angle_3d(joints[f"{lead}_hip"][:, frame],
                    joints[f"{lead}_knee"][:, frame],
                    joints[f"{lead}_ankle"][:, frame])


def est_knee_series(df, lead):
    hx = df[f"{lead}_hip_x"].to_numpy(float);  hy = df[f"{lead}_hip_y"].to_numpy(float)
    kx = df[f"{lead}_knee_x"].to_numpy(float); ky = df[f"{lead}_knee_y"].to_numpy(float)
    ax = df[f"{lead}_ankle_x"].to_numpy(float); ay = df[f"{lead}_ankle_y"].to_numpy(float)
    return M._angle(hx, hy, kx, ky, ax, ay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    obp_diff_col = "lead_knee_extension_from_fp_to_br"
    poi_idx = poi.set_index("session_pitch")[obp_diff_col].to_dict()

    # truths (3D, once per pitch)
    T_abs_br, T_abs_fp, T_diff, T_obp = [], [], [], []
    # estimates per azimuth
    E_abs_br = {az: [] for az in AZIMUTHS}
    E_abs_fp = {az: [] for az in AZIMUTHS}
    E_diff   = {az: [] for az in AZIMUTHS}
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

            t_br = knee3d(joints, lead, rel)
            t_fp = knee3d(joints, lead, fp)
            T_abs_br.append(t_br)
            T_abs_fp.append(t_fp)
            T_diff.append(t_br - t_fp)
            T_obp.append(poi_idx.get(r.session_pitch, np.nan))

            for az in AZIMUTHS:
                df = O.project_view(joints, azimuth_deg=az)
                knee = est_knee_series(df, lead)
                E_abs_br[az].append(float(knee[rel]))
                E_abs_fp[az].append(float(knee[fp]))
                E_diff[az].append(float(knee[rel] - knee[fp]))
            done += 1
        except Exception:
            fail += 1
        if done and done % 100 == 0:
            print(f"  ...{done} processed")
    print(f"processed {done} / failed {fail}\n")

    def r2(est, truth):
        e = np.asarray(est, float); t = np.asarray(truth, float)
        m = np.isfinite(e) & np.isfinite(t)
        if m.sum() <= 2:
            return np.nan
        return np.corrcoef(e[m], t[m])[0, 1] ** 2

    rows = [
        ("knee ABS @br  vs 3D-abs",  E_abs_br, T_abs_br),
        ("knee ABS @fp  vs 3D-abs",  E_abs_fp, T_abs_fp),
        ("knee DIFF     vs 3D-diff", E_diff,   T_diff),
    ]
    print("=" * 78)
    print("[Fair, same 3D-truth basis] knee variants by azimuth (r2)")
    print("=" * 78)
    hdr = f"{'variant':26s}" + "".join(f"{az:>7d}" for az in AZIMUTHS) + "   best"
    print(hdr); print("-" * len(hdr))
    for label, E, T in rows:
        line = f"{label:26s}"; best_az, best = None, -1
        for az in AZIMUTHS:
            v = r2(E[az], T)
            if pd.notna(v) and v > best:
                best, best_az = v, az
            line += f"{v:7.2f}"
        line += f"   {best_az} ({best:.2f})"
        print(line)

    print("\n@0deg, all on identical footing:")
    print(f"  ABS @br : r2 = {r2(E_abs_br[0], T_abs_br):.3f}")
    print(f"  ABS @fp : r2 = {r2(E_abs_fp[0], T_abs_fp):.3f}")
    print(f"  DIFF    : r2 = {r2(E_diff[0],   T_diff):.3f}")
    print("\nCross-check (different bases, for reference only):")
    print(f"  DIFF est vs OBP column : r2 = {r2(E_diff[0], T_obp):.3f}  (the old 0.62)")
    print(f"  3D-diff truth vs OBP   : r2 = {r2(T_diff, T_obp):.3f}  "
          f"(how well 3D-diff matches OBP's own diff)")


if __name__ == "__main__":
    main()