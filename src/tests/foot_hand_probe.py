"""
Diamond - foot/hand endpoint probe (projection-only, nothing adopted)
foot_hand_probe.py

Question (user idea 2026-07-08): the adopted metrics stop at the ankle/wrist.
Do more distal endpoints -- foot (toe/heel) and hand (finger) -- raise r2?

OBP c3d has the markers: LTOE/RTOE, LHEE/RHEE (foot), LFIN/RFIN (hand).
RTMPose Halpe26 already extracts big toe on real video; heel is available;
hand would need the Wholebody-133 model (real-video feasibility is a later,
separate check -- this probe is projection-only).

Variants (metrics.py is NOT modified; ground truth matches each definition):
  A) speed, like-for-like 3D-direct truth (2D point speed vs the SAME point's
     3D peak speed):
       spd_wrist    baseline  (reproduces adopted wrist_speed, r2=0.60 @0)
       spd_hand     FIN marker (more distal, closer to the ball)
       spd_handmid  midpoint(wrist,FIN) ~ hand center / ball position
  B) stride_length, OBP-column truth (poi stride_length). Normalization is
     /pixel_stature (the ADOPTED 0.82 one, = stride_pct_height) since
     2026-07-09; the first probe run used /body_scale (0.59 family) and its
     variant ranking needed re-checking under the adopted normalization:
       str_ankle    baseline  (fp + endpoints from ankle; 0.824 @0 expected)
       str_toe      fp + endpoints from big toe
       str_heel     fp + endpoints from heel
       str_hfp_ank  fp from heel (physical heel strike), endpoints ankle
                    (isolates event-detection vs endpoint choice)
       str_toelead  user variant 2026-07-08: trail anchor stays ankle-based
                    (rear foot's initial rubber position, unchanged), only the
                    LEAD endpoint measured at the forefoot (big toe)

Template: c3d_truth_test.py (events detected once at az=0, /scale and
/stature normalizations preserved from the adopted definitions).

Usage:
    python foot_hand_probe.py --limit 50
    python foot_hand_probe.py
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

EXTRA_MARKERS = {
    "left_toe":  ["LTOE"], "right_toe":  ["RTOE"],
    "left_heel": ["LHEE"], "right_heel": ["RHEE"],
    "left_hand": ["LFIN"], "right_hand": ["RFIN"],
}


def load_extended(path):
    c = ezc3d.c3d(path)
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    fps = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    pts = c["data"]["points"][:3]
    idx = {l: i for i, l in enumerate(labels)}
    joints = {}
    for name, mks in {**O.MARKER_MAP, **EXTRA_MARKERS}.items():
        mk = [m for m in mks if m in idx]
        if not mk:
            raise KeyError(f"missing marker: {name}")
        joints[name] = np.nanmean([pts[:, idx[m], :] for m in mk], axis=0)
    return joints, fps


# J-map variants: reuse foot_plant_frame / _xy unchanged, only swap which
# joint plays the "ankle" role.
J_ANKLE = M.JOINTS
J_TOE  = dict(M.JOINTS, l_an="left_toe",  r_an="right_toe")
J_HEEL = dict(M.JOINTS, l_an="left_heel", r_an="right_heel")


# ---------- 3D ground truths (once per pitch, azimuth-invariant) ----------
def truth_point_speed(joints, key, fps):
    p = joints[key]                                       # (3, n)
    v = np.linalg.norm(np.diff(p, axis=1), axis=0) * fps  # m/s
    return float(np.nanmax(v))


def truth_midpoint_speed(joints, k1, k2, fps):
    p = (joints[k1] + joints[k2]) / 2.0
    v = np.linalg.norm(np.diff(p, axis=1), axis=0) * fps
    return float(np.nanmax(v))


# ---------- 2D estimates ----------
def est_point_speed(df, name, fps, stat):
    x = df[f"{name}_x"].to_numpy(float)
    y = df[f"{name}_y"].to_numpy(float)
    return float(np.nanmax(M._speed(x, y, fps))) / stat


def est_midpoint_speed(df, n1, n2, fps, stat):
    x = (df[f"{n1}_x"].to_numpy(float) + df[f"{n2}_x"].to_numpy(float)) / 2.0
    y = (df[f"{n1}_y"].to_numpy(float) + df[f"{n2}_y"].to_numpy(float)) / 2.0
    return float(np.nanmax(M._speed(x, y, fps))) / stat


def est_stride(df, lead, fps, rel, J_fp, J_lead, J_trail, scale):
    """Adopted stride recipe (trail anchor -> lead endpoint at fp), with the
    fp-detection joint (J_fp) and the lead/trail endpoint joints swappable."""
    fp = M.foot_plant_frame(df, lead, fps, J_fp, rel)
    lead_key, trail_key = ("l_an", "r_an") if lead == "left" else ("r_an", "l_an")
    lead_x, _ = M._xy(df, lead_key, J_lead)
    trail_x, _ = M._xy(df, trail_key, J_trail)
    anchor = M.trail_anchor_x(trail_x, fp, fps)
    return float(abs(lead_x[fp] - anchor) / scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    poi_stride = dict(zip(poi["session_pitch"], poi["stride_length"]))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    SPEED = ["spd_wrist", "spd_hand", "spd_handmid"]
    STRIDE = ["str_ankle", "str_toe", "str_heel", "str_hfp_ank", "str_toelead"]
    T = {k: [] for k in SPEED + ["stride"]}
    E = {k: {az: [] for az in AZIMUTHS} for k in SPEED + STRIDE}
    done = fail = 0

    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = load_extended(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
            df0 = O.project_view(joints, azimuth_deg=0.0)
            rel = M.release_frame(df0, arm, fps, M.JOINTS)
            fp0 = M.foot_plant_frame(df0, lead, fps, M.JOINTS, rel)
            if rel <= fp0 + 1 or fp0 < 3:
                fail += 1; continue

            wkey, hkey = f"{arm}_wrist", f"{arm}_hand"
            T["spd_wrist"].append(truth_point_speed(joints, wkey, fps))
            T["spd_hand"].append(truth_point_speed(joints, hkey, fps))
            T["spd_handmid"].append(truth_midpoint_speed(joints, wkey, hkey, fps))
            T["stride"].append(poi_stride.get(r.session_pitch, np.nan))

            for az in AZIMUTHS:
                df = O.project_view(joints, azimuth_deg=az)
                stat = M.pixel_stature(df, M.JOINTS)
                E["spd_wrist"][az].append(est_point_speed(df, wkey, fps, stat))
                E["spd_hand"][az].append(est_point_speed(df, hkey, fps, stat))
                E["spd_handmid"][az].append(est_midpoint_speed(df, wkey, hkey, fps, stat))
                E["str_ankle"][az].append(est_stride(df, lead, fps, rel, J_ANKLE, J_ANKLE, J_ANKLE, stat))
                E["str_toe"][az].append(est_stride(df, lead, fps, rel, J_TOE, J_TOE, J_TOE, stat))
                E["str_heel"][az].append(est_stride(df, lead, fps, rel, J_HEEL, J_HEEL, J_HEEL, stat))
                E["str_hfp_ank"][az].append(est_stride(df, lead, fps, rel, J_HEEL, J_ANKLE, J_ANKLE, stat))
                E["str_toelead"][az].append(est_stride(df, lead, fps, rel, J_ANKLE, J_TOE, J_ANKLE, stat))
            done += 1
        except Exception:
            fail += 1
        if done and done % 100 == 0:
            print(f"  ...{done} processed", flush=True)
    print(f"processed {done} / failed {fail}\n")

    def r2(est_list, truth_list):
        e = np.asarray(est_list, float); t = np.asarray(truth_list, float)
        m = np.isfinite(e) & np.isfinite(t)
        if m.sum() <= 2:
            return np.nan
        r = np.corrcoef(e[m], t[m])[0, 1]
        return r * r

    PLAN = [
        ("spd wrist (baseline)",  "spd_wrist",   "spd_wrist"),
        ("spd hand (FIN)",        "spd_hand",    "spd_hand"),
        ("spd handmid (wr+FIN)",  "spd_handmid", "spd_handmid"),
        ("stride ankle (basel.)", "str_ankle",   "stride"),
        ("stride toe",            "str_toe",     "stride"),
        ("stride heel",           "str_heel",    "stride"),
        ("stride heelFP+ankle",   "str_hfp_ank", "stride"),
        ("stride ankAnchor+toe",  "str_toelead", "stride"),
    ]

    print("=" * 80)
    print("[foot/hand endpoint probe] 2D estimate vs matched truth, by azimuth (r2)")
    print("  speed truth = same point's 3D peak speed; stride truth = OBP column")
    print("=" * 80)
    hdr = f"{'variant':24s}" + "".join(f"{az:>7d}" for az in AZIMUTHS) + "   best"
    print(hdr); print("-" * len(hdr))
    for label, ekey, tkey in PLAN:
        line = f"{label:24s}"; best_az, best = None, -1
        for az in AZIMUTHS:
            val = r2(E[ekey][az], T[tkey])
            if pd.notna(val) and val > best:
                best, best_az = val, az
            line += f"{val:7.2f}"
        line += f"   {best_az} ({best:.2f})"
        print(line)

    print("\nKey comparisons @0deg:")
    print(f"  speed  like-for-like: wrist={r2(E['spd_wrist'][0], T['spd_wrist']):.3f}"
          f"  hand={r2(E['spd_hand'][0], T['spd_hand']):.3f}"
          f"  handmid={r2(E['spd_handmid'][0], T['spd_handmid']):.3f}")
    print(f"  speed  cross (diag.): hand2D-vs-wrist3D={r2(E['spd_hand'][0], T['spd_wrist']):.3f}"
          f"  wrist2D-vs-hand3D={r2(E['spd_wrist'][0], T['spd_hand']):.3f}")
    print(f"  stride: ankle={r2(E['str_ankle'][0], T['stride']):.3f}"
          f"  toe={r2(E['str_toe'][0], T['stride']):.3f}"
          f"  heel={r2(E['str_heel'][0], T['stride']):.3f}"
          f"  heelFP+ankle={r2(E['str_hfp_ank'][0], T['stride']):.3f}"
          f"  ankAnchor+toeLead={r2(E['str_toelead'][0], T['stride']):.3f}")


if __name__ == "__main__":
    main()
