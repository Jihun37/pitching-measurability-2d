"""Why does stride_angle get WORSE with GT events?

Re-screening the rejected candidates showed stride_angle at az90 falling from
r2 0.889 (our detected foot plant) to 0.493 (the OBP fp_100 landmark). An adopted
metric losing half its r2 when handed a better event is the opposite of the
expected direction, and it has to be explained before a GT-event map can be
treated as the measurability ceiling.

Two candidate explanations, both testable here:
  1. WRONG LANDMARK. OBP ships two foot-plant events, fp_10 (10% load) and
     fp_100 (100%). If the stride_angle column is defined at the other one, the
     "better" event is simply the wrong instant.
  2. SELF-REFERENTIAL ADVANTAGE. Our detector is derived from the same projected
     2D pose the metric is read from, so its error correlates with the pose and
     can flatter the fit -- the trap the ledger already flags for truth columns.
     If that is what is happening, a FIXED offset from the GT event (same frame
     for every pitch, no pose input) should not recover the r2, while our
     detector does.

Run:  conda activate diamond; cd src\\analysis; python gt_event_paradox.py
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

AZ = 90
EL = 0
LIMIT = 411


def r2(e, t):
    e, t = np.asarray(e, float), np.asarray(t, float)
    m = np.isfinite(e) & np.isfinite(t)
    return np.corrcoef(e[m], t[m])[0, 1] ** 2 if m.sum() > 4 else np.nan


def main():
    gt = load_gt_events()
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    root = os.path.join(config.OBP_DATA_DIR, "c3d")

    keys = ["ours_side", "ours_front", "gt_fp100", "gt_fp10",
            "gt_fp100_m5", "gt_fp100_m10", "gt_fp100_p5", "gt_pkh80"]
    acc = {k: {"e": [], "t": []} for k in keys}
    dfp = []
    done = 0
    for r in md.itertuples(index=False):
        if done >= LIMIT:
            break
        sp = r.session_pitch
        g = gt.get(sp)
        if sp not in poi.index or not g or not {"rel", "fp", "fp10", "pkh"} <= set(g):
            continue
        path = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            continue
        try:
            joints, fps = load_feet(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
            df = project_cam(joints, AZ, EL)
            rel = M.release_frame(df, arm, fps, M.JOINTS, view="frontal")
            f_side = int(M.foot_plant_frame(df, lead, fps, M.JOINTS, rel))
            f_front = int(M.foot_plant_frame(df, lead, fps, M.JOINTS, rel,
                                             view="front"))
        except Exception:
            continue
        tv = poi.loc[sp, "stride_angle"]
        frames = {
            "ours_side": f_side,
            "ours_front": f_front,
            "gt_fp100": g["fp"],
            "gt_fp10": g["fp10"],
            "gt_fp100_m5": g["fp"] - 5,
            "gt_fp100_m10": g["fp"] - 10,
            "gt_fp100_p5": g["fp"] + 5,
            # a pose-free offset with the SAME construction as our detector's
            # prior, but anchored on GT events only
            "gt_pkh80": int(round(g["pkh"] + 0.80 * (g["rel"] - g["pkh"]))),
        }
        for k, f in frames.items():
            if f is None or f < 1 or f >= len(df):
                continue
            acc[k]["e"].append(M.stride_angle_2d(df, arm, int(f), M.JOINTS))
            acc[k]["t"].append(tv)
        dfp.append(dict(sp=sp, ours_front=f_front, gt_fp100=g["fp"],
                        gt_fp10=g["fp10"],
                        d_front_100=f_front - g["fp"],
                        d_front_10=f_front - g["fp10"]))
        done += 1

    print(f"pitches {done}   stride_angle vs OBP stride_angle column, "
          f"az{AZ}/el{EL}\n")
    print(f"{'foot-plant frame source':>26} {'n':>5} {'r2':>8}")
    for k in keys:
        print(f"{k:>26} {len(acc[k]['e']):>5} {r2(*acc[k].values()):>8.3f}")

    d = pd.DataFrame(dfp)
    print("\nour frontal detector minus each GT landmark (frames @360fps):")
    for c in ("d_front_100", "d_front_10"):
        v = d[c].to_numpy(float)
        print(f"  {c:>12}  median {np.median(v):>6.1f}  SD {v.std():>6.1f}")

    print("\nreading: if gt_fp10 or a shifted gt_fp100 recovers the r2, the "
          "landmark choice was wrong;\nif only OUR frames do, the detector's "
          "error is correlated with the pose it is read from.")


if __name__ == "__main__":
    main()
