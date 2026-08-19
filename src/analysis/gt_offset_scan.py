"""Is there ANY fixed instant that explains stride_angle's r2?

gt_event_paradox showed no GT landmark reaching our detector's 0.836 at az90.
This scans r2 against a fixed frame offset from the GT foot plant across a wide
range. Every offset is the SAME shift for every pitch, so each point is a
candidate time-instant DEFINITION, evaluated without any pose input.

  - if some offset reaches ~0.84, our detector is simply finding a differently
    defined instant, and that instant can be stated as a definition;
  - if the whole curve stays far below, no fixed instant explains it, and the
    r2 comes from our detector CHOOSING a per-pitch frame using the same 2D pose
    the metric is then read from -- a self-referential advantage, and the honest
    geometric ceiling is the curve's maximum, not 0.836.

Run:  conda activate diamond; cd src\\analysis; python gt_offset_scan.py
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

AZ, EL = 90, 0
OFFSETS = list(range(-60, 61, 5))
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

    acc = {o: {"e": [], "t": []} for o in OFFSETS}
    ours = {"e": [], "t": []}
    done = 0
    for r in md.itertuples(index=False):
        if done >= LIMIT:
            break
        sp = r.session_pitch
        g = gt.get(sp)
        if sp not in poi.index or not g or "fp" not in g:
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
            ff = int(M.foot_plant_frame(df, lead, fps, M.JOINTS, rel, view="front"))
        except Exception:
            continue
        tv = poi.loc[sp, "stride_angle"]
        ours["e"].append(M.stride_angle_2d(df, arm, ff, M.JOINTS))
        ours["t"].append(tv)
        for o in OFFSETS:
            f = g["fp"] + o
            if 1 <= f < len(df):
                acc[o]["e"].append(M.stride_angle_2d(df, arm, int(f), M.JOINTS))
                acc[o]["t"].append(tv)
        done += 1

    vals = {o: r2(*acc[o].values()) for o in OFFSETS}
    best_o = max(vals, key=lambda k: (vals[k] if np.isfinite(vals[k]) else -1))
    print(f"pitches {done}   stride_angle, az{AZ}/el{EL}\n")
    print(f"{'offset from GT fp':>19} {'ms':>7} {'r2':>8}")
    for o in OFFSETS:
        mark = "  <- best" if o == best_o else ""
        print(f"{o:>19} {o/360*1000:>7.0f} {vals[o]:>8.3f}{mark}")
    print(f"\n{'our frontal detector':>19} {'':>7} {r2(*ours.values()):>8.3f}")
    print(f"\nbest FIXED instant : {vals[best_o]:.3f} at {best_o:+d} frames "
          f"({best_o/360*1000:+.0f} ms from GT foot plant)")
    print("our per-pitch frame: %.3f" % r2(*ours.values()))


if __name__ == "__main__":
    main()
