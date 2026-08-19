"""Why does the stride metric collapse when handed the GT foot plant?

Under GT events stride falls from r2 0.839 to 0.546 and loses every usable zone,
which is implausible for a horizontal distance at foot plant. Three explanations:

  1. our ESTIMATOR is wrong -- ruled out by reading it: est_stride_anchor measures
     the trail foot's pre-motion anchor to the lead ankle, not both ankles at fp
     (that variant exists but is a rejected row).
  2. the GT landmark is off.
  3. our foot-plant DEFINITION differs from the instant OBP measured the column at.

The test needs no camera model and no 3D reconstruction. Once the lead foot is
planted its horizontal position PLATEAUS, so any frame after true contact must give
the same stride -- the metric is insensitive to the exact frame inside the plateau.
Measure where each candidate frame sits relative to that plateau: a frame BEFORE
the plateau is still travelling and under-reads the stride.

Run:  conda activate diamond; cd src\\analysis; python stride_gt_definition.py
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

AZ, EL = 180, 0          # stride's own best azimuth
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

    est = {k: {"e": [], "t": []} for k in ("ours", "gt", "plateau")}
    rows = []
    done = 0
    for r in md.itertuples(index=False):
        if done >= LIMIT:
            break
        sp = r.session_pitch
        g = gt.get(sp)
        if sp not in poi.index or not g or not {"fp", "rel"} <= set(g):
            continue
        path = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            continue
        try:
            j, fps = load_feet(path)
            arm = O.detect_throwing_arm(j, fps)
            lead = "left" if arm == "right" else "right"
            trail = "right" if lead == "left" else "left"
            df = project_cam(j, AZ, EL)
            rel = M.release_frame(df, arm, fps, M.JOINTS)
            f_ours = int(M.foot_plant_frame(df, lead, fps, M.JOINTS, rel))
        except Exception:
            continue
        f_gt, f_rel = int(g["fp"]), int(g["rel"])
        la = df[f"{lead}_ankle_x"].to_numpy(float)
        ta = df[f"{trail}_ankle_x"].to_numpy(float)
        stat = M.pixel_stature(df, M.JOINTS)
        tv = poi.loc[sp, "stride_length"]

        def stride_at(f):
            return abs(la[f] - M.trail_anchor_x(ta, f, fps)) / stat

        # plateau = the lead ankle's settled position over [fp_gt, release]
        seg = la[min(f_gt, f_ours):f_rel + 1]
        if seg.size < 3:
            continue
        plat = float(np.nanmedian(la[max(f_gt, f_ours):f_rel + 1]))
        trav = float(np.nanmax(la[:f_rel + 1]) - np.nanmin(la[:f_rel + 1])) + 1e-9

        est["ours"]["e"].append(stride_at(f_ours)); est["ours"]["t"].append(tv)
        est["gt"]["e"].append(stride_at(f_gt)); est["gt"]["t"].append(tv)
        est["plateau"]["e"].append(abs(plat - M.trail_anchor_x(ta, f_rel, fps)) / stat)
        est["plateau"]["t"].append(tv)
        rows.append(dict(sp=sp, f_ours=f_ours, f_gt=f_gt, d=f_ours - f_gt,
                         gap_ours=(la[f_ours] - plat) / trav,
                         gap_gt=(la[f_gt] - plat) / trav))
        done += 1

    d = pd.DataFrame(rows)
    print(f"pitches {done}   truth = OBP stride_length column, az{AZ}/el{EL}\n")
    print(f"{'stride measured at':>26} {'r2':>8}")
    for k in ("ours", "gt", "plateau"):
        print(f"{k:>26} {r2(*est[k].values()):>8.3f}")

    print("\nlead-ankle position at each candidate frame, as a fraction of the")
    print("settled plateau (0 = already planted, negative = still travelling):")
    for c in ("gap_ours", "gap_gt"):
        v = d[c].to_numpy(float)
        v = v[np.isfinite(v)]
        print(f"  {c:>9}  median {np.median(v):>+7.3f}  "
              f"IQR {np.percentile(v,25):+.3f}..{np.percentile(v,75):+.3f}")
    print(f"\n  our fp - GT fp : median {np.median(d.d):+.1f} frames  "
          f"SD {d.d.std():.1f}")
    print("\nreading: if the GT frame sits well before the plateau it is measuring a")
    print("foot still in flight -- a DEFINITION difference, not a broken landmark.")


if __name__ == "__main__":
    main()
