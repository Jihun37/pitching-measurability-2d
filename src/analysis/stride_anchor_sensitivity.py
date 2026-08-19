"""Which half of the stride estimator is frame-sensitive: the lead ankle or the
trail anchor?

Both our detected foot plant and the OBP landmark sit on the lead ankle's settled
plateau (median +0.013 and +0.009 of travel), so the lead-ankle reading cannot
explain stride's collapse from r2 0.824 to 0.392 under GT events. The other input
is trail_anchor_x, which searches trail_x[:fp+1] for the pre-motion quiet window
using a threshold defined as 20% of THAT SEGMENT'S own peak speed -- so changing fp
changes the threshold, the window length, and therefore the anchor.

Cross the two inputs to isolate it, and score the event-free variant that takes the
anchor over the whole delivery.

Run:  conda activate diamond; cd src\\analysis; python stride_anchor_sensitivity.py
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

AZ, EL = 180, 0
LIMIT = 411
KEYS = ["ankle_ours__anchor_ours", "ankle_gt__anchor_gt",
        "ankle_gt__anchor_ours", "ankle_ours__anchor_gt",
        "ankle_gt__anchor_rel", "ankle_plateau__anchor_rel"]


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

    acc = {k: {"e": [], "t": []} for k in KEYS}
    anc_diff = []
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
            f_o = int(M.foot_plant_frame(df, lead, fps, M.JOINTS, rel))
        except Exception:
            continue
        f_g, f_r = int(g["fp"]), int(g["rel"])
        la = df[f"{lead}_ankle_x"].to_numpy(float)
        ta = df[f"{trail}_ankle_x"].to_numpy(float)
        stat = M.pixel_stature(df, M.JOINTS)
        tv = poi.loc[sp, "stride_length"]

        a_o = M.trail_anchor_x(ta, f_o, fps)
        a_g = M.trail_anchor_x(ta, f_g, fps)
        a_r = M.trail_anchor_x(ta, f_r, fps)
        plat = float(np.nanmedian(la[max(f_o, f_g):f_r + 1]))
        anc_diff.append((a_g - a_o) / stat)

        vals = {
            "ankle_ours__anchor_ours": abs(la[f_o] - a_o),
            "ankle_gt__anchor_gt":     abs(la[f_g] - a_g),
            "ankle_gt__anchor_ours":   abs(la[f_g] - a_o),
            "ankle_ours__anchor_gt":   abs(la[f_o] - a_g),
            "ankle_gt__anchor_rel":    abs(la[f_g] - a_r),
            "ankle_plateau__anchor_rel": abs(plat - a_r),
        }
        for k, v in vals.items():
            acc[k]["e"].append(v / stat)
            acc[k]["t"].append(tv)
        done += 1

    print(f"pitches {done}   truth = OBP stride_length, az{AZ}/el{EL}\n")
    print(f"{'lead ankle from':>16} {'trail anchor from':>18} {'r2':>8}")
    for k in KEYS:
        a, b = k.split("__")
        print(f"{a.replace('ankle_',''):>16} {b.replace('anchor_',''):>18} "
              f"{r2(*acc[k].values()):>8.3f}")
    v = np.array(anc_diff, float)
    v = v[np.isfinite(v)]
    print(f"\nanchor(GT fp) - anchor(our fp), in statures: "
          f"median {np.median(v):+.4f}  SD {v.std():.4f}  "
          f"|.|>0.02: {np.mean(np.abs(v) > 0.02):.1%}")
    print("\nreading: whichever COLUMN changes the r2 is the frame-sensitive input.")


if __name__ == "__main__":
    main()
