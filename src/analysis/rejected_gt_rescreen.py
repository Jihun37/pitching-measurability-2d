"""Re-screen the REJECTED candidates with GT events (ledger §2-A).

Those candidates were scored with our own detected events. Foot plant is our
noisiest event (SD ~15 frames = 42 ms vs the OBP landmark, only 41-72% within
3 frames), and most of the rejected candidates are anchored exactly there --
so a rejection could be a detector artefact rather than a projection limit.
The precedent is ledger §2-B: five of six COM-curve candidates jump from
0.11-0.47 to 0.88-0.99 the moment OBP events are used.

For every candidate this prints r2 with OUR events and with GT events, at each
azimuth, and flags any that crosses the usable floor (0.60) only under GT.

Run:  conda activate diamond; cd src\\analysis; python rejected_gt_rescreen.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3", "../tests"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config, metrics as M, obp_project as O
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from obp_gt_events import load_gt_events
from extended_metrics_test import compute_extended, TRUTH

AZS = [0, 30, 60, 90, 120, 150, 180]
EL = 0
FLOOR = 0.60
LIMIT = 411

CANDS = [k for k, (col, kind) in TRUTH.items() if col is not None]


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

    acc = {(c, az, mode): {"e": [], "t": []}
           for c in CANDS for az in AZS for mode in ("ours", "gt")}
    done = 0
    for r in md.itertuples(index=False):
        if done >= LIMIT:
            break
        sp = r.session_pitch
        g = gt.get(sp)
        if sp not in poi.index or not g or not {"rel", "fp"} <= set(g):
            continue
        path = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            continue
        try:
            joints, fps = load_feet(path)
            arm = O.detect_throwing_arm(joints, fps)
        except Exception:
            continue
        for az in AZS:
            try:
                df = project_cam(joints, az, EL)
            except Exception:
                continue
            for mode, ev in (("ours", None), ("gt", {"rel": g["rel"], "fp": g["fp"]})):
                try:
                    vals = compute_extended(df, fps, arm, events=ev)
                except Exception:
                    continue
                for c in CANDS:
                    col = TRUTH[c][0]
                    if c not in vals or col not in poi.columns:
                        continue
                    acc[(c, az, mode)]["e"].append(vals[c])
                    acc[(c, az, mode)]["t"].append(poi.loc[sp, col])
        done += 1

    print(f"pitches {done}   (r2 vs the OBP truth column, el=0)\n")
    hdr = f"{'candidate':>28} " + "".join(f"{('az%d' % a):>16}" for a in AZS)
    print(hdr)
    print(f"{'':>28} " + "".join(f"{'ours':>8}{'GT':>8}" for _ in AZS))
    revived = []
    for c in CANDS:
        line = f"{c:>28} "
        best_o = best_g = np.nan
        for az in AZS:
            o = r2(*acc[(c, az, "ours")].values())
            gg = r2(*acc[(c, az, "gt")].values())
            best_o = np.nanmax([best_o, o])
            best_g = np.nanmax([best_g, gg])
            line += f"{o:>8.3f}{gg:>8.3f}"
        print(line)
        if np.isfinite(best_g) and best_g >= FLOOR and not (best_o >= FLOOR):
            revived.append((c, best_o, best_g))

    print("\nCROSSES THE 0.60 FLOOR ONLY WITH GT EVENTS "
          "(= rejected for detector error, not geometry):")
    if revived:
        for c, o, g_ in revived:
            print(f"  {c:>28}   ours {o:.3f}  ->  GT {g_:.3f}")
    else:
        print("  none")


if __name__ == "__main__":
    main()
