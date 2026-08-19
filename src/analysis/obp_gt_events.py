"""OBP ground-truth EVENT frames (peak knee height / foot plant / release).

The angle map asks a geometric question -- can this metric be measured from this
viewpoint -- so it must not be answered with our own event detectors: a detector
failure and a projection impossibility then look identical. OBP ships the events
itself in full_sig/landmarks.csv, so use those as the ceiling and treat detector
error as a separate layer on top.

Columns available per pitch (n=411):
    pkh_time     peak knee height        410 present
    fp_10_time   foot plant, 10% load    403 present (8 missing)
    fp_100_time  foot plant, 100% load   403 present
    MER_time     max external rotation   411
    BR_time      ball release            411
    MIR_time     max internal rotation   411

The landmark time grid is the SAME grid as the c3d clip (verified: row count is
equal for all 403 scored pitches), so the index of the matching time is directly a
c3d frame index.

TRAP: missing events are stored as time = 0, not NaN -- always filter t > 0, or the
event lands on frame 0 and every metric reading it silently reads the windup.

    from obp_gt_events import load_gt_events
    gt = load_gt_events()                  # {session_pitch: {"pkh":…, "fp":…}}
"""
import os, sys, zipfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

EVENTS = {"pkh": "pkh_time", "fp10": "fp_10_time", "fp": "fp_100_time",
          "mer": "MER_time", "rel": "BR_time", "mir": "MIR_time"}


def load_gt_events(events=None):
    """{session_pitch: {name: frame_index}} for every event present (time > 0)."""
    want = events or list(EVENTS)
    cols = ["session_pitch", "time"] + [EVENTS[e] for e in want]
    zp = os.path.join(config.OBP_DATA_DIR, "full_sig", "landmarks.zip")
    with zipfile.ZipFile(zp) as z:
        with z.open("landmarks.csv") as f:
            lm = pd.read_csv(f, usecols=cols)

    out = {}
    for sp, g in lm.groupby("session_pitch", sort=False):
        g = g.sort_values("time")
        t = g["time"].to_numpy(float)
        d = {}
        for e in want:
            v = float(g[EVENTS[e]].iloc[0])
            if np.isfinite(v) and v > 0:          # 0 == missing, never a real event
                d[e] = int(np.argmin(np.abs(t - v)))
        if d:
            out[sp] = d
    return out


def gt_frame_count():
    """{session_pitch: number of landmark rows} -- for the grid-alignment check."""
    zp = os.path.join(config.OBP_DATA_DIR, "full_sig", "landmarks.zip")
    with zipfile.ZipFile(zp) as z:
        with z.open("landmarks.csv") as f:
            lm = pd.read_csv(f, usecols=["session_pitch", "time"])
    return lm.groupby("session_pitch").size().to_dict()


if __name__ == "__main__":
    # Report: how far our detectors sit from the GT events, per view.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
    import metrics as M, obp_project as O
    from master_angle_table import load_feet
    from hss_elevation_test import project_cam
    from angle_zone_sweep import release_view

    AZS = [0, 90]
    gt = load_gt_events()
    nrows = gt_frame_count()
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    root = os.path.join(config.OBP_DATA_DIR, "c3d")

    rows = []
    aligned = misaligned = 0
    for r in md.itertuples(index=False):
        sp = r.session_pitch
        if sp not in gt:
            continue
        p = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(p):
            continue
        try:
            j, fps = load_feet(p)
            arm = O.detect_throwing_arm(j, fps)
            lead = "left" if arm == "right" else "right"
        except Exception:
            continue
        g = gt[sp]
        for az in AZS:
            df = project_cam(j, az, 0)
            if az == AZS[0]:
                if nrows.get(sp) == len(df):
                    aligned += 1
                else:
                    misaligned += 1
            try:
                rel = M.release_frame(df, arm, fps, M.JOINTS,
                                      view=release_view(az, 0))
                fp_s = M.foot_plant_frame(df, lead, fps, M.JOINTS, rel)
                fp_f = M.foot_plant_frame(df, lead, fps, M.JOINTS, rel, view="front")
                pkh = M.peak_knee_height_frame(df, lead, fp_s, M.JOINTS)
            except Exception:
                continue
            rows.append(dict(az=az, sp=sp,
                             d_rel=rel - g.get("rel", np.nan),
                             d_fp_side_100=fp_s - g.get("fp", np.nan),
                             d_fp_side_10=fp_s - g.get("fp10", np.nan),
                             d_fp_front_100=fp_f - g.get("fp", np.nan),
                             d_fp_front_10=fp_f - g.get("fp10", np.nan),
                             d_pkh=pkh - g.get("pkh", np.nan)))

    d = pd.DataFrame(rows)
    print(f"pitches with GT events : {len(gt)}")
    print(f"landmark/c3d row count : aligned {aligned}  misaligned {misaligned}\n")
    print("detector - GT, in frames @360fps   (median / SD / share |err|<=3f)")
    for az in AZS:
        s = d[d.az == az]
        print(f"\n  az {az}:")
        for c in [c for c in d.columns if c.startswith("d_")]:
            v = s[c].to_numpy(float)
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            print(f"    {c:>16}  n={v.size:>4}  median {np.median(v):>7.1f}  "
                  f"SD {v.std():>6.1f}  <=3f {np.mean(np.abs(v) <= 3):>6.1%}")
