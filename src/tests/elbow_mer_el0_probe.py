"""Elbow Flex @MER from a GROUND-LEVEL camera (el=0): why the whole row dies.

The GT-clean map gives el=0 a best cell of r2=0.33 (az90/270) and 0.06 at the
pure side az0, against 0.886 at the az330/el60 anchor. This probe measures the
cause instead of asserting it, at the MER frame:

  tilt      angle between the arm PLANE normal (upper arm x forearm) and the
            camera axis. 0 deg = the plane faces the camera and the projected
            angle equals the true one; 90 deg = the plane is edge-on and the
            three joints collapse onto a line, where the measured angle is
            whatever the noise says.
  fore      projected / true forearm length (1.0 = no foreshortening).
  err       2D reading minus the OBP truth, per pitch.

Not a new definition and not a new map: it reads metrics.elbow_flexion_2d at the
GT MER frame (obp_gt_events) on the gt_clean population, so its r2 column must
land on the published map cells.

Run:  conda activate diamond; cd src\\tests; python elbow_mer_el0_probe.py
      python elbow_mer_el0_probe.py --limit 400
"""
import os, sys, argparse
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../stage2", "../stage3", "../analysis"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)

import config
import obp_project as O
import metrics as M
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from obp_gt_events import load_gt_events
from gt_landmark_outlier_effect import outlier_pitches

TRUTH_COL = "elbow_flexion_mer"
VIEWS = [(0, 0), (90, 0), (270, 0), (0, 15), (0, 30), (0, 45), (330, 45),
         (330, 60), (330, 75)]


def cam_axis(az_deg, el_deg):
    """Viewing direction of project_cam (same formula, single source of truth
    for the convention: az increases 3B -> home, az90 = front)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([-np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    gt = load_gt_events(["mer"])
    bad = outlier_pitches()

    rec = {v: {"tilt": [], "fore": [], "est": [], "tru": []} for v in VIEWS}
    done = 0
    for r in md.itertuples(index=False):
        if done >= a.limit:
            break
        sp = r.session_pitch
        if sp in bad or sp not in gt or "mer" not in gt[sp] or sp not in poi.index:
            continue
        truth = float(poi.loc[sp, TRUTH_COL])
        if not np.isfinite(truth):
            continue
        path = os.path.join(config.OBP_DATA_DIR, "c3d",
                            f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            continue
        joints, fps = load_feet(path)
        arm = O.detect_throwing_arm(joints, fps)
        mer = gt[sp]["mer"]
        sh = joints[f"{arm}_shoulder"][:, mer].astype(float)
        el3 = joints[f"{arm}_elbow"][:, mer].astype(float)
        wr = joints[f"{arm}_wrist"][:, mer].astype(float)
        u, f = el3 - sh, wr - el3
        nrm = np.cross(u, f)
        nrm = nrm / (np.linalg.norm(nrm) + 1e-9)

        for v in VIEWS:
            ax = cam_axis(*v)
            tilt = np.degrees(np.arccos(min(1.0, abs(float(nrm @ ax)))))
            df = project_cam(joints, *v)
            pw = np.array([df[f"{arm}_wrist_x"].iloc[mer], df[f"{arm}_wrist_y"].iloc[mer]])
            pe = np.array([df[f"{arm}_elbow_x"].iloc[mer], df[f"{arm}_elbow_y"].iloc[mer]])
            rec[v]["tilt"].append(tilt)
            rec[v]["fore"].append(float(np.linalg.norm(pw - pe) / np.linalg.norm(f)))
            rec[v]["est"].append(M.elbow_flexion_2d(df, arm, mer, M.JOINTS))
            rec[v]["tru"].append(truth)
        done += 1
        if done % 50 == 0:
            print(f"  ...{done}")

    print(f"\nn = {done} pitches (gt_clean, GT MER frame)\n")
    print(f"{'view':>12}{'plane tilt':>12}{'forearm':>10}{'r2':>8}"
          f"{'bias':>9}{'MAE':>8}{'err SD':>9}")
    print(f"{'':>12}{'(0=face-on)':>12}{'proj/true':>10}{'':>8}"
          f"{'deg':>9}{'deg':>8}{'deg':>9}")
    print("-" * 68)
    out = []
    for v in VIEWS:
        d = rec[v]
        e = np.asarray(d["est"], float); t = np.asarray(d["tru"], float)
        m = np.isfinite(e) & np.isfinite(t)
        r2 = np.corrcoef(e[m], t[m])[0, 1] ** 2
        err = e[m] - t[m]
        row = dict(az=v[0], el=v[1], tilt=float(np.mean(d["tilt"])),
                   fore=float(np.mean(d["fore"])), r2=float(r2),
                   bias=float(np.mean(err)), mae=float(np.mean(np.abs(err))),
                   err_sd=float(np.std(err, ddof=1)))
        out.append(row)
        print(f"  az{v[0]:>3d}/el{v[1]:>2d}{row['tilt']:>12.1f}{row['fore']:>10.2f}"
              f"{row['r2']:>8.2f}{row['bias']:>9.1f}{row['mae']:>8.1f}"
              f"{row['err_sd']:>9.1f}")

    p = os.path.join(config.OBP_VALIDATION_DIR, "elbow_mer_el0_probe.csv")
    pd.DataFrame(out).to_csv(p, index=False)
    print(f"\nsaved -> {p}")
    print("Compare the r2 column against angle_zone_sweep_gt_clean.csv (subset of "
          "pitches, so close but not bit-identical).")


if __name__ == "__main__":
    main()
