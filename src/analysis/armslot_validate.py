"""
Arm slot, validated on its own terms.

The definition here is the shoulder-to-hand vector against the vertical
(Escamilla & Fleisig). The release's arm_slot column is a FOREARM projection
angle, a different quantity wearing the same name, so the truth is built directly
from the 3D coordinates instead of taken from the column:

    truth = angle( the Y and Z components of (hand - shoulder), vertical Z )

The 2D estimate at each camera azimuth is then compared against that truth,
giving an r-squared per azimuth. It should peak at the front, near 90 degrees,
and collapse at the side.

Usage:
  python armslot_validate.py                 # the whole release, as an r2 table
  python armslot_validate.py --c3d <path>    # one clip, estimate against truth
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import obp_project as O
import metrics as M

AZIMUTHS = [0, 15, 30, 45, 60, 75, 90]


def truth_3d_frontal(joints, arm, rel):
    """The 3D coronal-plane arm slot: the angle (hand - shoulder) makes with the
    vertical, using its [horizontal Y, vertical Z] components.
    """
    S = joints[f"{arm}_shoulder"][:, rel]
    W = joints[f"{arm}_wrist"][:, rel]
    vec = W - S                       # (X, Y, Z): X is the pitch direction, Y lateral, Z vertical
    run = abs(vec[1])                 # in-plane horizontal component, Y
    rise = vec[2]                     # vertical component, Z
    return float(np.degrees(np.arctan2(run, rise)))


def estimate_2d(joints, arm, rel, az):
    df = O.project_view(joints, azimuth_deg=az)
    sx, sy = df[f"{arm}_shoulder_x"].iloc[rel], df[f"{arm}_shoulder_y"].iloc[rel]
    wx, wy = df[f"{arm}_wrist_x"].iloc[rel],   df[f"{arm}_wrist_y"].iloc[rel]
    return float(np.degrees(np.arctan2(abs(wx-sx), (sy-wy))))


def release_of(joints, arm, fps):
    df0 = O.project_view(joints, azimuth_deg=0.0)
    return M.release_frame(df0, arm, fps, M.JOINTS)


def one(path):
    joints, fps = O.load_c3d_joints(path)
    arm = O.detect_throwing_arm(joints, fps)
    rel = release_of(joints, arm, fps)
    truth = truth_3d_frontal(joints, arm, rel)
    est = {az: estimate_2d(joints, arm, rel, az) for az in AZIMUTHS}
    return truth, est


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = np.nansum((y-yhat)**2); ss_tot = np.nansum((y-np.nanmean(y))**2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3d", default=None)
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    if a.c3d and not a.batch:
        truth, est = one(a.c3d)
        print(f"3D coronal truth arm slot = {truth:.1f} deg")
        for az in sorted(est):
            print(f"  az={az:2d}deg  2D estimate {est[az]:5.1f}  (error {est[az]-truth:+.1f})")
        return

    import config
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    truths, ests = [], {az: [] for az in AZIMUTHS}
    n = fail = 0
    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit: break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path): fail += 1; continue
        try:
            t, e = one(path)
            truths.append(t)
            for az in AZIMUTHS: ests[az].append(e[az])
            n += 1
        except Exception:
            fail += 1
    print(f"done {n} / failed {fail}\n")
    # Save the truth distribution, for the z-score comparison against real
    # frontal video.
    try:
        import config as _cfg
        os.makedirs(_cfg.OBP_VALIDATION_DIR, exist_ok=True)
        pd.DataFrame({"arm_slot_truth": truths}).to_csv(
            os.path.join(_cfg.OBP_VALIDATION_DIR, "armslot_ref.csv"), index=False)
        print(f"arm slot truth distribution -> {os.path.join(_cfg.OBP_VALIDATION_DIR,'armslot_ref.csv')}"
              f"  (mean {np.mean(truths):.1f} / std {np.std(truths):.1f})\n")
    except Exception as _e:
        pass
    print(f"arm slot definition = shoulder->hand against the vertical (truth from the 3D coronal plane)")
    print(f"{'azimuth':>8} {'r2':>8} {'MAE':>8}")
    for az in AZIMUTHS:
        e = np.array(ests[az]); t = np.array(truths)
        print(f"{az:6d}° {r2(t,e):8.3f} {np.nanmean(np.abs(e-t)):8.1f}")
    print("\n-> r2 should peak near the front (90 deg) and collapse at the side (0 deg)")


if __name__ == "__main__":
    main()