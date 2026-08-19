"""Is our foot-plant residual a DETECTION defect or a DEFINITION gap?

fp_detect_probe ruled out the fallback (it never fires) and showed three
physical redesigns all LOSE to the incumbent, which already puts 61% of pitches
within 3 frames of OBP fp_100 but has a ~5% tail beyond 30 frames. Before
tuning further, test the competing explanation:

  OBP fp_100 = the instant the lead force plate reads 100% body weight
               (a KINETIC event: full weight acceptance)
  our fp     = the instant the lead foot visually stops moving
               (a KINEMATIC event: contact / foot planted)

Those are not the same instant, and the delay between them is the pitcher's
loading rate -- invisible to any camera. If that is the residual, it is a
definition wall, not an engineering target, and no 2D detector can close it.

Decomposition (n=408):
  1. Build a 3D KINEMATIC foot plant from the c3d lead ankle (what a perfect
     camera could in principle see), independent of the force plate.
  2. our 2D fp  vs  3D kinematic fp   -> genuine 2D DETECTION error (the part
     that is actually improvable).
  3. 3D kinematic fp  vs  OBP fp_100  -> the kinematic-to-kinetic DEFINITION
     gap (the part that is not).
  4. Does our error vs fp_100 track the pitcher's loading span
     (fp_10 -> fp_100)? A correlation confirms (3) drives the tail.
"""
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(HERE, "..", "stage3"))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))

import config                                    # noqa: E402
import metrics as M                              # noqa: E402
import obp_project as O                          # noqa: E402
from master_angle_table import load_feet         # noqa: E402

LM_ZIP = os.path.join(config.OBP_DATA_DIR, "full_sig", "landmarks.zip")


def fp_kinematic_3d(joints, lead, fps, rel):
    """3D kinematic foot plant: the lead ankle stops moving and stays down.
    Uses the raw c3d (no projection, no pose noise), so it is the ceiling any
    camera-based detector could reach. Definition: within [0, rel], the first
    frame after the ankle's descent where its 3D speed falls below 8% of the
    stride peak and remains below through release."""
    a = joints["left_ankle" if lead == "left" else "right_ankle"]
    end = max(5, int(rel))
    v = np.linalg.norm(np.diff(a[:, :end + 1], axis=1), axis=0) * fps
    v = np.concatenate([[v[0]], v])
    w = max(3, int(round(0.03 * fps)))
    v = np.convolve(v, np.ones(w) / w, mode="same")
    pk = np.nanmax(v)
    if not np.isfinite(pk) or pk <= 0:
        return None
    lo = v < 0.08 * pk
    lo[:int(0.35 * end)] = False           # landing cannot precede the leg lift
    stay = np.flatnonzero(np.cumsum(~lo[::-1])[::-1] == 0)
    return int(stay[0]) if len(stay) else None


def main():
    print("loading OBP event times ...")
    with zipfile.ZipFile(LM_ZIP) as z:
        with z.open("landmarks.csv") as f:
            lm = pd.read_csv(f, usecols=["session_pitch", "time",
                                         "fp_10_time", "fp_100_time"])
    ev = {}
    for sp, g in lm.groupby("session_pitch"):
        g = g.sort_values("time")
        t = g.time.to_numpy(float)
        ev[sp] = (int(np.argmin(np.abs(t - float(g.fp_10_time.iloc[0])))),
                  int(np.argmin(np.abs(t - float(g.fp_100_time.iloc[0])))))
    print(f"  {len(ev)} pitches\n")

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    rows = []
    for r in md.itertuples(index=False):
        sp = r.session_pitch
        if sp not in ev:
            continue
        p = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(p):
            continue
        try:
            j, fps = load_feet(p)
            arm = O.detect_throwing_arm(j, fps)
            lead = "left" if arm == "right" else "right"
            df0 = O.project_view(j, azimuth_deg=0.0)
            rel = M.release_frame(df0, arm, fps, M.JOINTS)
            if rel < 5:
                continue
            fp2d = M.foot_plant_frame(df0, lead, fps, M.JOINTS, rel)
            fp3d = fp_kinematic_3d(j, lead, fps, rel)
            if fp3d is None:
                continue
            g10, g100 = ev[sp]
            rows.append(dict(sp=sp, fp2d=fp2d, fp3d=fp3d, g10=g10, g100=g100,
                             d_2d_3d=fp2d - fp3d,
                             d_3d_100=fp3d - g100,
                             d_2d_100=fp2d - g100,
                             load_span=g100 - g10))
        except Exception:
            continue

    d = pd.DataFrame(rows)
    print(f"n = {len(d)} pitches, frames @360Hz (1 frame = 2.8 ms)\n")

    def stat(col, name):
        v = d[col].to_numpy(float); av = np.abs(v)
        print(f"  {name:<38}{np.median(v):>+8.1f}{av.mean():>9.1f}"
              f"{np.percentile(av,90):>8.1f}{100*(av<=3).mean():>7.0f}%"
              f"{100*(av>30).mean():>7.0f}%")

    print(f"  {'comparison':<38}{'median':>8}{'mean|.|':>9}{'p90':>8}"
          f"{'<=3f':>8}{'>30f':>8}")
    stat("d_2d_100", "our 2D fp      vs OBP fp_100")
    stat("d_2d_3d", "our 2D fp      vs 3D kinematic fp   [DETECTION]")
    stat("d_3d_100", "3D kinematic fp vs OBP fp_100      [DEFINITION]")

    print(f"\n  OBP loading span fp_10->fp_100: median {d.load_span.median():.0f}"
          f"  mean {d.load_span.mean():.1f}  p90 {np.percentile(d.load_span,90):.0f} frames")

    r = np.corrcoef(d.load_span, np.abs(d.d_2d_100))[0, 1]
    r3 = np.corrcoef(d.load_span, np.abs(d.d_3d_100))[0, 1]
    print(f"\n  corr(loading span, |our error vs fp_100|)          r = {r:+.3f}")
    print(f"  corr(loading span, |3D-kinematic error vs fp_100|) r = {r3:+.3f}")

    tail = d[np.abs(d.d_2d_100) > 30]
    print(f"\n  tail pitches (|err vs fp_100| > 30f): n={len(tail)} "
          f"({100*len(tail)/len(d):.1f}%)")
    if len(tail):
        print(f"    their |2D vs 3D-kinematic| error: median "
              f"{np.median(np.abs(tail.d_2d_3d)):.1f}  mean {np.mean(np.abs(tail.d_2d_3d)):.1f}")
        print(f"    their loading span:               median "
              f"{tail.load_span.median():.0f} (vs {d.load_span.median():.0f} overall)")

    dst = os.path.join(config.ROOT, "data", "outputs", "obp_validation",
                       "fp_definition_split.csv")
    d.to_csv(dst, index=False)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
