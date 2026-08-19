"""Foot-plant detector v2 — clean ground truth + persistence fix.

Two corrections to the first pass (fp_detect_probe / fp_definition_split):

1. GROUND-TRUTH BUG (mine, not the detector's). Some OBP pitches carry
   fp_10_time = fp_100_time = 0, i.e. the force-plate event is missing. The
   probe mapped time 0.0 to frame 0 and scored the detector against it, which
   manufactured "catastrophic" errors of +300..+700 frames and inflated
   mean|.| and SD (74.8). Those pitches are excluded here.

2. The REAL defect, once the fake tail is gone: in the mid-band (6 < |err| <= 30
   frames) 77 of 80 pitches fire EARLY. The incumbent returns cand[0] -- the
   FIRST frame satisfying its 3-way threshold conjunction -- so a momentary
   qualifying blip during the stride triggers it prematurely. Fix: require the
   conjunction to PERSIST (the foot is planted and stays planted), which is the
   physical meaning of the event.

Scored against OBP fp_100_time on valid-GT pitches only.
"""
import argparse
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


def _prep(df, lead, rel):
    """Shared signals of the incumbent rule: oriented forward travel, ground
    proximity, vertical speed -- so variants differ only in SELECTION."""
    ak = "l_an" if lead == "left" else "r_an"
    x, y = M._xy(df, ak, M.JOINTS)
    x = np.asarray(x, float); y = np.asarray(y, float)
    end = max(3, int(rel))
    fwd = x[:end + 1] - x[0]
    if np.nanmax(np.abs(fwd)) < 1e-9:
        return None
    if abs(np.nanmin(fwd)) > abs(np.nanmax(fwd)):
        fwd = -fwd
    ysub = y[:end + 1]
    ylo, yhi = np.nanmin(ysub), np.nanmax(ysub)
    ynorm = (y - ylo) / (yhi - ylo + 1e-9)
    return end, fwd, ynorm, y


def fp_persist(df, lead, fps, rel, hold_s=0.03, ynorm_thr=0.97, vy_frac=0.15):
    """The incumbent conjunction, but the first frame from which it HOLDS
    continuously for `hold_s` seconds -- a planted foot stays planted, whereas
    a momentary blip during the stride does not."""
    P = _prep(df, lead, rel)
    fb = max(0, int(rel) - int(0.13 * fps))
    if P is None:
        return fb
    end, fwd, ynorm, y = P
    xmax = np.nanmax(fwd)
    vy = np.abs(np.concatenate([[0.0], np.diff(y)])) * fps
    vypk = np.nanmax(vy[:end + 1]) + 1e-9
    ok = np.zeros(end + 1, bool)
    for f in range(3, end):
        ok[f] = (fwd[f] > 0.70 * xmax and ynorm[f] > ynorm_thr
                 and vy[f] < vy_frac * vypk)
    hold = max(2, int(round(hold_s * fps)))
    run = 0
    for f in range(3, end + 1):
        run = run + 1 if ok[f] else 0
        if run >= hold:
            return f - hold + 1
    return int(np.flatnonzero(ok)[0]) if ok.any() else fb


def fp_persist_relaxed(df, lead, fps, rel, hold_s=0.03):
    """Same persistence rule with a looser ground gate (ynorm > 0.90). The 0.97
    gate is measured against the ankle minimum over [0, rel]; if the ankle dips
    slightly lower late in the window, the true landing never reaches 0.97 and
    selection is pushed onto whatever noise qualifies first."""
    return fp_persist(df, lead, fps, rel, hold_s=hold_s, ynorm_thr=0.90)


DETECTORS = {
    "incumbent": lambda df, l, f, r: M.foot_plant_frame(df, l, f, M.JOINTS, r),
    "D persist30ms": fp_persist,
    "E persist+relax": fp_persist_relaxed,
    "F persist50ms": lambda df, l, f, r: fp_persist(df, l, f, r, hold_s=0.05),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    print("loading OBP event times ...")
    with zipfile.ZipFile(LM_ZIP) as z:
        with z.open("landmarks.csv") as f:
            lm = pd.read_csv(f, usecols=["session_pitch", "time",
                                         "fp_10_time", "fp_100_time"])
    ev, bad = {}, 0
    for sp, g in lm.groupby("session_pitch"):
        g = g.sort_values("time")
        t = g.time.to_numpy(float)
        f100 = float(g.fp_100_time.iloc[0]); f10 = float(g.fp_10_time.iloc[0])
        if not np.isfinite(f100) or f100 <= 0 or not np.isfinite(f10) or f10 <= 0:
            bad += 1
            continue                     # missing force-plate event -> no GT
        ev[sp] = int(np.argmin(np.abs(t - f100)))
    print(f"  {len(ev)} pitches with valid foot-plant GT "
          f"({bad} excluded: fp time missing/zero)\n")

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    rows = []
    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
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
            row = {"sp": sp, "user": int(r.user), "gt": ev[sp]}
            for nm, fn in DETECTORS.items():
                row[nm] = fn(df0, lead, fps, rel) - ev[sp]
            rows.append(row)
        except Exception:
            continue
        if len(rows) and len(rows) % 100 == 0:
            print(f"  ...{len(rows)} processed")

    d = pd.DataFrame(rows)
    print(f"\nn = {len(d)} pitches, frames @360Hz (1 frame = 2.8 ms)\n")
    print(f"  {'detector':<18}{'median':>8}{'mean|.|':>9}{'SD':>8}{'p90|.|':>8}"
          f"{'<=3f':>7}{'<=6f':>7}{'>30f':>7}")
    for nm in DETECTORS:
        v = d[nm].to_numpy(float); av = np.abs(v)
        print(f"  {nm:<18}{np.median(v):>+8.1f}{av.mean():>9.1f}{v.std():>8.1f}"
              f"{np.percentile(av, 90):>8.1f}{100*(av<=3).mean():>6.0f}%"
              f"{100*(av<=6).mean():>6.0f}%{100*(av>30).mean():>6.0f}%")

    dst = os.path.join(config.ROOT, "data", "outputs", "obp_validation",
                       "fp_detect_v2.csv")
    d.to_csv(dst, index=False)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
