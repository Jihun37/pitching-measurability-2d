"""Foot-plant detector diagnosis + candidate comparison (paper-prep).

cog_curve_probe showed foot plant is the weak link in the event set: vs OBP's
fp_100_time our detector runs median -1.0 frame but mean|.| 16.1 @360Hz, i.e.
the central tendency is fine and a heavy tail is doing the damage. That tail
kills 5 COM-curve candidates (they score ~0.15 with our events, 0.94-0.998 with
OBP's), so foot plant is the single highest-leverage event to fix.

The incumbent (metrics.foot_plant_frame) is a 3-way hard-threshold conjunction --
forward progress > 0.70*max AND ynorm > 0.97 AND |vy| < 0.15*peak -- returning
the FIRST qualifying frame, and silently falling back to release - 130 ms when
no frame qualifies. Hypothesis: the fallback (and the strict ynorm>0.97 gate)
produce the tail.

Candidates tested here are physical rather than threshold-tuned: after foot
plant the lead ankle comes to REST and stays there, so the landing is the ONSET
OF THE TERMINAL QUIET PERIOD (A), or equivalently the moment lead-ankle forward
travel SATURATES at its plateau (B). C combines them.

Scored against OBP fp_100_time (established in scratch/fp_event_match.py as the
event our kinematic detector corresponds to: median -1 frame vs +6 for fp_10).
Touches no adopted definition -- candidates live here until a result justifies
wiring one in.
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


def _ankle(df, lead):
    ak = "l_an" if lead == "left" else "r_an"
    x, y = M._xy(df, ak, M.JOINTS)
    return np.asarray(x, float), np.asarray(y, float)


def _fwd(x, end):
    """Lead-ankle forward travel, oriented positive toward the plate."""
    f = x[:end + 1] - x[0]
    if np.nanmax(np.abs(f)) < 1e-9:
        return None
    return -f if abs(np.nanmin(f)) > abs(np.nanmax(f)) else f


def fp_incumbent_instrumented(df, lead, fps, rel):
    """metrics.foot_plant_frame, re-implemented to report WHICH branch fired."""
    x, y = _ankle(df, lead)
    end = max(3, int(rel))
    fallback = max(0, int(rel) - int(0.13 * fps))
    fwd = _fwd(x, end)
    if fwd is None:
        return fallback, "degenerate"
    xmax = np.nanmax(fwd)
    ysub = y[:end + 1]
    ylo, yhi = np.nanmin(ysub), np.nanmax(ysub)
    ynorm = (y - ylo) / (yhi - ylo + 1e-9)
    vy = np.abs(np.concatenate([[0.0], np.diff(y)])) * fps
    vypk = np.nanmax(vy[:end + 1]) + 1e-9
    cand = [f for f in range(3, end)
            if fwd[f] > 0.70 * xmax and ynorm[f] > 0.97 and vy[f] < 0.15 * vypk]
    return (cand[0], "threshold") if cand else (fallback, "FALLBACK")


def _smooth_speed(x, y, fps, win_s=0.03):
    """Lead-ankle speed, boxcar-smoothed over ~30 ms (noise, not shape)."""
    s = np.hypot(np.gradient(x), np.gradient(y)) * fps
    w = max(3, int(round(win_s * fps)))
    k = np.ones(w) / w
    return np.convolve(s, k, mode="same")


def fp_quiet_onset(df, lead, fps, rel, q=0.12, min_s=0.04):
    """A: onset of the TERMINAL quiet period of the lead ankle.
    After landing the ankle is planted and stops moving, so scan back from
    release for the longest sustained low-speed run and take its first frame."""
    x, y = _ankle(df, lead)
    end = max(5, int(rel))
    s = _smooth_speed(x, y, fps)[:end + 1]
    if not np.isfinite(s).any():
        return max(0, int(rel) - int(0.13 * fps))
    thr = q * np.nanmax(s)
    quiet = s < thr
    min_len = max(2, int(round(min_s * fps)))
    # collect runs, keep those in the back half (landing cannot precede the lift)
    runs, i = [], 0
    while i <= end:
        if quiet[i]:
            j = i
            while j + 1 <= end and quiet[j + 1]:
                j += 1
            if (j - i + 1) >= min_len:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    runs = [r for r in runs if r[0] > 0.35 * end]
    if not runs:
        return max(0, int(rel) - int(0.13 * fps))
    return runs[-1][0]          # onset of the LAST sustained quiet run


def fp_fwd_saturate(df, lead, fps, rel, frac=0.95):
    """B: the frame lead-ankle forward travel SATURATES at its stride plateau.
    Plateau = median over the last 50 ms before release (the foot is planted);
    landing = first frame reaching `frac` of it that never falls back below."""
    x, y = _ankle(df, lead)
    end = max(5, int(rel))
    fwd = _fwd(x, end)
    if fwd is None:
        return max(0, int(rel) - int(0.13 * fps))
    k = max(2, int(round(0.05 * fps)))
    plateau = float(np.nanmedian(fwd[max(0, end - k):end + 1]))
    if not np.isfinite(plateau) or plateau <= 0:
        return max(0, int(rel) - int(0.13 * fps))
    ok = fwd >= frac * plateau
    # first index from which it stays true through the end
    stay = np.flatnonzero(np.cumsum(~ok[::-1])[::-1] == 0)
    return int(stay[0]) if len(stay) else max(0, int(rel) - int(0.13 * fps))


def fp_combined(df, lead, fps, rel):
    """C: saturation gives the landing instant; the terminal quiet run confirms
    the foot really stopped. Take the later of the two when they are close
    (saturation can trigger on a late drift), else trust saturation."""
    a = fp_quiet_onset(df, lead, fps, rel)
    b = fp_fwd_saturate(df, lead, fps, rel)
    if abs(a - b) <= int(round(0.06 * fps)):
        return max(a, b)
    return b


DETECTORS = {
    "incumbent": lambda df, l, f, r: fp_incumbent_instrumented(df, l, f, r)[0],
    "A quiet-onset": fp_quiet_onset,
    "B fwd-saturate": fp_fwd_saturate,
    "C combined": fp_combined,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    print("loading OBP event times ...")
    with zipfile.ZipFile(LM_ZIP) as z:
        with z.open("landmarks.csv") as f:
            lm = pd.read_csv(f, usecols=["session_pitch", "time", "fp_100_time"])
    ev = {}
    for sp, g in lm.groupby("session_pitch"):
        g = g.sort_values("time")
        t = g.time.to_numpy(float)
        ev[sp] = int(np.argmin(np.abs(t - float(g.fp_100_time.iloc[0]))))
    print(f"  {len(ev)} pitches\n")

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    rows = []
    done = 0
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
            _, branch = fp_incumbent_instrumented(df0, lead, fps, rel)
            row = {"sp": sp, "branch": branch, "gt": ev[sp], "rel": rel}
            for nm, fn in DETECTORS.items():
                row[nm] = fn(df0, lead, fps, rel) - ev[sp]
            rows.append(row)
            done += 1
        except Exception:
            continue
        if done and done % 100 == 0:
            print(f"  ...{done} processed")

    d = pd.DataFrame(rows)
    print(f"\nn = {len(d)} pitches, frames @360Hz (1 frame = 2.8 ms)\n")

    print("[1] incumbent branch usage")
    for b, c in d.branch.value_counts().items():
        sub = d[d.branch == b]["incumbent"]
        print(f"  {b:<12} n={c:>4} ({100*c/len(d):4.1f}%)   "
              f"median {np.median(sub):+6.1f}  mean|.| {np.mean(np.abs(sub)):6.1f}")

    print("\n[2] detector comparison (error vs OBP fp_100)")
    print(f"  {'detector':<16}{'median':>8}{'mean|.|':>9}{'p90|.|':>8}"
          f"{'<=3f':>7}{'<=6f':>7}{'>30f':>7}")
    for nm in DETECTORS:
        v = d[nm].to_numpy(float)
        av = np.abs(v)
        print(f"  {nm:<16}{np.median(v):>+8.1f}{av.mean():>9.1f}"
              f"{np.percentile(av,90):>8.1f}"
              f"{100*(av<=3).mean():>6.0f}%{100*(av<=6).mean():>6.0f}%"
              f"{100*(av>30).mean():>6.0f}%")

    dst = os.path.join(config.ROOT, "data", "outputs", "obp_validation",
                       "fp_detect_probe.csv")
    d.to_csv(dst, index=False)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
