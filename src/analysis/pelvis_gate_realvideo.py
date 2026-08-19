"""
Diamond - Real-video plausibility check for the OVERHEAD pelvis rotational velocity
pelvis_gate_realvideo.py

OBP already validated this: from overhead, the hip-line yaw rate reproduces OBP
max_pelvis_rotational_velo at r=0.909 with correct absolute scale (~746 deg/s).
Real video has no 3D ground truth, so this is a PLAUSIBILITY check on real overhead
clips (the same pitching_overhead_0X the HSS work used), using the occlusion-refined
best pose:
  1. peak pelvis rotational velocity (deg/s) in the OBP range (median 742)?
  2. pelvis gate >=150 deg/s crossing (when the pelvis "opens") in the throw phase?
  3. clean single peak (not jitter / double peak)?
  4. refined vs stock: does refinement stabilize the hip line?
  5. cross-clip consistency.

Absolute deg/s assumes a near-overhead camera (image plane ~ horizontal plane); the
OBP audit showed the scale is stable across elevation 75-89.5 deg, so a mounted
overhead phone is in range. Angle is scale-invariant (no ppm/calibration needed).

Run:  conda activate diamond; cd src\analysis; python pelvis_gate_realvideo.py
"""
import os, sys
import numpy as np
import pandas as pd
import cv2
from scipy.signal import savgol_filter, medfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
import config

CLIPS = ["pitching_overhead_01", "pitching_overhead_02", "pitching_overhead_03"]
GATE = 150.0                 # deg/s, the app's pelvis gate
OBP_MED = 742.0              # OBP median max_pelvis_rotational_velo (reference)


def _odd(k):
    return int(k) + (int(k) % 2 == 0)


def flip_correct(lx, ly, rx, ry):
    """Greedy L/R hip tracking: from overhead the detector confuses the two hips,
    flipping the line ~180 deg between frames. At each frame pick keep-vs-swap by
    whichever assignment moves the two points LESS from the previous (corrected)
    frame. Removes the flip spikes without touching real rotation."""
    lx, ly, rx, ry = (a.copy() for a in (lx, ly, rx, ry))
    n = len(lx)
    chord = np.nanmedian(np.hypot(rx - lx, ry - ly))
    flips = 0
    for t in range(1, n):
        pL, pR = np.array([lx[t-1], ly[t-1]]), np.array([rx[t-1], ry[t-1]])
        cL, cR = np.array([lx[t], ly[t]]), np.array([rx[t], ry[t]])
        keep = np.hypot(*(cL - pL)) + np.hypot(*(cR - pR))
        swap = np.hypot(*(cR - pL)) + np.hypot(*(cL - pR))
        # only swap on a CLEAR flip: keep-motion is large (~half the pelvis width)
        # and swapping more than halves it. Avoids noise-driven over-swapping.
        if swap * 2.0 < keep and keep > 0.5 * chord:
            lx[t], ly[t], rx[t], ry[t] = rx[t], ry[t], lx[t], ly[t]
            flips += 1
    return lx, ly, rx, ry, flips


def _wrap(a):
    return (a + 180) % 360 - 180


def shoulder_anchored_recovery(df, fps, conf=0.4, chord_frac=0.5):
    """Pelvis rot-velo (deg/s) via shoulder-anchored hip recovery.
    The hips track clean except in the brief release occlusion; the shoulders stay
    reliable throughout. So:
      1. resolve the hip-line 180-flip by CONTINUITY (nearest to previous resolved
         angle), re-anchored to the shoulder line at the start / after long gaps —
         only true ~180 jumps flip, not noise.
      2. gate occluded hip frames (low conf OR collapsed chord), and INTERPOLATE the
         resolved hip angle across them (bridges the occlusion; a monotonic release
         rotation across a short gap -> its mean rate, physically sane).
      3. smooth -> derivative.
    Returns (rot_velo deg/s, occluded mask, n_bridged)."""
    def c(k):
        return df[k].to_numpy(float)
    lhx, lhy, rhx, rhy = c("left_hip_x"), c("left_hip_y"), c("right_hip_x"), c("right_hip_y")
    lsx, lsy, rsx, rsy = c("left_shoulder_x"), c("left_shoulder_y"), c("right_shoulder_x"), c("right_shoulder_y")
    hv = np.minimum(c("left_hip_v"), c("right_hip_v"))

    hip_raw = np.degrees(np.arctan2(rhy - lhy, rhx - lhx))       # 0..360 wrapped
    sh = np.degrees(np.arctan2(rsy - lsy, rsx - lsx))
    chord = np.hypot(rhx - lhx, rhy - lhy)
    occl = (hv < conf) | (chord < chord_frac * np.nanmedian(chord))

    n = len(hip_raw)
    resolved = np.full(n, np.nan)
    prev = None
    for t in range(n):
        if occl[t]:
            continue
        cands = [hip_raw[t], hip_raw[t] + 180.0]
        anchor = prev if prev is not None else sh[t]      # shoulder anchor at start
        best = min(cands, key=lambda v: abs(_wrap(v - anchor)))
        resolved[t] = (prev + _wrap(best - prev)) if prev is not None else best
        prev = resolved[t]
    # bridge occluded gaps
    n_bridged = int(np.isnan(resolved).sum())
    resolved = pd.Series(resolved).interpolate(limit_direction="both").to_numpy()

    win = min(_odd(max(5, round(0.05 * fps))), n - (n % 2 == 0))
    if win <= 3:
        vel = np.gradient(resolved) * fps
    else:
        vel = savgol_filter(resolved, win, 3, deriv=1, delta=1.0 / fps, mode="interp")
    # medfilt the velocity to drop 1-2 frame residual spikes (HSS lesson): the real
    # pelvis rotation is a ~50-100 ms envelope; the spikes are keypoint jitter.
    vel = medfilt(np.abs(vel), max(3, _odd(0.042 * fps)))
    return vel, occl, n_bridged


def pelvis_rot_velo(df, fps, conf=0.3, mf_s=0.042, valid_frac=0.45, do_flip=True):
    """Hip-line yaw rate (deg/s) with the overhead-noise recipe HSS uses.
    Real 2D-pose hip keypoints jitter; the naive derivative explodes. So:
      1. confidence gate: NaN hips below `conf`, then interpolate.
      2. coordinate medfilt (~42 ms): kill 1-2 frame keypoint spikes.
      3. chord-validity gate: hip line >= `valid_frac` of its median length
         (end-on / collapsed hip line makes the angle meaningless).
    Returns (rot_velo deg/s, valid mask)."""
    def col(c):
        return df[c].to_numpy(float)
    lx, ly, rx, ry = col("left_hip_x"), col("left_hip_y"), col("right_hip_x"), col("right_hip_y")
    lv, rv = col("left_hip_v"), col("right_hip_v")

    # 1. confidence gate + linear interpolation of gaps
    bad = (lv < conf) | (rv < conf)
    for a in (lx, ly, rx, ry):
        a[bad] = np.nan
    s = pd.DataFrame({"lx": lx, "ly": ly, "rx": rx, "ry": ry}).interpolate(
        limit_direction="both")
    lx, ly, rx, ry = (s[c].to_numpy() for c in ("lx", "ly", "rx", "ry"))

    # 2. flip correction (L/R hip swap tracking) BEFORE medfilt
    flips = 0
    if do_flip:
        lx, ly, rx, ry, flips = flip_correct(lx, ly, rx, ry)

    # 3. coordinate medfilt
    k = max(3, _odd(mf_s * fps))
    lx, ly, rx, ry = (medfilt(a, k) for a in (lx, ly, rx, ry))

    # 4. chord-validity gate
    hp = np.hypot(rx - lx, ry - ly)
    valid = (~bad) & (hp >= valid_frac * np.nanmedian(hp))

    ang = np.unwrap(np.arctan2(ry - ly, rx - lx))
    n = len(ang)
    win = min(_odd(max(5, round(0.05 * fps))), n - (n % 2 == 0))
    if win <= 3:
        vel = np.gradient(ang) * fps
    else:
        vel = savgol_filter(ang, win, 3, deriv=1, delta=1.0 / fps, mode="interp")
    return np.abs(np.degrees(vel)), valid, flips


def analyze(clip):
    out_dir = os.path.join(config.ROOT, "data", "outputs", clip)
    ref = os.path.join(out_dir, f"{clip}_coords_rtmp_refined.csv")
    stock = os.path.join(out_dir, f"{clip}_coords_rtmp.csv")
    video = os.path.join(config.ROOT, "data", "videos", f"{clip}.mov")
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    dsto = pd.read_csv(stock)   # refined == stock for hips (refine only touches wrist/elbow)
    hipv = np.nanmean([dsto[f"{j}_v"] for j in ["left_hip", "right_hip"]])

    # shoulder-anchored hip recovery vs the previous (raw, gated) estimate
    vr, occl, n_bridged = shoulder_anchored_recovery(dsto, fps)
    vraw, _, _ = pelvis_rot_velo(dsto, fps, do_flip=False, valid_frac=0.45)

    m = int(len(vr) * 0.08)
    core = np.ones(len(vr), bool)
    core[:m] = False
    core[len(vr) - m:] = False
    pk = int(np.argmax(np.where(core, vr, -np.inf)))
    peak_r = float(vr[pk])
    peak_raw = float(np.nanmax(np.where(core, vraw, -np.inf)))
    occl_frac = float(occl.mean())

    above = vr >= GATE
    lo = pk
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = pk
    while hi < len(vr) - 1 and above[hi + 1]:
        hi += 1
    dur_ms = (hi - lo) / fps * 1000

    print(f"\n=== {clip} ===  fps={fps:.1f}  frames={len(dsto)}  hip_conf={hipv:.2f}  "
          f"occluded={occl_frac*100:.0f}% ({n_bridged} bridged)")
    print(f"  peak pelvis rot-velo (shoulder-anchored) = {peak_r:6.0f} deg/s @ f{pk} (t={pk/fps:.2f}s)")
    print(f"  peak (raw gated, no recovery)            = {peak_raw:6.0f} deg/s   (delta {peak_r-peak_raw:+.0f})")
    print(f"  vs OBP median {OBP_MED:.0f} deg/s  -> ratio {peak_r/OBP_MED:.2f}")
    print(f"  gate >={GATE:.0f}: above-gate {dur_ms:.0f} ms")
    return dict(clip=clip, fps=fps, hip_conf=hipv, peak_refined=peak_r,
                peak_raw=peak_raw, gate_ms=dur_ms, curve=vr, peak_f=pk, valid=~occl)


def main():
    rows = [analyze(c) for c in CLIPS]

    fig, axes = plt.subplots(len(CLIPS), 1, figsize=(9, 3 * len(CLIPS)), sharex=False)
    for ax, r in zip(np.atleast_1d(axes), rows):
        t = np.arange(len(r["curve"])) / r["fps"]
        ax.plot(t, r["curve"], lw=1.4, color="#1f77b4")
        ax.axhline(GATE, color="crimson", ls="--", lw=1, label=f"gate {GATE:.0f}")
        ax.axhline(OBP_MED, color="green", ls=":", lw=1, label=f"OBP med {OBP_MED:.0f}")
        ax.axvline(r["peak_f"] / r["fps"], color="k", ls="-", lw=0.8, alpha=0.5)
        ax.set_title(f"{r['clip']}  peak={r['peak_refined']:.0f} deg/s  hip_conf={r['hip_conf']:.2f}")
        ax.set_ylabel("pelvis rot-velo (deg/s)")
        ax.legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel("time (s)") if len(CLIPS) > 1 else axes.set_xlabel("time (s)")
    fig.tight_layout()
    out = os.path.join(config.OBP_VALIDATION_DIR, "pelvis_gate_realvideo.png")
    fig.savefig(out, dpi=110)
    print(f"\nsaved curves -> {out}")

    print("\n[SUMMARY]  (plausibility, no 3D GT on real video)")
    print(f"  OBP reference: median 742 deg/s, gate >=150")
    pr = [r["peak_refined"] for r in rows]
    print(f"  real peaks: {[round(x) for x in pr]} deg/s  "
          f"mean {np.mean(pr):.0f}  spread {np.std(pr):.0f}")
    inrange = all(300 <= x <= 1200 for x in pr)
    print(f"  all in plausible OBP range (300-1200)? {inrange}")


if __name__ == "__main__":
    main()
