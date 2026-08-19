"""
Diamond - wrist_speed peak: jitter or real? (backbone disagreement probe)

Motivation: on pitching_lateral_02 the peak wrist speed reads 13.84/stature
with MediaPipe vs 11.79 with RTMPose (z -0.08 vs -2.00 against the OBP
clean-projection distribution). No 3D truth exists for the clip, so neither
number can be assumed correct. Two discriminating tests that need no truth:

1. PEAK SHAPE: overlay both backbones' wrist-speed series around the peak.
   Jitter peaks are 1-2 frame spikes (huge prominence over neighbors, tiny
   FWHM); real whip peaks span the physical acceleration time.
2. SMOOTHING SENSITIVITY: sweep the SG window in TIME (ms, fps-invariant)
   and watch the normalized peak decay. CONTROL = clean OBP mocap
   projections (zero detector noise): their decay curve is the shape of
   real motion under smoothing. The backbone that decays much faster than
   the mocap control is selling jitter.

Run:  cd src\tests
      python wrist_peak_diag.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(HERE, "..", "stage3"))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import config
import metrics as M
import obp_project as O
from smoother import smooth_coordinates
from hss_elevation_test import project_cam

CLIP = "pitching_lateral_02"
FPS = 120.0
ARM = "right"
WIN_MS = [0, 25, 50, 75, 100, 150]      # SG window in time; 0 = raw
N_OBP_CTRL = 5
OUT_DIR = os.path.join(config.ROOT, "data", "outputs", CLIP)


def sg_frames(ms, fps):
    if ms <= 0:
        return 0
    w = int(round(ms / 1000.0 * fps))
    return max(5, w // 2 * 2 + 1)       # smoother needs odd >= polyorder+2


def wrist_speed_series(df, fps, arm=ARM):
    wx, wy = M._xy(df, f"{'r' if arm == 'right' else 'l'}_wr", M.JOINTS)
    return M._speed(wx, wy, fps)


def peak_stats(spd, fps):
    pk = int(np.nanargmax(spd))
    v = float(spd[pk])
    nb = np.concatenate([spd[max(0, pk - 3):pk], spd[pk + 1:pk + 4]])
    prom = v / (np.nanmean(nb) + 1e-9)             # spike indicator
    half = v / 2.0
    a = pk
    while a > 0 and spd[a - 1] >= half:
        a -= 1
    b = pk
    while b < len(spd) - 1 and spd[b + 1] >= half:
        b += 1
    fwhm_ms = (b - a + 1) / fps * 1000.0
    return pk, v, prom, fwhm_ms


def sweep(raw, fps, stature_ref=None):
    """Peak wrist speed (/stature) for each SG window; also the w-min series."""
    vals, series = {}, {}
    for ms in WIN_MS:
        w = sg_frames(ms, fps)
        df = raw if w == 0 else smooth_coordinates(raw, window=w)
        spd = wrist_speed_series(df, fps)
        stat = stature_ref or M.pixel_stature(df, M.JOINTS)
        vals[ms] = float(np.nanmax(spd)) / stat
        series[ms] = spd / stat
    return vals, series


def main():
    # ── real clip, both backbones ────────────────────────
    res = {}
    for sfx, tag in (("", "mediapipe"), ("_rtmp", "rtmp")):
        raw = pd.read_csv(os.path.join(OUT_DIR, f"{CLIP}_coords{sfx}.csv"))
        if "nose_x" in raw.columns and "head_x" not in raw.columns:
            raw = raw.rename(columns={"nose_x": "head_x", "nose_y": "head_y",
                                      "nose_v": "head_v"})
        vals, series = sweep(raw, FPS)
        res[tag] = (vals, series)

    # ── OBP clean-projection controls (side view) ────────
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    ctrl = []
    for r in md.itertuples(index=False):
        if len(ctrl) >= N_OBP_CTRL:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            continue
        try:
            joints, fps3 = O.load_c3d_joints(path)
            arm3 = O.detect_throwing_arm(joints, fps3)
            df = project_cam(joints, 0, 0)
            # controls carry no confidence columns; add dummies so the
            # smoother's visibility gate is a no-op
            for j in set(c[:-2] for c in df.columns if c.endswith("_x")):
                df[f"{j}_v"] = 1.0
            vals = {}
            for ms in WIN_MS:
                w = sg_frames(ms, fps3)
                d = df if w == 0 else smooth_coordinates(df, window=w)
                wx, wy = M._xy(d, f"{'r' if arm3 == 'right' else 'l'}_wr", M.JOINTS)
                spd = M._speed(wx, wy, fps3)
                vals[ms] = float(np.nanmax(spd))
            ctrl.append(vals)
        except Exception:
            continue

    # ── report: peak shape at the pipeline-standard window (~58ms=7f@120) ──
    print(f"=== {CLIP}  peak shape (pipeline window w=7 frames ~58 ms) ===")
    print(f"{'backbone':<10s} {'raw peak':>9s} {'raw prom':>9s} {'raw FWHM':>9s}"
          f" {'w7 peak':>8s} {'w7 prom':>8s} {'w7 FWHM':>9s}")
    for tag in ("mediapipe", "rtmp"):
        raw = pd.read_csv(os.path.join(OUT_DIR, f"{CLIP}_coords{'' if tag == 'mediapipe' else '_rtmp'}.csv"))
        spd_raw = wrist_speed_series(raw, FPS)
        _, v0, p0, f0 = peak_stats(spd_raw, FPS)
        sm = smooth_coordinates(raw, window=7)
        spd7 = wrist_speed_series(sm, FPS)
        _, v7, p7, f7 = peak_stats(spd7, FPS)
        st = M.pixel_stature(sm, M.JOINTS)
        print(f"{tag:<10s} {v0/st:>9.2f} {p0:>9.2f} {f0:>7.0f}ms"
              f" {v7/st:>8.2f} {p7:>8.2f} {f7:>7.0f}ms")

    # ── report: smoothing decay, normalized to the 25 ms point ──
    print(f"\n=== smoothing sensitivity  (peak value / value@25ms) ===")
    hdr = f"{'window':<10s}" + "".join(f"{ms:>8d}ms" for ms in WIN_MS)
    print(hdr)
    for tag in ("mediapipe", "rtmp"):
        vals = res[tag][0]
        base = vals[25]
        print(f"{tag:<10s}" + "".join(f"{vals[ms]/base:>10.3f}" for ms in WIN_MS))
    C = np.array([[c[ms] for ms in WIN_MS] for c in ctrl], float)
    Cn = C / C[:, 1:2]
    print(f"{'obp-clean':<10s}" + "".join(f"{v:>10.3f}" for v in Cn.mean(0))
          + f"   (n={len(ctrl)}, sd@150ms {Cn[:, -1].std():.3f})")

    # ── figures ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"mediapipe": "green", "rtmp": "magenta"}
    for tag in ("mediapipe", "rtmp"):
        spd = res[tag][1][50]                       # ~pipeline smoothing
        pk = int(np.nanargmax(spd))
        x = (np.arange(len(spd)) - pk) / FPS * 1000
        axes[0].plot(x, spd, color=colors[tag], label=f"{tag} (50 ms SG)")
    axes[0].set_xlim(-400, 400)
    axes[0].set_xlabel("ms from own peak"); axes[0].set_ylabel("wrist speed /stature")
    axes[0].set_title(f"{CLIP}: peak shape"); axes[0].legend()

    for tag in ("mediapipe", "rtmp"):
        vals = res[tag][0]
        axes[1].plot(WIN_MS, [vals[ms] / vals[25] for ms in WIN_MS],
                     "o-", color=colors[tag], label=tag)
    axes[1].plot(WIN_MS, Cn.mean(0), "s--", color="gray",
                 label=f"OBP clean (n={len(ctrl)})")
    axes[1].fill_between(WIN_MS, Cn.min(0), Cn.max(0), color="gray", alpha=0.15)
    axes[1].set_xlabel("SG window (ms)"); axes[1].set_ylabel("peak / peak@25ms")
    axes[1].set_title("smoothing decay vs clean-mocap control"); axes[1].legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"{CLIP}_wristpeak_diag.png")
    fig.savefig(out, dpi=120)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
