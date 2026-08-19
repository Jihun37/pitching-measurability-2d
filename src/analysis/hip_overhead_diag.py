"""
Diamond - Diagnose WHY overhead hip keypoints are noisy (feasibility of a fix)
hip_overhead_diag.py

The pelvis-gate real-video check failed: peaks 4-10x too high, hip_conf ~0.65.
Before asking "can we improve overhead hip detection", find out HOW the hips are
bad. Three candidate failure modes, each with a different fix:
  A. jitter around the right place   -> temporal smoothing / tracking fixes it
  B. left/right hip SWAP (180 flips) -> association/tracking fixes it (cheap)
  C. genuinely occluded / wrong      -> needs a better detector (hard)

Per clip (stock RTMPose = same hips as refined): hip-line angle, length, min-conf,
frame-to-frame jitter vs the (better) shoulder line, and a 180-flip count.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

CLIPS = ["pitching_overhead_01", "pitching_overhead_02", "pitching_overhead_03"]


def line_stats(df, a, b):
    ax, ay = df[f"{a}_x"].to_numpy(float), df[f"{a}_y"].to_numpy(float)
    bx, by = df[f"{b}_x"].to_numpy(float), df[f"{b}_y"].to_numpy(float)
    ang = np.degrees(np.arctan2(by - ay, bx - ax))
    length = np.hypot(bx - ax, by - ay)
    conf = np.minimum(df[f"{a}_v"].to_numpy(float), df[f"{b}_v"].to_numpy(float))
    return ang, length, conf, (ax, ay, bx, by)


def jitter(coords):
    ax, ay, bx, by = coords
    ja = np.hypot(np.diff(ax), np.diff(ay))
    jb = np.hypot(np.diff(bx), np.diff(by))
    return np.concatenate([[0], (ja + jb) / 2])


def analyze(clip):
    d = pd.read_csv(os.path.join(config.ROOT, "data", "outputs", clip,
                                 f"{clip}_coords_rtmp.csv"))
    ha, hl, hc, hco = line_stats(d, "left_hip", "right_hip")
    sa, sl, sc, sco = line_stats(d, "left_shoulder", "right_shoulder")
    hj, sj = jitter(hco), jitter(sco)

    # left/right hip swap: frame-to-frame angle jump near +-180
    dang = np.abs(np.diff(np.unwrap(np.radians(ha))))
    dang = np.degrees(dang)
    flips = int(np.sum(dang > 120))
    hl_med = np.nanmedian(hl)
    collapse = float(np.mean(hl < 0.45 * hl_med))

    print(f"\n=== {clip} ===  frames={len(d)}")
    print(f"  hip  line: len med={hl_med:.0f}px  conf med={np.nanmedian(hc):.2f}  "
          f"jitter med={np.nanmedian(hj):.1f}px p95={np.nanpercentile(hj,95):.1f}px")
    print(f"  shldr line: len med={np.nanmedian(sl):.0f}px  conf med={np.nanmedian(sc):.2f}  "
          f"jitter med={np.nanmedian(sj):.1f}px p95={np.nanpercentile(sj,95):.1f}px")
    print(f"  hip vs shoulder jitter ratio (med) = {np.nanmedian(hj)/max(np.nanmedian(sj),1e-6):.2f}x")
    print(f"  hip-line collapse (<45% median len): {collapse*100:.1f}% of frames")
    print(f"  large angle jumps >120deg/frame (L/R swap or flip): {flips}")
    return dict(clip=clip, ha=ha, hl=hl, hc=hc, hj=hj, sj=sj, hl_med=hl_med)


def main():
    rows = [analyze(c) for c in CLIPS]
    fig, axes = plt.subplots(len(CLIPS), 1, figsize=(10, 3.2 * len(CLIPS)))
    for ax, r in zip(np.atleast_1d(axes), rows):
        f = np.arange(len(r["ha"]))
        ax.plot(f, r["ha"], color="#1f77b4", lw=1, label="hip-line angle (deg)")
        ax2 = ax.twinx()
        ax2.plot(f, r["hl"], color="#ff7f0e", lw=1, alpha=0.7, label="hip-line len (px)")
        ax2.axhline(0.45 * r["hl_med"], color="crimson", ls="--", lw=0.8)
        ax.set_title(f"{r['clip']}  hip angle (blue) + hip-line length (orange), "
                     f"red=45% collapse threshold")
        ax.set_ylabel("angle (deg)")
        ax2.set_ylabel("length (px)")
    axes[-1].set_xlabel("frame") if len(CLIPS) > 1 else axes.set_xlabel("frame")
    fig.tight_layout()
    out = os.path.join(config.OBP_VALIDATION_DIR, "hip_overhead_diag.png")
    fig.savefig(out, dpi=110)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
