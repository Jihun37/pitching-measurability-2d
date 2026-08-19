"""
Split one clip into its pitches, and measure how consistent they are.

Where a clip holds several pitches, say five:
  1. find the pitch peaks in the wrist-speed series (scipy find_peaks);
  2. cut the coordinates into [pre seconds before, post seconds after] each
     peak. The default 3 seconds before is not arbitrary: the trail anchor needs
     the still period before the windup;
  3. run metrics.compute_candidates on each segment, giving per-pitch values;
  4. report mean, SD and CV across pitches, and the knee-angle consistency
     readout.

This assumes the camera is on a tripod for the whole clip, so nothing about the
viewpoint changes between pitches. That makes it CLEANER consistency data than
separate clips would give, not worse.

Usage, after extraction and smoothing:
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(_HERE, "..", "stage3"))

import config
import metrics as M

GRADE = {
    "lead_knee_angle":     "A (SD r2=0.96)",
    "knee_ext_velo_br":    "B (SD r2=0.44)",
    "release_height":      "B (SD r2=0.38)",
    "stride_pct_height":   "reference (SD r2=0.37)",
    "wrist_speed":         "reference (SD r2=0.20)",
    "trunk_anterior_tilt": "reference (SD r2=0.16)",
}
SHOW = list(GRADE.keys())


def detect_arm(df):
    def pk(j):
        x = df[f"{j}_x"].to_numpy(float); y = df[f"{j}_y"].to_numpy(float)
        return np.nanmax(np.hypot(np.diff(x), np.diff(y)))
    return "right" if pk("right_wrist") >= pk("left_wrist") else "left"


def wrist_speed_series(df, arm, fps):
    x = df[f"{arm}_wrist_x"].to_numpy(float)
    y = df[f"{arm}_wrist_y"].to_numpy(float)
    s = np.hypot(np.diff(x), np.diff(y)) * fps
    s = np.concatenate([[0.0], s])
    return pd.Series(s).interpolate(limit_direction="both").to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", required=True, help="smoothed coordinate CSV for the whole clip")
    ap.add_argument("--fps", type=float, default=120.0)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--pitches", type=int, default=5, help="how many pitches to expect")
    ap.add_argument("--min-gap", type=float, default=3.0,
                    help="minimum seconds between pitches; adjust if too few or too many peaks are found")
    ap.add_argument("--pre", type=float, default=3.0,
                    help="seconds before the peak; must cover the still period the trail anchor needs")
    ap.add_argument("--post", type=float, default=1.0, help="seconds after the peak")
    ap.add_argument("--plot", action="store_true",
                    help="write a wrist-speed and peaks png, to check the detection")
    a = ap.parse_args()

    df = pd.read_csv(a.coords)
    if "nose_x" in df.columns and "head_x" not in df.columns:
        df = df.rename(columns={"nose_x": "head_x", "nose_y": "head_y",
                                "nose_v": "head_v"})
    arm = a.arm or detect_arm(df)
    spd = wrist_speed_series(df, arm, a.fps)

    # 1) find the pitch peaks
    dist = max(int(a.min_gap * a.fps), 1)
    height = 0.45 * np.nanmax(spd)            # only a peak above 45% of the clip maximum counts as a pitch
    peaks, props = find_peaks(spd, distance=dist, height=height)
    # More peaks than pitches expected: keep the n highest.
    if len(peaks) > a.pitches:
        order = np.argsort(props["peak_heights"])[::-1][:a.pitches]
        peaks = np.sort(peaks[order])
    print(f"arm={arm}  frames={len(df)}  pitch peaks found: {len(peaks)} "
          f"(expected {a.pitches})")
    for i, p in enumerate(peaks):
        print(f"  pitch {i+1}: frame {p}  (t={p/a.fps:.1f}s, "
              f"speed={spd[p]:.0f}px/s)")
    if len(peaks) < 2:
        print("\n[stopping] fewer than two peaks. Lower --min-gap, or check with --plot.")
        return
    if len(peaks) != a.pitches:
        print(f"\n[note] not the {a.pitches} expected. Check the detection with --plot.")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4))
        t = np.arange(len(spd)) / a.fps
        ax.plot(t, spd, lw=0.8)
        ax.plot(peaks / a.fps, spd[peaks], "rv", ms=10)
        ax.axhline(height, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("time (s)"); ax.set_ylabel("wrist speed (px/s)")
        ax.set_title(f"pitch peaks: {len(peaks)}")
        png = a.coords.replace(".csv", "_peaks.png")
        fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)
        print(f"peak plot -> {png}")

    # 2) split into segments, then 3) measure each
    pre = int(a.pre * a.fps); post = int(a.post * a.fps)
    rows = []
    for i, p in enumerate(peaks):
        lo, hi = max(0, p - pre), min(len(df), p + post)
        seg = df.iloc[lo:hi].reset_index(drop=True)
        try:
            cand = {k: v for k, (v, _) in
                    M.compute_candidates(seg, fps=a.fps, arm=arm,
                                         view="side").items()}
        except Exception as e:
            print(f"  pitch {i+1}: failed ({e})")
            continue
        row = {"pitch": i + 1, "frame_lo": lo, "frame_hi": hi,
               "peak_frame": int(p)}
        row.update({m: cand.get(m, np.nan) for m in SHOW})
        rows.append(row)

    res = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("[per-pitch measurements]")
    print("=" * 78)
    print(res.to_string(index=False))

    # 4) consistency
    print("\n" + "=" * 78)
    print("[consistency across pitches]  mean / SD / CV%")
    print()
    print(f"{'quantity':22s}{'mean':>9s}{'SD':>9s}{'CV%':>8s}   grade")
    print("-" * 78)
    for m in SHOW:
        v = res[m].to_numpy(float); v = v[np.isfinite(v)]
        if len(v) < 2:
            continue
        mean, sd = v.mean(), v.std(ddof=1)
        cv = abs(sd / mean * 100) if mean != 0 else np.nan
        print(f"{m:22s}{mean:>9.2f}{sd:>9.3f}{cv:>8.1f}   {GRADE[m]}")

    ka = res["lead_knee_angle"].to_numpy(float); ka = ka[np.isfinite(ka)]
    if len(ka) >= 2:
        sd = ka.std(ddof=1)
        print("\n" + "=" * 78)
        print("[release consistency -- lead knee angle, the validated consistency quantity]")
        sd = float(np.nanstd(ka))
        print(f"  knee angle SD across pitches = {sd:.2f} deg  (n={len(ka)})")
        print(f"  for reference, the OBP elite mean true SD is 4.3 deg.")
        print(f"  a smaller SD means the lead-leg block repeats, pitch to pitch.")

    out = a.coords.replace("_smoothed.csv", "_pitches.csv") \
                  .replace("_coords.csv", "_pitches.csv")
    res.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()