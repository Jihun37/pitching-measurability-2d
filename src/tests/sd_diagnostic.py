"""
Diamond - SD recovery diagnostic
sd_diagnostic.py

Question: why does stride (value r2=0.82) recover within-pitcher SD poorly
(SD r2=0.37) while knee angle recovers it almost perfectly (0.96)?

Two competing explanations:
  (H-signal) Pitchers don't DIFFER much in their true stride variability
             (elite lab pitches are all highly repeatable) -> nothing to
             recover. Not a measurement failure.
  (H-noise)  Our stride SD is inflated by per-pitch measurement noise
             (e.g. anchor detection jitter) -> we add fake variability.

This decomposes SD recovery for each metric into observable pieces:

  truth_SD_spread : std ACROSS pitchers of each pitcher's TRUE within-SD.
                    small  -> little signal to recover (favors H-signal).
  our_SD_mean     : mean of our within-SD;  truth_SD_mean for comparison.
                    our >> truth -> we inflate variance (favors H-noise).
  err_SD          : within-pitcher SD of (our_estimate - truth) per pitch,
                    averaged over pitchers = our per-pitch measurement noise.
  SNR             : truth_SD_spread / err_SD  (higher -> easier recovery).

Also reports the per-pitch value r2 on the SAME repeated-pitch subset, so the
value-level vs SD-level gap is on identical data.

Usage:
    python sd_diagnostic.py --min-pitches 4
"""
import os, sys, argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(_HERE, "..", "stage3"))

import config

# reuse the per-pitch table the consistency test already saved
PERPITCH = os.path.join(config.OBP_VALIDATION_DIR, "repeatability_perpitch.csv")

METRICS = ["lead_knee_angle", "stride_length", "trunk_anterior_tilt",
           "knee_ext_velo_br", "wrist_speed", "release_height"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pitches", type=int, default=4)
    ap.add_argument("--perpitch", default=PERPITCH)
    a = ap.parse_args()

    if not os.path.exists(a.perpitch):
        print(f"[need] {a.perpitch}\n  run consistency_test.py first "
              f"(it saves repeatability_perpitch.csv)")
        return
    df = pd.read_csv(a.perpitch)
    cnt = df.groupby("user").size()
    keep = cnt[cnt >= a.min_pitches].index
    sub = df[df["user"].isin(keep)]
    print(f"pitchers with >= {a.min_pitches} pitches: {len(keep)}  "
          f"(pitches {len(sub)})\n")

    print("=" * 100)
    print("[SD recovery decomposition]  why some metrics recover within-pitcher SD and others don't")
    print("=" * 100)
    print(f"{'metric':22s}{'val_r2':>8s}{'SD_r2':>7s}"
          f"{'truSD_mean':>11s}{'ourSD_mean':>11s}"
          f"{'truSD_spread':>13s}{'err_SD':>8s}{'SNR':>7s}")
    print("-" * 100)

    rows = []
    for m in METRICS:
        e, t = f"est_{m}", f"tru_{m}"
        d = sub[["user", e, t]].dropna()
        if d.empty:
            print(f"{m:22s}  (no data)")
            continue

        # per-pitch value r2 on this subset
        val_r2 = d[e].corr(d[t]) ** 2

        # per-pitcher within SDs
        g = d.groupby("user").agg(
            esd=(e, "std"), tsd=(t, "std"),
            # per-pitch error SD within pitcher
            n=(e, "size"))
        g = g.dropna()
        # measurement noise: SD of (est - truth) within each pitcher, then avg
        err = (d.assign(diff=d[e] - d[t])
                 .groupby("user")["diff"].std().dropna())

        sd_r2 = g["esd"].corr(g["tsd"]) ** 2
        tru_sd_mean = g["tsd"].mean()
        our_sd_mean = g["esd"].mean()
        tru_sd_spread = g["tsd"].std()          # across-pitcher spread of true SD
        err_sd = err.mean()
        snr = tru_sd_spread / err_sd if err_sd > 0 else np.nan

        print(f"{m:22s}{val_r2:>8.2f}{sd_r2:>7.2f}"
              f"{tru_sd_mean:>11.3f}{our_sd_mean:>11.3f}"
              f"{tru_sd_spread:>13.3f}{err_sd:>8.3f}{snr:>7.2f}")
        rows.append({"metric": m, "value_r2": val_r2, "sd_r2": sd_r2,
                     "truth_sd_mean": tru_sd_mean, "our_sd_mean": our_sd_mean,
                     "truth_sd_spread": tru_sd_spread, "err_sd": err_sd,
                     "snr": snr})

    print("\nHow to read:")
    print("  truSD_spread small + SD_r2 low  -> H-signal: pitchers barely")
    print("     differ in true variability; nothing to recover (NOT our fault).")
    print("  ourSD_mean >> truSD_mean, or err_SD large -> H-noise: our")
    print("     per-pitch noise inflates SD; measurement-side fixable.")
    print("  SNR (truSD_spread / err_SD): higher -> SD recovery easier.")
    print("     knee_angle should show high SNR; stride/trunk likely low.")

    out = os.path.join(config.OBP_VALIDATION_DIR, "sd_diagnostic.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()