"""
Diamond - absolute-accuracy stats table (evaluation-metric expansion, ROADMAP #2).

r2 alone is scale/offset-invariant, so it cannot show whether a 2D estimate
agrees with the ground truth in ABSOLUTE terms (a constant offset still gives
r2=1). This script adds the absolute-agreement layer on top of the official r2
zone sweep. Units were verified pairwise-aligned for the 9 original adopted
metrics (scratch/unit_check.py, 2026-07-17: c3d is in metres, OBP stride is a
stature fraction like ours, wrist speed already height-scaled); COG Fwd Velo
(adopted 2026-07-18) uses the identical m/s height-scaling as wrist speed, so it
is unit-consistent by construction. Remaining offsets are DEFINITION gaps
(projection bias), which is exactly what these stats expose.

Per (metric, az, el) cell:
  n      finite est/truth pairs
  r2     Pearson r^2 (sanity: must reproduce angle_zone_sweep.csv)
  bias   mean(est - truth), native units (signed systematic offset)
  mae    mean |est - truth|, native units
  rmse   sqrt(mean (est - truth)^2), native units
  nmae_sd   mae / SD(truth) * 100  [%]. Error vs the between-subject SPREAD:
            "can the 2D measure resolve subjects?" SD over the metric's FULL GT
            population (one denominator per metric, comparable across views).
            NOTE: tightly-clustered metrics (wrist speed SD~1.7 m/s) show a high
            nmae_sd even at ~10% relative error - read it WITH nmae_mean.
  nmae_mean mae / |mean(truth)| * 100  [%]. Error vs the metric's MAGNITUDE
            (the intuitive relative error). Undefined-ish for signed metrics
            whose mean sits near 0 - here every GT mean is well away from 0.
  ccc    Lin's concordance correlation coefficient (absolute agreement:
         penalises both offset and scale mismatch, dimensionless)

All stats are RAW (uncalibrated) CLEAN-PROJECTION agreement under the selected
sweep convention. They are a geometric/definition ceiling, not real-phone
deployment accuracy: RTMPose error, viewpoint error, and the complete deployed
event/offset pipeline are absent. A high r2 with a low CCC identifies systematic
disagreement and motivates calibration research; it does not prove that a
calibration will transfer out of sample.

Input:  angle_zone_pairs.csv.gz  from  angle_zone_sweep.py --dump
        (detect-once paper convention; --suffix _redetect for the per-view-
        redetection diagnostic if that dump exists; neither is the complete
        offset-rule real-phone pipeline)
Output: absacc_table{suffix}.csv (full 15deg x 7-el grid)
        + adopted-view summary in native units
        + best-observed-cell diagnostic (explicitly selection-optimistic).

Run:  cd src\analysis
      python absacc_table.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
import config
# shared with the r2 sweep so the two never disagree about what a circular metric
# is; without it the sanity check below fails on every stride-angle cell
from angle_map_2d import unwrap_circular, CIRCULAR

# native unit of each adopted metric's est/truth pair (verified by unit_check)
UNIT = {
    "Lead Knee Angle [O]":  "deg",
    "Stride (anchor) [O]":  "xstature",
    "Trunk Tilt (ant) [O]": "deg",
    "Wrist Speed [O]":      "m/s",
    "Arm Slot [O]":         "deg",
    "Release Height [O]":   "m",
    "Hip-Shoulder Sep [O]": "deg",
    "Pelvis Rot Velo [O]":  "deg/s",
    "COG Fwd Velo [O]":     "m/s",
    "COG Velo @PKH [O]":    "m/s",
    "Release Ext [O]":      "m",
    "Stride Angle [O]":     "deg",
    # GT-event rescreen adoptions (2026-07-24); scored from the GT dump only
    "Elbow Flex @MER [O]":      "deg",
    "Glove Sh Abd @MER [O]":    "deg",
    "Torso Rot @BR [O]":        "deg",
    "Torso Lat Tilt @MER [O]":  "deg",
}

# Pre-specified paper anchors from docs/PAPER_MASTER.md section 3.1. These are
# kept separate from the same-sample r2 argmax so the main summary does not
# acquire selection optimism. They describe the detect-once clean-projection
# table, not the offset-rule deployment map.
# Overhead anchor decision (2026-07-18): (az0, el85) for BOTH overhead metrics.
# Section 3.1 pre-specifies only "overhead el75 peak" and leaves azimuth open;
# filling it with the same-sample argmax (75/75, 270/75) would leak selection
# optimism into the anchor table. At el=85 azimuth is nearly degenerate
# (HSS r2 spread 0.02, Pelvis 0.04 across all 24 azimuths), and (0, 85) is the
# project's historical overhead row - it reproduces the long-quoted official
# "overhead HSS r2=0.63" (now r2=0.629 at n=411). The el75-peak cells appear in the
# best-observed-cell diagnostic below.
# Trunk anchor moved el0 -> el15 (2026-07-18, LOCO-verified): a slightly
# elevated side view reduces sagittal foreshortening of the anterior lean.
# Raw improves (MAE 7.15->4.85, CCC 0.686->0.832, bias -6.58->-2.84) AND the
# gain survives leave-one-pitcher-out calibration (best MAE 4.61->4.11,
# CCC 0.827->0.856, pitcher_bias_sd 5.87->5.44). CAVEAT: el15 is also the
# r2-argmax elevation, so the VIEWPOINT choice shares the full-sample selection
# caveat of every other section-3.1 anchor; the CALIBRATION gain is out-of-fold
# (loco_calibration.py), so it is not in-sample overfit. Trunk stays a partial-
# recovery metric (per-pitcher offset spread ~5.4 deg persists).
ADOPTED_VIEW = {
    "Lead Knee Angle [O]":  (195, 0),
    "Stride (anchor) [O]":  (180, 0),
    "Trunk Tilt (ant) [O]": (0, 15),
    "Wrist Speed [O]":      (0, 0),
    "Arm Slot [O]":         (90, 0),
    "Release Height [O]":   (0, 0),
    "Hip-Shoulder Sep [O]": (0, 85),
    "Pelvis Rot Velo [O]":  (0, 85),
    # COG forward velocity: side view (az0/el0). Forward = pitching-direction
    # axis, seen in-plane from the side. Adopted 2026-07-18 (Winter whole-body
    # COM, r2~0.75). The "point-only, no noise-valid zone" note that used to
    # sit here is RETIRED 2026-07-29; see paper Sec. IV-B.
    "COG Fwd Velo [O]":     (0, 0),
    # COG velocity at peak knee height: side view too. Adopted 2026-07-18
    # (Winter COM @ new pkh event + SG derivative, r2 0.889, raw CCC 0.905).
    "COG Velo @PKH [O]":    (0, 0),
    # Release extension: pure side view. PRE-SPECIFIED as az0/el0 (the side
    # anchor the other sagittal metrics use), NOT the same-sample argmax cell
    # from the adoption probe. Sagittal-only: r2 collapses to ~0.01 by az90.
    # Adopted 2026-07-20.
    "Release Ext [O]":      (0, 0),
    # Stride angle: FRONT view (az90/el0). Lead-to-trail ankle line orientation
    # at foot plant. A transverse/horizontal quantity whose sign flips with
    # handedness, so it only recovers under the reflecting loader (12->13,
    # adopted 2026-07-23; r2 0.914, raw CCC 0.04 = ~90 deg convention offset,
    # LOCO CCC 0.956, MAE 1.29 deg). CALIBRATE group. NOTE this metric is even
    # better ELEVATED (el85, azimuth-independent: r2 0.966, LOCO CCC 0.985,
    # MAE 0.64 deg) -- the ground-front anchor is kept here because ADOPTED_VIEW
    # is one cell per metric and the ground view is the deployable one; the
    # elevated figures are reported alongside in the ledger.
    "Stride Angle [O]":     (90, 0),
    # GT-event rescreen adoptions (2026-07-24), pre-specified zone-center anchors.
    # Present in the GT sweep only. Full probe: analysis/tier1_adoption_probe.py.
    # Elbow flexion at MER: DIRECT (raw CCC 0.84, LOCO CCC 0.94), independent 0.83.
    "Elbow Flex @MER [O]":   (330, 60),
    # Glove-arm abduction at MER: CALIBRATE (LOCO CCC 0.90), independent 0.68.
    "Glove Sh Abd @MER [O]": (75, 15),
    # Trunk transverse rotation at release: overhead-recovered, CALIBRATE (LOCO CCC
    # 0.89, ~90deg convention offset). Partly overlaps adopted transverse (indep 0.31).
    "Torso Rot @BR [O]":     (90, 85),
    # Trunk lateral (coronal) tilt at MER: DIRECT (raw & LOCO CCC 0.91). Adopted for
    # MEASURABILITY (a distinct anatomical axis), not incremental prediction: its
    # independence from the adopted set is low (indep 0.24). User decision 2026-07-24.
    "Torso Lat Tilt @MER [O]": (90, 30),
}


def print_summary(df, sd, mu, title, choose):
    print("=" * 100)
    print(title)
    print("=" * 100)
    hdr = (f"{'metric':<24}{'az/el':>7}{'unit':>10}{'GT mean+-SD':>16}"
           f"{'bias':>9}{'MAE':>8}{'RMSE':>8}{'NMAE/SD':>9}{'NMAE/mu':>9}"
           f"{'CCC':>7}{'r2':>7}")
    print(hdr); print("-" * len(hdr))
    for metric, g in df.groupby("metric", sort=False):
        p = choose(metric, g)
        u = UNIT.get(metric, "?")
        print(f"{metric:<24}{f'{int(p.az)}/{int(p.el)}':>7}{u:>10}"
              f"{f'{mu[metric]:.2f}+-{sd[metric]:.2f}':>16}"
              f"{p.bias:>+9.2f}{p.mae:>8.2f}{p.rmse:>8.2f}"
              f"{p.nmae_sd:>8.1f}%{p.nmae_mean:>8.1f}%{p.ccc:>7.3f}{p.r2:>7.3f}")


def cell_stats(e, t, sd_t, mean_t):
    """All per-cell stats over finite pairs. sd_t / mean_t = metric-wide GT
    SD / mean (fixed denominators so cells are comparable across views)."""
    e = np.asarray(e, float); t = np.asarray(t, float)
    m = np.isfinite(e) & np.isfinite(t)
    n = int(m.sum())
    if n <= 2:
        return {"n": n, "r2": np.nan, "bias": np.nan, "mae": np.nan,
                "rmse": np.nan, "nmae_sd": np.nan, "nmae_mean": np.nan,
                "ccc": np.nan}
    e, t = e[m], t[m]
    d = e - t
    r = np.corrcoef(e, t)[0, 1]
    cov = ((e - e.mean()) * (t - t.mean())).mean()          # ddof=0 (Lin)
    ccc = 2 * cov / (e.var() + t.var() + (e.mean() - t.mean()) ** 2)
    mae = float(np.abs(d).mean())
    return {"n": n, "r2": r * r, "bias": float(d.mean()), "mae": mae,
            "rmse": float(np.sqrt((d ** 2).mean())),
            "nmae_sd": mae / sd_t * 100.0 if sd_t > 0 else np.nan,
            "nmae_mean": mae / abs(mean_t) * 100.0 if abs(mean_t) > 1e-9 else np.nan,
            "ccc": float(ccc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="",
                    help="pairs-file suffix, e.g. _redetect (default: official "
                         "detect-once dump)")
    ap.add_argument("--clean", action="store_true",
                    help="drop the pitches whose GT foot-plant landmark is broken "
                         "(gt_landmark_outlier_effect.outlier_pitches) -- this is "
                         "the paper population for the GT map. Output suffix gains "
                         "_clean.")
    a = ap.parse_args()

    pairs_p = os.path.join(config.OBP_VALIDATION_DIR,
                           f"angle_zone_pairs{a.suffix}.csv.gz")
    if not os.path.exists(pairs_p):
        sys.exit(f"missing {pairs_p} - run angle_zone_sweep.py --dump first")
    dp = pd.read_csv(pairs_p)
    out_suffix = a.suffix
    if a.clean:
        # imported lazily: gt_landmark_outlier_effect imports ADOPTED_VIEW from
        # this module, so a top-level import here would be circular
        from gt_landmark_outlier_effect import outlier_pitches
        bad = outlier_pitches()
        dp = dp[~dp.session_pitch.isin(bad)]
        out_suffix = f"{a.suffix}_clean"
        print(f"clean: dropped {len(bad)} implausible-GT-foot-plant pitches")
    print(f"pairs: {len(dp)} rows, {dp.session_pitch.nunique()} pitches, "
          f"{dp.metric.nunique()} metrics\n")

    # one GT-SD denominator per metric, over unique pitches (truth repeats
    # across the az/el grid, so dedupe before taking the SD)
    gt = dp.groupby(["metric", "session_pitch"]).truth.first()
    sd = gt.groupby("metric").std(ddof=0)
    mu = gt.groupby("metric").mean()

    out = []
    for (metric, az, el), g in dp.groupby(["metric", "az", "el"]):
        e = unwrap_circular(g.est) if metric in CIRCULAR else g.est
        s = cell_stats(e, g.truth, float(sd[metric]), float(mu[metric]))
        out.append({"metric": metric, "az": az, "el": el, **s})
    df = pd.DataFrame(out)

    # sanity: the recomputed r2 must reproduce the official sweep csv
    sweep_p = os.path.join(config.OBP_VALIDATION_DIR,
                           f"angle_zone_sweep{out_suffix}.csv")
    if os.path.exists(sweep_p):
        off = pd.read_csv(sweep_p).rename(columns={"r2": "r2_official"})
        mg = df.merge(off, on=["metric", "az", "el"], how="inner")
        dmax = (mg.r2 - mg.r2_official).abs().max()
        print(f"sanity vs {os.path.basename(sweep_p)}: "
              f"max |r2 - official| = {dmax:.6f} over {len(mg)} cells"
              + ("" if dmax < 1e-6 else "   << MISMATCH - conventions drifted"))
    else:
        print(f"sanity skipped: {sweep_p} not found")

    outp = os.path.join(config.OBP_VALIDATION_DIR, f"absacc_table{out_suffix}.csv")
    df.round(4).to_csv(outp, index=False)
    print(f"saved -> {outp}\n")

    def adopted(metric, g):
        az, el = ADOPTED_VIEW[metric]
        p = g[(g.az == az) & (g.el == el)]
        if len(p) != 1:
            raise ValueError(f"missing adopted cell for {metric}: az={az}, el={el}")
        return p.iloc[0]

    print_summary(
        df, sd, mu,
        "[CLEAN-PROJECTION ABSOLUTE AGREEMENT @ PRE-SPECIFIED PAPER ANCHOR]  "
        "raw (uncalibrated)",
        adopted,
    )

    print()
    print_summary(
        df, sd, mu,
        "[BEST OBSERVED CELL - DIAGNOSTIC ONLY, SAME-SAMPLE SELECTION OPTIMISTIC]",
        lambda _metric, g: g.loc[g.r2.idxmax()],
    )
    print("\nNMAE/SD = MAE / SD(GT population) [error vs subject spread];  "
          "NMAE/mu = MAE / mean(GT) [relative error].")
    print("CCC = Lin's concordance (1 = perfect absolute agreement); "
          "r2 vs CCC gap indicates uncalibrated offset/scale disagreement.")
    print("Arm Slot at az=90/el=0 is a clean-projection synthetic identity "
          "(2D coronal definition equals the 3D-direct definition), not "
          "evidence of perfect RTMPose or real-phone agreement.")


if __name__ == "__main__":
    main()
