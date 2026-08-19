"""How much does the GT-event map cost itself with its own broken landmarks?

`fp_100` is unusable for two of the 100 OBP pitchers: across 8 pitches the landmark
puts foot plant anywhere from 4% to 110% of the peak-knee-height-to-release interval,
while every one of the other 391 pitches sits in k = [0.763, 0.884] and our detector
reports a normal 0.79-0.84 for the same throws (docs/legacy_pre_dedup/GT_EVENT_MAP_HANDOFF.md 5). The
GT-event map is therefore scored partly against noise it did not have to accept.

Recomputes the GT map's r2 with those pitches dropped, for the metrics that actually
read the foot plant. Requires:
    angle_zone_sweep.py --gt-events --dump      -> angle_zone_pairs_gt.csv.gz
    scratch/fp_front_error_anatomy.py           -> fp_front_error_anatomy.csv

Run:  conda activate diamond; cd src\\analysis; python gt_landmark_outlier_effect.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
import numpy as np, pandas as pd
import config
from angle_map_2d import unwrap_circular, CIRCULAR
from absacc_table import ADOPTED_VIEW

# Outliers are flagged by a robust rule on k ALONE, which is built from GT
# landmarks only -- so the filter never consults our detector and the argument is
# not circular. (Reading the range off the pitches our detector agreed with would
# have been.) median k = 0.8276, MAD-scaled SD = 0.0249; +-3 through +-5 MAD all
# flag the identical 8 pitches from the identical 2 pitchers, and a pure-physics
# rule (k outside [0.5, 0.95]) flags 5 of the same 8. The verdict does not depend
# on where the threshold is put.
N_MAD = 3.0


def outlier_pitches(n_mad=N_MAD, verbose=False):
    """session_pitch ids whose GT foot plant is implausible, by the k rule above.

    Single source of the gt_clean population: absacc_table / loco_calibration /
    within_pitcher_agreement all import this so the paper map and the accuracy
    layer are scored on the SAME pitches (n=394 of the 401 with GT events)."""
    an = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR,
                                  "fp_front_error_anatomy.csv"))
    k = an.k_true.to_numpy(float)
    med = float(np.median(k))
    mad = float(np.median(np.abs(k - med)) * 1.4826)
    bad = set(an.loc[(an.k_true < med - n_mad * mad) |
                     (an.k_true > med + n_mad * mad), "sp"])
    if verbose:
        print(f"k median {med:.4f}, MAD-SD {mad:.4f}, +-{n_mad:g} MAD -> "
              f"{len(bad)} implausible pitches")
    return bad


def r2(e, t, circular=False):
    e, t = np.asarray(e, float), np.asarray(t, float)
    if circular:
        e = unwrap_circular(e)
    m = np.isfinite(e) & np.isfinite(t)
    return np.corrcoef(e[m], t[m])[0, 1] ** 2 if m.sum() > 4 else np.nan


def main():
    V = config.OBP_VALIDATION_DIR
    an = pd.read_csv(os.path.join(V, "fp_front_error_anatomy.csv"))
    k = an.k_true.to_numpy(float)
    med = float(np.median(k))
    mad = float(np.median(np.abs(k - med)) * 1.4826)
    lo, hi = med - N_MAD * mad, med + N_MAD * mad
    print(f"k = (fp-pkh)/(rel-pkh) over {len(k)} pitches: median {med:.4f}, "
          f"MAD-SD {mad:.4f}\n  +-{N_MAD:g} MAD -> keep [{lo:.3f}, {hi:.3f}]")
    bad = outlier_pitches()
    print(f"pitches with an implausible GT foot plant: {len(bad)}")
    print(f"  {sorted(bad)}")
    print(f"  from users: {sorted(an[an.sp.isin(bad)].user.unique())}\n")

    dp = pd.read_csv(os.path.join(V, "angle_zone_pairs_gt.csv.gz"))
    print(f"GT pairs: {len(dp)} rows, {dp.session_pitch.nunique()} pitches\n")

    print("=" * 78)
    print("GT-event map at the adopted view, with and without the broken landmarks")
    print("=" * 78)
    print(f"{'metric':>24} {'az/el':>7} {'all':>8} {'dropped':>9} {'delta':>8} {'n drop':>7}")
    for metric, (az, el) in ADOPTED_VIEW.items():
        g = dp[(dp.metric == metric) & (dp.az == az) & (dp.el == el)]
        if g.empty:
            continue
        circ = metric in CIRCULAR
        keep = g[~g.session_pitch.isin(bad)]
        a, b = r2(g.est, g.truth, circ), r2(keep.est, keep.truth, circ)
        n = g.session_pitch.nunique() - keep.session_pitch.nunique()
        mark = "   <--" if abs(b - a) > 0.005 else ""
        print(f"{metric:>24} {f'{az}/{el}':>7} {a:>8.3f} {b:>9.3f} {b-a:>+8.3f} {n:>7}{mark}")

    print("\nmetrics that read the foot plant at all: Stride Angle (frame of the "
          "ankle line)\nand Release Ext (trail-anchor quiet window). Everything else "
          "is release- or\npkh-anchored, so dropping these pitches only changes its n.")

    # The corrected GT map: the same grid with the implausible landmarks dropped.
    # Recomputed from the dump, so it needs no second sweep and is exactly what a
    # --gt-events run would produce on the filtered population.
    keep = dp[~dp.session_pitch.isin(bad)]
    out = []
    for (metric, az, el), g in keep.groupby(["metric", "az", "el"]):
        out.append({"metric": metric, "az": az, "el": el,
                    "r2": r2(g.est, g.truth, metric in CIRCULAR)})
    corr = pd.DataFrame(out)
    p = os.path.join(V, "angle_zone_sweep_gt_clean.csv")
    corr.to_csv(p, index=False)
    print(f"\nsaved -> {p}  ({len(corr)} cells, "
          f"{keep.session_pitch.nunique()} pitches)")

    det = pd.read_csv(os.path.join(V, "angle_zone_sweep.csv")).rename(
        columns={"r2": "r2_det"})
    raw = pd.read_csv(os.path.join(V, "angle_zone_sweep_gt.csv")).rename(
        columns={"r2": "r2_gt"})
    m = (corr.rename(columns={"r2": "r2_gtc"})
             .merge(det, on=["metric", "az", "el"])
             .merge(raw, on=["metric", "az", "el"]))
    print("\n" + "=" * 78)
    print("usable cells (r2 >= 0.6) out of 168, and best cell")
    print("=" * 78)
    print(f"{'metric':>24} {'det':>5} {'GT':>5} {'GTclean':>8}   "
          f"{'best det':>9} {'best GTclean':>13}")
    for metric, g in m.groupby("metric"):
        print(f"{metric:>24} {(g.r2_det>=0.6).sum():>5} {(g.r2_gt>=0.6).sum():>5} "
              f"{(g.r2_gtc>=0.6).sum():>8}   {g.r2_det.max():>9.3f} "
              f"{g.r2_gtc.max():>13.3f}")


if __name__ == "__main__":
    main()
