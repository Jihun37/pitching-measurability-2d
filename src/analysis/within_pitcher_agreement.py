"""Within-pitcher vs between-pitcher measurement agreement (paper-prep probe).

The official angle map reports POOLED r2 (our 2D estimate vs 3D truth over all
411 pitches). A pooled r2 can be high purely because pitchers differ in body
size / style: the pipeline may rank pitchers correctly yet fail to track a
SINGLE pitcher's pitch-to-pitch changes. Deployment (an app coaching one
pitcher) needs the latter.

This script decomposes, at each metric's pre-specified paper anchor:
  - pooled r2            : official number (sanity check, must match the table)
  - between-pitcher r2   : r2 of pitcher-MEAN est vs pitcher-MEAN truth
  - within-pitcher r2    : r2 of pitcher-CENTERED est vs pitcher-CENTERED truth
  - truth ICC (between)  : fraction of TRUTH variance that is between-pitcher
                           (if ~1, there is almost no within-pitcher signal to
                           track, so a low within-r2 is expected, not a failure)
  - n_pitchers, mean pitches/pitcher

Reads the existing dump (no new metric definitions, no table writes):
  data/outputs/obp_validation/angle_zone_pairs{suffix}.csv.gz
Anchors are imported from absacc_table.ADOPTED_VIEW (single source of truth).

Convention (2026-07-24): the PAPER numbers come from the GT-event dump on the
gt_clean population (`--suffix _gt --clean`) -- the same pitches the published
map is scored on, and the only mode in which the four GT-only adoptions
(Elbow Flex @MER etc.) have rows at all. No-argument = deployment layer.

Run:  cd srcnalysis
      python within_pitcher_agreement.py --suffix _gt --clean   # paper
      python within_pitcher_agreement.py                        # deployment
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))          # analysis/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # src/
import config  # noqa: E402
from absacc_table import ADOPTED_VIEW  # noqa: E402

VALID = os.path.join(config.ROOT, "data", "outputs", "obp_validation")
META = os.path.join(config.OBP_DATA_DIR, "metadata.csv")


def r2(x, y):
    """Pearson r^2 between two 1-D arrays (nan-safe, needs >=3 varying points)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    r = np.corrcoef(x, y)[0, 1]
    return r * r


def icc_between(truth, grp):
    """Fraction of truth variance that is between-group (one-way ANOVA style).
    var_between / (var_between + var_within), pitcher = group."""
    d = pd.DataFrame({"y": truth, "g": grp}).dropna()
    grand = d.y.mean()
    gm = d.groupby("g").y
    # between: sum n_i (mean_i - grand)^2 ; within: sum (y - mean_i)^2
    ss_between = (gm.count() * (gm.mean() - grand) ** 2).sum()
    ss_within = ((d.y - d.groupby("g").y.transform("mean")) ** 2).sum()
    tot = ss_between + ss_within
    return ss_between / tot if tot > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="",
                    help="pairs-file suffix ('_gt' = OBP landmark events = paper "
                         "convention; default = detected events)")
    ap.add_argument("--clean", action="store_true",
                    help="drop pitches with a broken GT foot-plant landmark "
                         "(matches angle_zone_sweep_gt_clean.csv)")
    a = ap.parse_args()

    pairs = pd.read_csv(os.path.join(VALID, f"angle_zone_pairs{a.suffix}.csv.gz"))
    out_suffix = a.suffix
    if a.clean:
        from gt_landmark_outlier_effect import outlier_pitches
        bad = {str(b) for b in outlier_pitches()}
        pairs = pairs[~pairs.session_pitch.astype(str).isin(bad)]
        out_suffix = f"{a.suffix}_clean"
        print(f"clean: dropped {len(bad)} implausible-GT-foot-plant pitches")
    meta = pd.read_csv(META)[["session_pitch", "user"]]
    pairs = pairs.merge(meta, on="session_pitch", how="left")
    if pairs.user.isna().any():
        miss = pairs.session_pitch[pairs.user.isna()].nunique()
        print(f"WARNING: {miss} session_pitch had no pitcher mapping (dropped)")
        pairs = pairs.dropna(subset=["user"])

    rows = []
    for metric, (az, el) in ADOPTED_VIEW.items():
        g = pairs[(pairs.metric == metric) & (pairs.az == az) & (pairs.el == el)]
        if g.empty:
            print(f"  (no dump rows for {metric} @ az{az}/el{el})")
            continue
        g = g.dropna(subset=["est", "truth"])
        # pitcher-mean and pitcher-centered forms
        gm_est = g.groupby("user").est.transform("mean")
        gm_tru = g.groupby("user").truth.transform("mean")
        cen_est = g.est - gm_est
        cen_tru = g.truth - gm_tru
        per = g.groupby("user").agg(est=("est", "mean"), truth=("truth", "mean"))

        # only pitchers with >=2 pitches contribute within-pitcher signal
        counts = g.groupby("user").size()
        multi = counts[counts >= 2].index
        gmulti = g[g.user.isin(multi)]
        cen_est_m = cen_est[g.user.isin(multi)]
        cen_tru_m = cen_tru[g.user.isin(multi)]

        rows.append({
            "metric": metric.replace(" [O]", ""),
            "az/el": f"{az}/{el}",
            "n": len(g),
            "n_pitch": g.user.nunique(),
            "pit/pchr": round(len(g) / g.user.nunique(), 1),
            "pooled_r2": r2(g.est, g.truth),
            "between_r2": r2(per.est, per.truth),
            "within_r2": r2(cen_est_m, cen_tru_m),
            "truth_ICC": icc_between(g.truth.values, g.user.values),
            "truth_within_SD": gmulti.assign(c=cen_tru_m).c.std(),
            "truth_total_SD": g.truth.std(),
        })

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("=" * 120)
    print("WITHIN- vs BETWEEN-PITCHER AGREEMENT  @ pre-specified paper anchors")
    print("  pooled_r2   : official map number (2D est vs 3D truth, all pitches)")
    print("  between_r2  : pitcher-mean est vs pitcher-mean truth (ranks pitchers)")
    print("  within_r2   : pitcher-centered est vs truth (tracks ONE pitcher's changes)")
    print("  truth_ICC   : fraction of TRUTH variance that is between-pitcher")
    print("                (near 1 => little within-pitcher signal exists to track)")
    print("=" * 120)
    fmt = {c: "{:.3f}".format for c in
           ["pooled_r2", "between_r2", "within_r2", "truth_ICC"]}
    fmt.update({c: "{:.4g}".format for c in ["truth_within_SD", "truth_total_SD"]})
    print(out.to_string(index=False, formatters=fmt))

    dst = os.path.join(VALID, f"within_pitcher_agreement{out_suffix}.csv")
    out.to_csv(dst, index=False)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
