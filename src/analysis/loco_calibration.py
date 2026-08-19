"""
Diamond - LOCO calibration test: does the projection bias GENERALISE?

absacc_table.py showed systematic clean-projection biases (wrist speed +10%,
HSS +25deg, knee-ext-velo x0.8 compression, ...). That table only proves the
bias EXISTS on average; it does not prove that correcting it transfers to an
unseen pitcher. This script answers exactly that, on the existing 408-pitch
OBP dump, per metric, at the PRE-SPECIFIED anchor viewpoint only (imported
from absacc_table.ADOPTED_VIEW - fixed BEFORE this test, so no same-sample
viewpoint selection leaks into the calibration verdict).

Protocol (leave-one-PITCHER-out, folds = OBP subject id `user`):
  for each held-out pitcher: fit the calibration on all other pitchers'
  pitches, apply it to the held-out pitcher's 2D values, pool the
  out-of-fold predictions, then score the pool against 3D/OBP ground truth.

Calibration models (each fitted per fold, per metric):
  ratio    corrected = k * est,        k = mean(truth) / mean(est)
  offset   corrected = est + b,        b = mean(truth - est)
  linear   corrected = a * est + b,    OLS fit
Raw (uncalibrated) is scored on the same pairs as the baseline.

Scored: bias, MAE, RMSE, CCC, r2 (pooled out-of-fold), plus
  pitcher_bias_sd = SD over pitchers of the per-pitcher mean residual.
  This is the direct test of "the offset differs per pitcher, so a global
  correction fails": a calibration only truly works if MAE drops AND the
  between-pitcher bias spread collapses, not just the pooled mean.

Interpretation guard: this is still CLEAN-PROJECTION data (no RTMPose noise,
no viewpoint/event error). A metric that fails LOCO here fails deployment
calibration a fortiori; a metric that passes still needs the real-video
phone-vs-mocap test before any deployment accuracy claim.

Input:  angle_zone_pairs{suffix}.csv.gz (angle_zone_sweep.py --dump) + metadata.csv
Output: loco_calibration{suffix}.csv (long: metric x model x stats)
        + printed raw-vs-corrected comparison per metric.

Convention (2026-07-24): the PAPER numbers come from the GT-event dump on the
gt_clean population (`--suffix _gt --clean`), so the calibration layer is scored
on exactly the pitches the published map is scored on, and the four GT-only
adoptions (Elbow Flex @MER etc.) are covered here too. The no-argument run stays
the deployment (detected-event) layer.

Run:  cd src\analysis
      python loco_calibration.py --suffix _gt --clean    # paper
      python loco_calibration.py                         # deployment
"""
import os, sys, argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
import config
from absacc_table import ADOPTED_VIEW, UNIT

MODELS = ("ratio", "offset", "linear")


def fit_apply(model, e_tr, t_tr, e_te):
    if model == "ratio":
        return t_tr.mean() / e_tr.mean() * e_te
    if model == "offset":
        return e_te + (t_tr - e_tr).mean()
    a, b = np.polyfit(e_tr, t_tr, 1)
    return a * e_te + b


def score(pred, t, users):
    d = pred - t
    r = np.corrcoef(pred, t)[0, 1]
    cov = ((pred - pred.mean()) * (t - t.mean())).mean()        # ddof=0 (Lin)
    ccc = 2 * cov / (pred.var() + t.var() + (pred.mean() - t.mean()) ** 2)
    per = pd.Series(d).groupby(pd.Series(users)).mean()
    return {"bias": float(d.mean()), "mae": float(np.abs(d).mean()),
            "rmse": float(np.sqrt((d ** 2).mean())),
            "ccc": float(ccc), "r2": float(r * r),
            "pitcher_bias_sd": float(per.std(ddof=0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="",
                    help="pairs-file suffix. '_gt' = the OBP landmark-event dump, "
                         "which is the paper convention (default: detected events, "
                         "the deployment layer)")
    ap.add_argument("--clean", action="store_true",
                    help="drop the pitches whose GT foot-plant landmark is broken, "
                         "matching angle_zone_sweep_gt_clean.csv. Output gains _clean.")
    a = ap.parse_args()

    pairs_p = os.path.join(config.OBP_VALIDATION_DIR,
                           f"angle_zone_pairs{a.suffix}.csv.gz")
    if not os.path.exists(pairs_p):
        sys.exit(f"missing {pairs_p} - run angle_zone_sweep.py --dump first")
    dp = pd.read_csv(pairs_p)
    out_suffix = a.suffix
    if a.clean:
        from gt_landmark_outlier_effect import outlier_pitches
        bad = outlier_pitches()
        dp = dp[~dp.session_pitch.astype(str).isin({str(b) for b in bad})]
        out_suffix = f"{a.suffix}_clean"
        print(f"clean: dropped {len(bad)} implausible-GT-foot-plant pitches")
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    dp["session_pitch"] = dp.session_pitch.astype(str)
    md["session_pitch"] = md.session_pitch.astype(str)
    dp = dp.merge(md[["session_pitch", "user"]].drop_duplicates("session_pitch"),
                  on="session_pitch", how="left")
    if dp.user.isna().any():
        sys.exit("pairs with no metadata user id - session_pitch mismatch")

    out = []
    print(f"pitches {dp.session_pitch.nunique()}, pitchers {dp.user.nunique()}\n")
    print("=" * 96)
    print("[LOCO CALIBRATION @ PRE-SPECIFIED ANCHOR]  leave-one-pitcher-out, "
          "pooled out-of-fold scores")
    print("=" * 96)
    for metric, (az, el) in ADOPTED_VIEW.items():
        g = dp[(dp.metric == metric) & (dp.az == az) & (dp.el == el)]
        m = np.isfinite(g.est.to_numpy(float)) & np.isfinite(g.truth.to_numpy(float))
        g = g[m]
        # ADOPTED_VIEW carries the GT-only rescreen adoptions (Elbow Flex @MER etc.),
        # which are absent from the detected dump -- skip rather than crash. Their
        # LOCO verdict lives in analysis/tier1_adoption_probe.py.
        if len(g) < 5:
            continue
        e = g.est.to_numpy(float); t = g.truth.to_numpy(float)
        users = g.user.to_numpy()
        preds = {mo: np.full(len(g), np.nan) for mo in MODELS}
        for u in np.unique(users):
            te = users == u; tr = ~te
            for mo in MODELS:
                preds[mo][te] = fit_apply(mo, e[tr], t[tr], e[te])

        rows = {"raw": score(e, t, users)}
        for mo in MODELS:
            rows[mo] = score(preds[mo], t, users)
        for name, s in rows.items():
            out.append({"metric": metric, "az": az, "el": el, "model": name,
                        "n": len(g), **s})

        u_ = UNIT.get(metric, "?")
        print(f"\n{metric}  @az={az}/el={el}  [{u_}]  n={len(g)}, "
              f"{len(np.unique(users))} pitchers")
        hdr = (f"  {'model':<8}{'bias':>9}{'MAE':>8}{'RMSE':>8}{'CCC':>8}"
               f"{'r2':>7}{'pitcher_bias_sd':>17}")
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for name, s in rows.items():
            print(f"  {name:<8}{s['bias']:>+9.3f}{s['mae']:>8.3f}"
                  f"{s['rmse']:>8.3f}{s['ccc']:>8.3f}{s['r2']:>7.3f}"
                  f"{s['pitcher_bias_sd']:>17.3f}")

    outp = os.path.join(config.OBP_VALIDATION_DIR,
                        f"loco_calibration{out_suffix}.csv")
    pd.DataFrame(out).round(4).to_csv(outp, index=False)
    print(f"\nsaved -> {outp}")
    print("\npitcher_bias_sd = SD over pitchers of per-pitcher mean residual "
          "(between-pitcher offset spread).")
    print("A calibration generalises only if MAE drops AND pitcher_bias_sd "
          "shrinks vs raw; pooled bias~0 alone is not success.")
    print("Clean-projection data: a LOCO pass here is necessary, not "
          "sufficient, for deployment accuracy (real phone-vs-mocap pending).")


if __name__ == "__main__":
    main()
