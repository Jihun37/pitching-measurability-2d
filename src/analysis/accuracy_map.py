"""ABSOLUTE-ACCURACY layer for the whole evaluation set, per cell.

WHAT THIS REPLACES, and why it had to be rebuilt. The accuracy layer used to be
three scripts (`absacc_table.py`, `loco_calibration.py`, `within_pitcher_agreement.py`)
that all looped over `absacc_table.ADOPTED_VIEW`, a hand-written dict of 16 metrics
each pinned to ONE pre-specified viewpoint. That was the right shape when the paper
claimed 16 adopted metrics at 16 anchors. It is the wrong shape for a graded MAP:
the map scores 47 estimator rows at all 168 cells, so an accuracy layer that covers
16 rows at 1 cell each cannot be joined to it, and after the 2026-07-29 dedup four
of those 16 keys no longer exist in the dump at all.

This script covers **every evaluation row at every cell**, on exactly the pitches
and with exactly the conventions `gate_map.py` uses, so the two tables join on
(metric, source, az, el) with no reconciliation step.

WHAT IS HERE THAT THE MAP DOES NOT ALREADY HAVE. `gate_map.csv` already carries the
leave-one-pitcher-out calibrated CCC, MAE and per-pitcher bias SD per cell, so the
old `loco_calibration.py` is subsumed and is NOT reimplemented. What the map lacks,
and this adds:

    bias        mean(est - truth), signed, native units -- the map reports |error|
                only, so a systematic offset is invisible in it
    rmse        outlier-sensitive companion to MAE
    nmae_sd     MAE / SD(truth) over the metric's own scored population. The
                readable form: at 1.0 the typical error equals the spread the
                measurement has to resolve, so it cannot separate two pitchers
    nmae_mean   MAE / |mean(truth)|, the intuitive relative error
    within_r2   r-squared after centring each pitcher -- following ONE pitcher over
                time, which a pooled statistic can fake by ranking pitchers
    between_r2  r-squared of the per-pitcher means, the ranking half of the same split
    truth_icc   share of truth variance lying between pitchers. Read within_r2
                WITH this: a metric whose truth barely moves within a pitcher has
                little within-pitcher signal to recover, which is a property of the
                quantity, not a failure of the estimate

SELF-VERIFICATION. `n`, `r2`, raw CCC and raw MAE are recomputed here independently
and asserted equal to `gate_map.csv`'s columns of the same name. If the two ever
disagree the script fails rather than writing a table, because a silent divergence
between the accuracy layer and the map is precisely the bug this design removes.

RAW, CLEAN-PROJECTION. Every statistic is uncalibrated agreement on exact
projections of mocap joints under ground-truth event anchors. It is a geometric and
definitional ceiling, not phone accuracy: no pose error, no viewpoint error, no
detected events. Calibrated agreement is the map's `ccc` column.

Input:  angle_zone_pairs_gt.csv.gz + rejected_gt_pairs.csv.gz, gate_map.csv
Output: accuracy_map_gt_clean.csv (long: metric x source x az x el)
        + a per-row summary at each row's best-CCC cell, printed
Run:  conda activate diamond; cd src\\analysis; python accuracy_map.py
"""
import os, sys, argparse
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config
from angle_map_2d import CIRCULAR, unwrap_circular
from gt_landmark_outlier_effect import outlier_pitches

V = config.OBP_VALIDATION_DIR
KEY = ["metric", "source", "az", "el"]
MIN_N = 30          # same floor gate_map.score_cell applies


def ccc(e, t):
    se, st = e.std(), t.std()
    cov = ((e - e.mean()) * (t - t.mean())).mean()
    den = se ** 2 + st ** 2 + (e.mean() - t.mean()) ** 2
    return 2 * cov / den if den > 0 else np.nan


def r2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def icc_between(t, g):
    d = pd.DataFrame({"y": t, "g": g}).dropna()
    if d.empty:
        return np.nan
    grand = d.y.mean()
    gm = d.groupby("g").y
    ssb = (gm.count() * (gm.mean() - grand) ** 2).sum()
    ssw = ((d.y - d.groupby("g").y.transform("mean")) ** 2).sum()
    tot = ssb + ssw
    return ssb / tot if tot > 0 else np.nan


def cell_stats(e, t, g):
    ok = np.isfinite(e) & np.isfinite(t)
    e, t, g = e[ok], t[ok], g[ok]
    if len(e) < MIN_N or e.std() < 1e-9 or t.std() < 1e-9:
        return None
    d = e - t
    df = pd.DataFrame({"e": e, "t": t, "g": g})
    per = df.groupby("g").agg(e=("e", "mean"), t=("t", "mean"))
    cnt = df.groupby("g").size()
    multi = cnt[cnt >= 2].index                 # only these carry within signal
    mm = df.g.isin(multi)
    cen_e = (df.e - df.groupby("g").e.transform("mean"))[mm]
    cen_t = (df.t - df.groupby("g").t.transform("mean"))[mm]
    sd_t, mean_t = t.std(), abs(t.mean())
    mae = float(np.abs(d).mean())
    return dict(
        n=len(e), n_pitcher=int(df.g.nunique()),
        r2=r2(e, t), raw_ccc=ccc(e, t),
        bias=float(d.mean()), raw_mae=mae, rmse=float(np.sqrt((d ** 2).mean())),
        nmae_sd=mae / sd_t if sd_t > 0 else np.nan,
        nmae_mean=mae / mean_t if mean_t > 1e-9 else np.nan,
        within_r2=r2(cen_e, cen_t), between_r2=r2(per.e, per.t),
        truth_icc=icc_between(t, g), truth_sd=float(sd_t),
        truth_within_sd=float(cen_t.std()) if len(cen_t) > 2 else np.nan,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="accuracy_map_gt_clean.csv")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the gate_map cross-check (diagnostics only)")
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    user_of = dict(zip(md.session_pitch, md.user))
    keep = {sp for sp in md.session_pitch if sp not in outlier_pitches()}
    from mer_proxy_map import map_population
    keep &= set(map_population())
    print(f"population: {len(keep)} pitches (gt_clean, frozen map ids)")

    parts = []
    for f, src in (("angle_zone_pairs_gt.csv.gz", "adopted"),
                   ("rejected_gt_pairs.csv.gz", "screened")):
        p = os.path.join(V, f)
        if not os.path.exists(p):
            sys.exit(f"missing {f} -- build the GT dumps first")
        d = pd.read_csv(p)
        d = d[d.session_pitch.isin(keep)]
        d["source"] = src
        parts.append(d)
    # identical intersection to gate_map: the two sweeps gate pitches differently
    common = set.intersection(*[set(p.session_pitch) for p in parts])
    parts = [p[p.session_pitch.isin(common)] for p in parts]
    d = pd.concat(parts, ignore_index=True)
    d["user"] = d.session_pitch.map(user_of)
    d = d[d.user.notna()]
    print(f"pairs: {len(d):,} rows, {d.metric.nunique()} metrics, "
          f"{d.session_pitch.nunique()} pitches\n")

    rows = []
    for (metric, src), gm in d.groupby(["metric", "source"], sort=True):
        codes = gm.user.astype("category").cat.codes.to_numpy()
        e_all = gm.est.to_numpy(float); t_all = gm.truth.to_numpy(float)
        az_all = gm.az.to_numpy(int); el_all = gm.el.to_numpy(int)
        circ = metric.strip() in CIRCULAR
        for az in np.unique(az_all):
            ma = az_all == az
            for el in np.unique(el_all[ma]):
                m = ma & (el_all == el)
                e_cell = unwrap_circular(e_all[m]) if circ else e_all[m]
                s = cell_stats(e_cell, t_all[m], codes[m])
                if s is not None:
                    rows.append(dict(metric=metric, source=src, az=int(az),
                                     el=int(el), **s))
        print(f"  scored {metric}")
    out = pd.DataFrame(rows)

    # ---- cross-check against the map -------------------------------------
    if not a.no_verify:
        gp = os.path.join(V, "gate_map.csv")
        g = pd.read_csv(gp)[KEY + ["n", "r2", "raw_ccc", "raw_mae"]]
        j = out.merge(g, on=KEY, suffixes=("", "_gate"), how="inner")
        if len(j) != len(g):
            sys.exit(f"cell-set mismatch: {len(j)} joined vs {len(g)} in gate_map")
        # RELATIVE tolerance, deliberately. gate_map.csv is written with
        # float_format="%.6g", so a stored value carries six significant digits and
        # nothing more: an MAE in deg/s of order 1e4 comes back rounded by ~5e-2,
        # which is a formatting artefact of the file, not a disagreement between the
        # two computations. An absolute tolerance therefore fails on exactly the
        # large-magnitude velocity rows and passes everywhere else, which would be a
        # misleading check. 1e-5 relative is two orders looser than %.6g needs.
        worst, RTOL = {}, 1e-5
        for c in ("n", "r2", "raw_ccc", "raw_mae"):
            a_, b_ = j[c].to_numpy(float), j[f"{c}_gate"].to_numpy(float)
            scale = np.maximum(np.abs(b_), 1e-12)
            worst[c] = float(np.nanmax(np.abs(a_ - b_) / scale))
        print("\nCROSS-CHECK vs gate_map.csv (max RELATIVE delta, %.6g storage):")
        for c, v in worst.items():
            print(f"  {c:<8} {v:.3e}")
        bad = [c for c, v in worst.items() if v > RTOL]
        if bad:
            sys.exit(f"accuracy layer disagrees with the map on {bad}")
        print(f"  {len(j):,} cells agree within {RTOL:g} relative")

    p = os.path.join(V, a.out)
    out.to_csv(p, index=False, float_format="%.6g")
    print(f"\nsaved -> {p}  ({len(out):,} cells, {out.metric.nunique()} rows)")

    # ---- per-row summary at the row's best-CCC cell -----------------------
    ls = os.path.join(V, "layer_summary.csv")
    if os.path.exists(ls):
        s = pd.read_csv(ls)
        s = s[s.map_cells > 0]
        recs = []
        for r in s.itertuples(index=False):
            az, el = (int(x) for x in str(r.best_ccc_view).split("/"))
            c = out[(out.metric == r.metric) & (out.az == az) & (out.el == el)]
            if c.empty:
                continue
            c = c.iloc[0]
            recs.append(dict(metric=r.metric, view=f"{az}/{el}",
                             grade=r.metric_grade, loco_ccc=r.best_ccc,
                             bias=c.bias, mae=c.raw_mae, nmae_sd=c.nmae_sd,
                             pooled_r2=c.r2, between_r2=c.between_r2,
                             within_r2=c.within_r2, truth_icc=c.truth_icc))
        t = pd.DataFrame(recs).sort_values("loco_ccc", ascending=False)
        print("\n" + "=" * 104)
        print("PER-ROW ACCURACY at each row's best-CCC cell")
        print("  (the VIEW is chosen by the same-sample CCC argmax, so it carries "
              "the map's own selection caveat)")
        print("=" * 104)
        print(t.to_string(index=False,
                          float_format=lambda x: f"{x:.3f}"))
        t.to_csv(os.path.join(V, "accuracy_bestcell_gt_clean.csv"),
                 index=False, float_format="%.6g")
        print(f"\nsaved -> {os.path.join(V, 'accuracy_bestcell_gt_clean.csv')}")


if __name__ == "__main__":
    main()
