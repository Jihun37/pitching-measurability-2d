"""An FP-ONLY viewpoint routing rule, chosen and evaluated honestly.

Both foot-plant detectors are kept. What was missing is a rule that says which one to
use where, fitted on something a deployed system can actually see.

WHY NOT THE CELLWISE ORACLE. `deploy_map.py` reports a per-cell "oracle" that picks
whichever detector produced the better METRIC CCC in that cell. That peeks at the
answer twice over -- it uses the held-out truth, and it uses the metric outcome
rather than anything measurable at capture time. It is an upper bound and is kept
here only as one.

THE RULE. Per cell, choose the detector with the smaller **debiased foot-plant MAE**
-- mean |err - median(err)| in frames @360 Hz, against the OBP fp_100 landmark, with
release detected per view by the adopted `release_view()` rule. Debiased because a
constant offset is absorbed downstream by the LOCO calibration, so only the spread
is the detector's fault. **Ties break on p90 of the same debiased error** (a detector
that matches on average but has a worse tail is the worse detector); an exact
remaining tie falls back to `side`, the incumbent.

Note the selection criterion is EVENT ERROR, not metric outcome. That is deliberate:
it never touches the quantity being validated, so the rule is fittable from a
detector-only calibration set.

THE EVALUATION IS PITCHER-LEVEL CROSS-VALIDATED. The rule is refitted inside every
fold on the TRAINING pitchers only, then applied to the HELD-OUT pitchers' pitches.
Each pitch therefore receives an estimate from a routing decision made without any
data from its own pitcher, and the assembled out-of-fold vector is scored by the
map's own gate (`gate_map.score_cell`, itself leave-one-pitcher-out). Two schemes are
run because they answer slightly different questions:
    lopo    leave-one-PITCHER-out (98 folds) -- matches the project's LOCO convention
    5fold   grouped 5-fold -- a harder test, ~20 % of pitchers unseen at once

Reported side by side: side-only, frontal-only, the existing `release_view` wedge,
the CV routing (the RESULT), the full-data fit of the same rule (what would ship),
and the cellwise metric oracle (upper bound only).

Outputs: fp_routing_rule.csv, fp_routing_cv_cells.csv, gate_map_deploy_fp_cv.csv
Run:  conda activate diamond; cd src\\analysis; python fp_routing_cv.py
"""
import os, sys, argparse
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config
from gate_map import score_cell
from mer_proxy_map import map_population
from angle_zone_sweep import release_view
from angle_map_2d import CIRCULAR, unwrap_circular

V = config.OBP_VALIDATION_DIR
KEY = ["metric", "source", "az", "el"]
DET = ("side", "frontal")
TIE_EPS = 1e-9
PAIRS = {  # detector -> (adopted dump, screened dump)
    "side":    ("angle_zone_pairs_redetect.csv.gz",
                "rejected_gt_pairs_detected.csv.gz"),
    "frontal": ("angle_zone_pairs_redetect_fpfrontal.csv.gz",
                "rejected_gt_pairs_detected_fpfrontal.csv.gz"),
}


_RULE_CACHE = {}


def fp_view(az, el):
    """THE FP ROUTING RULE: which foot-plant detector to use at (az, el).

    A FITTED TABLE, not a hand-written wedge -- read from fp_routing_rule.csv, which
    `main()` writes by choosing per cell on debiased fp MAE (p90 tie-break). Nearest
    grid cell wins for off-grid viewpoints. Returns "side" or "frontal", the same
    vocabulary `metrics.foot_plant_frame(view=...)` takes.

    Parallel to `angle_zone_sweep.release_view` and deliberately NOT the same
    function: the release rule is a single front wedge, this one is frontal in TWO
    arcs about 180 deg apart (front AND rear), because the stride runs along the
    camera axis at both. They agree in only 64 % of cells.

    NOT WIRED INTO DEPLOYMENT. `metrics.compute_candidates` still calls the side
    detector unconditionally; adopting this rule there is a separate decision."""
    if not _RULE_CACHE:
        p = os.path.join(V, "fp_routing_rule.csv")
        if not os.path.exists(p):
            return "side"
        for r in pd.read_csv(p).itertuples(index=False):
            _RULE_CACHE[(int(r.az), int(r.el))] = str(r.detector)
    if not _RULE_CACHE:
        return "side"
    az = int(az) % 360
    best = min(_RULE_CACHE, key=lambda c: (min(abs(c[0] - az), 360 - abs(c[0] - az))
                                           + abs(c[1] - el)))
    return _RULE_CACHE[best]


def load_fp_errors(pop):
    """Per-pitch foot-plant error, frames @360Hz, per (cell, detector). Anchor
    det_rel = release detected per view, i.e. the deployment configuration."""
    d = pd.read_csv(os.path.join(V, "event_error_map_pairs.csv.gz"))
    d = d[(d.event == "fp") & (d.anchor == "det_rel") & d.sp.isin(pop)].copy()
    d["f360"] = d.err_ms * 360.0 / 1000.0
    return d[["az", "el", "detector", "sp", "f360"]]


def choose(err, train_sp):
    """Per cell, pick the detector by debiased MAE on `train_sp`; ties -> p90.

    Returns (rule, stats) where rule[(az, el)] = 'side' | 'frontal'."""
    t = err[err.sp.isin(train_sp)]
    g = t.groupby(["az", "el", "detector"]).f360
    med = g.transform("median")
    t = t.assign(dev=(t.f360 - med).abs())
    agg = (t.groupby(["az", "el", "detector"]).dev
             .agg(mae="mean", p90=lambda s: float(np.percentile(s, 90)))
             .reset_index())
    w = agg.pivot(index=["az", "el"], columns="detector",
                  values=["mae", "p90"])
    rule, ties = {}, 0
    for (az, el), r in w.iterrows():
        ms, mf = r[("mae", "side")], r[("mae", "frontal")]
        if not np.isfinite(mf):
            rule[(az, el)] = "side"; continue
        if not np.isfinite(ms):
            rule[(az, el)] = "frontal"; continue
        if abs(ms - mf) <= TIE_EPS:
            ties += 1
            ps, pf = r[("p90", "side")], r[("p90", "frontal")]
            rule[(az, el)] = "frontal" if pf < ps - TIE_EPS else "side"
        else:
            rule[(az, el)] = "side" if ms < mf else "frontal"
    return rule, w, ties


def folds(users, scheme, seed=0):
    """Pitcher-level folds: [(train_users, test_users), ...]."""
    u = np.array(sorted(set(users)))
    if scheme == "lopo":
        return [(set(u) - {x}, {x}) for x in u]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(u))
    parts = np.array_split(idx, 5)
    return [(set(u[np.setdiff1d(idx, p)]), set(u[p])) for p in parts]


def load_pairs(pop):
    """Both detectors' per-pitch estimates, merged on (metric, source, cell, pitch)."""
    out = []
    for det, (ad, sc) in PAIRS.items():
        parts = []
        for f, src in ((ad, "adopted"), (sc, "screened")):
            p = os.path.join(V, f)
            if not os.path.exists(p):
                sys.exit(f"missing {f} -- build the detected dumps first")
            d = pd.read_csv(p)
            d["source"] = src
            parts.append(d)
        d = pd.concat(parts, ignore_index=True)
        d = d[d.session_pitch.isin(pop)]
        d = d.rename(columns={"est": f"est_{det}"})
        cols = KEY + ["session_pitch", f"est_{det}"] + (["truth"] if det == "side"
                                                        else [])
        out.append(d[cols])
    m = out[0].merge(out[1], on=KEY + ["session_pitch"], how="outer")
    return m


def score_policy(pairs, pick, codes_of, floor, strong):
    """pick: Series aligned with pairs, values 'side'/'frontal'. Returns cell table."""
    est = np.where(pick.to_numpy() == "frontal",
                   pairs.est_frontal.to_numpy(float),
                   pairs.est_side.to_numpy(float))
    d = pairs[KEY + ["session_pitch", "truth"]].assign(est=est)
    rows = []
    for k, g in d.groupby(KEY, sort=False):
        codes = g.session_pitch.map(codes_of).to_numpy()
        e = g.est.to_numpy(float)
        if k[0].strip() in CIRCULAR:
            e = unwrap_circular(e)
        s = score_cell(e, g.truth.to_numpy(float), codes, floor, strong)
        if s is None:
            continue
        rows.append(dict(zip(KEY, k), **s))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.75)
    ap.add_argument("--strong", type=float, default=0.80)
    a = ap.parse_args()

    pop = map_population()
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    md = md[md.session_pitch.isin(pop)]
    user_of = dict(zip(md.session_pitch, md.user))
    codes_of = {sp: c for sp, c in
                zip(md.session_pitch, md.user.astype("category").cat.codes)}
    print(f"population {len(pop)} pitches, {md.user.nunique()} pitchers")

    err = load_fp_errors(pop)
    print(f"fp errors: {len(err):,} rows over "
          f"{err.groupby(['az','el']).ngroups} cells x {len(DET)} detectors\n")

    # ---- the shipped rule: fit on everything -------------------------------
    full_rule, full_stats, full_ties = choose(err, set(pop))
    R = full_stats.copy()
    R.columns = [f"{a_}_{b}" for a_, b in R.columns]
    R = R.reset_index()
    R["detector"] = [full_rule[(r.az, r.el)] for r in R.itertuples(index=False)]
    R["margin_f360"] = (R.mae_side - R.mae_frontal).abs()
    R["release_rule"] = [release_view(int(r.az), int(r.el))
                         for r in R.itertuples(index=False)]
    R.to_csv(os.path.join(V, "fp_routing_rule.csv"), index=False,
             float_format="%.6g")
    nf = int((R.detector == "frontal").sum())
    print(f"[RULE fitted on all 394]  frontal in {nf} of {len(R)} cells "
          f"({100*nf/len(R):.0f}%), exact MAE ties {full_ties}")
    agree = int((R.detector == R.release_rule).sum())
    print(f"  agrees with the release_view wedge in {agree} of {len(R)} cells "
          f"({100*agree/len(R):.0f}%) -- the fp rule is NOT the release rule")

    print("\n  where the fp rule says FRONTAL (el rows x az):")
    piv = R.pivot(index="el", columns="az", values="detector")
    for el in sorted(piv.index):
        line = "".join("F" if piv.loc[el, az] == "frontal" else "."
                       for az in sorted(piv.columns))
        print(f"    el{int(el):>2}  {line}")
    print("    az " + "".join(str((a_ // 15) % 10) for a_ in sorted(piv.columns)))

    # ---- cross-validated assignment ---------------------------------------
    pairs = load_pairs(pop)
    print(f"\npairs merged: {len(pairs):,} rows, "
          f"{pairs.groupby(KEY).ngroups} metric-cells")
    sp_user = pairs.session_pitch.map(user_of)

    assign = {}
    for scheme in ("lopo", "5fold"):
        pick = pd.Series(index=pairs.index, dtype=object)
        flips = 0
        for tr, te in folds(md.user, scheme):
            rule, _, _ = choose(err[err.sp.isin(
                {s for s in pop if user_of.get(s) in tr})],
                {s for s in pop if user_of.get(s) in tr})
            flips += sum(1 for c in rule if rule[c] != full_rule[c])
            m = sp_user.isin(te).to_numpy()
            sub = pairs.loc[m, ["az", "el"]]
            pick.loc[m] = [rule[(r.az, r.el)] for r in sub.itertuples(index=False)]
        assign[scheme] = pick
        print(f"  {scheme}: per-fold cell decisions differing from the full-data "
              f"fit: {flips} of {len(folds(md.user, scheme)) * len(R)}")

    # ---- score every policy ------------------------------------------------
    policies = {
        "side_only": pd.Series("side", index=pairs.index),
        "frontal_only": pd.Series("frontal", index=pairs.index),
        "release_view_rule": pd.Series(
            [release_view(int(r.az), int(r.el))
             for r in pairs[["az", "el"]].itertuples(index=False)],
            index=pairs.index),
        "fp_rule_fitted": pd.Series(
            [full_rule[(r.az, r.el)]
             for r in pairs[["az", "el"]].itertuples(index=False)],
            index=pairs.index),
        "fp_rule_cv_lopo": assign["lopo"],
        "fp_rule_cv_5fold": assign["5fold"],
    }
    gt = pd.read_csv(os.path.join(V, "gate_map.csv"))
    onmap = gt[gt.gate_pass][KEY].copy()

    res, tables = {}, {}
    for name, pick in policies.items():
        t = score_policy(pairs, pick, codes_of, a.floor, a.strong)
        tables[name] = t
        j = onmap.merge(t[KEY + ["grade", "ccc"]], on=KEY, how="left")
        res[name] = dict(strong=int((j.grade == "strong").sum()),
                         moderate=int((j.grade == "moderate").sum()))
        print(f"  scored {name}")
    tables["fp_rule_cv_lopo"].to_csv(
        os.path.join(V, "gate_map_deploy_fp_cv.csv"), index=False,
        float_format="%.6g")

    # cellwise metric oracle, from deploy_map -- upper bound only
    orc = os.path.join(V, "gate_map_deploy_oracle.csv")
    if os.path.exists(orc):
        j = onmap.merge(pd.read_csv(orc)[KEY + ["grade"]], on=KEY, how="left")
        res["cellwise_oracle (UPPER BOUND)"] = dict(
            strong=int((j.grade == "strong").sum()),
            moderate=int((j.grade == "moderate").sum()))

    n_on = len(onmap)
    print("\n" + "=" * 92)
    print(f"FP ROUTING -- deployable cells over the {n_on} GT-map cells, n=394, "
          f"98 pitchers")
    print("=" * 92)
    hdr = f"{'policy':<34}{'strong':>8}{'moderate':>10}{'total':>8}{'kept':>8}"
    print(hdr); print("-" * len(hdr))
    for name, r in res.items():
        tot = r["strong"] + r["moderate"]
        print(f"{name:<34}{r['strong']:>8}{r['moderate']:>10}{tot:>8}"
              f"{tot/n_on:>8.3f}")
    print("-" * len(hdr))
    print("  fp_rule_cv_* is THE RESULT: the routing decision for every pitch was")
    print("  made without any data from that pitch's own pitcher. fp_rule_fitted is")
    print("  the same rule fitted on all 394 (what would ship); the gap between them")
    print("  is the optimism of fitting the routing. The cellwise oracle picks by")
    print("  METRIC OUTCOME and is not attainable -- it is printed as a ceiling.")

    out = pd.DataFrame([dict(policy=k, **v) for k, v in res.items()])
    out["total"] = out.strong + out.moderate
    out["kept"] = out.total / n_on
    out.to_csv(os.path.join(V, "fp_routing_cv_cells.csv"), index=False,
               float_format="%.6g")
    print(f"\nsaved -> {os.path.join(V, 'fp_routing_rule.csv')}")
    print(f"saved -> {os.path.join(V, 'fp_routing_cv_cells.csv')}")
    print(f"saved -> {os.path.join(V, 'gate_map_deploy_fp_cv.csv')}")


if __name__ == "__main__":
    main()
