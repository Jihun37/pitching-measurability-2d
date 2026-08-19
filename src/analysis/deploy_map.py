"""THE DEPLOYABLE MAP: what survives when the events are DETECTED, not given.

The graded map (`gate_map.csv`) is built on OBP landmarks, so it answers
"is this measurable from this viewpoint" -- an ORACLE claim. This file answers the
other question, "can we actually deliver it from a phone", by re-detecting release
and foot plant on every projected view and re-scoring the identical gate.

THE FROZEN MAP IS NEVER TOUCHED. Everything here is written to its own files.

Foot plant has NO adopted routing rule (open decision), so it is not guessed:
both detectors are run over the whole grid and the two policies are composed
per cell afterwards --

    fp_side      the side detector everywhere (what the code does today)
    fp_frontal   the frontal detector everywhere
    rule         FIXED, angle-based: frontal inside release_view()'s wedge
                 (az 60-120, el<=15), side elsewhere. This is the only fixed
                 angle rule that exists in the codebase; it is the RELEASE rule
                 reused, and the oracle gap below is the price of that reuse.
    oracle       per cell, whichever detector scores better. An UPPER BOUND, not
                 a deliverable -- it needs the answer to pick the detector.

Release always follows the adopted `release_view()` rule in all four.

Two independent routes to a deployability verdict, and comparing them is the point:
  EMPIRICAL   re-score the gate on detected events (the four policies above)
  PREDICTED   per cell, the metric's event tolerance (event_tolerance_cells.csv,
              +-12 f) against the detector's achieved error at that same cell
              (event_error_map.csv). Deployable when tolerance >= achieved error.
The agreement between them says whether "tolerance" is a usable abstraction at all
or just a story.

Inputs (build them first):
    angle_zone_sweep.py --redetect --dump  [--fp-strategy frontal]
    rejected_gt_full_sweep.py --dump --detected  [--fp-strategy frontal]
    event_tolerance_full.py
Outputs: gate_map_deploy_<policy>.csv, deploy_map_cells.csv, deploy_map_summary.csv,
         figure via viz/fig_deploy_map.py
Run:  conda activate diamond; cd src\\analysis; python deploy_map.py
      python deploy_map.py --score      (re-run the gate on the detected dumps)
"""
import os, sys, argparse, subprocess
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config
from angle_zone_sweep import release_view

V = config.OBP_VALIDATION_DIR
GATE = os.path.join(V, "gate_map.csv")
TOL = os.path.join(V, "event_tolerance_cells.csv")
ERR = os.path.join(V, "event_error_map.csv")
RANK = {"limited": 0, "moderate": 1, "strong": 2}
KEY = ["metric", "source", "az", "el"]

# policy -> (adopted pairs, screened pairs)
DUMPS = {
    "fp_side":    ("angle_zone_pairs_redetect.csv.gz",
                   "rejected_gt_pairs_detected.csv.gz"),
    "fp_frontal": ("angle_zone_pairs_redetect_fpfrontal.csv.gz",
                   "rejected_gt_pairs_detected_fpfrontal.csv.gz"),
}


def score(policy, ad, sc):
    """Delegate to gate_map.py so the deployed cells are scored by the SAME gate
    the frozen map uses -- CLI, not a re-implementation."""
    out = os.path.join(V, f"gate_map_deploy_{policy}.csv")
    cmd = [sys.executable, "gate_map.py", "--adopted-pairs", ad,
           "--screen-pairs", sc, "--out-suffix", f"_deploy_{policy}",
           "--pop-frozen"]
    print(f"  scoring {policy} ...")
    r = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit(f"gate_map failed for {policy}")
    return out


def compose(side, frontal):
    """rule = fixed angle routing; oracle = per-cell best of the two."""
    m = side.merge(frontal, on=KEY, suffixes=("_s", "_f"), how="outer")
    use_f_rule = m.apply(lambda r: release_view(int(r.az), int(r.el)) == "frontal",
                         axis=1)
    better_f = (m.ccc_f.fillna(-9) > m.ccc_s.fillna(-9))
    out = {}
    for name, pick_f in (("rule", use_f_rule), ("oracle", better_f)):
        rows = []
        for col in ("ccc", "r2", "grade", "verdict", "gate_pass", "spike", "n"):
            a, b = m.get(f"{col}_s"), m.get(f"{col}_f")
            if a is None:
                continue
            rows.append(pd.Series(np.where(pick_f, b, a), name=col))
        d = pd.concat([m[KEY]] + rows, axis=1)
        d["fp_detector"] = np.where(pick_f, "frontal", "side")
        out[name] = d
    return out


def achieved_error(pol_fp):
    """Achieved detector error per (cell, event), in frames @360Hz, DEBIASED.

    Debiased on purpose. `tol` is measured by re-scoring the whole gate at each
    shifted anchor, and that re-scoring includes the LOCO calibration -- so a
    CONSTANT event-frame offset is already absorbed and only the spread has to fit
    inside the tolerance band. `event_error_map.csv`'s `mae_f360` is the RAW mean
    |error|, which would double-count that bias, so the spread is recomputed here
    from the per-pitch dump on the frozen 394.

    Release follows the adopted `release_view()` rule; foot plant follows the
    policy being scored; peak knee height is read off its own detector (it is
    ~0.2 f and never binding)."""
    from mer_proxy_map import map_population
    pop = map_population()
    d = pd.read_csv(os.path.join(V, "event_error_map_pairs.csv.gz"))
    d = d[d.sp.isin(pop)]
    d["f360"] = d.err_ms * 360.0 / 1000.0

    def spread(sub):
        g = sub.groupby(["az", "el"]).f360
        return (g.apply(lambda s: float(np.abs(s - s.median()).mean()))
                if len(sub) else pd.Series(dtype=float))

    rel_s = spread(d[(d.event == "rel") & (d.detector == "side")])
    rel_f = spread(d[(d.event == "rel") & (d.detector == "frontal")])
    rel_rule = pd.Series({k: (rel_f.get(k, np.nan)
                              if release_view(k[0], k[1]) == "frontal"
                              else rel_s.get(k, np.nan)) for k in rel_s.index})
    fp = spread(d[(d.event == "fp") & (d.detector == pol_fp)
                  & (d.anchor == "det_rel")])
    pk = spread(d[(d.event == "pkh") & (d.anchor == f"det_fp_{pol_fp}")])

    # per-cell DISTRIBUTION of |debiased error|, for the coverage criterion below
    def dist(sub):
        out = {}
        for k, g in sub.groupby(["az", "el"]):
            v = g.f360.to_numpy(float)
            out[k] = np.abs(v - np.median(v))
        return out
    dists = {
        "rel": dist(d[(d.event == "rel") & (d.detector == "side")]),
        "fp": dist(d[(d.event == "fp") & (d.detector == pol_fp)
                     & (d.anchor == "det_rel")]),
        "pkh": dist(d[(d.event == "pkh") & (d.anchor == f"det_fp_{pol_fp}")])}
    dists["rel"].update(dist(d[(d.event == "rel") & (d.detector == "frontal")
                               & d.apply(lambda r: release_view(r.az, r.el)
                                         == "frontal", axis=1)]))
    return ({"rel": rel_rule, "fp": fp,
             "pkh": pk if len(pk) else rel_rule * np.nan}, dists)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true",
                    help="re-run gate_map on the detected dumps (slow); omit to "
                         "reuse gate_map_deploy_*.csv")
    a = ap.parse_args()

    maps = {}
    for pol, (ad, sc) in DUMPS.items():
        p = os.path.join(V, f"gate_map_deploy_{pol}.csv")
        if a.score or not os.path.exists(p):
            for f in (ad, sc):
                if not os.path.exists(os.path.join(V, f)):
                    sys.exit(f"missing dump {f} -- see the module docstring")
            score(pol, ad, sc)
        maps[pol] = pd.read_csv(p)
        print(f"  {pol}: {len(maps[pol])} cells, "
              f"{int((maps[pol].grade=='strong').sum())} strong")

    comp = compose(maps["fp_side"], maps["fp_frontal"])
    for name, d in comp.items():
        d.to_csv(os.path.join(V, f"gate_map_deploy_{name}.csv"), index=False,
                 float_format="%.6g")
        maps[name] = d
    gt = pd.read_csv(GATE)

    # ---- per-cell join -----------------------------------------------------
    # The ADOPTED deployed policy is the FP-specific LOPO routing
    # (analysis/fp_routing_cv.py). `rule` (the release_view heuristic reused for fp)
    # and the cellwise metric `oracle` are kept only as comparison columns -- the
    # oracle picks by metric outcome and is a ceiling, never a policy.
    cv_p = os.path.join(V, "gate_map_deploy_fp_cv.csv")
    if os.path.exists(cv_p):
        maps["lopo"] = pd.read_csv(cv_p)
    else:
        print("  !! gate_map_deploy_fp_cv.csv missing -- run fp_routing_cv.py")
        maps["lopo"] = maps["rule"]
    cells = gt[KEY + ["ccc", "grade", "gate_pass"]].rename(
        columns={"ccc": "ccc_gt", "grade": "grade_gt", "gate_pass": "onmap_gt"})
    for pol in ("fp_side", "fp_frontal", "rule", "lopo", "oracle"):
        d = maps[pol][KEY + ["ccc", "grade"]].rename(
            columns={"ccc": f"ccc_{pol}", "grade": f"grade_{pol}"})
        cells = cells.merge(d, on=KEY, how="left")

    # ---- predicted deployability from tolerance vs achieved error ----------
    if os.path.exists(TOL):
        tol = pd.read_csv(TOL)
        err, dists = achieved_error("side")
        rec = []
        for r in tol.itertuples(index=False):
            if r.reference == "lockstep":
                continue
            ach = err.get(r.reference)
            e = (ach.get((int(r.az), int(r.el)), np.nan)
                 if ach is not None else np.nan)
            # COVERAGE: share of pitches whose |debiased anchor error| fits inside
            # the tolerance band. A fairer comparator than tol - MAE, because the
            # band has to hold for the pitch, not for the average pitch.
            dv = dists.get(r.reference, {}).get((int(r.az), int(r.el)))
            cov = (float(np.mean(dv <= r.tol_map)) if dv is not None
                   and r.tol_map >= 0 and len(dv) else np.nan)
            rec.append(dict(metric=r.metric, source=r.source, az=r.az, el=r.el,
                            reference=r.reference, tol_map=r.tol_map,
                            achieved=e, margin=r.tol_map - e, coverage=cov))
        T = pd.DataFrame(rec)
        # a cell is predicted deployable when EVERY reference it uses is covered
        worst = (T.groupby(KEY)
                  .agg(min_margin=("margin", "min"),
                       min_coverage=("coverage", "min"),
                       worst_ref=("margin", lambda s: T.loc[s.idxmin(), "reference"]
                                  if s.notna().any() else ""))
                  .reset_index())
        cells = cells.merge(worst, on=KEY, how="left")
        T.to_csv(os.path.join(V, "deploy_tolerance_vs_error.csv"), index=False,
                 float_format="%.6g")
    else:
        cells["min_margin"] = np.nan; cells["min_coverage"] = np.nan
        cells["worst_ref"] = ""
        print("  !! event_tolerance_cells.csv missing -- predicted layer skipped")

    cells.to_csv(os.path.join(V, "deploy_map_cells.csv"), index=False,
                 float_format="%.6g")

    # ---- per-metric summary ------------------------------------------------
    rows = []
    on = cells[cells.onmap_gt]
    for (metric, src), g in cells.groupby(["metric", "source"]):
        o = g[g.onmap_gt]
        def cnt(pol, gr):
            return int((o[f"grade_{pol}"] == gr).sum())
        gt_strong = int((o.grade_gt == "strong").sum())
        gt_mod = int((o.grade_gt == "moderate").sum())
        rows.append(dict(
            metric=metric, source=src,
            oracle_gate_cells=len(o), gt_strong=gt_strong, gt_moderate=gt_mod,
            dep_strong=cnt("lopo", "strong"),
            dep_moderate=cnt("lopo", "moderate"),
            dep_total=cnt("lopo", "strong") + cnt("lopo", "moderate"),
            dep_strong_rule=cnt("rule", "strong"),
            dep_moderate_rule=cnt("rule", "moderate"),
            dep_strong_oracle=cnt("oracle", "strong"),
            dep_moderate_oracle=cnt("oracle", "moderate"),
            dep_strong_side=cnt("fp_side", "strong"),
            dep_strong_frontal=cnt("fp_frontal", "strong"),
            d_strong_vs_rule=cnt("lopo", "strong") - cnt("rule", "strong"),
            d_moderate_vs_rule=cnt("lopo", "moderate") - cnt("rule", "moderate"),
            d_total_vs_rule=(cnt("lopo", "strong") + cnt("lopo", "moderate"))
                            - (cnt("rule", "strong") + cnt("rule", "moderate")),
            retention=(cnt("lopo", "strong") + cnt("lopo", "moderate")) /
                      max(1, len(o)),
            retention_strong=cnt("lopo", "strong") / max(1, gt_strong)
                             if gt_strong else np.nan,
            rule_vs_oracle=(cnt("oracle", "strong") + cnt("oracle", "moderate"))
                           - (cnt("lopo", "strong") + cnt("lopo", "moderate")),
            worst_ref=(o.worst_ref.mode().iloc[0]
                       if "worst_ref" in o and o.worst_ref.notna().any()
                       and len(o.worst_ref.mode()) else ""),
            min_margin_med=(float(o.min_margin.median())
                            if "min_margin" in o else np.nan)))
    S = pd.DataFrame(rows).sort_values("oracle_gate_cells", ascending=False)
    S.to_csv(os.path.join(V, "deploy_map_summary.csv"), index=False,
             float_format="%.6g")
    report(S, cells)


def report(S, cells):
    pd.set_option("display.width", 240)
    print("\n" + "=" * 122)
    print("DEPLOYABLE MAP  --  GT-event (oracle) map vs the same gate on DETECTED "
          "release + foot plant, n=394")
    print("=" * 122)
    hdr = (f"{'metric':<40}{'GT S':>6}{'GT M':>6}{'GT tot':>7}"
           f"{'dep S':>7}{'dep M':>7}{'dep tot':>8}{'keep':>6}{'keepS':>7}"
           f"{'vs heur S/M/tot':>18}{'fail ev':>9}")
    print(hdr); print("-" * len(hdr))
    for r in S.itertuples(index=False):
        vs = (str(r.d_strong_vs_rule) + "/" + str(r.d_moderate_vs_rule)
              + "/" + str(r.d_total_vs_rule))
        ks = r.retention_strong if np.isfinite(r.retention_strong) else 0.0
        print(f"{r.metric:<40}{r.gt_strong:>6}{r.gt_moderate:>6}"
              f"{r.oracle_gate_cells:>7}{r.dep_strong:>7}{r.dep_moderate:>7}"
              f"{r.dep_total:>8}{r.retention:>6.2f}{ks:>7.2f}"
              f"{vs:>18}{str(r.worst_ref):>9}")
    print("-" * len(hdr))
    tot_o = int(S.oracle_gate_cells.sum()); tot_s = int(S.gt_strong.sum())
    keep = int(S.dep_total.sum())
    tvs = (str(int(S.d_strong_vs_rule.sum())) + "/"
           + str(int(S.d_moderate_vs_rule.sum())) + "/"
           + str(int(S.d_total_vs_rule.sum())))
    print(f"{'TOTAL':<40}{tot_s:>6}{int(S.gt_moderate.sum()):>6}{tot_o:>7}"
          f"{int(S.dep_strong.sum()):>7}{int(S.dep_moderate.sum()):>7}"
          f"{keep:>8}{keep/max(1,tot_o):>6.2f}"
          f"{S.dep_strong.sum()/max(1,tot_s):>7.2f}{tvs:>18}")
    print("")
    print("  DEPLOYED POLICY = FP-specific LOPO routing (fp_routing_cv.py).")
    print("  'vs heur' = change against the release_view heuristic reused for fp.")
    print("")
    print("  GT S/M/tot = the frozen map cells. dep S/M/tot = the same cells under")
    print("  DETECTED events with FP-specific LOPO routing -- each held-out pitcher")
    print("  did not contribute to its own routing selection. keep = dep tot/GT tot,")
    print("  keepS = dep S/GT S. fail ev = the reference with the smallest tolerance")
    print("  margin (diagnostic only; tolerance is not a deployability claim).")

    print("\n" + "=" * 122)
    print("LOPO ROUTING vs the METRIC-ORACLE CEILING -- what routing cannot reach")
    print("=" * 122)
    d = cells[cells.onmap_gt].copy()
    for pol in ("lopo", "oracle"):
        d[f"r_{pol}"] = d[f"grade_{pol}"].map(RANK).fillna(0)
    gap = d[d.r_oracle > d.r_lopo]
    print(f"  cells where the metric oracle beats LOPO routing: {len(gap)} "
          f"of {len(d)} ({100*len(gap)/max(1,len(d)):.0f}%)")
    if len(gap):
        by = gap.groupby("el").size()
        print(f"  by elevation: " + "  ".join(f"el{int(k)}:{v}" for k, v in by.items()))
        print(f"  by azimuth (top): " +
              "  ".join(f"az{int(k)}:{v}" for k, v in
                        gap.groupby("az").size().sort_values(ascending=False)
                        .head(8).items()))

    if "min_margin" in cells:
        print("\n" + "=" * 122)
        print("PREDICTED (tolerance >= achieved error) vs EMPIRICAL (re-scored gate)")
        print("=" * 122)
        d = cells[cells.onmap_gt & cells.min_margin.notna()].copy()
        d["emp"] = d.grade_lopo.isin(("strong", "moderate"))
        print(f"  cells with both verdicts: {len(d)}")
        for name, pred in (("margin >= 0  (tol vs debiased MAE)",
                            d.min_margin >= 0),
                           ("coverage >= 0.90 (share of pitches inside the band)",
                            d.min_coverage >= 0.90)):
            tp = int((pred & d.emp).sum()); fp_ = int((pred & ~d.emp).sum())
            fn = int((~pred & d.emp).sum()); tn = int((~pred & ~d.emp).sum())
            n = max(1, len(d))
            print(f"")
            print(f"  criterion: {name}")
            print(f"    predicted OK & survived   {tp:>5}   "
                  f"predicted OK & died     {fp_:>5}")
            print(f"    predicted FAIL & survived {fn:>5}   "
                  f"predicted FAIL & died   {tn:>5}")
            print(f"    agreement {100*(tp+tn)/n:.0f}%")
        print("")
        print("  WHY IT IS ONLY THIS GOOD, and it is not a bug in either layer:")
        print("  tolerance is measured by shifting EVERY pitch by the same k, which")
        print("  the LOCO calibration then partly absorbs. The real detector error is")
        print("  RANDOM per pitch, which calibration cannot absorb -- it adds")
        print("  variance instead of bias. So a systematic-offset tolerance")
        print("  SYSTEMATICALLY OVER-STATES what a noisy anchor survives, and the")
        print("  empirical deployed map, not the tolerance band, is the verdict.")
    print(f"\nsaved -> {os.path.join(V, 'deploy_map_cells.csv')}")
    print(f"saved -> {os.path.join(V, 'deploy_map_summary.csv')}")


if __name__ == "__main__":
    main()
