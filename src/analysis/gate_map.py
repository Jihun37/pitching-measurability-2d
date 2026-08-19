"""The GATE map: run the FULL adoption gate at EVERY viewpoint cell, not just at one
pre-specified anchor.

Why this exists (2026-07-27, user's call). The paper's question is "which measurement
can be made from which angle", so the viewpoint is the RESULT, not a nuisance to be
fixed before scoring. The old procedure answered a different question -- "should this
metric be adopted" -- by freezing one anchor per metric and scoring there, which throws
away the field the paper is about, and it let a metric fail for a thin-zone anchor
rather than for anything about the metric. The r2 sweep stays as the SCREENING layer;
this is the layer the map is built from.

  layer 1  screening   r2 >= 0.60 per cell     angle_zone_sweep_gt_clean.csv (adopted)
                                               rejected_gt_full_sweep.csv    (screened)
  layer 2  gate        LOCO CCC >= 0.80 per cell   THIS FILE -> gate_map.csv

NO REDUNDANCY FILTER (standing instruction). A cell passes or fails on its own
measurement. Colinearity with an adopted metric is carried as an annotation column so
it can be read, never as an exclusion.

Per cell, leave-one-PITCHER-out calibration, closed form for all three models so the
whole 8-9k-cell field is affordable:
    offset   b_g = mean(t - e) over the OTHER pitchers
    ratio    k_g = mean(t) / mean(e) over the others
    linear   least squares on the others
Reported per cell: raw and calibrated CCC / MAE / pitcher-bias SD, the winning model,
and the verdict. A cell PASSES when the best out-of-fold CCC >= 0.80.

DIRECT vs CALIBRATE mirrors the metric-level probe: DIRECT = raw CCC already >= 0.80;
CALIBRATE = calibration lifts CCC over the floor AND drops both MAE and pitcher-bias
SD. A cell that only clears with a model that does not shrink pitcher bias is marked
PASS(weak) -- it is still a pass, just flagged.

Isolated passes are annotated (`spike`), never dropped: a single 15-degree cell is
narrower than our own real-video viewpoint error (median 15 degrees, holdout), so a
zone of one cell cannot be aimed at in practice even though it is a true measurement.

Population = the gt_clean pitches (gt_landmark_outlier_effect.outlier_pitches), the
same filter every paper number uses.

Inputs (build them first):
    angle_zone_sweep.py --gt-events --dump      -> angle_zone_pairs_gt.csv.gz  (16 adopted)
    rejected_gt_full_sweep.py --dump            -> rejected_gt_pairs.csv.gz    (36 screened)

Run:  conda activate diamond; cd src\\analysis; python gate_map.py
      python gate_map.py --floor 0.80 --metrics "Elbow Flex @MER [O],max_elbow_flexion"
"""
import os, sys, argparse
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config
from gt_landmark_outlier_effect import outlier_pitches
# Circular metrics must be put on one branch before ANY scoring, exactly as the
# official map does -- without it Stride Angle's r2 differs from
# angle_zone_sweep_gt_clean.csv by up to 0.98 at the wrapping views, and the
# calibration would be fitted to a split sample. Screened columns are scored
# unwrapped-free, matching how rejected_gt_full_sweep screened them.
from angle_map_2d import CIRCULAR, unwrap_circular

ADOPTED_PAIRS = os.path.join(config.OBP_VALIDATION_DIR, "angle_zone_pairs_gt.csv.gz")
SCREEN_PAIRS = os.path.join(config.OBP_VALIDATION_DIR, "rejected_gt_pairs.csv.gz")
OUT = os.path.join(config.OBP_VALIDATION_DIR, "gate_map.csv")
AZ_STEP = 15


def ccc(pred, t):
    """Lin's concordance: agreement, not just correlation."""
    if len(pred) < 3:
        return np.nan
    vp, vt = pred.var(), t.var()
    cov = ((pred - pred.mean()) * (t - t.mean())).mean()
    den = vp + vt + (pred.mean() - t.mean()) ** 2
    return float(2 * cov / den) if den > 0 else np.nan


def loco_predictions(e, t, g):
    """Leave-one-group-out predictions for all three calibration models, in closed
    form. g = integer pitcher codes. Returns {model: pred array}."""
    n = len(e)
    ng = np.bincount(g)
    # per-group sums
    se = np.bincount(g, weights=e); st = np.bincount(g, weights=t)
    see = np.bincount(g, weights=e * e); set_ = np.bincount(g, weights=e * t)
    Se, St, See, Set = se.sum(), st.sum(), see.sum(), set_.sum()
    N = float(n)

    # "others" aggregates for each group
    on = N - ng
    oe, ot = Se - se, St - st
    oee, oet = See - see, Set - set_
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_e, mean_t = oe / on, ot / on
        b_off = mean_t - mean_e                       # offset
        k_rat = np.where(np.abs(mean_e) > 1e-12, ot / oe, np.nan)
        den = oee - oe * oe / on
        slope = np.where(np.abs(den) > 1e-12, (oet - oe * ot / on) / den, np.nan)
        icept = mean_t - slope * mean_e
    bad = on < 5                                       # too little training data
    for arr in (b_off, k_rat, slope, icept):
        arr[bad] = np.nan
    return {"offset": e + b_off[g],
            "ratio": k_rat[g] * e,
            "linear": slope[g] * e + icept[g]}


def stats(pred, t, g):
    m = np.isfinite(pred) & np.isfinite(t)
    if m.sum() < 10:
        return dict(ccc=np.nan, mae=np.nan, pbsd=np.nan)
    p, tt, gg = pred[m], t[m], g[m]
    d = p - tt
    per = np.bincount(gg, weights=d) / np.maximum(np.bincount(gg), 1)
    per = per[np.bincount(gg) > 0]
    return dict(ccc=ccc(p, tt), mae=float(np.abs(d).mean()),
                pbsd=float(per.std()))


def grade_of(ccc, strong, moderate):
    """strong / moderate / limited. The map is GRADED, not gated: a single pass
    line invited exactly the criticism that the line was chosen to admit a favoured
    metric. Note CCC <= |r| always, so a grade carries an association floor inside
    it: r2 >= 0.64 at the strong contour, which is ABOVE the 0.60 screen, but only
    r2 >= 0.5625 at moderate, which is BELOW it.
    ⚠ CORRECTED 2026-07-29 -- this docstring used to claim the two layers "can only
    disagree in one direction (association without agreement, never the reverse)".
    That is FALSE at the moderate contour, and it had propagated into a draft of
    paper Sec. III-D. gate_pass is (grade != "limited"), i.e. CCC alone; r2 is
    reported, never applied as an inclusion filter. Both directions occur in the
    frozen map: 20 cells hold association without agreement (the `hatch` flag),
    and 57 of the 400 moderate cells, over 20 rows, clear CCC 0.75 with r2 < 0.60."""
    if not np.isfinite(ccc):
        return "limited"
    return "strong" if ccc >= strong else ("moderate" if ccc >= moderate
                                           else "limited")


MODELS = ("offset", "ratio", "linear")

# ⚠ DEFAULT TRUE, and it must stay that way. Thirteen scripts import score_cell directly
# (event_tolerance_map, event_tolerance, event_tolerance_full, composite_event_offsets,
# mer_proxy_map, internal_peak_sweep, fp_routing_cv, hss_anchor_probe, stride_plateau_2d,
# motion_onset_candidates, setup_anchor_regression, adopted_tolerance_parity,
# elbow_mer_jitter). They never see gate_map's CLI, so a flag-driven default of False
# would silently score them under the RETIRED leaky selection while gate_map.csv itself
# was nested -- a mixed-generation map, which is exactly what the freeze exists to
# prevent. It happened once, on 2026-08-08, and event_tolerance_map.csv had to be rebuilt.
# --legacy-selection turns it off, for diagnosis only.
NESTED = True


def nested_predictions(e, t, g):
    """NESTED correction-model selection (adopted 2026-08-08, user's call).

    The plain path below picks the winning model by the argmax of the out-of-fold CCC
    taken over ALL pitchers -- including the held-out pitcher's own out-of-fold
    prediction. Parameter fitting was pitcher-blind; MODEL SELECTION WAS NOT. Measured
    before the switch: nested costs 24 graded cells of 1,500 and moves no row across the
    retained line, which is smaller than the cluster-bootstrap interval on the same count
    ([1481, 1533]). The leak was removed outright rather than reported as a limitation.

        outer   hold pitcher p out entirely
        inner   leave-one-pitcher-out WITHIN the rest, score each model, take the argmax
        apply   fit the chosen model on everything except p, predict p

    All three models are closed form, so the inner leave-two-out fits are rank-1 downdates
    of the same per-pitcher sufficient statistics and the whole G x G grid is vectorised.

    Returns (pred, fold_choice) with fold_choice giving m*(p) per pitcher."""
    G = int(g.max()) + 1
    ng = np.bincount(g, minlength=G).astype(float)
    se = np.bincount(g, weights=e, minlength=G)
    st = np.bincount(g, weights=t, minlength=G)
    see = np.bincount(g, weights=e * e, minlength=G)
    set_ = np.bincount(g, weights=e * t, minlength=G)
    N, Se, St, See, Set = float(len(e)), se.sum(), st.sum(), see.sum(), set_.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        on = N - ng[:, None] - ng[None, :]
        oe = Se - se[:, None] - se[None, :]
        ot = St - st[:, None] - st[None, :]
        oee = See - see[:, None] - see[None, :]
        oet = Set - set_[:, None] - set_[None, :]
        mean_e, mean_t = oe / on, ot / on
        b_off = mean_t - mean_e
        k_rat = np.where(np.abs(mean_e) > 1e-12, ot / oe, np.nan)
        den = oee - oe * oe / on
        slope = np.where(np.abs(den) > 1e-12, (oet - oe * ot / on) / den, np.nan)
        icept = mean_t - slope * mean_e
    bad = on < 5
    for arr in (b_off, k_rat, slope, icept):
        arr[bad] = np.nan
        np.fill_diagonal(arr, np.nan)

    q = g
    inner = {"offset": e[:, None] + b_off[:, q].T,
             "ratio": k_rat[:, q].T * e[:, None],
             "linear": slope[:, q].T * e[:, None] + icept[:, q].T}
    valid = (g[:, None] != np.arange(G)[None, :])
    tt = np.where(valid, t[:, None], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        mt, vt = np.nanmean(tt, axis=0), np.nanvar(tt, axis=0)
        inner_ccc = np.full((len(MODELS), G), -9.0)
        for mi, mo in enumerate(MODELS):
            pp = np.where(valid, inner[mo], np.nan)
            mp, vp = np.nanmean(pp, axis=0), np.nanvar(pp, axis=0)
            cov = np.nanmean((pp - mp) * (tt - mt), axis=0)
            dd = vp + vt + (mp - mt) ** 2
            c = np.where(dd > 0, 2 * cov / dd, np.nan)
            inner_ccc[mi] = np.where(np.isfinite(c), c, -9.0)

    pick = np.argmax(inner_ccc, axis=0)

    # ONE MODEL PER CELL -- the modal blind choice, applied to EVERY fold.
    #
    # ⚠ Do not revert to per-fold models. Letting each outer fold keep its own m*(p)
    # assembles a prediction vector in which different pitchers carry different affine
    # transforms, which agrees with the truth worse than any single model does, and it
    # makes the cell violently unstable whenever two models are near-tied: measured
    # 2026-08-08, a 1e-5 perturbation of the inputs moved 473 of 5,544 cells by more than
    # 0.01 CCC and 124 by more than 0.1, worst case 0.64, and the worst cells were exactly
    # the near-ties (torso_lateral_tilt_mer 180/15 split 50 offset / 48 ratio, CCC swinging
    # -0.012 <-> -0.653). The same comparison on the pre-nested map differed by 1.1e-05
    # with zero cells changing grade.
    #
    # Residual dependence, stated exactly: each m*(p) is chosen strictly blind to p, but
    # the MODE is taken over all folds, so a pitcher influences the model family through
    # the other folds' inner fits. The fitted PARAMETERS stay leave-one-pitcher-out, so
    # the only non-blind channel is a single ternary choice informed by 98 pitchers --
    # far weaker than the retired rule, which put the held-out pitcher's own out-of-fold
    # prediction directly into the selection statistic.
    mode = int(np.bincount(pick, minlength=len(MODELS)).argmax())
    outer = loco_predictions(e, t, g)
    return outer[MODELS[mode]], pick


def score_cell(e, t, g, floor, strong=0.80):
    ok = np.isfinite(e) & np.isfinite(t)
    e, t, g = e[ok], t[ok], g[ok]
    if len(e) < 30 or e.std() < 1e-9 or t.std() < 1e-9:
        return None
    gu, g = np.unique(g, return_inverse=True)
    raw = stats(e, t, g)
    preds = loco_predictions(e, t, g)
    cal = {mo: stats(p, t, g) for mo, p in preds.items()}
    folds = None
    if NESTED:
        npred, pick = nested_predictions(e, t, g)
        b = stats(npred, t, g)
        folds = np.bincount(pick, minlength=3)
        # the cell no longer has ONE winning model; report the modal fold choice
        best_mo = MODELS[int(np.argmax(folds))]
    else:
        best_mo = max(cal, key=lambda mo: (cal[mo]["ccc"]
                                           if np.isfinite(cal[mo]["ccc"]) else -9))
        b = cal[best_mo]
    r = float(np.corrcoef(e, t)[0, 1] ** 2)
    grade = grade_of(b["ccc"], strong, floor)
    if grade == "limited":
        verdict = "fail"
    elif raw["ccc"] >= floor:
        verdict = "DIRECT"
    elif b["mae"] < raw["mae"] and b["pbsd"] < raw["pbsd"]:
        verdict = "CALIBRATE"
    else:
        # clears the line, but the calibration does not shrink per-pitcher bias --
        # it will not generalise to a new pitcher's individual tracking
        verdict = "PASS(weak)"
    return dict(n=len(e), n_pitcher=len(gu), r2=r,
                raw_ccc=raw["ccc"], raw_mae=raw["mae"], raw_pbsd=raw["pbsd"],
                ccc=b["ccc"], mae=b["mae"], pbsd=b["pbsd"], model=best_mo,
                ccc_offset=cal["offset"]["ccc"], ccc_ratio=cal["ratio"]["ccc"],
                ccc_linear=cal["linear"]["ccc"], grade=grade, verdict=verdict,
                # under --nested the model is chosen PER FOLD, so `model` above is the
                # modal choice and these are the fold counts behind it
                folds_offset=int(folds[0]) if folds is not None else -1,
                folds_ratio=int(folds[1]) if folds is not None else -1,
                folds_linear=int(folds[2]) if folds is not None else -1,
                # the only direction the two layers can disagree in
                hatch=bool(r >= 0.60 and b["ccc"] < floor),
                gate_pass=grade != "limited")


def load_pairs(path, source, keep):
    if not os.path.exists(path):
        print(f"  !! missing {os.path.basename(path)} -- skipping {source}")
        return None
    d = pd.read_csv(path)
    d = d[d.session_pitch.isin(keep)]
    d["source"] = source
    return d


def mark_spikes(g):
    """Annotate passing cells with no passing 15-degree neighbour (azimuth wraps)."""
    P = {(int(r.az), int(r.el)) for r in g.itertuples(index=False) if r.gate_pass}
    els = sorted({int(x) for x in g.el.unique()})
    out = []
    for r in g.itertuples(index=False):
        if not r.gate_pass:
            out.append(False); continue
        az, el = int(r.az), int(r.el)
        nb = [((az + d) % 360, el) for d in (-AZ_STEP, AZ_STEP)]
        i = els.index(el)
        nb += [(az, els[j]) for j in (i - 1, i + 1) if 0 <= j < len(els)]
        out.append(not any(p in P for p in nb))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.75,
                    help="MODERATE line: cells below it are not on the map")
    ap.add_argument("--strong", type=float, default=0.80, help="STRONG line")
    ap.add_argument("--metrics", default=None, help="comma list, for a quick look")
    ap.add_argument("--legacy-selection", action="store_true",
                    help="DIAGNOSIS ONLY. Restore the pre-2026-08-08 behaviour, in which "
                         "the winning correction model is chosen by an argmax over "
                         "out-of-fold CCC that includes the held-out pitcher -- i.e. "
                         "model selection is NOT pitcher-blind. The adopted protocol is "
                         "nested and is the default; see nested_predictions().")
    ap.add_argument("--out-suffix", default="",
                    help="write gate_map<suffix>.csv instead, so a second floor "
                         "can be scored without overwriting the frozen table")
    ap.add_argument("--screen-pairs", default=None,
                    help="read a NON-official screened dump (e.g. "
                         "rejected_gt_pairs_detected.csv.gz). The adopted dump is "
                         "then skipped unless --adopted-pairs is given too, so the "
                         "run scores the screened layer alone.")
    ap.add_argument("--adopted-pairs", default=None,
                    help="likewise for the adopted dump")
    ap.add_argument("--pop-frozen", action="store_true",
                    help="lock the population to the frozen map's 394 ids "
                         "(mer_proxy_map.map_population) instead of deriving it "
                         "from whichever dumps are present. Needed whenever only "
                         "one dump is loaded, or a comparison would silently be "
                         "scored on 395 pitches against the map's 394.")
    a = ap.parse_args()
    global NESTED
    NESTED = not a.legacy_selection
    print("correction-model selection: "
          + ("NESTED (pitcher-blind, adopted 2026-08-08)" if NESTED
             else "LEGACY per-cell argmax over all out-of-fold predictions "
                  "-- NOT pitcher-blind, diagnosis only"))
    out_path = OUT.replace(".csv", f"{a.out_suffix}.csv")
    adopted_p = (os.path.join(config.OBP_VALIDATION_DIR, a.adopted_pairs)
                 if a.adopted_pairs else
                 (None if a.screen_pairs else ADOPTED_PAIRS))
    screen_p = (os.path.join(config.OBP_VALIDATION_DIR, a.screen_pairs)
                if a.screen_pairs else SCREEN_PAIRS)

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    user_of = dict(zip(md.session_pitch, md.user))
    bad = outlier_pitches()
    keep = {sp for sp in md.session_pitch if sp not in bad}
    print(f"gt_clean population: {len(keep)} of {len(md)} pitches "
          f"({len(bad)} landmark outliers removed)")
    if a.pop_frozen:
        from mer_proxy_map import map_population
        keep &= set(map_population())
        print(f"  --pop-frozen: locked to the frozen map's {len(keep)} ids")

    parts = [p for p in (load_pairs(adopted_p, "adopted", keep)
                         if adopted_p else None,
                         load_pairs(screen_p, "screened", keep)) if p is not None]
    if not parts:
        sys.exit("no pair dumps found -- build them first (see module docstring)")
    # The two sweeps have different pitch-inclusion gates (angle_zone_sweep also
    # requires pkh and rel > fp+1), so gt_clean alone still leaves the adopted rows
    # on 394 pitches and the screened rows on 395. Intersect, so EVERY metric in the
    # map is scored on exactly the same pitches and a cell count is comparable
    # across rows.
    common = set.intersection(*[set(p.session_pitch) for p in parts])
    for i, p in enumerate(parts):
        n0 = p.session_pitch.nunique()
        parts[i] = p[p.session_pitch.isin(common)]
        if n0 != len(common):
            print(f"  {p.source.iloc[0]}: {n0} -> {len(common)} pitches "
                  f"(intersected with the other dump)")
    d = pd.concat(parts, ignore_index=True)
    if a.metrics:
        want = {m.strip() for m in a.metrics.split(",")}
        d = d[d.metric.isin(want)]
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
                s = score_cell(e_cell, t_all[m], codes[m], a.floor, a.strong)
                if s is None:
                    continue
                rows.append(dict(metric=metric, source=src, az=int(az),
                                 el=int(el), **s))
        print(f"  scored {metric}")

    out = pd.DataFrame(rows)
    out["spike"] = False
    for (metric, src), g in out.groupby(["metric", "source"]):
        out.loc[g.index, "spike"] = mark_spikes(g)
    # BUG FIX 2026-07-27: this wrote OUT, not out_path, so --out-suffix computed a
    # filename and then overwrote the frozen gate_map.csv anyway -- the exact
    # accident the flag exists to prevent.
    out.to_csv(out_path, index=False, float_format="%.6g")
    print(f"\nsaved -> {out_path}")

    print("\n" + "=" * 100)
    print(f"GRADED MAP  (strong CCC>={a.strong:.2f} / moderate >={a.floor:.2f}, "
          f"per cell, GT events, gt_clean)")
    print("=" * 100)
    print(f"{'metric':40}{'src':>9}{'gate cells':>11}{'r2 cells':>10}"
          f"{'best cell':>11}{'best CCC':>10}{'spikes':>8}")
    print("-" * 100)
    summ = []
    for (metric, src), g in out.groupby(["metric", "source"]):
        ok = g[g.gate_pass]
        r2c = int((g.r2 >= 0.60).sum())
        if len(ok):
            b = ok.loc[ok.ccc.idxmax()]
            cell = f"{int(b.az)}/{int(b.el)}"; cc = f"{b.ccc:.3f}"
        else:
            b = g.loc[g.ccc.idxmax()] if g.ccc.notna().any() else None
            cell = "-" if b is None else f"({int(b.az)}/{int(b.el)})"
            cc = "-" if b is None else f"({b.ccc:.3f})"
        print(f"{metric:40}{src:>9}{len(ok):>11}{r2c:>10}{cell:>11}{cc:>10}"
              f"{int(g.spike.sum()):>8}")
        summ.append(dict(metric=metric, source=src, gate_cells=len(ok),
                         r2_cells=r2c, spikes=int(g.spike.sum())))
    s = pd.DataFrame(summ)
    print("-" * 100)
    print(f"metrics with at least one gate cell: {(s.gate_cells > 0).sum()} "
          f"of {len(s)}")
    print(f"total gate cells {int(s.gate_cells.sum())} vs r2 cells "
          f"{int(s.r2_cells.sum())} "
          f"({100 * s.gate_cells.sum() / max(1, s.r2_cells.sum()):.0f}% of the "
          f"screening layer survives the gate)")


if __name__ == "__main__":
    main()
