"""Diamond - task 6 of the event rebuild: IS OUR FOOT PLANT AIMED AT THE WRONG
LANDMARK?

OBP ships two foot-plant landmarks and we have only ever scored against one:

    fp_10_time    10 % load        the foot first accepts weight
    fp_100_time   100 % load       the foot is fully loaded          <- our target

They are 7 frames apart at the median (19 ms, SD 5.1, p95 19 f, n=402), and only
8 % of pitches have them within 3 frames. That is the same order as our detector's
debiased error (6.09 f360), so "our foot plant is noisy" and "our foot plant is
aimed at the other landmark" predict almost the same summary statistic. Nobody has
separated them. This is a DEFINITION check, not a new event -- neither landmark is
new and nothing here adds a detector.

Two independent questions, and they can disagree:

  A  WHAT DOES THE DETECTOR HIT?   Write det = fp10 + alpha * gap, with
     gap = fp100 - fp10. alpha = 0 means we are actually finding fp_10 and the
     "error" is a constant definitional offset; alpha = 1 means we really are
     finding fp_100 and the error is variance. alpha is the slope of
     (det - fp10) on gap, which is exactly 1 when the detector's noise is
     independent of the gap and 0 when the detector rides fp_10.
     Reconstructed from event_error_map_pairs.csv.gz (err_ms = det - fp100), so
     this half needs no new sweep.

  B  WHAT DO THE METRICS WANT?   Re-read all 19 fp-referencing map rows anchored at
     fp_10 instead and re-score the gate. If a poi column is defined at 10 % load,
     its CCC will jump; if it is defined at 100 %, it will fall. This is the only
     test that says which landmark each TRUTH COLUMN means, as opposed to which one
     our detector happens to reach. Needs the two sweeps re-run with --fp-target
     fp10 and is driven by --part b.

Outputs: fp_target_detector.csv (A), fp_target_map.csv (B)
Run:  conda activate diamond; cd src\\analysis
      python fp_target_check.py --part a
      python fp_target_check.py --part b     (after the fp10 sweeps + gate_map)
"""
import os, sys, argparse
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config
from obp_gt_events import load_gt_events
from mer_proxy_map import map_population

V = config.OBP_VALIDATION_DIR
PAIRS = os.path.join(V, "event_error_map_pairs.csv.gz")
F360 = 360.0
PHONE_F = 3.0          # 1 frame at 120 fps, in 360 Hz frames


def debiased_mae(e):
    """MAE after removing the sample's own median offset -- the part a constant
    definitional shift CANNOT explain."""
    e = e[np.isfinite(e)]
    return float(np.abs(e - np.median(e)).mean()) if e.size else np.nan


def part_a(a):
    if not os.path.exists(PAIRS):
        sys.exit(f"missing {PAIRS} -- run event_error_sweep.py --map --dump first")
    gt = load_gt_events()
    pop = map_population()
    both = {sp: (v["fp"], v["fp10"]) for sp, v in gt.items()
            if {"fp", "fp10"} <= set(v) and sp in pop}
    print(f"pitches with BOTH fp landmarks, inside the frozen map population: "
          f"{len(both)}")

    d = pd.read_csv(PAIRS)
    d = d[(d.event == "fp") & d.sp.isin(both)].copy()
    print(f"fp rows in the dump: {len(d):,}  "
          f"detectors {sorted(d.detector.unique())}  "
          f"anchors {sorted(d.anchor.unique())}")

    d["fp100"] = d.sp.map(lambda s: both[s][0])
    d["fp10"] = d.sp.map(lambda s: both[s][1])
    d["gap"] = d.fp100 - d.fp10
    # err_ms was written as (det - gt_fp100)/fps*1000. Every OBP c3d is 360 Hz
    # (POINT RATE, verified) so the inversion is exact -- but the dump was written
    # at %.6g, which drops the decimals once |err_ms| >= 1000, i.e. for the handful
    # of catastrophic misses. Rounding still lands on the right frame; the guard
    # only has to confirm the rounding is UNAMBIGUOUS (residual well under 0.5).
    fr = d.err_ms * F360 / 1000.0
    resid = np.abs(fr - np.round(fr))
    lossy = int((resid > 0.02).sum())
    print(f"frame reconstruction: max |residual| = {resid.max():.3f}, "
          f"{lossy} of {len(d):,} rows ({lossy / len(d):.3%}) lost sub-frame "
          f"precision to %.6g -- all of them |err| > 900 ms")
    if resid.max() >= 0.25:
        sys.exit("err_ms cannot be inverted unambiguously; abort rather than "
                 "report a guess")
    d["det"] = d.fp100 + np.round(fr).astype(int)
    d["e100"] = d.det - d.fp100
    d["e10"] = d.det - d.fp10

    # alpha is a least-squares SLOPE, so it is dominated by the catastrophic misses
    # (|err| up to 3 s) unless they are trimmed. Untrimmed it reads ~0.5 for the side
    # detector at every elevation, i.e. "halfway between the landmarks", which is an
    # OUTLIER ARTEFACT and not a fact about the detector -- trimmed it reads 0.85-1.0
    # above the ground. Both are kept so the fragility stays visible; the trimmed one
    # is the number to quote.
    TRIM = 30                       # frames; |e100| > 30 f (83 ms) is not a foot plant
    rows = []
    for (az, el, det, anch), g in d.groupby(["az", "el", "detector", "anchor"]):
        gap = g.gap.to_numpy(float)
        e10 = g.e10.to_numpy(float); e100 = g.e100.to_numpy(float)
        m = np.isfinite(gap) & np.isfinite(e10)
        if m.sum() < 30 or np.var(gap[m]) < 1e-9:
            continue
        k = m & (np.abs(e100) <= TRIM)
        ok = k.sum() >= 30 and np.var(gap[k]) > 1e-9
        rows.append(dict(az=az, el=el, detector=det, anchor=anch, n=int(m.sum()),
                         kept=float(k.sum() / m.sum()),
                         alpha=float(np.polyfit(gap[k], e10[k], 1)[0]) if ok else np.nan,
                         alpha_raw=float(np.polyfit(gap[m], e10[m], 1)[0]),
                         bias100=float(np.median(e100)), bias10=float(np.median(e10)),
                         dmae100=debiased_mae(e100[k]) if ok else np.nan,
                         dmae10=debiased_mae(e10[k]) if ok else np.nan,
                         dmae100_raw=debiased_mae(e100), dmae10_raw=debiased_mae(e10),
                         p1f100=float(np.mean(np.abs(e100 - np.median(e100)) <= PHONE_F)),
                         p1f10=float(np.mean(np.abs(e10 - np.median(e10)) <= PHONE_F))))
    C = pd.DataFrame(rows)
    out = os.path.join(V, "fp_target_detector.csv")
    C.to_csv(out, index=False, float_format="%.6g")

    print("\n" + "=" * 100)
    print("[A] WHAT DOES THE DETECTOR HIT?   det = fp10 + alpha*gap   "
          "(alpha 0 = fp_10, 1 = fp_100)")
    print("=" * 100)
    hdr = (f"{'detector':>10}{'anchor':>16}{'cells':>7}{'kept':>7}"
           f"{'alpha':>8}{'(raw)':>8}{'bias100':>9}{'bias10':>8}{'dMAE100':>9}"
           f"{'dMAE10':>8}{'p1f100':>8}{'p1f10':>7}")
    print(hdr); print("-" * len(hdr))
    for (det, anch), g in C.groupby(["detector", "anchor"]):
        print(f"{det:>10}{anch:>16}{len(g):>7}{g.kept.median():>7.2f}"
              f"{g.alpha.median():>8.3f}{g.alpha_raw.median():>8.3f}"
              f"{g.bias100.median():>9.1f}{g.bias10.median():>8.1f}"
              f"{g.dmae100.median():>9.2f}{g.dmae10.median():>8.2f}"
              f"{g.p1f100.median():>8.3f}{g.p1f10.median():>7.3f}")
    print("\n  bias/dMAE in frames @360Hz; dMAE is debiased, so a pure definitional")
    print("  offset is already removed from it -- if dMAE10 << dMAE100 the two")
    print("  landmarks are NOT interchangeable by a constant. p1f = share within one")
    print("  120 fps phone frame of the sample's own median.")

    print("\n  alpha distribution over all cells:")
    q = C.alpha.quantile([.05, .25, .5, .75, .95])
    print("    " + "  ".join(f"p{int(k*100):02d} {v:+.3f}" for k, v in q.items()))
    print(f"    share of cells with alpha < 0.5 (closer to fp_10): "
          f"{float((C.alpha < 0.5).mean()):.3f}")
    print(f"    share with alpha > 0.5 (closer to fp_100): "
          f"{float((C.alpha > 0.5).mean()):.3f}")

    print("\n  BY ELEVATION, both detectors, each on its own detected release "
          "(the deployment configuration):")
    hdr2 = (f"{'detector':>9}{'el':>5}{'alpha':>8}{'bias100':>9}{'bias10':>8}"
            f"{'dMAE100':>9}{'dMAE10':>8}{'better':>8}{'p1f100':>8}{'p1f10':>7}")
    print(hdr2); print("-" * len(hdr2))
    for det in ("side", "frontal"):
        s = C[(C.detector == det) & (C.anchor == "det_rel")]
        for el, g in s.groupby("el"):
            b = ("fp_10" if g.dmae10.median() < g.dmae100.median() - 0.05 else
                 ("fp_100" if g.dmae100.median() < g.dmae10.median() - 0.05
                  else "tie"))
            print(f"{det:>9}{int(el):>5}{g.alpha.median():>8.3f}"
                  f"{g.bias100.median():>9.1f}{g.bias10.median():>8.1f}"
                  f"{g.dmae100.median():>9.2f}{g.dmae10.median():>8.2f}"
                  f"{b:>8}{g.p1f100.median():>8.3f}{g.p1f10.median():>7.3f}")
    print("\n  'better' = which target leaves the smaller DEBIASED spread, i.e. which")
    print("  one our detector is actually tracking once a constant offset is allowed.")

    # Falsifiable check of the "the detector rides fp_10" reading. If det = fp10 +
    # const + eps with eps independent of the gap, then err100 = eps - gap exactly,
    # so Var(err100) MUST equal Var(err10) + Var(gap). Where that identity holds the
    # gap is pure ADDED noise against fp_100 and no offset can remove it; where it
    # fails the detector is genuinely tracking something between the landmarks.
    # TRIMMED, and it has to be. Raw per-elevation variance is useless here: the
    # catastrophic misses put Var(e10) at ~5,400 f2 (SD 74 f) by el85, which dwarfs
    # Var(gap)=26 and drives the ratio to 1.00 whatever the detector is doing. Keep
    # only detections inside +-TRIM frames of fp_100 (a miss of 83 ms is not a
    # foot plant at all), and score per CELL so the 24 azimuths cannot pool their
    # outliers together.
    TRIM = 30
    print(f"\n  [variance identity]  if the detector rides fp_10, "
          f"Var(e100) = Var(e10) + Var(gap).  |e100| <= {TRIM} f only")
    hdr3 = (f"{'detector':>9}{'el':>5}{'kept':>7}{'Var(e10)':>10}{'Var(gap)':>10}"
            f"{'sum':>9}{'Var(e100)':>11}{'ratio':>8}{'alpha':>8}{'verdict':>16}")
    print(hdr3); print("-" * len(hdr3))
    dt = d[d.e100.abs() <= TRIM]
    for det in ("side", "frontal"):
        for el in sorted(d.el.unique()):
            sub = dt[(dt.detector == det) & (dt.anchor == "det_rel") & (dt.el == el)]
            raw = d[(d.detector == det) & (d.anchor == "det_rel") & (d.el == el)]
            per = []
            for az, g in sub.groupby("az"):
                if len(g) < 30 or np.var(g.gap) < 1e-9:
                    continue
                v10 = float(np.var(g.e10)); vg = float(np.var(g.gap))
                v100 = float(np.var(g.e100))
                per.append((v10, vg, v100, v100 / (v10 + vg),
                            float(np.polyfit(g.gap, g.e10, 1)[0])))
            if not per:
                continue
            P = np.array(per)
            v10, vg, v100, ratio, alpha = np.median(P, axis=0)
            vd = ("rides fp_10" if 0.85 <= ratio <= 1.15 else
                  ("tracks fp_100" if ratio < 0.85 else "neither"))
            print(f"{det:>9}{int(el):>5}{len(sub) / max(len(raw), 1):>7.2f}"
                  f"{v10:>10.1f}{vg:>10.1f}{v10 + vg:>9.1f}{v100:>11.1f}"
                  f"{ratio:>8.2f}{alpha:>8.2f}{vd:>16}")
    print("\n  'kept' = share of detections surviving the trim -- read the verdict")
    print("  only where it is high; a low share means the detector is mostly missing")
    print("  outright and the target question does not arise. ratio ~1 = the")
    print("  fp_10 -> fp_100 interval is pure added noise for us and NO constant")
    print("  offset removes it; ratio < 1 = we genuinely follow part of it.")
    print(f"\nsaved -> {out}")


def fp_dependent_rows():
    """Every map row whose value can change when 'fp' resolves to the other
    landmark, derived from the CODE rather than from event_inventory.csv.

    Two sources, both mechanical:
      screened  CANDS[c] whose event key is 'fp', PLUS every callable (window-max)
                observable -- those read the window [fp, rel] no matter what event
                key they carry, which is exactly what the inventory's per-row
                `reference` field cannot express.
      adopted   the estimators that read ctx['fp'] (est_release_ext / t3_release_ext
                / est_stride_angle; est_cog_pkh only when ctx['pkh'] is absent,
                which it never is under --gt-events, hence COG Velo @PKH's exact
                0.0000 below -- a prediction, not a coincidence)."""
    from rejected_gt_full_sweep import CANDS
    WINDOW_OBS = {"elbow_ext_velo_max", "elbow_flex_max", "shoulder_line_velo_max",
                  "hip_line_velo_max", "shoulder_line_min", "hz_abd_throw_max",
                  "torso_pelvis_timing", "knee_ext_velo_max", "knee_ext_fp_to_br"}
    rows = {c for c, (obs, ev) in CANDS.items()
            if ev == "fp" or obs in WINDOW_OBS}
    rows |= {"Release Ext [O]", "Stride Angle [O]", "COG Velo @PKH [O]"}
    return rows


def part_b(a):
    base = os.path.join(V, "gate_map.csv")
    alt = os.path.join(V, "gate_map_fp10.csv")
    if not os.path.exists(alt):
        sys.exit(f"missing {alt}\nBuild it first:\n"
                 "  python rejected_gt_full_sweep.py --dump --fp-target fp10\n"
                 "  python angle_zone_sweep.py --gt-events --dump --fp-target fp10\n"
                 "  python gate_map.py --screen-pairs rejected_gt_pairs_fp10.csv.gz "
                 "--adopted-pairs angle_zone_pairs_gt_fp10.csv.gz "
                 "--out-suffix _fp10 --pop-frozen")
    b = pd.read_csv(base); f = pd.read_csv(alt)
    uses_fp = fp_dependent_rows()
    KEY = ["metric", "source", "az", "el"]
    m = b.merge(f, on=KEY, suffixes=("_100", "_10"))
    m["uses_fp"] = m.metric.isin(uses_fp)

    print("=" * 104)
    print("[B] WHAT DO THE METRICS WANT?   every fp-referencing row re-anchored at "
          "fp_10")
    print("=" * 104)
    ctl = m[~m.uses_fp]
    dmax = float((ctl.ccc_100 - ctl.ccc_10).abs().max())
    ngr = int((ctl.grade_100 != ctl.grade_10).sum())
    moved = set(m.loc[(m.ccc_100 - m.ccc_10).abs() > 1e-9, "metric"])
    print(f"\n[control] fp-dependent rows derived from the CODE (CANDS event keys + "
          f"window-max observables + the adopted estimators that read ctx['fp']): "
          f"{len(uses_fp)}")
    print(f"          rows that do NOT depend on fp: {ctl.metric.nunique()} "
          f"metrics, {len(ctl)} cells")
    print(f"          max |dCCC| = {dmax:.10f}, grade changes = {ngr}   "
          + ("OK" if dmax == 0 and ngr == 0 else "!! LEAK"))
    print(f"          rows that actually moved: {len(moved)}; all fp-dependent? "
          + ("YES" if moved <= uses_fp else f"NO -> {sorted(moved - uses_fp)}"))
    print("\n  NOTE event_inventory.csv indexes only the 40 rows holding a gate "
          "cell, so\n  it is NOT a valid dependency list -- the 12 FAIL rows are "
          "absent from it and\n  10 of them do read fp. Deriving from the code is "
          "the only safe control here.")

    ORD = {"limited": 0, "moderate": 1, "strong": 2}
    rows = []
    for (metric, src), g in m[m.uses_fp].groupby(["metric", "source"]):
        d = g.ccc_10 - g.ccc_100
        s100 = int((g.grade_100 == "strong").sum())
        s10 = int((g.grade_10 == "strong").sum())
        g100 = int(g.gate_pass_100.sum()); g10 = int(g.gate_pass_10.sum())
        up = int((g.grade_10.map(ORD) > g.grade_100.map(ORD)).sum())
        dn = int((g.grade_10.map(ORD) < g.grade_100.map(ORD)).sum())
        rows.append(dict(metric=metric, source=src, cells=len(g),
                         strong_fp100=s100, strong_fp10=s10, d_strong=s10 - s100,
                         gate_fp100=g100, gate_fp10=g10,
                         best_ccc_fp100=float(g.ccc_100.max()),
                         best_ccc_fp10=float(g.ccc_10.max()),
                         dccc_med=float(d.median()), dccc_max=float(d.max()),
                         cells_up=up, cells_down=dn,
                         prefers=("fp_10" if s10 > s100 else
                                  ("fp_100" if s100 > s10 else "tie"))))
    R = pd.DataFrame(rows).sort_values("d_strong")
    out = os.path.join(V, "fp_target_map.csv")
    R.to_csv(out, index=False, float_format="%.6g")

    hdr = (f"{'metric':<40}{'cells':>6}{'S@100':>7}{'S@10':>6}{'dS':>5}"
           f"{'best@100':>10}{'best@10':>9}{'dCCC med':>10}{'up/down':>10}"
           f"{'prefers':>9}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in R.itertuples(index=False):
        print(f"{r.metric:<40}{r.cells:>6}{r.strong_fp100:>7}{r.strong_fp10:>6}"
              f"{r.d_strong:>+5}{r.best_ccc_fp100:>10.3f}{r.best_ccc_fp10:>9.3f}"
              f"{r.dccc_med:>+10.4f}{f'{r.cells_up}/{r.cells_down}':>10}"
              f"{r.prefers:>9}")
    tot100, tot10 = int(R.strong_fp100.sum()), int(R.strong_fp10.sum())
    print("-" * len(hdr))
    lbl = "TOTAL over the fp-dependent rows"
    print(f"{lbl:<40}{int(R.cells.sum()):>6}{tot100:>7}"
          f"{tot10:>6}{tot10 - tot100:>+5}")
    print(f"\n  rows preferring fp_10: {(R.prefers=='fp_10').sum()}   "
          f"fp_100: {(R.prefers=='fp_100').sum()}   tie: {(R.prefers=='tie').sum()}")
    print(f"\nsaved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "both"], default="both")
    a = ap.parse_args()
    if a.part in ("a", "both"):
        part_a(a)
    if a.part in ("b", "both"):
        print()
        part_b(a)


if __name__ == "__main__":
    main()
