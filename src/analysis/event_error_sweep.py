"""
Diamond - event frame-error sweep: WHY per-view event detection fails.

Follow-up to angle_zone_sweep --oracle (docs/EVENT_REDETECT_SWEEP_HANDOFF.md).
The oracle proved Arm Slot / Release Height oblique cells are DETECTOR-LIMITED
(no existing strategy recovers them) while detect-once r2~0.9-1.0 shows the
projection geometry is fine - the whole gap is event error. This sweep
measures the STRUCTURE of that error so the fix can be chosen:

  bias-dominated  (|median| large, spread small)  -> per-view offset correction
  variance-dominated (spread large)               -> needs a new detection signal

Ground truth events = the el=0/az=0 side-view detection (the convention every
Level-A validation used; matches 3D-validated release). For every (az, el)
view and BOTH strategies, release + foot plant are re-detected and the frame
error is recorded in ms.

Outputs (OBP_VALIDATION_DIR):
  - event_error_sweep.csv   per (az, el, strategy, event): median/IQR/p33/n
Report: error grids per strategy + detail table at the zone cells the final
deployment-honest map lost (loaded from angle_zone_sweep[_redetect].csv).

--map mode (2026-07-27): the FULL per-cell event error map, all four events the
gate map's 34 strong rows actually read -- release, foot plant, peak knee height
and the MER proxy -- each measured at every one of the 168 cells and for every
detector that could serve it. The legacy mode above covers only rel/fp and calls
foot_plant_frame WITHOUT a view, so its "frontal" fp rows are the SIDE detector
anchored on a frontal release; the frontal foot-plant detector adopted 2026-07-26
had never been scored per cell at all.

  - event_error_map.csv   per (az, el, event, detector, anchor, pop)

Two things the map separates that a single number cannot:
  ANCHOR   fp takes a release, pkh takes an fp -> each is measured both on the GT
           anchor (the detector's OWN error) and on the detected anchor (what
           deployment actually gets, errors compounded).
  POP      all = every pitch with the GT event; clean = gt_clean (n=394, the paper
           population). The 8 excluded pitches have broken OBP fp landmarks and
           dominate the fp error moments (docs memory: fp SD 15 -> 3.7 frames).

MER is not detected and never will be: it rides rel - k, k = 11 frames @360 Hz.
Its error therefore decomposes exactly into [release detection error] (per cell)
+ [proxy residual] (view-independent), and --map reports both.

Run:  cd src\\analysis
      python event_error_sweep.py --map --limit 10    (smoke test)
      python event_error_sweep.py --map
"""
import os, sys, argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)

import config
import obp_project as O
import metrics as M
from master_angle_table import load_feet
from angle_zone_sweep import AZ, EL, project_cam, release_view
from obp_gt_events import load_gt_events
from mer_proxy_map import map_population

STRATS = ("side", "frontal")
EVENTS = ("rel", "fp")

# --- --map mode -----------------------------------------------------------
# (event, detector, anchor). anchor "-" = the detector needs no other event.
MAP_VARIANTS = [
    ("rel", "side",    "-"),
    ("rel", "frontal", "-"),
    ("fp",  "side",    "gt_rel"),      # detector's own error
    ("fp",  "frontal", "gt_rel"),
    ("fp",  "side",    "det_rel"),     # deployment: anchored on its own release
    ("fp",  "frontal", "det_rel"),
    ("pkh", "argmin",  "gt_fp"),       # detector's own error
    ("pkh", "argmin",  "det_fp_side"),
    ("pkh", "argmin",  "det_fp_frontal"),
    ("mer", "side",    "rel_minus_k"),
    ("mer", "frontal", "rel_minus_k"),
]
MER_LAG_F360 = 11.0        # rel - mer, frames at 360 Hz (scratch/mer_timing_probe)
PHONE_FPS = 120.0          # the frame rate every deployment claim is made at


def collect(limit=None, gt=False):
    """gt=False: reference = our own az0/el0 side detection (the original
    convention, which measures per-view CONSISTENCY).
    gt=True:  reference = the OBP landmark events, i.e. per-view ACCURACY. Use
    this one to decide where a detector needs fixing -- consistency with a
    possibly-wrong az0 answer is not the same question (2026-07-24)."""
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    ev_gt = load_gt_events() if gt else None

    err = {(az, el, s, ev): [] for az in AZ for el in EL
           for s in STRATS for ev in EVENTS}
    done = fail = 0
    for i, r in enumerate(md.itertuples(index=False)):
        if limit and i >= limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = load_feet(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
            if ev_gt is not None:
                g = ev_gt.get(r.session_pitch)
                if not g or not {"rel", "fp"} <= set(g):
                    fail += 1; continue
                rel, fp = int(g["rel"]), int(g["fp"])
            else:
                df0 = O.project_view(joints, azimuth_deg=0.0)
                rel = M.release_frame(df0, arm, fps, M.JOINTS)
                fp = M.foot_plant_frame(df0, lead, fps, M.JOINTS, rel)
            if rel <= fp + 1 or fp < 3:
                fail += 1; continue

            for az in AZ:
                for el in EL:
                    df = project_cam(joints, az, el)
                    for s in STRATS:
                        try:
                            rel_v = M.release_frame(df, arm, fps, M.JOINTS,
                                                    view=s)
                            fp_v = M.foot_plant_frame(df, lead, fps,
                                                      M.JOINTS, rel_v)
                            ok = rel_v > fp_v + 1 and fp_v >= 3
                        except Exception:
                            ok = False
                        e_rel = (rel_v - rel) / fps * 1000 if ok else np.nan
                        e_fp = (fp_v - fp) / fps * 1000 if ok else np.nan
                        err[(az, el, s, "rel")].append(e_rel)
                        err[(az, el, s, "fp")].append(e_fp)
            done += 1
        except Exception:
            fail += 1
        if done and done % 50 == 0:
            print(f"  ...{done} processed")
    print(f"processed {done} / failed {fail}\n")
    return err


def aggregate(err):
    rows = []
    for (az, el, s, ev), v in err.items():
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size < 3:
            rows.append({"az": az, "el": el, "strategy": s, "event": ev,
                         "n": int(v.size), "median_ms": np.nan,
                         "iqr_ms": np.nan, "p_within_33ms": np.nan})
            continue
        q25, q50, q75 = np.percentile(v, [25, 50, 75])
        rows.append({"az": az, "el": el, "strategy": s, "event": ev,
                     "n": int(v.size), "median_ms": round(q50, 1),
                     "iqr_ms": round(q75 - q25, 1),
                     "p_within_33ms": round(float(np.mean(np.abs(v) <= 33)), 3)})
    return pd.DataFrame(rows)


def collect_map(limit=None):
    """Per-pitch signed frame error (ms) for every (cell, variant) in MAP_VARIANTS.

    Reference is the OBP landmark for that event -- accuracy, never consistency
    with our own az0 answer. Lists stay index-aligned with `sps` (a per-cell
    failure appends NaN rather than skipping), so a population filter is a mask.

    Returns (err, sps, proxy_resid) where proxy_resid[i] = the view-INDEPENDENT
    MER proxy residual ((gt_rel - k) - gt_mer) in frames at 360 Hz."""
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    ev_gt = load_gt_events()

    err = {(az, el, e, d, a): [] for az in AZ for el in EL
           for (e, d, a) in MAP_VARIANTS}
    sps, proxy_resid = [], []
    done = fail = 0
    for i, r in enumerate(md.itertuples(index=False)):
        if limit and i >= limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        g = ev_gt.get(r.session_pitch)
        # release + foot plant are required to even run the detectors; pkh and mer
        # are optional (410 / 411 of 411 present) and NaN out on their own rows.
        if not g or not {"rel", "fp"} <= set(g):
            fail += 1; continue
        try:
            joints, fps = load_feet(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
        except Exception:
            fail += 1; continue

        g_rel, g_fp = int(g["rel"]), int(g["fp"])
        g_pkh = int(g["pkh"]) if "pkh" in g else None
        g_mer = int(g["mer"]) if "mer" in g else None
        k = int(round(MER_LAG_F360 / 360.0 * fps))
        ms = lambda a, b: (a - b) / fps * 1000.0 if (a is not None and
                                                     b is not None) else np.nan

        sps.append(r.session_pitch)
        proxy_resid.append((g_rel - k - g_mer) if g_mer is not None else np.nan)

        for az in AZ:
            for el in EL:
                vals = {}
                try:
                    df = project_cam(joints, az, el)
                    rel = {s: int(M.release_frame(df, arm, fps, M.JOINTS, view=s))
                           for s in STRATS}
                    fp_gt = {s: int(M.foot_plant_frame(df, lead, fps, M.JOINTS,
                                                       g_rel, view=s))
                             for s in STRATS}
                    fp_det = {s: int(M.foot_plant_frame(df, lead, fps, M.JOINTS,
                                                        rel[s], view=s))
                              for s in STRATS}
                    pkh = {"gt_fp": int(M.peak_knee_height_frame(df, lead, g_fp,
                                                                 M.JOINTS))}
                    for s in STRATS:
                        pkh[f"det_fp_{s}"] = int(M.peak_knee_height_frame(
                            df, lead, fp_det[s], M.JOINTS))

                    for s in STRATS:
                        vals[("rel", s, "-")] = ms(rel[s], g_rel)
                        vals[("fp", s, "gt_rel")] = ms(fp_gt[s], g_fp)
                        vals[("fp", s, "det_rel")] = ms(fp_det[s], g_fp)
                        vals[("mer", s, "rel_minus_k")] = ms(rel[s] - k, g_mer)
                    for a, v in pkh.items():
                        vals[("pkh", "argmin", a)] = ms(v, g_pkh)
                except Exception:
                    pass
                for (e, d, a) in MAP_VARIANTS:
                    err[(az, el, e, d, a)].append(vals.get((e, d, a), np.nan))
        done += 1
        if done % 25 == 0:
            print(f"  ...{done} processed")

    print(f"processed {done} / failed {fail}\n")
    return err, sps, np.asarray(proxy_resid, float)


def aggregate_map(err, sps, pops):
    """One row per (cell, variant, population). Frames are reported at BOTH rates:
    f360 = c3d frames (the unit event_tolerance.py uses), and the p1f/p2f/p3f
    columns are phone frames at PHONE_FPS (the unit a deployment claim is in)."""
    sps = np.asarray(sps)
    rows = []
    for pop, keep in pops.items():
        mask = np.isin(sps, list(keep)) if keep is not None else np.ones(len(sps), bool)
        for (az, el, e, d, a), v in err.items():
            x = np.asarray(v, float)[mask]
            x = x[np.isfinite(x)]
            row = {"az": az, "el": el, "event": e, "detector": d, "anchor": a,
                   "pop": pop, "n": int(x.size),
                   "is_ruled": (d == release_view(az, el)) if e in ("rel", "mer")
                               else ""}
            if x.size < 3:
                row.update({c: np.nan for c in
                            ("median_ms", "iqr_ms", "mae_ms", "mae_f360",
                             "p1f", "p2f", "p3f", "p_within_33ms")})
            else:
                q25, q50, q75 = np.percentile(x, [25, 50, 75])
                mae = float(np.mean(np.abs(x)))
                one = 1000.0 / PHONE_FPS
                row.update({
                    "median_ms": round(q50, 1), "iqr_ms": round(q75 - q25, 1),
                    "mae_ms": round(mae, 1),
                    "mae_f360": round(mae / 1000.0 * 360.0, 2),
                    "p1f": round(float(np.mean(np.abs(x) <= one)), 3),
                    "p2f": round(float(np.mean(np.abs(x) <= 2 * one)), 3),
                    "p3f": round(float(np.mean(np.abs(x) <= 3 * one)), 3),
                    "p_within_33ms": round(float(np.mean(np.abs(x) <= 33)), 3)})
            rows.append(row)
    return pd.DataFrame(rows)


def map_grid(df, event, detector, anchor, col, pop="clean"):
    """Compact az x el grid of one statistic for one variant."""
    sub = df[(df.event == event) & (df.detector == detector)
             & (df.anchor == anchor) & (df["pop"] == pop)]
    print(f"\n[{event.upper()} {col}]  detector={detector}  anchor={anchor}  "
          f"pop={pop}   (. = n<3)")
    print("        " + "".join(f"{az:>5d}" for az in AZ))
    for el in EL:
        line = f"  el={el:>2d} "
        for az in AZ:
            c = sub[(sub.az == az) & (sub.el == el)]
            v = c[col].iloc[0] if len(c) else np.nan
            line += f"{v:>5.0f}" if pd.notna(v) else "    ."
        print(line)


def report_map(df, proxy_resid, pop="clean"):
    d = df[df["pop"] == pop]
    print("=" * 96)
    print(f"EVENT ERROR MAP  ({pop} population, {int(d.n.max())} pitches, "
          f"{len(AZ)}x{len(EL)} cells)")
    print("=" * 96)
    print("\nPer variant, over the 168 cells (MAE in c3d frames @360Hz):")
    hdr = f"{'event':>5} {'detector':>9} {'anchor':>15} {'MAE med':>8}" \
          f"{'MAE p90':>9}{'|med| med':>11}{'IQR med':>9}{'p1f med':>9}{'p3f med':>9}"
    print(hdr); print("-" * len(hdr))
    for (e, dt, a) in MAP_VARIANTS:
        s = d[(d.event == e) & (d.detector == dt) & (d.anchor == a)]
        if not len(s) or s.mae_f360.isna().all():
            continue
        print(f"{e:>5} {dt:>9} {a:>15} {s.mae_f360.median():>8.2f}"
              f"{s.mae_f360.quantile(0.9):>9.2f}"
              f"{s.median_ms.abs().median():>11.1f}{s.iqr_ms.median():>9.1f}"
              f"{s.p1f.median():>9.2f}{s.p3f.median():>9.2f}")
    print("\n  MAE med/p90 = median / 90th pct of per-cell MAE, c3d frames @360Hz")
    print(f"  |med| med, IQR med = ms;  p1f/p3f = share within 1/3 phone frames "
          f"@{PHONE_FPS:.0f}fps")

    pr = proxy_resid[np.isfinite(proxy_resid)]
    if pr.size:
        print(f"\nMER proxy residual ((gt_rel - k) - gt_mer), view-INDEPENDENT, "
              f"n={pr.size}:")
        print(f"  median {np.median(pr):+.2f} f  |  SD {pr.std(ddof=1):.2f} f  |  "
              f"MAE {np.mean(np.abs(pr)):.2f} f  |  "
              f"within +-3f {np.mean(np.abs(pr) <= 3):.1%}")
        print("  MER cell error = this residual + that cell's release error.")

    print("\nBy elevation (MAE, c3d frames @360Hz, median over azimuth):")
    piv = d.pivot_table(index="el", columns=["event", "detector", "anchor"],
                        values="mae_f360", aggfunc="median")
    print(piv.round(2).to_string())

    print("\nRELEASE-RULE CHECK: is release_view() picking the better detector?")
    r = d[d.event == "rel"].pivot_table(index=["az", "el"], columns="detector",
                                        values="mae_f360")
    r = r.dropna()
    ruled = [release_view(az, el) for az, el in r.index]
    r["ruled"] = ruled
    r["ruled_mae"] = [r[c].iloc[i] for i, c in enumerate(ruled)]
    r["best_mae"] = r[["side", "frontal"]].min(axis=1)
    print(f"  cells where the rule picks the worse detector: "
          f"{int((r.ruled_mae > r.best_mae + 1e-9).sum())} / {len(r)}")
    print(f"  mean MAE lost to the rule: "
          f"{(r.ruled_mae - r.best_mae).mean():.2f} frames @360Hz")

    print("\nFOOT PLANT: side vs frontal detector, per elevation (MAE f360, "
          "anchor=gt_rel)")
    f = d[(d.event == "fp") & (d.anchor == "gt_rel")].pivot_table(
        index="el", columns="detector", values="mae_f360", aggfunc="median")
    f["frontal_wins_cells"] = [
        int((d[(d.event == "fp") & (d.anchor == "gt_rel") & (d.el == el)]
             .pivot_table(index="az", columns="detector", values="mae_f360")
             .pipe(lambda t: t["frontal"] < t["side"])).sum())
        for el in f.index]
    print(f.round(2).to_string())

    print("\nANCHOR COST (deployment - own error, MAE f360, median over cells):")
    for e, pairs in (("fp", [("side", "gt_rel", "det_rel"),
                             ("frontal", "gt_rel", "det_rel")]),
                     ("pkh", [("argmin", "gt_fp", "det_fp_side"),
                              ("argmin", "gt_fp", "det_fp_frontal")])):
        for dt, a0, a1 in pairs:
            s0 = d[(d.event == e) & (d.detector == dt) & (d.anchor == a0)]
            s1 = d[(d.event == e) & (d.detector == dt) & (d.anchor == a1)]
            if len(s0) and len(s1):
                print(f"  {e:>4} {dt:>8}: {a0} {s0.mae_f360.median():.2f} -> "
                      f"{a1} {s1.mae_f360.median():.2f}  "
                      f"({s1.mae_f360.median() - s0.mae_f360.median():+.2f})")


def grid(df, strategy, event, col):
    """Compact az x el grid of one error statistic."""
    print(f"\n[{event.upper()} {col}]  strategy={strategy}   (ms, . = n<3)")
    print("        " + "".join(f"{az:>5d}" for az in AZ))
    sub = df[(df.strategy == strategy) & (df.event == event)]
    for el in EL:
        line = f"  el={el:>2d} "
        for az in AZ:
            c = sub[(sub.az == az) & (sub.el == el)]
            v = c[col].iloc[0] if len(c) else np.nan
            line += f"{v:>5.0f}" if pd.notna(v) else "    ."
        print(line)


def lost_cells():
    """Unique (az, el) cells the final deployment-honest map lost, with the
    metrics lost there (from the official sweep CSVs)."""
    V = config.OBP_VALIDATION_DIR
    once = pd.read_csv(os.path.join(V, "angle_zone_sweep.csv")
                       ).rename(columns={"r2": "once"})
    rule = pd.read_csv(os.path.join(V, "angle_zone_sweep_redetect.csv")
                       ).rename(columns={"r2": "rule"})
    m = once.merge(rule, on=["metric", "az", "el"])
    lost = m[((m.once >= 0.6) & ~(m.rule >= 0.6))
             | ((m.once >= 0.8) & ~(m.rule >= 0.8))]
    out = {}
    for r in lost.itertuples():
        out.setdefault((r.az, r.el), []).append(r.metric.split(" [")[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gt", action="store_true",
                    help="reference = OBP landmark events (per-view ACCURACY) "
                         "instead of our az0 side detection (consistency). "
                         "Outputs get a _gt suffix.")
    ap.add_argument("--map", action="store_true",
                    help="full per-cell map over all four events (rel / fp / pkh / "
                         "MER proxy) and every detector+anchor -> event_error_map.csv")
    ap.add_argument("--dump", action="store_true",
                    help="--map only: also write the per-pitch errors "
                         "(event_error_map_pairs.csv.gz) for the axis-3 join.")
    a = ap.parse_args()

    if a.map:
        err, sps, proxy = collect_map(a.limit)
        # clean = the frozen map population, READ rather than re-derived here.
        # Subtracting the k-rule outliers from whatever collect_map gathered gives
        # 395, a retired count: collect_map keeps every pitch carrying the GT event,
        # whereas the frozen gate also requires pkh present AND rel > fp + 1. That
        # one pitch (1371_2) has an empty foot-plant-to-release window, so the map
        # never scores it. See population_freeze.py for the 411/-10/-7/=394 chain.
        pops = {"all": None, "clean": set(sps) & set(map_population())}
        df = aggregate_map(err, sps, pops)
        out = os.path.join(config.OBP_VALIDATION_DIR, "event_error_map.csv")
        df.to_csv(out, index=False)
        print(f"saved -> {out}   ({len(df):,} rows)")
        if a.dump:
            recs = []
            for (az, el, e, d, an), v in err.items():
                recs.append(pd.DataFrame({"az": az, "el": el, "event": e,
                                          "detector": d, "anchor": an,
                                          "sp": sps, "err_ms": v}))
            dp = os.path.join(config.OBP_VALIDATION_DIR,
                              "event_error_map_pairs.csv.gz")
            # NO float_format here. err_ms is a multiple of 1000/360 ms, and two
            # phone frames at 120 fps is EXACTLY six c3d frames = 16.666... ms.
            # The next decimal place is a 6, so ANY finite decimal rounding rounds
            # UP and lands just ABOVE that threshold: %.4g gives 16.67 and even
            # %.10g gives 16.66666667. Either way every pitch sitting exactly on
            # the boundary drops out of a p2f recomputed from this dump, while p1f
            # (8.333) and p3f (25.0) round the safe way and hide the defect.
            # Pandas' default repr round-trips exactly. Leave it alone.
            pd.concat(recs, ignore_index=True).to_csv(dp, index=False)
            print(f"saved -> {dp}")
        report_map(df, proxy, pop="clean")
        for e, dt, an in (("fp", "side", "gt_rel"), ("fp", "frontal", "gt_rel"),
                          ("pkh", "argmin", "gt_fp"), ("mer", "side", "rel_minus_k")):
            map_grid(df, e, dt, an, "mae_f360")
        return

    df = aggregate(collect(a.limit, gt=a.gt))
    sfx = "_gt" if a.gt else ""
    out = os.path.join(config.OBP_VALIDATION_DIR, f"event_error_sweep{sfx}.csv")
    df.to_csv(out, index=False)
    print(f"saved -> {out}")

    for s in STRATS:
        grid(df, s, "rel", "median_ms")
        grid(df, s, "rel", "iqr_ms")

    print("\n" + "=" * 96)
    print("[LOST CELLS]  release error at cells the deployment-honest map lost")
    print("  bias-dominated -> offset fix; spread-dominated -> new signal needed")
    print("=" * 96)
    cells = lost_cells()
    print(f"{'cell':14s}{'side med/IQR':>16s}{'frontal med/IQR':>18s}"
          f"{'fp-side med/IQR':>18s}  lost metrics")
    print("-" * 96)
    for (az, el) in sorted(cells):
        def stat(s, ev):
            c = df[(df.az == az) & (df.el == el)
                   & (df.strategy == s) & (df.event == ev)]
            if not len(c) or pd.isna(c.median_ms.iloc[0]):
                return "    -    "
            return f"{c.median_ms.iloc[0]:>+5.0f}/{c.iqr_ms.iloc[0]:<4.0f}"
        mets = ", ".join(sorted(set(cells[(az, el)])))
        print(f"az={az:>3d} el={el:>2d} {stat('side','rel'):>16s}"
              f"{stat('frontal','rel'):>18s}{stat('side','fp'):>18s}  {mets}")


if __name__ == "__main__":
    main()
