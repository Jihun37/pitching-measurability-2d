"""
Diamond - fine azimuth sweep for threshold-based valid zones.

Refines the 45deg-step 2D angle map (angle_map_2d.py) to 15deg azimuth steps so
that r2-threshold crossings can be located, turning the "3 stations" framing
into per-metric VALID ZONES: contiguous azimuth arcs where a metric clears an
absolute r2 floor, with the original stations kept as optimal anchors.

Tiers:  r2 >= 0.80 -> "reliable",  r2 >= 0.60 -> "usable" (the floor already
defended in the paper via overhead HSS r2=0.63), below -> invalid.

Reuse (no re-implementation):
  - adopted metric rows (estimators + truths): angle_map_2d.adopted_rows
  - robust projector: hss_elevation_test.project_cam (handles el up to 90)
  - loader: master_angle_table.load_feet

Conventions (identical to angle_map_2d, so 45deg grid points must reproduce it):
  - Pearson r2 vs poi-column or 3D-direct truth, full-dataset pooled.
  - Events detected ONCE on the el=0/az=0 side view and reused across views
    (paper-table convention, distinct from the deployment per-view rule).

Performance: each view is projected ONCE and the DataFrame is shared by all
estimators (angle_map_2d re-projected per metric; same math, 9x fewer projections).

Outputs (OBP_VALIDATION_DIR):
  - angle_zone_sweep.csv   raw r2 per (metric, az, el), az step 15deg
  - angle_zone_table.csv   valid arcs per (metric, el, tier) with interpolated edges
  - angle_zone_pairs.csv.gz  (--dump only) per-pitch est/truth pairs, long
    format, consumed by absacc_table.py for absolute-accuracy stats

Run:  cd src\\analysis
      python angle_zone_sweep.py --limit 10    (smoke test)
      python angle_zone_sweep.py
      python angle_zone_sweep.py --redetect    (deployment-honest variant)

--redetect: per-view event RE-DETECTION (deployment convention, 2026-07-10).
  Events are re-detected on each projected view with the strategy rule in
  release_view() (refined via the --oracle diagnostic; see its docstring).
  The el=0/az=0 gate still decides pitch inclusion so the population matches
  the detect-once run; a failed detection at one view NaNs that view only.
  Outputs get a _redetect suffix (official files untouched) and the run ends
  with a diff vs the detect-once map. Spec: docs/EVENT_REDETECT_SWEEP_HANDOFF.md

--oracle: strategy ORACLE diagnostic (2026-07-10, follow-up to --redetect).
  Detects events with BOTH strategies at every view and evaluates every
  estimator under each, so cells the azimuth rule lost can be classified:
  recoverable by a better strategy choice (free fix: change the rule) vs
  detector-limited (both strategies fail: needs a new detection signal).
  Requires angle_zone_sweep.csv and angle_zone_sweep_redetect.csv to exist.
  Output: angle_zone_sweep_oracle.csv (long format with a strategy column).
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
from hss_elevation_test import project_cam
from angle_map_2d import adopted_rows, gt_only_rows, unwrap_circular, CIRCULAR
from obp_gt_events import load_gt_events

AZ_STEP = 15
AZ = list(range(0, 360, AZ_STEP))
EL = [0, 15, 30, 45, 60, 75, 85]   # 15deg el rows added 2026-07-10: the zone
                                   # el-boundaries were only known to +-30deg,
                                   # and station_classify trains on this grid
TIERS = [("reliable", 0.80), ("usable", 0.60)]


def release_view(az, el):
    """Strategy rule for per-view release detection, REFINED from the oracle
    diagnostic (user decision 2026-07-10): frontal only in the front
    sector az 60-120 at el<=60 (OFFICIAL az convention 2026-07-15:
    az90 = home/catcher, probe scratch/obp_axis_probe.py); side everywhere
    else (including the entire rear half and all el>=75, where side
    empirically wins).
    Supersedes the first azimuth-only rule (0/180+-45 side, 90/270+-45
    frontal), whose rear-frontal assignment the oracle proved wrong.
    KNOWN OPEN EXCEPTION (not yet adopted): at el=0 the rear sector is an
    exact u-flip mirror of the front (oracle values identical), so az
    255-285/el=0 genuinely prefers frontal (Arm Slot usable+reliable,
    Release Height reliable survive only with it) - kept out for now to
    keep the rule simple; see docs/EVENT_REDETECT_SWEEP_HANDOFF.md.

    2026-07-24 (user-approved): the elevation bound tightened from el<=60 to
    el<=15. Scored against the OBP LANDMARK release instead of our own az0
    detection (event_error_sweep.py --gt, n=411), frontal COLLAPSES above el15 --
    at az 90-120 x el 30-60 its release IQR is ~200 ms where side is 3 ms, and
    the old rule routed exactly those ten cells to frontal. The ground rule is
    deliberately untouched: at el<=15 the governing evidence is the real-video
    holdout, not clean projection. Cost of the old rule, measured end to end:
    Torso Lat Tilt @MER read r2 0.007 with frontal vs 0.798 with side at its
    az90/el30 anchor (docs/legacy_pre_dedup/GT_EVENT_MAP_HANDOFF.md 6f)."""
    return "frontal" if 60 <= az <= 120 and el <= 15 else "side"


def r2(e, t, circular=False):
    """Pearson r2. circular=True first rotates the estimate onto one branch: for an
    angle metric the linear correlation is undefined across the atan2 cut, and the
    failure is silent (see angle_map_2d.recentre_circular)."""
    e = np.asarray(e, float); t = np.asarray(t, float)
    if circular:
        e = unwrap_circular(e)
    m = np.isfinite(e) & np.isfinite(t)
    return np.corrcoef(e[m], t[m])[0, 1] ** 2 if m.sum() > 2 else np.nan


def circular_arcs(vals, thr):
    """Contiguous azimuth arcs (with wraparound) where r2 >= thr.
    vals: r2 array aligned with AZ. Returns list of (i_start, i_end) inclusive
    grid-index runs; a full ring returns [(0, len-1)] flagged by caller."""
    ok = [(pd.notna(v) and v >= thr) for v in vals]
    n = len(ok)
    if all(ok):
        return [(0, n - 1)], True
    if not any(ok):
        return [], False
    arcs = []
    # start scanning from a False index so wraparound runs are not split
    start = next(i for i in range(n) if not ok[i])
    i = 0
    while i < n:
        j = (start + i) % n
        if ok[j]:
            k = i
            while k + 1 < n and ok[(start + k + 1) % n]:
                k += 1
            arcs.append(((start + i) % n, (start + k) % n))
            i = k + 1
        else:
            i += 1
    return arcs, False


def interp_edge(az_in, r_in, az_out, r_out, thr):
    """Linear r2-threshold crossing between an inside and an outside grid point."""
    if pd.isna(r_out) or r_in == r_out:
        return float(az_out)
    f = (r_in - thr) / (r_in - r_out)          # 0 at inside point, 1 at outside
    d = (az_out - az_in + 540) % 360 - 180     # signed shortest step
    return (az_in + f * d) % 360


def zone_table(df_r2):
    """Valid arcs per (metric, el, tier) from the fine-sweep r2 table."""
    out = []
    for (metric, el), g in df_r2.groupby(["metric", "el"]):
        g = g.set_index("az").reindex(AZ)
        vals = g["r2"].to_numpy(float)
        for tier, thr in TIERS:
            arcs, full = circular_arcs(vals, thr)
            for i0, i1 in arcs:
                if full:
                    lo, hi, width = 0.0, 360.0, 360.0
                else:
                    prev, nxt = (i0 - 1) % len(AZ), (i1 + 1) % len(AZ)
                    lo = interp_edge(AZ[i0], vals[i0], AZ[prev], vals[prev], thr)
                    hi = interp_edge(AZ[i1], vals[i1], AZ[nxt], vals[nxt], thr)
                    width = (hi - lo) % 360
                peak_i = int(np.nanargmax(vals)) if np.isfinite(vals).any() else -1
                out.append({"metric": metric, "el": el, "tier": tier,
                            "az_lo": round(lo, 1), "az_hi": round(hi, 1),
                            "width_deg": round(width, 1),
                            "peak_az": AZ[peak_i] if peak_i >= 0 else np.nan,
                            "peak_r2": round(float(np.nanmax(vals)), 3)
                                       if np.isfinite(vals).any() else np.nan})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--redetect", action="store_true",
                    help="re-detect events per view (deployment convention)")
    ap.add_argument("--oracle", action="store_true",
                    help="evaluate BOTH detection strategies at every view")
    ap.add_argument("--dump", action="store_true",
                    help="also save per-pitch est/truth pairs (long format) "
                         "for absolute-accuracy stats (absacc_table.py)")
    ap.add_argument("--gt-events", action="store_true",
                    help="use the OBP landmark events (pkh / fp_100 / BR) instead "
                         "of our detectors -- isolates the projection question "
                         "from detector error. Outputs get a _gt suffix.")
    ap.add_argument("--fp-strategy", choices=["side", "frontal"], default="side",
                    help="which foot-plant detector --redetect uses at EVERY view "
                         "(release still follows release_view()). No fp routing "
                         "rule is adopted, so both are run and composed per cell "
                         "in deploy_map.py.")
    ap.add_argument("--fp-target", choices=["fp", "fp10"], default="fp",
                    help="which OBP foot-plant landmark 'fp' resolves to in "
                         "--gt-events mode: fp_100 (default, the map's convention) "
                         "or fp_10. Task 6 -- a DEFINITION check, no new event and "
                         "no new detector. Outputs get an extra _fp10 tag.")
    a = ap.parse_args()
    if a.fp_target != "fp" and not a.gt_events:
        ap.error("--fp-target only applies to --gt-events (the detected path has "
                 "no landmark to choose)")
    if a.oracle and a.dump:
        ap.error("--dump is not supported with --oracle")
    if a.oracle:
        a.redetect = False   # oracle supersedes the rule-based path
    if a.gt_events and (a.redetect or a.oracle):
        ap.error("--gt-events replaces per-view detection; drop --redetect/--oracle")

    # GT events answer the map's actual question -- is this metric measurable from
    # this viewpoint -- without our detector error folded in. Events are view-
    # INDEPENDENT here by construction (they come from the 3D landmarks), so there
    # is nothing to re-detect per view.
    gt_ev = load_gt_events() if a.gt_events else None

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    # GT-only metrics (elbow flex / glove abduction at MER, torso rotation) join the
    # map only when GT events are available -- two of them need the MER event.
    rows = adopted_rows() + (gt_only_rows() if a.gt_events else [])
    est = {(ri, az, el): [] for ri in range(len(rows)) for az in AZ for el in EL}
    STRATS = ("side", "frontal")
    est_o = ({(ri, az, el, s): [] for ri in range(len(rows))
              for az in AZ for el in EL for s in STRATS} if a.oracle else None)
    tru = {ri: [] for ri in range(len(rows))}
    dump_rows = []
    done = fail = redet_fail = 0

    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = load_feet(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
            trail = "right" if lead == "left" else "left"
            df0 = O.project_view(joints, azimuth_deg=0.0)
            if gt_ev is not None:
                g = gt_ev.get(r.session_pitch)
                if not g or not {"rel", "fp", "pkh"} <= set(g):
                    fail += 1; continue
                rel, fp, pkh = g["rel"], g["fp"], g["pkh"]
                if a.fp_target != "fp":
                    if a.fp_target not in g:
                        fail += 1; continue
                    fp = g[a.fp_target]
                if rel <= fp + 1 or fp < 3:
                    fail += 1; continue
            else:
                rel = M.release_frame(df0, arm, fps, M.JOINTS)
                fp = M.foot_plant_frame(df0, lead, fps, M.JOINTS, rel)
                if rel <= fp + 1 or fp < 3:
                    fail += 1; continue
                pkh = M.peak_knee_height_frame(df0, lead, fp, M.JOINTS)
            ctx = {"arm": arm, "lead": lead, "trail": trail,
                   "rel": rel, "fp": fp, "pkh": pkh, "fps": fps,
                   "height_m": float(r.session_height_m),
                   # MER / MIR are GT-only; None in detected mode so the GT-only
                   # estimators return NaN there rather than crashing.
                   "mer": (g.get("mer") if gt_ev is not None else None),
                   "mir": (g.get("mir") if gt_ev is not None else None)}
            sp = r.session_pitch

            for ri, (label, estfn, truth) in enumerate(rows):
                if isinstance(truth, tuple):
                    tval = truth[1](joints, ctx)
                else:
                    tval = poi.loc[sp, truth] if (sp in poi.index and truth in poi.columns) else np.nan
                tru[ri].append(tval)

            for az in AZ:
                for el in EL:
                    df = project_cam(joints, az, el)
                    if a.oracle:
                        for strat in STRATS:
                            try:
                                rel_v = M.release_frame(df, arm, fps, M.JOINTS,
                                                        view=strat)
                                fp_v = M.foot_plant_frame(df, lead, fps,
                                                          M.JOINTS, rel_v)
                            except Exception:
                                rel_v, fp_v = -1, -1
                            if rel_v <= fp_v + 1 or fp_v < 3:
                                redet_fail += 1
                                for ri in range(len(rows)):
                                    est_o[(ri, az, el, strat)].append(np.nan)
                                continue
                            pkh_v = M.peak_knee_height_frame(df, lead, fp_v, M.JOINTS)
                            ctx_v = {**ctx, "rel": rel_v, "fp": fp_v, "pkh": pkh_v}
                            for ri, (label, estfn, truth) in enumerate(rows):
                                try:
                                    est_o[(ri, az, el, strat)].append(estfn(df, ctx_v))
                                except Exception:
                                    est_o[(ri, az, el, strat)].append(np.nan)
                        continue
                    ctx_v = ctx
                    if a.redetect:
                        try:
                            rel_v = M.release_frame(df, arm, fps, M.JOINTS,
                                                    view=release_view(az, el))
                            fp_v = M.foot_plant_frame(df, lead, fps, M.JOINTS,
                                                      rel_v, view=a.fp_strategy)
                        except Exception:
                            rel_v, fp_v = -1, -1
                        if rel_v <= fp_v + 1 or fp_v < 3:
                            # failed detection at THIS view only: NaN the view,
                            # keep the pitch alive at every other view
                            redet_fail += 1
                            for ri in range(len(rows)):
                                est[(ri, az, el)].append(np.nan)
                            continue
                        pkh_v = M.peak_knee_height_frame(df, lead, fp_v, M.JOINTS)
                        ctx_v = {**ctx, "rel": rel_v, "fp": fp_v, "pkh": pkh_v}
                    for ri, (label, estfn, truth) in enumerate(rows):
                        try:
                            est[(ri, az, el)].append(estfn(df, ctx_v))
                        except Exception:
                            est[(ri, az, el)].append(np.nan)
            done += 1
            if a.dump:
                # collected only after the pitch fully succeeded, so [-1]
                # is guaranteed to be THIS pitch's value in every cell
                for ri, (label, _, _) in enumerate(rows):
                    t = tru[ri][-1]
                    for az in AZ:
                        for el in EL:
                            dump_rows.append((label.strip(), az, el, sp,
                                              est[(ri, az, el)][-1], t))
        except Exception:
            fail += 1
        if done and done % 50 == 0:
            print(f"  ...{done} processed")
    print(f"processed {done} / failed {fail}")
    if a.redetect or a.oracle:
        tot = done * len(AZ) * len(EL) * (len(STRATS) if a.oracle else 1)
        print(f"per-view detection failures: {redet_fail} / {tot} views "
              f"({100.0 * redet_fail / max(tot, 1):.1f}%)")
    print()

    if a.oracle:
        rows_o = []
        for ri, (label, _, _) in enumerate(rows):
            circ = label.strip() in CIRCULAR
            for az in AZ:
                for el in EL:
                    for strat in STRATS:
                        rows_o.append({"metric": label.strip(), "az": az,
                                       "el": el, "strategy": strat,
                                       "r2": r2(est_o[(ri, az, el, strat)],
                                                tru[ri], circ)})
        df_o = pd.DataFrame(rows_o)
        out = os.path.join(config.OBP_VALIDATION_DIR, "angle_zone_sweep_oracle.csv")
        df_o.to_csv(out, index=False)
        print(f"saved -> {out}")
        print_oracle_analysis(df_o)
        return

    out_rows = []
    for ri, (label, _, _) in enumerate(rows):
        circ = label.strip() in CIRCULAR
        for az in AZ:
            for el in EL:
                out_rows.append({"metric": label.strip(), "az": az, "el": el,
                                 "r2": r2(est[(ri, az, el)], tru[ri], circ)})
    df_r2 = pd.DataFrame(out_rows)
    suffix = (("_gt" if a.gt_events else ("_redetect" if a.redetect else ""))
              + ("" if a.fp_target == "fp" else f"_{a.fp_target}")
              + ("" if not a.redetect or a.fp_strategy == "side"
                 else f"_fp{a.fp_strategy}"))
    out = os.path.join(config.OBP_VALIDATION_DIR, f"angle_zone_sweep{suffix}.csv")
    df_r2.to_csv(out, index=False)
    print(f"saved -> {out}")

    if a.dump:
        dp = pd.DataFrame(dump_rows, columns=["metric", "az", "el",
                                              "session_pitch", "est", "truth"])
        outp = os.path.join(config.OBP_VALIDATION_DIR,
                            f"angle_zone_pairs{suffix}.csv.gz")
        dp.to_csv(outp, index=False)
        print(f"saved -> {outp}  ({len(dp)} pairs, {dp.session_pitch.nunique()} pitches)")

    zones = zone_table(df_r2)
    outz = os.path.join(config.OBP_VALIDATION_DIR, f"angle_zone_table{suffix}.csv")
    zones.to_csv(outz, index=False)
    print(f"saved -> {outz}\n")

    print("=" * 78)
    print("[VALID ZONES]  az arcs with r2 >= threshold (15deg sweep, edges interpolated)")
    print("=" * 78)
    for (metric, el), g in zones.groupby(["metric", "el"]):
        for tier, thr in TIERS:
            gt = g[g.tier == tier]
            if gt.empty:
                continue
            arcs = "  ".join(
                "full ring" if row.width_deg == 360 else
                f"[{row.az_lo:.0f}-{row.az_hi:.0f}] ({row.width_deg:.0f}deg)"
                for row in gt.itertuples())
            print(f"{metric:24s} el={el:>2d}  {tier:9s}(r2>={thr}): {arcs}")

    if a.redetect:
        print_redetect_diff(df_r2, zones)


def print_oracle_analysis(df_o):
    """Classify cells the azimuth rule lost: recoverable by strategy choice
    (some strategy clears the tier) vs detector-limited (both fail)."""
    base_p = os.path.join(config.OBP_VALIDATION_DIR, "angle_zone_sweep.csv")
    rule_p = os.path.join(config.OBP_VALIDATION_DIR, "angle_zone_sweep_redetect.csv")
    for p in (base_p, rule_p):
        if not os.path.exists(p):
            print(f"\n[ORACLE] missing {p} - run that sweep first")
            return

    wide = df_o.pivot_table(index=["metric", "az", "el"], columns="strategy",
                            values="r2").reset_index()
    mg = (wide
          .merge(pd.read_csv(base_p).rename(columns={"r2": "r2_once"}),
                 on=["metric", "az", "el"])
          .merge(pd.read_csv(rule_p).rename(columns={"r2": "r2_rule"}),
                 on=["metric", "az", "el"]))
    mg["r2_best"] = mg[["side", "frontal"]].max(axis=1)

    print("\n" + "=" * 78)
    print("[ORACLE]  cells lost by the az rule: strategy-fixable vs detector-limited")
    print("=" * 78)
    for tier, thr in TIERS[::-1]:                      # usable first
        print(f"\n--- {tier} tier (r2 >= {thr}) ---")
        print(f"{'metric':24s}{'lost':>6s}{'side ok':>9s}{'front ok':>10s}"
              f"{'either':>8s}{'DETECTOR-LIMITED':>18s}")
        print("-" * 76)
        any_lost = False
        for metric, g in mg.groupby("metric"):
            lost = g[(g.r2_once >= thr) & ~(g.r2_rule >= thr)]
            if lost.empty:
                continue
            any_lost = True
            unrec = lost[~(lost.r2_best >= thr)]
            print(f"{metric:24s}{len(lost):>6d}{(lost.side >= thr).sum():>9d}"
                  f"{(lost.frontal >= thr).sum():>10d}"
                  f"{(lost.r2_best >= thr).sum():>8d}{len(unrec):>18d}")
            if 0 < len(unrec) <= 12:
                cells = "  ".join(f"az={int(c.az)}/el={int(c.el)}"
                                  for c in unrec.itertuples())
                print(f"{'':24s}  limited at: {cells}")
        if not any_lost:
            print("  (no lost cells at this tier)")

    # Empirical strategy preference over event-DEPENDENT metrics only
    # (HSS + Wrist Speed estimators never read rel/fp -> uninformative).
    dep = mg[~mg.metric.str.contains("Hip-Shoulder|Wrist Speed")]
    print("\n[STRATEGY PREFERENCE]  mean r2 over event-dependent metrics per view")
    print("  S = side wins, F = frontal wins (margin > 0.02), . = tie/na")
    print("        " + "".join(f"{az:>4d}" for az in AZ))
    for el in EL:
        line = f"  el={el:>2d} "
        for az in AZ:
            c = dep[(dep.az == az) & (dep.el == el)]
            s, f = c.side.mean(), c.frontal.mean()
            ch = "." if (pd.isna(s) or pd.isna(f) or abs(s - f) <= 0.02) \
                 else ("S" if s > f else "F")
            line += f"{ch:>4s}"
        print(line)
    print("  current rule per el row:")
    for el in EL:
        line = f"  el={el:>2d} "
        line += "".join(f"{'S' if release_view(az, el) == 'side' else 'F':>4s}"
                        for az in AZ)
        print(line)


def print_redetect_diff(df_r2, zones):
    """Diff the redetect run against the official detect-once outputs."""
    base_p = os.path.join(config.OBP_VALIDATION_DIR, "angle_zone_sweep.csv")
    basez_p = os.path.join(config.OBP_VALIDATION_DIR, "angle_zone_table.csv")
    if not os.path.exists(base_p):
        print("\n[DIFF] detect-once angle_zone_sweep.csv not found - skipped")
        return

    base = pd.read_csv(base_p).rename(columns={"r2": "r2_once"})
    mg = df_r2.merge(base, on=["metric", "az", "el"], how="inner")
    mg["dr2"] = mg["r2"] - mg["r2_once"]

    print("\n" + "=" * 78)
    print("[DIFF vs detect-once]  d = r2(redetect) - r2(once), per-cell")
    print("=" * 78)
    anchor = mg[(mg.az == 0) & (mg.el == 0)]
    print(f"sanity az=0/el=0: mean |d| = {anchor.dr2.abs().mean():.4f} "
          f"(should be ~0: same events by construction)\n")
    print(f"{'metric':24s}{'mean d':>8s}{'min d':>8s}{'  worst cell':14s}"
          f"{'lost usable':>12s}{'lost reliable':>14s}")
    print("-" * 80)
    for metric, g in mg.groupby("metric"):
        w = g.loc[g.dr2.idxmin()]
        lost_u = ((g.r2_once >= 0.60) & ~(g.r2 >= 0.60)).sum()
        lost_r = ((g.r2_once >= 0.80) & ~(g.r2 >= 0.80)).sum()
        print(f"{metric:24s}{g.dr2.mean():>+8.3f}{g.dr2.min():>+8.3f}"
              f"  az={int(w.az):>3d},el={int(w.el):>2d}"
              f"{lost_u:>12d}{lost_r:>14d}")

    if os.path.exists(basez_p):
        basez = pd.read_csv(basez_p)
        w0 = basez.groupby(["metric", "el", "tier"]).width_deg.sum()
        w1 = zones.groupby(["metric", "el", "tier"]).width_deg.sum()
        cmp = pd.DataFrame({"once": w0, "redetect": w1}).fillna(0.0)
        cmp["delta"] = cmp.redetect - cmp.once
        changed = cmp[cmp.delta.abs() >= 5.0].sort_values("delta")
        print("\n[ZONE WIDTH CHANGES >= 5deg]  total valid-arc width per (metric, el, tier)")
        if changed.empty:
            print("  none - zone map is event-detection robust")
        else:
            for (metric, el, tier), row in changed.iterrows():
                print(f"  {metric:24s} el={el:>2d} {tier:9s} "
                      f"{row.once:6.1f} -> {row.redetect:6.1f}  ({row.delta:+.1f}deg)")


if __name__ == "__main__":
    main()
