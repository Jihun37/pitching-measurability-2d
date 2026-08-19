"""Axis 1, FULL: how much anchor error does every map cell absorb, out to +-12 f?

Supersedes `event_tolerance.py`, which had three limits the deployable map cannot
live with:
  * censored at +-3 f, while our foot-plant spread is 5-7 f360 -- so the intersection
    with the detector's real error was uncomputable
  * SCREENED rows only; the 16 adopted rows had no tolerance number at all
  * ONE offset applied to every event at once (lockstep), which
    `composite_event_offsets.py` proved hides the asymmetry on window rows

This file fixes all three in one pass, over both layers, at every cell that is on
the graded map.

WHAT IS PERTURBED, per row, taken from event_inventory.csv:
  fp / rel / pkh      shifted INDEPENDENTLY, one reference at a time
  lockstep            all of the row's references shifted together (comparable
                      with the old axis-1 numbers)
  mer_proxy           the MER rows ride release - 11 f, so their deployable anchor
                      is `rel`; shifting rel is their tolerance
  mer_true            GT-only (Elbow Flex @MER). No deployable tolerance exists --
                      reported as `oracle_only`, never as a number
  motion_onset / stride_plateau / hss_anchor / internal_peak
                      not events (2026-07-27) -- nothing to shift

WINDOW ROWS get their two ends separated: `fp` moves the window START, `rel` the
window END. That is the point of doing this independently -- lockstep always pairs
a release-end truncation with a foot-plant-end extension and cancels the very effect
being measured.

INTERNAL-PEAK ROWS additionally get the two failure modes told apart at every
offset, which a CCC number alone cannot do:
    win_trunc   the truth's peak instant fell OUTSIDE the shifted window
    false_peak  the window still contains it, but the 2D argmax jumped to a
                different hump (basin test, same rule as internal_peak_sweep)

Offsets are a NON-UNIFORM grid, fine where the action is and coarse out to the
limit: 0, +-1, +-2, +-3, +-4, +-6, +-8, +-10, +-12. `tol` is therefore reported on
that grid, and 12 means ">= 12".

Scoring is `gate_map.score_cell` -- the map's own gate, not a re-implementation.
Cells are restricted to those ON the graded map (grade != limited at offset 0),
because a tolerance for an already-failing cell is meaningless.

Outputs: event_tolerance_full.csv   (metric x cell x reference x offset)
         event_tolerance_cells.csv  (metric x cell x reference -> tol)
Run:  conda activate diamond; cd src\\analysis
      python event_tolerance_full.py --limit 20      (smoke, NOT a preview)
      python event_tolerance_full.py
"""
import os, sys, argparse, time
import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)

import config
import metrics as M
import obp_project as O
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from obp_gt_events import load_gt_events
from obp_gt_peaks import load_gt_peaks
from gate_map import score_cell
from mer_proxy_map import map_population
from angle_map_2d import (adopted_rows, gt_only_rows, CIRCULAR,
                          unwrap_circular, MER_LAG_S)
from rejected_gt_full_sweep import CANDS, all_observables, sample
from internal_peak_sweep import series_for, basin_peak, ROWS as IP_ROWS

OFFSETS = [-12, -10, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10, 12]
REFS = ("fp", "rel", "pkh", "lockstep")
RANK = {"limited": 0, "moderate": 1, "strong": 2}
J = M.JOINTS
GATE = os.path.join(config.OBP_VALIDATION_DIR, "gate_map.csv")
INV = os.path.join(config.OBP_VALIDATION_DIR, "event_inventory.csv")


def row_refs(inv):
    """{(metric, source): [shiftable references]} from the inventory, with the two
    MER classes mapped onto what a DEPLOYED system would actually shift."""
    out, oracle_only = {}, set()
    for (metric, src), g in inv.groupby(["metric", "source"]):
        have = set(g.reference.astype(str))
        refs = {r for r in have if r in ("fp", "rel", "pkh")}
        if "mer_proxy" in have:
            refs.add("rel")              # the proxy IS rel - 11 f
        # oracle-only means the GT event is the ONLY way in. Most MER rows carry
        # mer_true AND mer_proxy (map anchor vs deploy anchor) and ARE deployable;
        # only a row with mer_true and no proxy has no tolerance to measure.
        if "mer_true" in have and "mer_proxy" not in have:
            oracle_only.add((metric, src))
        if len(refs) > 1:
            refs.add("lockstep")
        out[(metric, src)] = sorted(refs, key=lambda x: REFS.index(x))
    return out, oracle_only


def shift(ev, ref, k, mer_proxy=False, fps=360.0):
    """Events with ONE reference moved by k frames (or all of them, for lockstep).

    mer_proxy=True re-derives MER from the SHIFTED release (rel - 11 f, the adopted
    deployment rule). Without it a MER row's estimator keeps reading the untouched GT
    landmark and its tolerance comes back as a flat line -- the same oracle leak that
    section 3 found in the map itself. With it, the offset-0 value is the PROXY
    baseline, not the map's GT-anchored one, so `ccc0` for these rows is deliberately
    lower than gate_map.csv; that gap is the oracle->deployable drop already
    quantified in mer_proxy_map.csv, not a regression."""
    if ref == "lockstep":
        out = {a: (b + k if a in ("fp", "rel", "pkh") else b)
               for a, b in ev.items()}
    else:
        out = {a: (b + k if a == ref else b) for a, b in ev.items()}
    if mer_proxy and out.get("rel") is not None:
        out["mer"] = int(round(out["rel"] - MER_LAG_S * fps))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--floor", type=float, default=0.75)
    ap.add_argument("--strong", type=float, default=0.80)
    a = ap.parse_args()

    gm = pd.read_csv(GATE)
    inv = pd.read_csv(INV)
    refs_of, oracle_only = row_refs(inv)
    proxy_rows = {(m, s2) for (m, s2), g in inv.groupby(["metric", "source"])
                  if "mer_proxy" in set(g.reference.astype(str))}
    onmap = gm[gm.gate_pass]
    # (metric, source) -> list of on-map cells
    cells_of = {k: sorted(set(zip(g.az.astype(int), g.el.astype(int))))
                for k, g in onmap.groupby(["metric", "source"])}
    todo = {k: refs_of.get(k, []) for k in cells_of if refs_of.get(k)}
    n_eval = sum(len(cells_of[k]) * len(v) * len(OFFSETS) for k, v in todo.items())
    print(f"on-map cells {len(onmap)} over {len(cells_of)} rows; "
          f"{len(todo)} rows have a shiftable reference")
    print(f"oracle-only (GT event, no deployable tolerance): "
          f"{sorted(m for m, _ in oracle_only)}")
    print(f"event-free / no shiftable reference: "
          f"{sorted(m for (m, s) in cells_of if not refs_of.get((m, s)))}")
    print(f"evaluations per pitch: {n_eval:,}\n")

    adopted = {l.strip(): (fn, t) for l, fn, t in adopted_rows() + gt_only_rows()}
    # internal-peak rows -> their OBP peak-frame key (poi-exact refs only)
    ip_ref = {lbl: rf for lbl, _, _, rf in IP_ROWS
              if rf not in ('wrist3d', 'com3d')}

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    root = os.path.join(config.OBP_DATA_DIR, "c3d")
    gt = load_gt_events(); gp = load_gt_peaks(); pop = map_population()
    # HARD GUARD. map_population() intersects two pair dumps; if either is being
    # rewritten by a concurrent sweep it can come back short or empty, and then
    # this script happily writes a full-SHAPED table of NaN that looks like a
    # result. That happened once (0 pitches, 35,666 all-"limited" rows). Fail loud.
    if not pop or len(pop) < 300:
        sys.exit(f"population is {0 if not pop else len(pop)} pitches -- refusing "
                 f"to run. map_population() reads angle_zone_pairs_gt.csv.gz and "
                 f"rejected_gt_pairs.csv.gz; do not run this while a sweep is "
                 f"rewriting either of them.")
    print(f"population {len(pop)} pitches")

    est = {(k, c, r, o): [] for k, v in todo.items() for c in cells_of[k]
           for r in v for o in OFFSETS}
    tru = {k: [] for k in todo}
    ipd = []                      # internal-peak decomposition records
    users, done, fail = [], 0, 0
    t0 = time.time()

    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and done >= a.limit:
            break
        sp = r.session_pitch
        if sp not in pop or sp not in poi.index:
            continue
        g = gt.get(sp)
        if not g or not {"fp", "rel", "pkh"} <= set(g):
            fail += 1; continue
        path = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            j, fps = load_feet(path)
            arm = O.detect_throwing_arm(j, fps)
        except Exception:
            fail += 1; continue
        lead = "left" if arm == "right" else "right"
        ev0 = {"fp": int(g["fp"]), "rel": int(g["rel"]), "pkh": int(g["pkh"]),
               "mer": g.get("mer"), "mir": g.get("mir")}
        ctx0 = {"arm": arm, "lead": lead,
                "trail": "right" if lead == "left" else "left",
                "fps": fps, "height_m": float(r.session_height_m),
                "mer": g.get("mer"), "mir": g.get("mir"), **ev0}
        users.append(r.user)
        for k in todo:
            metric, src = k
            if src == "adopted":
                _, truth = adopted[metric]
                tv = (truth[1](j, ctx0) if isinstance(truth, tuple)
                      else (float(poi.loc[sp, truth]) if truth in poi.columns
                            else np.nan))
            else:
                tv = float(poi.loc[sp, metric]) if metric in poi.columns else np.nan
            tru[k].append(tv)

        # project each needed view once, then walk every row/reference/offset
        need = sorted({c for k in todo for c in cells_of[k]})
        for (az, el) in need:
            df = project_cam(j, az, el)
            obs = None
            for k, rlist in todo.items():
                if (az, el) not in cells_of[k]:
                    continue
                metric, src = k
                if src == "screened" and obs is None:
                    obs, _ = all_observables(df, fps, lead)
                px = k in proxy_rows
                for ref in rlist:
                    for o in OFFSETS:
                        e = shift(ev0, ref, o, px, fps)
                        try:
                            if src == "screened":
                                # fp moves the window START, rel the window END
                                v = sample(obs, metric, e, fps,
                                           d_start=e["fp"] - ev0["fp"],
                                           d_end=e["rel"] - ev0["rel"])
                            else:
                                fn = adopted[metric][0]
                                v = fn(df, {**ctx0, **e})
                        except Exception:
                            v = np.nan
                        est[(k, (az, el), ref, o)].append(v)
            # Internal-peak decomposition: at each offset, WHICH failure mode.
            # Only the poi-exact rows qualify -- Wrist Speed and COG Fwd Velo have
            # no OBP peak frame (their reference is the 3D counterpart of our own
            # definition), so `gf` is absent and they are skipped rather than
            # decomposed against a proxy.
            for lbl, ipref in ip_ref.items():
                kk = next((x for x in todo if x[0] == lbl), None)
                if kk is None or (az, el) not in cells_of.get(kk, []):
                    continue
                gf = gp.get(sp, {}).get(ipref)
                if gf is None:
                    continue
                for ref in todo[kk]:
                    for o in OFFSETS:
                        e = shift(ev0, ref, o, kk in proxy_rows, fps)
                        try:
                            spec = series_for(lbl, df, {**ctx0, **e}, j)
                        except Exception:
                            spec = None
                        if spec is None:
                            continue
                        s, lo, hi, _ = spec
                        lo, hi = int(max(0, lo)), int(min(len(s) - 1, hi))
                        if hi <= lo:
                            continue
                        kh = lo + int(np.nanargmax(s[lo:hi + 1]))
                        kb = basin_peak(s, lo, hi, gf)
                        ipd.append((lbl, az, el, ref, o, sp,
                                    bool(lo <= gf <= hi),
                                    bool(kb is not None and kb == kh)))
        done += 1
        if done % 20 == 0:
            el_s = time.time() - t0
            print(f"  ...{done}  ({el_s / done:.2f} s/pitch, "
                  f"eta {(len(pop) - done) * el_s / done / 60:.0f} min)")
    print(f"processed {done} / failed {fail}   ({(time.time()-t0)/60:.1f} min)\n")

    codes = pd.Series(users).astype("category").cat.codes.to_numpy()
    rows, cellrows = [], []
    for k, rlist in todo.items():
        metric, src = k
        t = np.asarray(tru[k], float)
        circ = metric.strip() in CIRCULAR
        for c in cells_of[k]:
            for ref in rlist:
                gr = {}
                for o in OFFSETS:
                    e = np.asarray(est[(k, c, ref, o)], float)
                    if circ:
                        e = unwrap_circular(e)
                    s = score_cell(e, t, codes, a.floor, a.strong)
                    gr[o] = (s["grade"], s["ccc"]) if s else ("limited", np.nan)
                    rows.append(dict(metric=metric, source=src, az=c[0], el=c[1],
                                     reference=ref, offset=o, grade=gr[o][0],
                                     ccc=gr[o][1]))
                g0 = gr[0][0]
                rec = dict(metric=metric, source=src, az=c[0], el=c[1],
                           reference=ref, grade0=g0, ccc0=gr[0][1])
                for name, need in (("tol_map", 1), ("tol_strong", 2)):
                    if RANK[g0] < need:
                        rec[name] = -1
                        continue
                    tol = 0
                    for kk in [x for x in OFFSETS if x > 0]:
                        if all(RANK[gr[s2][0]] >= need for s2 in (-kk, kk)):
                            tol = kk
                        else:
                            break
                    rec[name] = tol
                cellrows.append(rec)
        print(f"  scored {metric} [{src}]")

    F = pd.DataFrame(rows)
    C = pd.DataFrame(cellrows)
    p1 = os.path.join(config.OBP_VALIDATION_DIR, "event_tolerance_full.csv")
    p2 = os.path.join(config.OBP_VALIDATION_DIR, "event_tolerance_cells.csv")
    F.to_csv(p1, index=False, float_format="%.6g")
    C.to_csv(p2, index=False, float_format="%.6g")
    if ipd:
        D = pd.DataFrame(ipd, columns=["metric", "az", "el", "reference", "offset",
                                       "session_pitch", "gt_in_window",
                                       "same_extremum"])
        p3 = os.path.join(config.OBP_VALIDATION_DIR, "event_tolerance_peakmode.csv")
        (D.groupby(["metric", "az", "el", "reference", "offset"])
           [["gt_in_window", "same_extremum"]].mean().reset_index()
           .to_csv(p3, index=False, float_format="%.6g"))
        print(f"saved -> {p3}")
    print(f"saved -> {p1}\nsaved -> {p2}")
    report(C, F)


def report(C, F):
    pd.set_option("display.width", 220)
    on = C[C.tol_map >= 0]
    print("\n" + "=" * 100)
    print("EVENT TOLERANCE to +-12 f, per reference, at every on-map cell")
    print("=" * 100)
    print(f"  (metric, cell, reference) triples on the map: {len(on)}")
    print("\n  distribution of tol_map (grid: 0,1,2,3,4,6,8,10,12):")
    vc = on.tol_map.value_counts().sort_index()
    for k, n in vc.items():
        lab = f">= {k}" if k == 12 else f"{k}"
        print(f"    survives +-{lab:>4} f : {n:>5}  ({100*n/len(on):.0f}%)")
    print(f"\n  cells that lose the MAP at +-1 f: "
          f"{int((on.tol_map == 0).sum())} ({100*(on.tol_map==0).mean():.0f}%)")
    st = C[C.tol_strong >= 0]
    print(f"  strong triples: {len(st)}, losing STRONG at +-1 f: "
          f"{int((st.tol_strong == 0).sum())}")

    print("\n  BY REFERENCE (median tol over the triples that use it):")
    hdr = f"{'reference':>10}{'triples':>9}{'tol med':>9}{'tol p10':>9}{'die@1f':>9}"
    print(hdr); print("-" * len(hdr))
    for ref, g in on.groupby("reference"):
        print(f"{ref:>10}{len(g):>9}{g.tol_map.median():>9.0f}"
              f"{g.tol_map.quantile(.10):>9.0f}"
              f"{float((g.tol_map == 0).mean()):>9.2f}")

    print("\n  PER METRIC (median tol_map per reference; -1 = row not on map there)")
    piv = on.pivot_table(index=["metric", "source"], columns="reference",
                         values="tol_map", aggfunc="median")
    print(piv.fillna(-1).astype(int).to_string())


if __name__ == "__main__":
    main()
