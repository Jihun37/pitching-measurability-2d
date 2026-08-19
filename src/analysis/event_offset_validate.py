"""
Diamond - az/el release-offset correction on the SIDE strategy: fit + validate.

Follow-up to event_error_sweep (docs/EVENT_REDETECT_SWEEP_HANDOFF.md): the
side-strategy release error at the lost oblique cells is BIAS-dominated
(median +19..+28 ms, IQR 3-8 ms, p33>=0.86 everywhere) while frontal is
TAIL-broken outside az~90 (10-45%% of pitches land ~200 ms off even where its
median looks clean). So the fix is a per-view offset on side, and frontal is
only defensible in the narrow front wedge.

Honest validation (no circularity): pitches are split into 2 folds by
metadata row parity; each pitch's events are corrected with the offset grid
fitted on the OTHER fold (per-cell median error, then a 3-point circular
median filter along az - edge-preserving smoothing). Two candidate detectors
are evaluated in the same run:

  rule : adopted release_view() rule (frontal az 60-120 & el<=60,
         side elsewhere), offset applied to the side cells only.
  side : side + offset EVERYWHERE (single-strategy detector - no view
         switching; wins if the corrected side matches frontal in front).

Foot plant is re-derived from the corrected release (it takes rel as input);
no separate fp offset (fp is healthy on the ground, small drift elevated-rear).

Outputs (OBP_VALIDATION_DIR):
  - event_release_offset.csv        deployment offset grid (ALL-pitch fit)
  - angle_zone_sweep_offset_rule.csv / _table_offset_rule.csv
  - angle_zone_sweep_offset_side.csv / _table_offset_side.csv
Report: fold-stability of the offset, lost-cell recovery vs the uncorrected
deployment-honest map, and zone-width comparison (ceiling / uncorrected /
rule+offset / side+offset).

Run:  cd src\\analysis
      python event_offset_validate.py --limit 20   (smoke test)
      python event_offset_validate.py
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
from angle_zone_sweep import AZ, EL, TIERS, project_cam, zone_table, r2, release_view
from angle_map_2d import adopted_rows

VARIANTS = ("rule", "side")


def circ_medfilt3(vals):
    """3-point circular median filter (edge-preserving az smoothing)."""
    n = len(vals)
    out = []
    for i in range(n):
        w = [v for v in (vals[(i - 1) % n], vals[i], vals[(i + 1) % n])
             if np.isfinite(v)]
        out.append(float(np.median(w)) if w else 0.0)
    return out


def iter_pitches(limit):
    """Yield gated pitches with the el0/az0 GT events (same population rule
    as every sweep in this family)."""
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    for i, r in enumerate(md.itertuples(index=False)):
        if limit and i >= limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            continue
        try:
            joints, fps = load_feet(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
            df0 = O.project_view(joints, azimuth_deg=0.0)
            rel = M.release_frame(df0, arm, fps, M.JOINTS)
            fp = M.foot_plant_frame(df0, lead, fps, M.JOINTS, rel)
            if rel <= fp + 1 or fp < 3:
                continue
        except Exception:
            continue
        yield i, r, joints, fps, arm, lead, rel, fp


def fit_offsets(limit):
    """Pass A: per-pitch side release errors -> per-fold offset grids
    (fitted on the opposite fold) + the all-pitch deployment grid."""
    errs = {(az, el): {0: [], 1: []} for az in AZ for el in EL}
    done = 0
    for i, r, joints, fps, arm, lead, rel, fp in iter_pitches(limit):
        fold = i % 2
        for az in AZ:
            for el in EL:
                df = project_cam(joints, az, el)
                try:
                    rs = M.release_frame(df, arm, fps, M.JOINTS, view="side")
                    e = (rs - rel) / fps * 1000.0
                except Exception:
                    e = np.nan
                errs[(az, el)][fold].append(e)
        done += 1
        if done % 50 == 0:
            print(f"  ...pass A {done}")
    print(f"pass A done: {done} pitches")

    def grid_from(pool_folds):
        out = {}
        for el in EL:
            med = []
            for az in AZ:
                v = np.concatenate([np.asarray(errs[(az, el)][f], float)
                                    for f in pool_folds])
                v = v[np.isfinite(v)]
                med.append(float(np.median(v)) if v.size >= 3 else np.nan)
            for az, m in zip(AZ, circ_medfilt3(med)):
                out[(az, el)] = m
        return out

    # offset applied to fold f = fitted on the other fold
    off = {0: grid_from([1]), 1: grid_from([0])}
    full = grid_from([0, 1])

    dmax = max(abs(off[0][k] - off[1][k]) for k in full)
    print(f"fold stability: max |offset(f0-fit) - offset(f1-fit)| = {dmax:.1f} ms")

    dep = pd.DataFrame([{"az": az, "el": el, "offset_ms": round(full[(az, el)], 1)}
                        for az in AZ for el in EL])
    out = os.path.join(config.OBP_VALIDATION_DIR, "event_release_offset.csv")
    dep.to_csv(out, index=False)
    print(f"saved -> {out}")
    return off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    off = fit_offsets(a.limit)

    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    rows = adopted_rows()
    est = {(v, ri, az, el): [] for v in VARIANTS for ri in range(len(rows))
           for az in AZ for el in EL}
    tru = {ri: [] for ri in range(len(rows))}
    done = 0

    for i, r, joints, fps, arm, lead, rel, fp in iter_pitches(a.limit):
        fold = i % 2
        ctx = {"arm": arm, "lead": lead,
               "trail": "right" if lead == "left" else "left",
               "rel": rel, "fp": fp, "fps": fps,
               "height_m": float(r.session_height_m)}
        sp = r.session_pitch
        for ri, (label, estfn, truth) in enumerate(rows):
            if isinstance(truth, tuple):
                tval = truth[1](joints, ctx)
            else:
                tval = (poi.loc[sp, truth]
                        if sp in poi.index and truth in poi.columns else np.nan)
            tru[ri].append(tval)

        for az in AZ:
            for el in EL:
                df = project_cam(joints, az, el)
                # corrected side events
                try:
                    rs = M.release_frame(df, arm, fps, M.JOINTS, view="side")
                    k = int(round(off[fold][(az, el)] * fps / 1000.0))
                    rc = int(np.clip(rs - k, 0, len(df) - 1))
                    fpc = M.foot_plant_frame(df, lead, fps, M.JOINTS, rc)
                    ok_side = rc > fpc + 1 and fpc >= 3
                except Exception:
                    ok_side = False
                events = {"side": (rc, fpc) if ok_side else None}
                # rule variant: frontal (uncorrected) in the front wedge
                if release_view(az, el) == "frontal":
                    try:
                        rf = M.release_frame(df, arm, fps, M.JOINTS,
                                             view="frontal")
                        fpf = M.foot_plant_frame(df, lead, fps, M.JOINTS, rf)
                        events["rule"] = (rf, fpf) if (rf > fpf + 1 and fpf >= 3) else None
                    except Exception:
                        events["rule"] = None
                else:
                    events["rule"] = events["side"]

                for v in VARIANTS:
                    ev = events[v]
                    if ev is None:
                        for ri in range(len(rows)):
                            est[(v, ri, az, el)].append(np.nan)
                        continue
                    ctx_v = {**ctx, "rel": ev[0], "fp": ev[1]}
                    for ri, (label, estfn, truth) in enumerate(rows):
                        try:
                            est[(v, ri, az, el)].append(estfn(df, ctx_v))
                        except Exception:
                            est[(v, ri, az, el)].append(np.nan)
        done += 1
        if done % 50 == 0:
            print(f"  ...pass B {done}")
    print(f"pass B done: {done} pitches\n")

    V = config.OBP_VALIDATION_DIR
    maps = {}
    for v in VARIANTS:
        out_rows = [{"metric": rows[ri][0].strip(), "az": az, "el": el,
                     "r2": r2(est[(v, ri, az, el)], tru[ri])}
                    for ri in range(len(rows)) for az in AZ for el in EL]
        df_r2 = pd.DataFrame(out_rows)
        df_r2.to_csv(os.path.join(V, f"angle_zone_sweep_offset_{v}.csv"),
                     index=False)
        zone_table(df_r2).to_csv(
            os.path.join(V, f"angle_zone_table_offset_{v}.csv"), index=False)
        maps[v] = df_r2
        print(f"saved -> angle_zone_sweep_offset_{v}.csv / angle_zone_table_offset_{v}.csv")

    once = pd.read_csv(os.path.join(V, "angle_zone_sweep.csv")
                       ).rename(columns={"r2": "once"})
    rede = pd.read_csv(os.path.join(V, "angle_zone_sweep_redetect.csv")
                       ).rename(columns={"r2": "rede"})
    mg = once.merge(rede, on=["metric", "az", "el"])
    for v in VARIANTS:
        mg = mg.merge(maps[v].rename(columns={"r2": v}),
                      on=["metric", "az", "el"])

    anchor = mg[(mg.az == 0) & (mg.el == 0)]
    print(f"\nsanity az=0/el=0 (offset=0 there): mean |side - once| = "
          f"{(anchor['side'] - anchor.once).abs().mean():.4f}")

    print("\n" + "=" * 82)
    print("[RECOVERY]  cells lost by the uncorrected map, recovered per detector")
    print("=" * 82)
    for tier, thr in TIERS[::-1]:
        print(f"\n--- {tier} (r2 >= {thr}) ---")
        print(f"{'metric':24s}{'lost':>6s}{'rule+off':>10s}{'side+off':>10s}")
        print("-" * 52)
        for metric, g in mg.groupby("metric"):
            lost = g[(g.once >= thr) & ~(g.rede >= thr)]
            if lost.empty:
                continue
            print(f"{metric:24s}{len(lost):>6d}"
                  f"{(lost['rule'] >= thr).sum():>10d}"
                  f"{(lost['side'] >= thr).sum():>10d}")

    print("\n[ZONE WIDTHS]  total valid-arc deg per (metric, el, tier)"
          " - rows where a corrected map differs from uncorrected by >= 5deg")
    ws = {}
    ws["once"] = pd.read_csv(os.path.join(V, "angle_zone_table.csv"))
    ws["rede"] = pd.read_csv(os.path.join(V, "angle_zone_table_redetect.csv"))
    for v in VARIANTS:
        ws[v] = pd.read_csv(os.path.join(V, f"angle_zone_table_offset_{v}.csv"))
    key = ["metric", "el", "tier"]
    w = pd.concat({k: t.groupby(key).width_deg.sum() for k, t in ws.items()},
                  axis=1).fillna(0.0)
    sel = w[(w["rule"] - w["rede"]).abs().ge(5) | (w["side"] - w["rede"]).abs().ge(5)]
    print(f"{'metric':24s}{'el':>4s} {'tier':9s}{'ceiling':>9s}{'uncorr':>8s}"
          f"{'rule+off':>10s}{'side+off':>10s}")
    print("-" * 78)
    for (metric, el, tier), row in sel.sort_values(
            ["metric", "el", "tier"]).iterrows():
        print(f"{metric:24s}{el:>4d} {tier:9s}{row['once']:>9.1f}{row['rede']:>8.1f}"
              f"{row['rule']:>10.1f}{row['side']:>10.1f}")


if __name__ == "__main__":
    main()
