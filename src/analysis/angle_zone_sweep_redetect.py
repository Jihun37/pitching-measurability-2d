"""
Diamond - zone sweep under the DEPLOYMENT event convention (per-view
re-detection), and the diff against the detect-once paper map.

angle_zone_sweep.py detects release/foot-plant ONCE on the clean el=0/az=0
side view and reuses them at every viewpoint - it isolates projection geometry
but is optimistic for event-dependent metrics: deployment has no perfect side
view and must find the events IN the off-axis video itself. This rerun
re-detects both events per (az, el) view:

  event strategy per azimuth (user rule, 2026-07-10):
    az within +-45 of 0/180   -> metrics.release_frame(view="side")
                                 (wrist-speed peak; ties at exactly 45 -> side)
    az within +-45 of 90/270  -> view="frontal" (arm-extension argmax)
  foot plant: metrics.foot_plant_frame on the same view (view-agnostic def).

A view whose detection fails the sanity gate (rel <= fp+1 or fp < 3)
contributes NaN for ALL metrics at that view (counted and reported as the
event-failure rate - that failure is itself deployment signal).

Output:
  - angle_zone_sweep_redetect.csv   (metric, az, el, r2  - same schema)
  - printed DIFF vs angle_zone_sweep.csv: usable-status flips (>=0.6 boundary)
    per metric x el, plus per-view event-failure rates.

Run:  cd src\\analysis
      python angle_zone_sweep_redetect.py --limit 20   (smoke test)
      python angle_zone_sweep_redetect.py
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
from angle_map_2d import adopted_rows
from angle_zone_sweep import AZ, EL, r2

FLOOR = 0.60


def event_view(az):
    """side/frontal event strategy per the user rule (ties at 45 -> side)."""
    d_side = min(abs((az - c + 180) % 360 - 180) for c in (0, 180))
    return "side" if d_side <= 45 else "frontal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    rows = adopted_rows()
    est = {(ri, az, el): [] for ri in range(len(rows)) for az in AZ for el in EL}
    tru = {ri: [] for ri in range(len(rows))}
    evfail = {(az, el): 0 for az in AZ for el in EL}
    done = fail = 0

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
            # keep the same per-pitch inclusion gate as the detect-once sweep
            # so both maps pool the identical pitch set (fair diff)
            df0 = O.project_view(joints, azimuth_deg=0.0)
            rel0 = M.release_frame(df0, arm, fps, M.JOINTS)
            fp0 = M.foot_plant_frame(df0, lead, fps, M.JOINTS, rel0)
            if rel0 <= fp0 + 1 or fp0 < 3:
                fail += 1; continue
            sp = r.session_pitch

            for ri, (label, estfn, truth) in enumerate(rows):
                if isinstance(truth, tuple):
                    ctx0 = {"arm": arm, "lead": lead, "trail": trail,
                            "rel": rel0, "fp": fp0, "fps": fps,
                            "height_m": float(r.session_height_m)}
                    tval = truth[1](joints, ctx0)
                else:
                    tval = poi.loc[sp, truth] if (sp in poi.index and truth in poi.columns) else np.nan
                tru[ri].append(tval)

            for az in AZ:
                for el in EL:
                    df = project_cam(joints, az, el)
                    try:
                        rel = M.release_frame(df, arm, fps, M.JOINTS,
                                              view=event_view(az))
                        fp = M.foot_plant_frame(df, lead, fps, M.JOINTS, rel)
                        ok = rel > fp + 1 and fp >= 3
                    except Exception:
                        ok = False
                    if not ok:
                        evfail[(az, el)] += 1
                        for ri in range(len(rows)):
                            est[(ri, az, el)].append(np.nan)
                        continue
                    ctx = {"arm": arm, "lead": lead, "trail": trail,
                           "rel": rel, "fp": fp, "fps": fps,
                           "height_m": float(r.session_height_m)}
                    for ri, (label, estfn, truth) in enumerate(rows):
                        try:
                            est[(ri, az, el)].append(estfn(df, ctx))
                        except Exception:
                            est[(ri, az, el)].append(np.nan)
            done += 1
        except Exception:
            fail += 1
        if done and done % 50 == 0:
            print(f"  ...{done} processed")
    print(f"processed {done} / failed {fail}\n")

    out_rows = []
    for ri, (label, _, _) in enumerate(rows):
        for az in AZ:
            for el in EL:
                out_rows.append({"metric": label.strip(), "az": az, "el": el,
                                 "r2": r2(est[(ri, az, el)], tru[ri])})
    df_new = pd.DataFrame(out_rows)
    out = os.path.join(config.OBP_VALIDATION_DIR, "angle_zone_sweep_redetect.csv")
    df_new.to_csv(out, index=False)
    print(f"saved -> {out}\n")

    # ---- event-failure rates (only views with any failures) ----
    print("[EVENT-DETECTION FAILURE RATE]  (per view, share of pitches)")
    bad = {k: v / max(1, done) for k, v in evfail.items() if v > 0}
    if not bad:
        print("  none - events detected on every view")
    else:
        for (az, el), fr in sorted(bad.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  az={az:>3d} el={el:>2d}  {fr:5.1%}")
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more views")

    # ---- diff vs the detect-once map ----
    old = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR,
                                   "angle_zone_sweep.csv"))
    m = old.merge(df_new, on=["metric", "az", "el"], suffixes=("_once", "_re"))
    m["ok_once"] = m["r2_once"] >= FLOOR
    m["ok_re"] = m["r2_re"] >= FLOOR
    print("\n" + "=" * 74)
    print(f"[DIFF vs detect-once]  usable-status flips at r2 >= {FLOOR}")
    print("=" * 74)
    print(f"{'metric':24s}{'el':>4s}  lost (once->re)              gained")
    print("-" * 74)
    any_flip = False
    for (metric, el), g in m.groupby(["metric", "el"]):
        lost = g[g.ok_once & ~g.ok_re]["az"].tolist()
        gained = g[~g.ok_once & g.ok_re]["az"].tolist()
        if lost or gained:
            any_flip = True
            print(f"{metric:24s}{el:>4d}  {str(lost):28s}  {str(gained)}")
    if not any_flip:
        print("  none - zones identical under per-view re-detection")
    d = (m["r2_once"] - m["r2_re"]).abs()
    print(f"\nmax |dr2| = {d.max():.3f}   mean |dr2| = {d.mean():.3f}   "
          f"cells with |dr2|>0.05: {(d > 0.05).sum()} / {len(m)}")


if __name__ == "__main__":
    main()
