"""
Diamond - velocity-metric noise probe under the DEPLOYMENT processing chain.

zone_edge_noise.py showed the three velocity metrics (wrist speed, knee ext
velo, pelvis rot velo) failing under iid noise at their zone PEAKS as well as
edges - i.e. global sensitivity of the unsmoothed clean-projection estimators,
not zone shrinkage. Deployment, however, never feeds raw jitter into a
derivative: real video is ~120 fps (config.FPS_DEFAULT) and every pipeline
smooths with stage2.smoother.smooth_coordinates(window=7, polyorder=2) first.

This probe reproduces that chain literally, with no invented parameters:
  1. decimate the 360 fps OBP projection by 3 -> exactly 120 fps
  2. add iid Gaussian keypoint noise at {0, 2, 4}px (sigma_m = px/300,
     same scale as noise_sweep.py); 0px isolates the sampling effect
  3. apply the deployment smoother as-is (SG window 7, polyorder 2)
  4. run the adopted estimators with fps=120 (events rescaled from the
     360 fps detection: rel//3, fp//3 - same detect-once convention)

Probed views: the usable-zone edge + peak angles of the three velocity
metrics (from angle_zone_table.csv, same selection as zone_edge_noise.py).

Output: OBP_VALIDATION_DIR/zone_edge_noise_velo.csv

Run:  cd src\\analysis
      python zone_edge_noise_velo.py --limit 10    (smoke test)
      python zone_edge_noise_velo.py
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
from smoother import smooth_coordinates
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from angle_map_2d import adopted_rows
from zone_edge_noise import edge_points, r2

DECIM = 3                  # 360 fps -> 120 fps (= config.FPS_DEFAULT)
FLOOR = 0.60
NOISE_PX = [0, 2, 4]       # @ppm=300; 0px = sampling-only reference
SEEDS = [0, 1, 2]
PPM_REF = 300.0
VELO_METRICS = ("Wrist Speed [O]", "Knee Ext Velo BR [O]", "Pelvis Rot Velo [O]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    zones = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR, "angle_zone_table.csv"))
    sweep = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR, "angle_zone_sweep.csv"))
    rows = adopted_rows()
    ri_of = {label.strip(): ri for ri, (label, _, _) in enumerate(rows)}

    targets, probes = {}, {}
    zsel = zones[(zones.tier == "usable") & (zones.metric.isin(VELO_METRICS))]
    for z in zsel.itertuples():
        ri = ri_of[z.metric]
        pts = [(az, "edge") for az in edge_points(z.az_lo, z.az_hi, z.width_deg)]
        pts.append((int(z.peak_az), "peak"))
        for az, role in pts:
            targets.setdefault((az, int(z.el)), set()).add(ri)
            k = (ri, z.metric, int(z.el), az)
            if probes.get(k) != "peak":
                probes[k] = role
    probes = sorted((k, role) for k, role in probes.items())
    print(f"{len(probes)} probes over {len(targets)} unique views\n")

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    est = {(ri, az, el, n, s): [] for (az, el), ris in targets.items()
           for ri in ris for n in NOISE_PX for s in SEEDS}
    tru = {ri: [] for ri in range(len(rows))}
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
            df0 = O.project_view(joints, azimuth_deg=0.0)
            rel = M.release_frame(df0, arm, fps, M.JOINTS)
            fp = M.foot_plant_frame(df0, lead, fps, M.JOINTS, rel)
            if rel <= fp + 1 or fp < 3:
                fail += 1; continue
            ctx = {"arm": arm, "lead": lead, "trail": trail,
                   "rel": rel // DECIM, "fp": fp // DECIM, "fps": fps / DECIM,
                   "height_m": float(r.session_height_m)}
            sp = r.session_pitch

            for ri, (label, estfn, truth) in enumerate(rows):
                if isinstance(truth, tuple):
                    tval = truth[1](joints, ctx)
                else:
                    tval = poi.loc[sp, truth] if (sp in poi.index and truth in poi.columns) else np.nan
                tru[ri].append(tval)

            for (az, el), ris in targets.items():
                clean = project_cam(joints, az, el).iloc[::DECIM].reset_index(drop=True)
                for n in NOISE_PX:
                    sig = n / PPM_REF
                    for s in SEEDS:
                        if n > 0:
                            rng = np.random.default_rng(s)
                            df = clean + rng.normal(0, sig, (len(clean), clean.shape[1]))
                        else:
                            if s != SEEDS[0]:
                                continue          # 0px is deterministic
                            df = clean
                        df = smooth_coordinates(df)   # deployment smoother as-is
                        for ri in ris:
                            try:
                                est[(ri, az, el, n, s)].append(rows[ri][1](df, ctx))
                            except Exception:
                                est[(ri, az, el, n, s)].append(np.nan)
            done += 1
        except Exception:
            fail += 1
        if done and done % 50 == 0:
            print(f"  ...{done} processed")
    print(f"processed {done} / failed {fail}\n")

    clean_r2 = sweep.set_index(["metric", "az", "el"])["r2"]
    out = []
    for (ri, metric, el, az), role in probes:
        rec = {"metric": metric, "el": el, "az": az, "role": role,
               "r2_360clean": round(float(clean_r2.get((metric, az, el), np.nan)), 3)}
        for n in NOISE_PX:
            seeds = SEEDS if n > 0 else SEEDS[:1]
            vals = [r2(est[(ri, az, el, n, s)], tru[ri]) for s in seeds]
            v = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else np.nan
            rec[f"r2_{n}px"] = round(v, 3)
            rec[f"holds_{n}px"] = bool(pd.notna(v) and v >= FLOOR)
        out.append(rec)
    df_out = pd.DataFrame(out)
    path_out = os.path.join(config.OBP_VALIDATION_DIR, "zone_edge_noise_velo.csv")
    df_out.to_csv(path_out, index=False)

    print("=" * 84)
    print(f"[VELO ZONE-EDGE NOISE, DEPLOYMENT CHAIN]  120fps + SG(7,2), floor {FLOOR}")
    print("=" * 84)
    print(f"{'metric':24s}{'el':>4s}{'az':>5s}{'role':>6s}{'360cl':>8s}"
          f"{'0px':>8s}{'2px':>8s}{'4px':>8s}  verdict")
    print("-" * 84)
    for rec in out:
        verdict = ("holds" if rec["holds_4px"] else
                   "holds@2px" if rec["holds_2px"] else
                   "sampling-only" if rec["holds_0px"] else "FAILS@0px")
        print(f"{rec['metric']:24s}{rec['el']:>4d}{rec['az']:>5d}{rec['role']:>6s}"
              f"{rec['r2_360clean']:>8.2f}{rec['r2_0px']:>8.2f}"
              f"{rec['r2_2px']:>8.2f}{rec['r2_4px']:>8.2f}  {verdict}")
    print(f"\nsaved -> {path_out}")


if __name__ == "__main__":
    main()
