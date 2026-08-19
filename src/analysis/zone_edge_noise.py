"""
Diamond - noise robustness at valid-zone boundary angles.

The valid zones (angle_zone_sweep.py) are measured on noiseless projections.
Real phone video carries pose jitter, so a zone edge that only just clears the
r2 floor may fall below it under noise. This probes exactly the edge angles:
for each usable arc, take the inside grid point nearest each interpolated edge
(full-ring zones get az 0/90 as representatives), inject Gaussian keypoint
noise, and check whether r2 stays >= 0.6.

Noise model: identical to analysis/noise_sweep.py (obp_project.project_view):
iid Gaussian per coordinate per frame, sigma = noise_px at ppm=300. project_cam
outputs meters (ppm=1), so sigma_m = noise_px / 300. Levels 2px and 4px
(~1.4% / ~2.8% of the 142px body-scale reference), seeds averaged.

Conventions otherwise identical to angle_zone_sweep (adopted rows, events
detected once on the clean el=0/az=0 side view).

Output: OBP_VALIDATION_DIR/zone_edge_noise.csv
        (metric, el, az, r2_clean, r2_2px, r2_4px, holds_2px, holds_4px)

Run:  cd src\\analysis
      python zone_edge_noise.py --limit 10    (smoke test)
      python zone_edge_noise.py
"""
import os, sys, argparse, math
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

AZ_STEP = 15
FLOOR = 0.60
NOISE_PX = [2, 4]          # @ppm=300, same scale as noise_sweep.py
SEEDS = [0, 1, 2]
PPM_REF = 300.0


def edge_points(az_lo, az_hi, width):
    """Inside grid points nearest the two interpolated edges of a usable arc."""
    if width >= 360:
        return [0, 90]                       # full ring: representative views
    lo = int(math.ceil(az_lo / AZ_STEP)) * AZ_STEP % 360
    hi = int(math.floor(az_hi / AZ_STEP)) * AZ_STEP % 360
    return sorted({lo, hi})


def r2(e, t):
    e = np.asarray(e, float); t = np.asarray(t, float)
    m = np.isfinite(e) & np.isfinite(t)
    return np.corrcoef(e[m], t[m])[0, 1] ** 2 if m.sum() > 2 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    zones = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR, "angle_zone_table.csv"))
    sweep = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR, "angle_zone_sweep.csv"))
    rows = adopted_rows()
    ri_of = {label.strip(): ri for ri, (label, _, _) in enumerate(rows)}

    # (az, el) -> set of metric indices to evaluate there.
    # Each usable arc contributes its two edge grid points; the zone PEAK is
    # probed too as a reference: an edge that fails while the peak holds means
    # the zone shrinks; a peak that also fails means the estimator is globally
    # noise-sensitive and the probe says nothing about the boundary per se.
    targets = {}          # view -> set(ri)
    probes = {}           # (ri, metric, el, az) -> role
    for z in zones[zones.tier == "usable"].itertuples():
        ri = ri_of[z.metric]
        pts = [(az, "edge") for az in edge_points(z.az_lo, z.az_hi, z.width_deg)]
        pts.append((int(z.peak_az), "peak"))
        for az, role in pts:
            key = (az, int(z.el))
            targets.setdefault(key, set()).add(ri)
            k = (ri, z.metric, int(z.el), az)
            if probes.get(k) != "peak":     # peak label wins on overlap
                probes[k] = role
    probes = sorted((k, role) for k, role in probes.items())
    print(f"{len(probes)} edge probes over {len(targets)} unique views\n")

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
                   "rel": rel, "fp": fp, "fps": fps,
                   "height_m": float(r.session_height_m)}
            sp = r.session_pitch

            for ri, (label, estfn, truth) in enumerate(rows):
                if isinstance(truth, tuple):
                    tval = truth[1](joints, ctx)
                else:
                    tval = poi.loc[sp, truth] if (sp in poi.index and truth in poi.columns) else np.nan
                tru[ri].append(tval)

            for (az, el), ris in targets.items():
                clean = project_cam(joints, az, el)
                for n in NOISE_PX:
                    sig = n / PPM_REF
                    for s in SEEDS:
                        rng = np.random.default_rng(s)
                        df = clean + rng.normal(0, sig, (len(clean), clean.shape[1]))
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
               "r2_clean": round(float(clean_r2.get((metric, az, el), np.nan)), 3)}
        for n in NOISE_PX:
            vals = [r2(est[(ri, az, el, n, s)], tru[ri]) for s in SEEDS]
            v = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else np.nan
            rec[f"r2_{n}px"] = round(v, 3)
            rec[f"holds_{n}px"] = bool(pd.notna(v) and v >= FLOOR)
        out.append(rec)
    df_out = pd.DataFrame(out)
    path_out = os.path.join(config.OBP_VALIDATION_DIR, "zone_edge_noise.csv")
    df_out.to_csv(path_out, index=False)

    print("=" * 74)
    print(f"[ZONE-EDGE NOISE]  r2 at usable-zone edges (floor {FLOOR}, seeds avg)")
    print("=" * 74)
    print(f"{'metric':24s}{'el':>4s}{'az':>5s}{'role':>6s}{'clean':>8s}{'2px':>8s}{'4px':>8s}  verdict")
    print("-" * 78)
    for rec in out:
        verdict = ("holds" if rec["holds_4px"] else
                   "holds@2px" if rec["holds_2px"] else "FAILS")
        print(f"{rec['metric']:24s}{rec['el']:>4d}{rec['az']:>5d}{rec['role']:>6s}"
              f"{rec['r2_clean']:>8.2f}{rec['r2_2px']:>8.2f}{rec['r2_4px']:>8.2f}  {verdict}")
    print(f"\nsaved -> {path_out}")


if __name__ == "__main__":
    main()
