"""
Diamond — consolidated camera-azimuth r2 sweep (adopted + rejected variants)
angle_sweep_full.py

WHY
  Several per-azimuth r2 tables were PRINTED by the individual test scripts but
  never persisted in one place. The paper needs a single "why 0deg / 90deg"
  table that includes BOTH the adopted metrics and the rejected (too-low)
  variants, each across all 7 camera angles. This script produces exactly that.

HOW (low risk)
  Every variant is computed by the SAME, already-validated function from the
  existing scripts — nothing is re-implemented here. Each c3d is loaded ONCE
  (with heels, so the heel-ground variant works), projected to all 7 azimuths,
  and every metric is evaluated on that one projection. One pass over OBP
  instead of running 5 scripts separately.

  Built-in correctness check: the recomputed core rows
  (stride_length / lateral_trunk_tilt / lead_knee_extension / wrist_peak_speed)
  must reproduce view_sweep_results.csv, and the knee-velocity / elbow rows must
  reproduce extended_metrics_view_sweep.csv. If they match, trust the new rows.

CONVENTIONS (kept identical to the source scripts)
  - Pearson r2  for poi-column and 3D-direct-column truths (all *_test.py).
  - determination R2 (1 - SSres/SStot) for arm_slot, matching armslot_validate.py.

OUTPUT
  OBP_VALIDATION_DIR/angle_sweep_full.csv     (metric x azimuth, tidy + wide)

USAGE
  python angle_sweep_full.py                  # full OBP
  python angle_sweep_full.py --limit 40       # quick smoke test
"""
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import config
import obp_project as O
import metrics as M

# --- reuse proven variant computations + truth maps (no re-implementation) ---
from hypothesis_test import (compute_variants as cv_hyp,
                             load_with_heels,
                             TRUTH as HYP_TRUTH)
from foot_ref_test import compute_variants as cv_stride
from extended_metrics_test import (compute_extended as cv_ext,
                                   TRUTH as EXT_TRUTH)
from armslot_validate import (truth_3d_frontal, estimate_2d, release_of,
                              r2 as armslot_R2)
from knee_fair_test import knee3d, est_knee_series

AZ = [0, 15, 30, 45, 60, 75, 90]

# core-4 (metrics.compute_candidates output keys) -> poi truth  (== view_sweep.MAP)
CORE_TRUTH = {
    "stride_length":       "stride_length",
    "lateral_trunk_tilt":  "torso_anterior_tilt_br",
    "lead_knee_extension": "lead_knee_extension_from_fp_to_br",
    "wrist_peak_speed":    "max_elbow_extension_velo",
}

# stride trail-anchor variants -> poi truth
STRIDE_TRUTH = {v: "stride_length"
                for v in ("stride_fp_both", "stride_trail_back", "stride_trail_start")}

# Pearson-r2 metrics whose 2D estimate lives in the wide feature frame and whose
# truth is a single poi column (resolved against poi at runtime). Candidate lists
# come straight from HYP_TRUTH; single strings from the others.
POI_TRUTH = {}
POI_TRUTH.update({k: [v] for k, v in CORE_TRUTH.items()})
POI_TRUTH.update({k: [v] for k, v in STRIDE_TRUTH.items()})
POI_TRUTH.update({k: (v if isinstance(v, list) else [v]) for k, v in HYP_TRUTH.items()})
POI_TRUTH.update({k: [v] for k, (v, _note) in EXT_TRUTH.items() if v is not None})

# metrics with NO Level-A truth (Level-B only) — reported as such, not as r2
NO_TRUTH = [k for k, (v, _n) in EXT_TRUTH.items() if v is None]

# paper bookkeeping: category + readable label (status is descriptive only)
META = {
    # adopted
    "knee_abs_br":         ("adopted",  "Lead Knee Angle (abs @release)"),
    "stride_trail_start":  ("adopted",  "Stride Length (trail-anchor)"),
    "lateral_trunk_tilt":  ("adopted",  "Trunk Tilt (anterior)"),
    "knee_ext_velo_br":    ("adopted",  "Knee Ext. Velocity (BR)"),
    "wrist_spd_stature":   ("adopted",  "Wrist Speed (/stature)"),
    "relh_ankle":          ("adopted",  "Release Height (ankle ground)"),
    # rejected / diagnostic variants
    "knee_abs_fp":         ("rejected", "Knee abs @fp (F3, low discrim.)"),
    "knee_ext_diff":       ("rejected", "Knee extension DIFF (fp->br)"),
    "lead_knee_extension": ("rejected", "Knee extension DIFF (poi col)"),
    "stride_fp_both":      ("rejected", "Stride both-feet @fp (baseline)"),
    "stride_trail_back":   ("rejected", "Stride trail rear-most anchor"),
    "trunk_tilt_br_vs_lateral": ("rejected", "Trunk tilt as LATERAL plane (F4)"),
    "trunk_tilt_br_vs_anterior":("rejected", "Trunk tilt as anterior (==adopted)"),
    "relh_heel":           ("rejected", "Release height heel ground (F6)"),
    "wrist_spd_scale":     ("rejected", "Wrist speed /body_scale (F7)"),
    "wrist_peak_speed":    ("rejected", "Wrist peak /body_scale (core dup)"),
    "stride_length":       ("rejected", "Stride both-feet (core dup of 0.44)"),
    "knee_ext_velo_max":   ("rejected", "Knee ext. velo @MAX (timing)"),
    "knee_ext_velo_fp":    ("rejected", "Knee ext. velo @FP (timing)"),
    "elbow_flexion_fp":    ("rejected", "Elbow flexion @fp (weak, multiview)"),
    "max_elbow_flexion":   ("rejected", "Max elbow flexion (F5)"),
    # special: arm slot (own R2 convention, frontal-optimal)
    "arm_slot":            ("adopted",  "Arm Slot (shoulder->hand vs vertical)"),
}


def pearson_r2(df, x, y):
    d = df[[x, y]].dropna()
    if len(d) <= 2:
        return np.nan
    r = d[x].corr(d[y])
    return r * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    # resolve poi truth columns once (first existing candidate wins)
    resolved = {}
    for m, cands in POI_TRUTH.items():
        hit = next((c for c in cands if c in poi.columns), None)
        resolved[m] = hit
        if hit is None:
            print(f"  !! no poi truth column for {m}  (candidates: {cands})")

    feat_rows = []          # wide per (pitch, az): all Pearson-truth 2D estimates
    knee3d_rows = []        # per pitch: 3D-direct knee absolute truth (br, fp)
    as_truth, as_est = [], {az: [] for az in AZ}   # arm slot (determination R2)
    done = fail = 0

    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1
            continue
        try:
            joints, fps = load_with_heels(path)      # ONE load, includes heels
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"

            # --- arm slot: 3D frontal truth + per-az 2D estimate (raw joints) ---
            rel_as = release_of(joints, arm, fps)
            t_as = truth_3d_frontal(joints, arm, rel_as)
            e_as = {az: estimate_2d(joints, arm, rel_as, az) for az in AZ}

            # --- 3D-direct knee absolute truth (az-independent) ---
            rel0 = M.release_frame(O.project_view(joints, azimuth_deg=0.0),
                                   arm, fps, M.JOINTS)
            fp0 = M.foot_plant_frame(O.project_view(joints, azimuth_deg=0.0),
                                     lead, fps, M.JOINTS, rel0)
            t_knee_br = knee3d(joints, lead, rel0)
            t_knee_fp = knee3d(joints, lead, fp0)

            # --- per-azimuth 2D estimates (one projection per az) ---
            ok_any = False
            for az in AZ:
                df = O.project_view(joints, azimuth_deg=az)
                feats = {"session_pitch": r.session_pitch, "azimuth": az}

                # core-4 (and any other compute_candidates keys; arm_slot dropped)
                cc = M.compute_candidates(df, fps=fps, arm=arm)
                feats.update({k: v for k, (v, _) in cc.items() if k != "arm_slot"})

                # rejected/diagnostic variant groups
                feats.update(cv_hyp(df, fps, arm))
                feats.update(cv_stride(df, fps, arm))
                feats.update(cv_ext(df, fps, arm))

                # knee absolute @br / @fp from the projected series
                ks = est_knee_series(df, lead)
                if rel0 < len(ks) and fp0 < len(ks):
                    feats["knee_abs_br"] = float(ks[rel0])
                    feats["knee_abs_fp"] = float(ks[fp0])

                feat_rows.append(feats)
                ok_any = True

            if ok_any:
                knee3d_rows.append({"session_pitch": r.session_pitch,
                                    "knee3d_br": t_knee_br, "knee3d_fp": t_knee_fp})
                as_truth.append(t_as)
                for az in AZ:
                    as_est[az].append(e_as[az])
                done += 1
        except Exception as e:
            fail += 1
        if done and done % 100 == 0:
            print(f"  ...{done} processed")

    print(f"\nprocessed {done} / failed {fail}")

    feat = pd.DataFrame(feat_rows)
    knee3d_df = pd.DataFrame(knee3d_rows)
    df = feat.merge(poi, on="session_pitch", how="inner", suffixes=("_our", ""))
    df = df.merge(knee3d_df, on="session_pitch", how="left")
    print(f"matched rows: {len(df)}   pitches: {df['session_pitch'].nunique()}\n")

    # ----- assemble tidy table: metric x azimuth r2 -----
    out = []

    # (a) Pearson-truth metrics (poi columns)
    for m, truth in resolved.items():
        if truth is None:
            continue
        oc = m + "_our" if (m + "_our") in df.columns else m
        if oc not in df.columns:
            continue
        row = {"metric": m, "truth_basis": "poi", "truth_col": truth}
        vals = []
        for az in AZ:
            sub = df[df["azimuth"] == az]
            v = pearson_r2(sub, oc, truth)
            row[f"r2_{az}"] = v
            vals.append(v if pd.notna(v) else -1)
        ba = AZ[int(np.argmax(vals))]
        row["best_az"], row["best_r2"] = ba, row[f"r2_{ba}"]
        out.append(row)

    # (b) 3D-direct knee absolute (br, fp) — Pearson vs 3D truth
    for m, truth in [("knee_abs_br", "knee3d_br"), ("knee_abs_fp", "knee3d_fp")]:
        if m not in df.columns:
            continue
        row = {"metric": m, "truth_basis": "3D-direct", "truth_col": truth}
        vals = []
        for az in AZ:
            sub = df[df["azimuth"] == az]
            v = pearson_r2(sub, m, truth)
            row[f"r2_{az}"] = v
            vals.append(v if pd.notna(v) else -1)
        ba = AZ[int(np.argmax(vals))]
        row["best_az"], row["best_r2"] = ba, row[f"r2_{ba}"]
        out.append(row)

    # (c) arm slot — determination R2 vs 3D frontal truth
    t_arr = np.array(as_truth, float)
    row = {"metric": "arm_slot", "truth_basis": "3D-frontal(R2)", "truth_col": "frontal_YZ"}
    vals = []
    for az in AZ:
        v = armslot_R2(t_arr, np.array(as_est[az], float))
        row[f"r2_{az}"] = v
        vals.append(v if pd.notna(v) else -1)
    ba = AZ[int(np.argmax(vals))]
    row["best_az"], row["best_r2"] = ba, row[f"r2_{ba}"]
    out.append(row)

    table = pd.DataFrame(out)

    # attach category/label, order adopted-first
    table["category"] = table["metric"].map(lambda k: META.get(k, ("other", k))[0])
    table["label"] = table["metric"].map(lambda k: META.get(k, ("other", k))[1])
    cat_order = {"adopted": 0, "rejected": 1, "other": 2}
    table = table.sort_values(
        by=["category", "best_r2"],
        key=lambda s: s.map(cat_order) if s.name == "category" else -s,
    ).reset_index(drop=True)

    cols = (["category", "metric", "label", "truth_basis", "truth_col"]
            + [f"r2_{az}" for az in AZ] + ["best_az", "best_r2"])
    table = table[cols]

    # ----- print -----
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("=" * 100)
    print("[Camera azimuth r2 sweep]   0deg=side(sagittal)  90deg=front(catcher)")
    print("  Pearson r2 for poi/3D-direct columns; arm_slot uses determination R2")
    print("=" * 100)
    hdr = f"{'metric':26s}" + "".join(f"{az:>7d}" for az in AZ) + "   best   cat"
    print(hdr); print("-" * len(hdr))
    for _, rr in table.iterrows():
        line = f"{rr['metric']:26s}"
        for az in AZ:
            v = rr[f"r2_{az}"]
            line += (f"{v:7.2f}" if pd.notna(v) else f"{'·':>7s}")
        line += f"   {int(rr['best_az'])} ({rr['best_r2']:.2f})  {rr['category']}"
        print(line)

    if NO_TRUTH:
        print("\n[no Level-A truth — reported as Level-B only, not in r2 table]")
        for m in NO_TRUTH:
            print(f"  {m}")

    out_csv = os.path.join(config.OBP_VALIDATION_DIR, "angle_sweep_full.csv")
    table.to_csv(out_csv, index=False)
    print(f"\nsaved -> {out_csv}")
    print("\nSANITY CHECK: stride_length / lateral_trunk_tilt / lead_knee_extension /")
    print("wrist_peak_speed rows should match view_sweep_results.csv;")
    print("knee_ext_velo_* / elbow_* rows should match extended_metrics_view_sweep.csv.")


if __name__ == "__main__":
    main()