"""
Diamond - Stride trail-anchor comparison test
stride_anchor_test.py

Hypothesis (Ji):
  The trail (pivot) foot is fixed on the rubber. By foot plant it has often
  been dragged forward, so measuring BOTH feet at the fp frame under-measures
  stride. The true stride is:
      lead ankle @ foot plant   <->   trail ankle at its FIXED rubber position.

Variants compared (all ankle-based; toe was rejected by foot_ref_test):
  stride_fp_both    : |lead[fp] - trail[fp]|            (current logic, baseline 0.59)
  stride_trail_back : trail anchor = its REAR-MOST x within [0..fp]
                      (rear = opposite of pitch direction; no event detection)
  stride_trail_start: trail anchor = median x of its pre-motion quiet window
                      (before trail-ankle speed exceeds 20% of its peak)

Events (rel, fp) come from metrics.py so all variants share timing.
metrics.py is NOT modified -- promote the winning definition afterwards.

Usage:
    python stride_anchor_test.py
    python stride_anchor_test.py --limit 50
"""
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
import config
import obp_project as O
import metrics as M

AZIMUTHS = [0, 15, 30, 45, 60, 75, 90]
TRUTH_COL = "stride_length"

VARIANTS = ["stride_fp_both", "stride_trail_back", "stride_trail_start"]


def _x(df, joint):
    return df[f"{joint}_x"].to_numpy(float)


def pitch_direction(lead_x, fp):
    """+1 if pitch direction is +x, -1 otherwise.
    Direction = where the lead foot travels from start to foot plant."""
    d = lead_x[fp] - np.nanmedian(lead_x[:max(3, fp // 4)])
    return 1.0 if d >= 0 else -1.0


def trail_anchor_back(trail_x, fp, direction):
    """Rear-most trail-ankle x within [0..fp] (rear = -direction)."""
    seg = trail_x[:fp + 1]
    if np.all(np.isnan(seg)):
        return trail_x[fp]
    # rear-most = minimum of (x * direction)
    i = int(np.nanargmin(seg * direction))
    return float(seg[i])


def trail_anchor_start(trail_x, fp, fps):
    """Median trail x over its pre-motion quiet window.
    Quiet = before trail-ankle |vx| first exceeds 20% of its own peak in [0..fp]."""
    seg = trail_x[:fp + 1]
    v = np.abs(np.diff(seg)) * fps
    if len(v) < 3 or np.all(np.isnan(v)):
        return float(np.nanmedian(seg[:5]))
    vpk = np.nanmax(v)
    if vpk <= 0:
        return float(np.nanmedian(seg))
    idx = np.where(v > 0.20 * vpk)[0]
    m = int(idx[0]) if len(idx) else len(seg) - 1
    m = max(3, m)                       # at least a few frames of anchor
    return float(np.nanmedian(seg[:m]))


def compute_variants(df, fps, arm):
    J = M.JOINTS
    lead = "left" if arm == "right" else "right"
    trail = "right" if lead == "left" else "left"

    rel = M.release_frame(df, arm, fps, J)
    fp = M.foot_plant_frame(df, lead, fps, J, rel)
    if rel <= fp + 1 or fp < 3:
        return {}

    scale = M.pixel_stature(df, J)
    lx = _x(df, f"{lead}_ankle")
    tx = _x(df, f"{trail}_ankle")
    d = pitch_direction(lx, fp)

    out = {}
    out["stride_fp_both"] = float(abs(lx[fp] - tx[fp]) / scale)
    out["stride_trail_back"] = float(
        abs(lx[fp] - trail_anchor_back(tx, fp, d)) / scale)
    out["stride_trail_start"] = float(
        abs(lx[fp] - trail_anchor_start(tx, fp, fps)) / scale)
    return out


def corr_r2(df, x, y):
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

    rows, done, fail = [], 0, 0
    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = O.load_c3d_joints(path)
            arm = O.detect_throwing_arm(joints, fps)
            for az in AZIMUTHS:
                df2d = O.project_view(joints, azimuth_deg=az)
                feats = compute_variants(df2d, fps, arm)
                if not feats:
                    continue
                rows.append({"session_pitch": r.session_pitch,
                             "azimuth": az, **feats})
            done += 1
        except Exception:
            fail += 1
        if done and done % 100 == 0:
            print(f"  ...{done} processed")
    print(f"processed {done} / failed {fail}\n")

    feat = pd.DataFrame(rows)
    df = feat.merge(poi[["session_pitch", TRUTH_COL]],
                    on="session_pitch", how="inner")
    print(f"matched rows: {len(df)}  pitches: {df['session_pitch'].nunique()}\n")

    print("=" * 78)
    print("[Level-A] stride trail-anchor variants vs OBP stride_length (r2)")
    print("=" * 78)
    hdr = f"{'variant':20s}" + "".join(f"{az:>8d}" for az in AZIMUTHS) + "    best"
    print(hdr); print("-" * len(hdr))
    for var in VARIANTS:
        line = f"{var:20s}"; best_az, best = None, -1
        for az in AZIMUTHS:
            sub = df[df["azimuth"] == az]
            r2 = corr_r2(sub, var, TRUTH_COL)
            if pd.notna(r2) and r2 > best:
                best, best_az = r2, az
            line += f"{r2:8.2f}"
        line += f"   {best_az} ({best:.2f})"
        print(line)

    # absolute scale check @0deg: mean of each variant vs OBP mean
    s0 = df[df["azimuth"] == 0]
    print(f"\n@0deg means (scale sanity, OBP mean = {s0[TRUTH_COL].mean():.3f}):")
    for var in VARIANTS:
        print(f"  {var:20s} mean={s0[var].mean():.3f}  "
              f"r2={corr_r2(s0, var, TRUTH_COL):.3f}")

    out_csv = os.path.join(config.OBP_VALIDATION_DIR, "stride_anchor_test.csv")
    feat.to_csv(out_csv, index=False)
    print(f"\nsaved -> {out_csv}")


if __name__ == "__main__":
    main()