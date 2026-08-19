"""
Diamond - Combined hypothesis test (H1-H3)
hypothesis_test.py

Tests three definition hypotheses in one batch, same pattern as
foot_ref_test / stride_anchor_test. metrics.py is NOT modified.

H1) Lead knee: ABSOLUTE angle at release (and at fp) should beat the
    fp->br DIFFERENCE (0.622), because a difference stacks the detection
    error of two frames while "lead leg block" is a single-instant quantity.
      variants: knee_at_br, knee_at_fp, knee_ext_diff (baseline)

H2) Trunk tilt plane check: our side-view tilt (mid-hip -> mid-shoulder in
    the image plane) is compared against BOTH OBP anterior tilt and any
    lateral tilt column, to settle which plane we are actually measuring.
      variants: trunk_tilt_br (current def) vs both truth planes

H3) Reference points: release height with HEEL-based ground (true contact)
    vs ANKLE-based ground (current); wrist peak speed normalized by
    stature vs body_scale (current).
      variants: relh_ankle, relh_heel, wrist_spd_scale, wrist_spd_stature

OBP poi column names for some truths are not 100% certain, so each truth is
a candidate LIST; the first column that exists is used, and all poi columns
matching the keywords are printed first so mappings can be corrected.

Usage:
    python hypothesis_test.py --limit 50
    python hypothesis_test.py
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
import ezc3d

AZIMUTHS = [0, 15, 30, 45, 60, 75, 90]

HEEL_MARKERS = {"left_heel": ["LHEE"], "right_heel": ["RHEE"]}

# our variant -> list of candidate OBP truth columns (first existing wins)
TRUTH = {
    "knee_at_br":        ["lead_knee_extension_br", "lead_knee_flexion_br",
                          "lead_knee_angle_br"],
    "knee_at_fp":        ["lead_knee_extension_fp", "lead_knee_flexion_fp",
                          "lead_knee_angle_fp"],
    "knee_ext_diff":     ["lead_knee_extension_from_fp_to_br"],
    "trunk_tilt_br_vs_anterior": ["torso_anterior_tilt_br"],
    "trunk_tilt_br_vs_lateral":  ["torso_lateral_tilt_br", "trunk_lateral_tilt_br",
                                  "torso_sideways_tilt_br"],
    "relh_ankle":        ["release_height", "rel_height", "release_pos_z"],
    "relh_heel":         ["release_height", "rel_height", "release_pos_z"],
    "wrist_spd_scale":   ["max_elbow_extension_velo"],
    "wrist_spd_stature": ["max_elbow_extension_velo"],
}

KEYWORDS = ["knee", "torso", "trunk", "release", "rel_"]


def load_with_heels(path):
    c = ezc3d.c3d(path)
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    fps = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    pts = c["data"]["points"][:3]
    idx = {l: i for i, l in enumerate(labels)}
    joints = {}
    for name, mks in {**O.MARKER_MAP, **HEEL_MARKERS}.items():
        mk = [m for m in mks if m in idx]
        if not mk:
            if name in HEEL_MARKERS:
                continue
            raise KeyError(f"missing marker: {name}")
        joints[name] = np.nanmean([pts[:, idx[m], :] for m in mk], axis=0)
    return joints, fps


def _xy(df, joint):
    return df[f"{joint}_x"].to_numpy(float), df[f"{joint}_y"].to_numpy(float)


def compute_variants(df, fps, arm):
    J = M.JOINTS
    lead = "left" if arm == "right" else "right"
    rel = M.release_frame(df, arm, fps, J)
    fp = M.foot_plant_frame(df, lead, fps, J, rel)
    if rel <= fp + 1 or fp < 3:
        return {}
    out = {}

    # ---- H1: knee absolute vs difference ----
    hx, hy = _xy(df, f"{lead}_hip")
    kx, ky = _xy(df, f"{lead}_knee")
    ax_, ay = _xy(df, f"{lead}_ankle")
    knee = M._angle(hx, hy, kx, ky, ax_, ay)
    out["knee_at_br"] = float(knee[rel])
    out["knee_at_fp"] = float(knee[fp])
    out["knee_ext_diff"] = float(knee[rel] - knee[fp])

    # ---- H2: trunk tilt (current definition, value reused for both truths) ----
    lhx, lhy = _xy(df, "left_hip"); rhx, rhy = _xy(df, "right_hip")
    lsx, lsy = _xy(df, "left_shoulder"); rsx, rsy = _xy(df, "right_shoulder")
    midhx, midhy = (lhx + rhx) / 2, (lhy + rhy) / 2
    midsx, midsy = (lsx + rsx) / 2, (lsy + rsy) / 2
    tilt = np.degrees(np.arctan2(midsx - midhx, -(midsy - midhy)))
    out["trunk_tilt_br_vs_anterior"] = float(tilt[rel])
    out["trunk_tilt_br_vs_lateral"] = float(tilt[rel])

    # ---- H3a: release height, ankle vs heel ground ----
    wkey = f"{arm}_wrist"
    wx, wy = _xy(df, wkey)
    scale = M.body_scale_px(df, J)
    lay = _xy(df, "left_ankle")[1]; ray = _xy(df, "right_ankle")[1]
    ground_an = np.nanmax(np.concatenate([lay, ray]))
    out["relh_ankle"] = float((ground_an - wy[rel]) / scale)
    if "left_heel_y" in df.columns and "right_heel_y" in df.columns:
        lhe = _xy(df, "left_heel")[1]; rhe = _xy(df, "right_heel")[1]
        ground_he = np.nanmax(np.concatenate([lhe, rhe]))
        out["relh_heel"] = float((ground_he - wy[rel]) / scale)

    # ---- H3b: wrist peak speed normalization ----
    spd = M._speed(wx, wy, fps)
    pk = float(np.nanmax(spd))
    stat = M.pixel_stature(df, J)
    out["wrist_spd_scale"] = pk / scale
    out["wrist_spd_stature"] = pk / stat

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

    # show available truth columns so mappings can be verified/corrected
    print("[poi columns matching keywords]")
    for kw in KEYWORDS:
        cols = [c for c in poi.columns if kw in c.lower()]
        if cols:
            print(f"  {kw:8s}: {cols}")
    print()

    resolved = {}
    for var, cands in TRUTH.items():
        hit = next((c for c in cands if c in poi.columns), None)
        resolved[var] = hit
        if hit is None:
            print(f"  !! no truth column found for {var} (candidates: {cands})")
    print()

    rows, done, fail = [], 0, 0
    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = load_with_heels(path)
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
    df = feat.merge(poi, on="session_pitch", how="inner", suffixes=("_our", ""))
    print(f"matched rows: {len(df)}  pitches: {df['session_pitch'].nunique()}\n")

    print("=" * 86)
    print("[Level-A] hypothesis variants vs OBP truth, by azimuth (r2)")
    print("=" * 86)
    hdr = f"{'variant':28s}" + "".join(f"{az:>7d}" for az in AZIMUTHS) + "   best"
    print(hdr); print("-" * len(hdr))
    summary = []
    for var, truth in resolved.items():
        if truth is None:
            print(f"{var:28s}  (no truth column -> skipped)")
            continue
        oc = var + "_our" if var + "_our" in df.columns else var
        if oc not in df.columns:
            print(f"{var:28s}  (not computed)")
            continue
        line = f"{var:28s}"; best_az, best = None, -1
        for az in AZIMUTHS:
            sub = df[df["azimuth"] == az]
            r2 = corr_r2(sub, oc, truth)
            if pd.notna(r2) and r2 > best:
                best, best_az = r2, az
            line += f"{r2:7.2f}"
        line += f"   {best_az} ({best:.2f})  [truth: {truth}]"
        print(line)
        summary.append({"variant": var, "truth": truth,
                        "best_az": best_az, "best_r2": best})

    print("\nHead-to-head @0deg:")
    s0 = df[df["azimuth"] == 0]
    pairs = [
        ("H1 knee", ["knee_ext_diff", "knee_at_fp", "knee_at_br"]),
        ("H2 trunk", ["trunk_tilt_br_vs_anterior", "trunk_tilt_br_vs_lateral"]),
        ("H3a relh", ["relh_ankle", "relh_heel"]),
        ("H3b wrist", ["wrist_spd_scale", "wrist_spd_stature"]),
    ]
    for label, vars_ in pairs:
        parts = []
        for v in vars_:
            t = resolved.get(v)
            oc = v + "_our" if v + "_our" in s0.columns else v
            if t and oc in s0.columns:
                parts.append(f"{v}={corr_r2(s0, oc, t):.3f}")
        print(f"  {label:10s} " + "   ".join(parts))

    out_csv = os.path.join(config.OBP_VALIDATION_DIR, "hypothesis_test.csv")
    feat.to_csv(out_csv, index=False)
    print(f"\nsaved -> {out_csv}")


if __name__ == "__main__":
    main()