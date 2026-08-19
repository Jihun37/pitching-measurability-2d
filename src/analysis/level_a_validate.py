"""
Level-A validation.

The question is how much of the OBP 3D truth our side-view 2D estimate recovers,
answered per quantity as r for the sign and r-squared for the agreement. It
replaces the provisional clean/depth/degenerate tag with a measured one.

A by-product is Level-B: each 2D quantity against measured ball speed, which is
a diagnostic rather than a validation.
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

OBP_DATA = config.OBP_DATA_DIR
FEATURES = os.path.join(config.OBP_VALIDATION_DIR, "candidate_features_obp.csv")  # written by obp_project.py

# Our 2D quantity -> the OBP 3D truth column. clean means a direct
# correspondence, loose a looser one.
MAP = {
    "arm_slot":            ("arm_slot",                          "clean"),
    # stride: the adopted 0.82 definition is the /stature version
    # (stride_pct_height); our "stride_length" (/body_scale) reads only 0.59.
    "stride_pct_height":   ("stride_length",                     "clean"),
    "lateral_trunk_tilt":  ("torso_anterior_tilt_br",            "clean"),
    "lead_knee_extension": ("lead_knee_extension_from_fp_to_br", "loose"),
    "wrist_peak_speed":    ("max_elbow_extension_velo",          "loose"),
}


def main():
    feat = pd.read_csv(FEATURES)
    # Our features and the OBP truth share names (arm_slot, stride_length), so
    # the merge would collide. Prefix ours.
    feat = feat.rename(columns={c: f"our_{c}" for c in feat.columns
                                if c != "session_pitch"})
    poi = pd.read_csv(os.path.join(OBP_DATA, "poi", "poi_metrics.csv"))
    df = feat.merge(poi, on="session_pitch", how="inner")
    print(f"pitches matched: {len(df)}\n")

    print("=" * 64)
    print("[Level-A] our 2D estimate  vs  the OBP 3D truth")
    print("=" * 64)
    print(f"{'our metric':22s}{'OBP truth':28s}{'r':>7s}{'r²':>7s}  type")
    print("-" * 64)
    rows = []
    for ours, (truth, kind) in MAP.items():
        oc = f"our_{ours}"
        if oc in df and truth in df:
            d = df[[oc, truth]].dropna()
            r = d[oc].corr(d[truth]) if len(d) > 2 else np.nan
            rows.append((ours, truth, r, kind))
    for ours, truth, r, kind in sorted(rows, key=lambda t: -abs(t[2])):
        verdict = ("strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak")
        print(f"{ours:22s}{truth:28s}{r:>7.3f}{r*r:>7.3f}  {kind:5s} [{verdict}]")

    print("\n" + "=" * 64)
    print("[Level-B diagnostic] our 2D estimate against measured ball speed")
    print("=" * 64)
    sp = "pitch_speed_mph"
    diag = []
    for ours in MAP:
        oc = f"our_{ours}"
        if oc in df:
            d = df[[oc, sp]].dropna()
            diag.append((ours, d[oc].corr(d[sp])))
    for ours, r in sorted(diag, key=lambda t: -abs(t[1])):
        print(f"  {ours:22s} r={r:+.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=FEATURES, help="the batch output csv from obp_project")
    a = ap.parse_args()
    FEATURES = a.features
    main()