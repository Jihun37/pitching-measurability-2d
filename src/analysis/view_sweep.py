"""
View sweep: camera azimuth against per-quantity Level-A r-squared.

Each c3d is loaded once, projected to several azimuths in memory, and measured,
so the r-squared against the OBP 3D truth is obtained per azimuth and the best
view per quantity falls out. Azimuth 0 is the side, 90 the front from the
catcher, and anything between is oblique.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import config
import obp_project as O
import metrics as M

OBP_DATA = config.OBP_DATA_DIR
AZIMUTHS = [0, 15, 30, 45, 60, 75, 90]
ELEV = 0.0

# Our quantity -> the OBP 3D truth column.
#  Note: arm_slot is excluded here. Its definition changed to shoulder-to-hand
#        against the vertical, which is not the OBP arm_slot column, a forearm
#        projection angle. armslot_validate.py handles that comparison.
MAP = {
    "stride_length":       "stride_length",
    "lateral_trunk_tilt":  "torso_anterior_tilt_br",
    "lead_knee_extension": "lead_knee_extension_from_fp_to_br",
    "wrist_peak_speed":    "max_elbow_extension_velo",
}


def main():
    md = pd.read_csv(os.path.join(OBP_DATA, "metadata.csv"))
    poi = pd.read_csv(os.path.join(OBP_DATA, "poi", "poi_metrics.csv"))
    c3d_root = os.path.join(OBP_DATA, "c3d")

    # az -> list of feature rows
    acc = {az: [] for az in AZIMUTHS}
    done = fail = 0
    for r in md.itertuples(index=False):
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = O.load_c3d_joints(path)
            arm = O.detect_throwing_arm(joints, fps)
        except Exception:
            fail += 1; continue
        for az in AZIMUTHS:
            try:
                df = O.project_view(joints, azimuth_deg=az, elevation_deg=ELEV)
                c = M.compute_candidates(df, fps=fps, arm=arm)
                row = {"session_pitch": r.session_pitch}
                row.update({k: v for k, (v, _) in c.items()})
                acc[az].append(row)
            except Exception:
                pass
        done += 1
        if done % 100 == 0:
            print(f"  ...{done} done")
    print(f"done {done} / failed {fail}\n")

    # r2 per azimuth
    table = {m: {} for m in MAP}
    for az in AZIMUTHS:
        feat = pd.DataFrame(acc[az])
        df = feat.merge(poi, on="session_pitch", how="inner", suffixes=("_our", ""))
        for ours, truth in MAP.items():
            oc = ours + "_our" if (ours + "_our") in df.columns else ours
            d = df[[oc, truth]].dropna()
            r = d[oc].corr(d[truth]) if len(d) > 2 else np.nan
            table[ours][az] = r * r if pd.notna(r) else np.nan

    # output
    print("=" * (16 + 8 * len(AZIMUTHS) + 14))
    hdr = "metric".ljust(20) + "".join(f"{az:>7d}°" for az in AZIMUTHS) + "   best"
    print("[Level-A r2  by camera azimuth]   (0=side, 90=front)")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for ours in MAP:
        vals = table[ours]
        best_az = max(AZIMUTHS, key=lambda a: (vals[a] if pd.notna(vals[a]) else -1))
        line = ours.ljust(20) + "".join(
            f"{(vals[a] if pd.notna(vals[a]) else float('nan')):>8.2f}" for a in AZIMUTHS)
        line += f"   {best_az}° ({vals[best_az]:.2f})"
        print(line)
        rows.append({"metric": ours, "best_az": best_az,
                     "best_r2": vals[best_az], **{f"r2_{a}": vals[a] for a in AZIMUTHS}})

    _out = os.path.join(config.OBP_VALIDATION_DIR, "view_sweep_results.csv")
    pd.DataFrame(rows).to_csv(_out, index=False)
    print(f"\nsaved -> {_out}")


if __name__ == "__main__":
    main()