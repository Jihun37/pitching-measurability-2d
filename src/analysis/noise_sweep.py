"""
Noise robustness: keypoint noise against Level-A r-squared, from the side.

A real phone recording carries pose jitter, which the clean projection has none
of. Gaussian noise in pixels is added to the projected coordinates to see how far
the three reliable quantities, and arm_slot, hold up. Noise is also reported as a
percentage of body_scale, about 142 px at 300 pixels per metre, and each level is
averaged over several seeds.
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
AZ = 0
NOISE_PX = [0, 1, 2, 4, 6, 8]      # about 0 to 5.6 per cent of a body_scale near 142 px
SEEDS = [0, 1, 2]
BODY_SCALE_REF = 142.0             # for reporting noise as a percentage

MAP = {
    "lateral_trunk_tilt":  "torso_anterior_tilt_br",
    "lead_knee_extension": "lead_knee_extension_from_fp_to_br",
    "stride_length":       "stride_length",
    "arm_slot":            "arm_slot",
}


def main():
    md = pd.read_csv(os.path.join(OBP_DATA, "metadata.csv"))
    poi = pd.read_csv(os.path.join(OBP_DATA, "poi", "poi_metrics.csv"))
    c3d_root = os.path.join(OBP_DATA, "c3d")

    # (noise, seed) -> list of feature rows
    acc = {(n, s): [] for n in NOISE_PX for s in SEEDS}
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
        for n in NOISE_PX:
            for s in SEEDS:
                try:
                    df = O.project_view(joints, azimuth_deg=AZ, noise_px=n, seed=s)
                    c = M.compute_candidates(df, fps=fps, arm=arm)
                    row = {"session_pitch": r.session_pitch}
                    row.update({k: v for k, (v, _) in c.items()})
                    acc[(n, s)].append(row)
                except Exception:
                    pass
        done += 1
        if done % 100 == 0:
            print(f"  ...{done} done")
    print(f"done {done} / failed {fail}\n")

    # r2 per noise level, averaged over seeds
    def r2_for(noise, ours, truth):
        vals = []
        for s in SEEDS:
            feat = pd.DataFrame(acc[(noise, s)])
            if feat.empty:
                continue
            df = feat.merge(poi, on="session_pitch", how="inner", suffixes=("_our", ""))
            oc = ours + "_our" if (ours + "_our") in df.columns else ours
            d = df[[oc, truth]].dropna()
            if len(d) > 2:
                rr = d[oc].corr(d[truth])
                if pd.notna(rr):
                    vals.append(rr * rr)
        return np.mean(vals) if vals else np.nan

    print("[side view (0 deg) noise robustness -- Level-A r2, averaged over seeds]")
    hdr = "noise".ljust(14) + "".join(f"{m[:10]:>12s}" for m in MAP)
    print(hdr); print("-" * len(hdr))
    rows = []
    for n in NOISE_PX:
        pct = n / BODY_SCALE_REF * 100
        cells = {ours: r2_for(n, ours, truth) for ours, truth in MAP.items()}
        line = f"{n}px(~{pct:.1f}%)".ljust(14) + "".join(f"{cells[m]:>12.2f}" for m in MAP)
        print(line)
        rows.append({"noise_px": n, "pct_body": round(pct, 1), **cells})
    _out = os.path.join(config.OBP_VALIDATION_DIR, "noise_sweep_results.csv")
    pd.DataFrame(rows).to_csv(_out, index=False)
    print(f"\nsaved -> {_out}")


if __name__ == "__main__":
    main()