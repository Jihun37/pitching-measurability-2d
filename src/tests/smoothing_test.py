"""
Smoothing robustness: inject noise, smooth, and see how much r-squared returns.

noise_sweep ran with no smoothing at all, which is the worst case. A real
recording is continuous in time, so a Savitzky-Golay filter recovers part of
what the noise cost. This sweeps noise against window width at azimuth 0 and
asks whether the three reliable quantities survive the noise level that broke
them there, about 2.8 per cent of stature, or 4 px. Windows are in frames at the
OpenBiomechanics rate of 360 Hz.
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
from smoother import smooth_coordinates

OBP_DATA = config.OBP_DATA_DIR
AZ = 0
NOISE_PX = [0, 2, 4, 6]              # ~0 / 1.4 / 2.8 / 4.2 % of body
SMOOTH = [None, 7, 15, 31]          # SG window in frames at 360 Hz; None means no smoothing
SEEDS = [0, 1, 2]

MAP = {
    "lateral_trunk_tilt":  "torso_anterior_tilt_br",
    "lead_knee_extension": "lead_knee_extension_from_fp_to_br",
    "stride_length":       "stride_length",
}


def main():
    md = pd.read_csv(os.path.join(OBP_DATA, "metadata.csv"))
    poi = pd.read_csv(os.path.join(OBP_DATA, "poi", "poi_metrics.csv"))
    c3d_root = os.path.join(OBP_DATA, "c3d")

    acc = {(n, w, s): [] for n in NOISE_PX for w in SMOOTH for s in SEEDS}
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
                df0 = O.project_view(joints, azimuth_deg=AZ, noise_px=n, seed=s)
                for w in SMOOTH:
                    df = df0 if w is None else smooth_coordinates(df0, window=w)
                    try:
                        c = M.compute_candidates(df, fps=fps, arm=arm)
                        row = {"session_pitch": r.session_pitch}
                        row.update({k: v for k, (v, _) in c.items()})
                        acc[(n, w, s)].append(row)
                    except Exception:
                        pass
        done += 1
        if done % 100 == 0:
            print(f"  ...{done} done")
    print(f"done {done} / failed {fail}\n")

    def r2(n, w, ours, truth):
        vals = []
        for s in SEEDS:
            feat = pd.DataFrame(acc[(n, w, s)])
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

    rows = []
    for ours, truth in MAP.items():
        print(f"=== {ours} (r²) ===")
        hdr = "noise".ljust(14) + "".join(
            f"{('SG'+str(w)) if w else 'none':>9s}" for w in SMOOTH)
        print(hdr); print("-" * len(hdr))
        for n in NOISE_PX:
            pct = n / 142.0 * 100
            cells = {w: r2(n, w, ours, truth) for w in SMOOTH}
            print(f"{n}px(~{pct:.1f}%)".ljust(14) +
                  "".join(f"{cells[w]:>9.2f}" for w in SMOOTH))
            rows.append({"metric": ours, "noise_px": n,
                         **{f"SG{w}" if w else "none": cells[w] for w in SMOOTH}})
        print()

    out = os.path.join(config.OBP_VALIDATION_DIR, "smoothing_results.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()