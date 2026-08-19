"""
Diamond - Repeatability / Consistency validation
repeatability_test.py

Question: when the same pitcher throws multiple pitches, does OUR 2D
within-pitcher variability recover the TRUE (3D) within-pitcher variability?
This is the platform's exclusive territory -- mocap labs measure one pitch
precisely; we measure many pitches repeatedly.

Three outputs, one pass over OBP c3d (side view, az=0, new metrics defs):

  A) Pitcher pitch-count distribution
       -> decides the minimum-pitches threshold (--min-pitches, default 4).

  B) Variance recovery (the core result)
       per pitcher: SD of our estimates  vs  SD of truth values
       -> correlate ACROSS pitchers -> r2 per metric.
       Interpretation: our SD = (true variability) + (our measurement noise).
       Metrics with high value-r2 (knee angle 0.94) should recover SD well;
       low value-r2 metrics may not -- the value-r2 vs SD-r2 comparison is
       itself a paper result.

  C) N-pitch averaging recovery curve
       pitcher trait = mean of truth over ALL pitches.
       our N-pitch average (random subsets, seed-avg) vs trait -> r2(N).
       -> grounds the product rule "profile = average of N clips".

Truth sources per metric:
  stride_length        poi: stride_length            (OBP-column)
  trunk_anterior_tilt  poi: torso_anterior_tilt_br   (OBP-column)
  knee_ext_velo_br     poi: lead_knee_extension_angular_velo_br (OBP-column)
  lead_knee_angle      3D direct (hip-knee-ankle @rel)
  wrist_speed          3D direct (wrist 3D peak speed)
  release_height       3D direct (wrist z - min foot z @rel)

Usage:
    python repeatability_test.py --limit 80      # smoke
    python repeatability_test.py
    python repeatability_test.py --min-pitches 5
"""
import os, sys, argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(_HERE, "..", "stage3"))

import config
import obp_project as O
import metrics as M

METRICS = ["lead_knee_angle", "stride_length", "trunk_anterior_tilt",
           "knee_ext_velo_br", "wrist_speed", "release_height"]
POI_TRUTH = {
    "stride_length":       "stride_length",
    "trunk_anterior_tilt": "torso_anterior_tilt_br",
    "knee_ext_velo_br":    "lead_knee_extension_angular_velo_br",
}
N_CURVE = [1, 2, 3, 5, 8]
SEEDS = [0, 1, 2, 3, 4]

# value-level r2 from prior validation (for the comparison column in B)
VALUE_R2 = {"lead_knee_angle": 0.94, "stride_length": 0.82,
            "trunk_anterior_tilt": 0.70, "knee_ext_velo_br": 0.65,
            "wrist_speed": 0.60, "release_height": 0.51}


def angle_3d(a, b, c):
    v1 = a - b; v2 = c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def truths_3d(joints, arm, lead, rel, fps):
    out = {}
    out["lead_knee_angle"] = angle_3d(joints[f"{lead}_hip"][:, rel],
                                      joints[f"{lead}_knee"][:, rel],
                                      joints[f"{lead}_ankle"][:, rel])
    w = joints[f"{arm}_wrist"]
    out["wrist_speed"] = float(np.nanmax(
        np.linalg.norm(np.diff(w, axis=1), axis=0) * fps))
    wz = w[2, rel]
    ground = np.nanmin([joints["left_ankle"][2, rel],
                        joints["right_ankle"][2, rel]])
    out["release_height"] = float(wz - ground)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-pitches", type=int, default=4,
                    help="SD analysis includes pitchers with >= this many pitches")
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    poi_idx = poi.set_index("session_pitch")
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
            lead = "left" if arm == "right" else "right"
            df = O.project_view(joints, azimuth_deg=0.0)
            rel = M.release_frame(df, arm, fps, M.JOINTS)
            fp = M.foot_plant_frame(df, lead, fps, M.JOINTS, rel)
            if rel <= fp + 1 or fp < 3:
                fail += 1; continue

            cand = {k: v for k, (v, _) in
                    M.compute_candidates(df, fps=fps, arm=arm).items()}
            t3 = truths_3d(joints, arm, lead, rel, fps)

            row = {"user": int(r.user), "session_pitch": r.session_pitch}
            for m in METRICS:
                row[f"est_{m}"] = cand.get(m, np.nan)
                if m in POI_TRUTH:
                    tcol = POI_TRUTH[m]
                    row[f"tru_{m}"] = (poi_idx.loc[r.session_pitch, tcol]
                                       if r.session_pitch in poi_idx.index
                                       and tcol in poi_idx.columns else np.nan)
                else:
                    row[f"tru_{m}"] = t3.get(m, np.nan)
            rows.append(row)
            done += 1
        except Exception:
            fail += 1
        if done and done % 100 == 0:
            print(f"  ...{done} processed")
    print(f"processed {done} / failed {fail}\n")

    df = pd.DataFrame(rows)

    # ---------- A) pitch-count distribution ----------
    cnt = df.groupby("user").size()
    print("=" * 66)
    print("[A] pitches per pitcher")
    print("=" * 66)
    dist = cnt.value_counts().sort_index()
    for k, v in dist.items():
        print(f"  {k:2d} pitches : {v:3d} pitchers")
    print(f"  total pitchers: {cnt.shape[0]}, "
          f">= {a.min_pitches} pitches: {(cnt >= a.min_pitches).sum()}")

    keep = cnt[cnt >= a.min_pitches].index
    sub = df[df["user"].isin(keep)]

    # ---------- B) variance (SD) recovery ----------
    print("\n" + "=" * 66)
    print(f"[B] within-pitcher SD recovery  (pitchers with >= "
          f"{a.min_pitches} pitches, n={len(keep)})")
    print("=" * 66)
    print(f"{'metric':22s}{'n_pit':>6s}{'r(SD)':>8s}{'r2(SD)':>8s}"
          f"{'value r2':>10s}")
    print("-" * 56)
    res_b = []
    diag = []
    for m in METRICS:
        g = sub.groupby("user").agg(
            esd=(f"est_{m}", "std"), tsd=(f"tru_{m}", "std")).dropna()
        if len(g) <= 2:
            print(f"{m:22s}{len(g):>6d}      (insufficient)")
            continue
        r = g["esd"].corr(g["tsd"])
        print(f"{m:22s}{len(g):>6d}{r:>8.3f}{r*r:>8.3f}"
              f"{VALUE_R2.get(m, float('nan')):>10.2f}")
        res_b.append({"metric": m, "n_pitchers": len(g),
                      "r_sd": r, "r2_sd": r * r,
                      "value_r2": VALUE_R2.get(m)})
        # --- SNR decomposition diagnostic ---
        signal = float(g["tsd"].std())          # spread of TRUE across-pitch SD between pitchers
        true_mean_sd = float(g["tsd"].mean())   # typical true across-pitch SD magnitude
        noise = float((g["esd"] - g["tsd"]).std())  # how far our SD estimate misses the truth
        bias = float((g["esd"] - g["tsd"]).mean())  # do we systematically over/under-estimate SD
        snr = signal / noise if noise > 1e-12 else float("nan")
        diag.append({"metric": m, "true_mean_sd": true_mean_sd,
                     "signal_betw_pitchers": signal, "noise_est_err": noise,
                     "bias": bias, "snr_decomp": snr})

    print("\n" + "=" * 66)
    print("[B-diag] SNR decomposition: is high SD-r2 from precise measurement"
          " or large true variability?")
    print("=" * 66)
    print(f"{'metric':22s}{'true_SD':>9s}{'signal':>9s}{'noise':>9s}"
          f"{'bias':>9s}{'SNR':>8s}")
    print("-" * 66)
    for d in diag:
        print(f"{d['metric']:22s}{d['true_mean_sd']:>9.3f}"
              f"{d['signal_betw_pitchers']:>9.3f}{d['noise_est_err']:>9.3f}"
              f"{d['bias']:>9.3f}{d['snr_decomp']:>8.2f}")
    pd.DataFrame(diag).to_csv(
        os.path.join(config.OBP_VALIDATION_DIR, "repeatability_snr_diag.csv"),
        index=False)

    # ---------- C) N-pitch averaging recovery curve ----------
    print("\n" + "=" * 66)
    print("[C] N-pitch averaging: our N-avg vs pitcher trait "
          "(truth mean of ALL pitches)")
    print("=" * 66)
    hdr = f"{'metric':22s}" + "".join(f"{'N='+str(n):>8s}" for n in N_CURVE)
    print(hdr); print("-" * len(hdr))
    res_c = []
    for m in METRICS:
        trait = sub.groupby("user")[f"tru_{m}"].mean()
        line = f"{m:22s}"
        rowc = {"metric": m}
        for n in N_CURVE:
            vals = []
            for s in SEEDS:
                rng = np.random.default_rng(s)
                est_means, traits = [], []
                for u, g in sub.groupby("user"):
                    e = g[f"est_{m}"].dropna().to_numpy()
                    if len(e) < n:
                        continue
                    pick = rng.choice(len(e), size=n, replace=False)
                    est_means.append(e[pick].mean())
                    traits.append(trait.loc[u])
                if len(est_means) > 2:
                    rr = np.corrcoef(est_means, traits)[0, 1]
                    vals.append(rr * rr)
            v = np.mean(vals) if vals else np.nan
            rowc[f"N{n}"] = v
            line += f"{v:>8.2f}"
        print(line)
        res_c.append(rowc)

    out1 = os.path.join(config.OBP_VALIDATION_DIR, "repeatability_sd.csv")
    out2 = os.path.join(config.OBP_VALIDATION_DIR, "repeatability_navg.csv")
    pd.DataFrame(res_b).to_csv(out1, index=False)
    pd.DataFrame(res_c).to_csv(out2, index=False)
    df.to_csv(os.path.join(config.OBP_VALIDATION_DIR,
                           "repeatability_perpitch.csv"), index=False)
    print(f"\nsaved -> {out1}\nsaved -> {out2}")


if __name__ == "__main__":
    main()