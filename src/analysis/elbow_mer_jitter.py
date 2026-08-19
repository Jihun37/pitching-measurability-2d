"""Elbow flexion at MER: sensitivity to a uniform anchor shift versus to
per-pitch anchor jitter, on the paper's own population and criterion.

WHY THIS REPLACES src/scratch/elbow_frame_sensitivity.py
  * that script scored n=403 on r^2; every paper layer is n=394 (gt_clean) on
    leave-one-pitcher-out CCC, so its numbers cannot go in the manuscript;
  * it rebuilt the flexion angle inline as `_angle(sh, el, wr)`, without the
    `180 -` the map applies. r^2 cannot see that -- CCC can, because CCC
    penalises bias and scale -- so the old table would not have been comparable
    with the map cell it describes. This imports all_observables() instead, the
    entry point that exists so a consumer cannot drift from the map definitions.

WHAT IS AND IS NOT MEASURED. Both experiments keep the 360 Hz trajectory intact
and move only the instant at which the value is read:
  uniform shift  every pitch is read k frames away from true MER, same k;
  jitter         every pitch is displaced independently, Gaussian, given SD.
Neither subsamples the trajectory, so neither measures the local signal loss a
genuinely coarser capture rate would also cause. A jitter SD is reported with a
variance-matched capture-rate reading only, never as a frame-rate experiment.

This is a single-cell case study at (az 330, el 60), not a map-wide result.

Run:  conda activate diamond; cd src\\analysis; python elbow_mer_jitter.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)

import numpy as np, pandas as pd
import config
import obp_project as O
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from obp_gt_events import load_gt_events
from mer_proxy_map import map_population
from rejected_gt_full_sweep import all_observables
from gate_map import score_cell, grade_of

CELL = (330, 60)
TRUTH = "elbow_flexion_mer"
DELTAS = list(range(-6, 7))
JITTER_SD = [0.0, 0.25, 0.43, 0.5, 0.75, 0.87, 1.0, 1.5, 1.73, 2.0, 3.0]
SEEDS = 40                      # one draw over 394 pitches is too noisy to quote
# A capture at F fps quantises an instant to +-0.5 phone frame, i.e. +-180/F frames
# of this 360 Hz grid. The uniform +-w quantisation error has SD w/sqrt(3), so the
# variance-matched Gaussian SD is (180/F)/sqrt(3). This is a reading of the jitter
# magnitude, NOT a subsampling experiment.
RATE_OF_SD = {0.43: 240, 0.87: 120, 1.73: 60}
OUT = os.path.join(config.OBP_VALIDATION_DIR, "elbow_mer_jitter.csv")


def ccc_of(e, t, g):
    s = score_cell(np.asarray(e, float), np.asarray(t, float), g, 0.75)
    return (np.nan, np.nan, "limited") if s is None else (
        s["ccc"], s["r2"], s["grade"])


def main():
    az, el = CELL
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi",
                                   "poi_metrics.csv")).set_index("session_pitch")
    gt = load_gt_events()
    keep = set(map_population())
    root = os.path.join(config.OBP_DATA_DIR, "c3d")

    vals = {d: [] for d in DELTAS}
    truth, users, slope = [], [], []
    for r in md.itertuples(index=False):
        sp = r.session_pitch
        if sp not in keep or sp not in poi.index:
            continue
        g = gt.get(sp)
        if not g or "mer" not in g:
            continue
        path = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            continue
        try:
            joints, fps = load_feet(path)
            df = project_cam(joints, az, el)
        except Exception:
            continue
        arm = O.detect_throwing_arm(joints, fps)
        lead = "left" if arm == "right" else "right"
        fx = all_observables(df, fps, lead)[0]["elbow_flex"]
        mer = int(g["mer"])
        if not (max(abs(min(DELTAS)), abs(max(DELTAS))) < mer <
                len(fx) - max(abs(min(DELTAS)), abs(max(DELTAS))) - 1):
            continue
        truth.append(float(poi.loc[sp, TRUTH])); users.append(int(r.user))
        slope.append(abs(fx[mer + 1] - fx[mer - 1]) / 2.0)
        for d in DELTAS:
            vals[d].append(fx[mer + d])

    t = np.asarray(truth, float)
    gu, gi = np.unique(np.asarray(users), return_inverse=True)
    sl = np.asarray(slope, float)
    print(f"population: n={len(t)} pitches, {len(gu)} pitchers   cell "
          f"az{az}/el{el}   truth {TRUTH}")
    print(f"|d(flexion)/dframe| at MER: median {np.median(sl):.2f} deg/f, "
          f"p90 {np.percentile(sl, 90):.2f}")
    print(f"ground-truth SD: {np.nanstd(t, ddof=1):.2f} deg  -> one frame is "
          f"{np.median(sl)/np.nanstd(t, ddof=1)*100:.0f} % of it\n")

    raw = np.asarray([vals[d] for d in DELTAS], float)
    di = {d: i for i, d in enumerate(DELTAS)}
    rows = []

    print("UNIFORM SHIFT (every pitch displaced by the same k)")
    print(f"{'k':>4}{'CCC':>9}{'r2':>8}  grade")
    for d in DELTAS:
        c, r2, gr = ccc_of(vals[d], t, gi)
        print(f"{d:>+4}{c:>9.3f}{r2:>8.3f}  {gr}")
        rows.append(dict(experiment="uniform_shift", k=d, jitter_sd=np.nan,
                         ccc=c, r2=r2, grade=gr, rate_fps=np.nan))
    base = ccc_of(vals[0], t, gi)[0]

    print(f"\nPER-PITCH JITTER (independent Gaussian displacement, "
          f"{SEEDS} draws)")
    print(f"{'SD (f)':>8}{'CCC':>9}{'+-':>7}{'r2':>8}   equivalent quantisation")
    for sd in JITTER_SD:
        cs, rs = [], []
        for s in range(SEEDS):
            rng = np.random.default_rng(s)
            if sd == 0:
                e = raw[di[0]]
            else:
                j = np.clip(np.rint(rng.normal(0, sd, len(t))).astype(int),
                            min(DELTAS), max(DELTAS))
                e = raw[[di[k] for k in j], np.arange(len(t))]
            c, r2, _ = ccc_of(e, t, gi)
            cs.append(c); rs.append(r2)
        lab = (f"~{RATE_OF_SD[sd]} fps" if sd in RATE_OF_SD else "")
        print(f"{sd:>8.2f}{np.mean(cs):>9.3f}{np.std(cs):>7.3f}"
              f"{np.mean(rs):>8.3f}   {lab}")
        rows.append(dict(experiment="jitter", k=np.nan, jitter_sd=sd,
                         ccc=np.mean(cs), ccc_sd=np.std(cs), r2=np.mean(rs),
                         grade=grade_of(np.mean(cs), 0.80, 0.75),
                         rate_fps=RATE_OF_SD.get(sd, np.nan)))

    print("\nPITCHER-MEAN AVERAGING (does repeating pitches recover it?)")
    for tag, sd in (("perfect anchor", 0.0), ("~60 fps jitter", 1.73)):
        rng = np.random.default_rng(0)
        if sd == 0:
            e = raw[di[0]]
        else:
            j = np.clip(np.rint(rng.normal(0, sd, len(t))).astype(int),
                        min(DELTAS), max(DELTAS))
            e = raw[[di[k] for k in j], np.arange(len(t))]
        d = pd.DataFrame({"u": users, "e": e, "t": t}).groupby("u").mean()
        pooled = ccc_of(e, t, gi)[0]
        pm = ccc_of(d.e.values, d.t.values, np.arange(len(d)))[0]
        print(f"  {tag:<16} pooled CCC {pooled:.3f} -> pitcher-mean CCC {pm:.3f}"
              f"   ({len(d)} pitchers)")
        rows.append(dict(experiment="pitcher_mean", k=np.nan, jitter_sd=sd,
                         ccc=pm, r2=np.nan, grade="", rate_fps=RATE_OF_SD.get(sd)))

    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.6g")
    print(f"\nbaseline CCC at the true anchor: {base:.3f}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
