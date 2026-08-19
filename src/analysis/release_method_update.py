"""
Diamond - cumulative per-angle release-method map (declared-az track).

User decision 2026-07-12: with the camera azimuth GIVEN (declared by the
shooter or a future high-accuracy classifier), the best release detector
differs per angle (oracle-az holdout: frontal wins at az90 where the
cross-strategy gate was self-sabotaging, side at 180/330, ball-separation
on the obliques; fitted map <=1f 40/60 vs global adopt 33/60). Because
each arc rests on only ~5 clips, the map is NOT hardcoded: every labeled
clip set is scored the same way into an EVIDENCE ledger, and the map is
rebuilt from the accumulated ledger - any arc can flip when new data
says so.

Files (data/outputs/):
  release_method_evidence.csv  one row per clip: true az, GT frame, and
                               the detected frame of every method
                               (side / side+off / frontal / deploy /
                               ballsep). Re-running on the same clips
                               replaces their rows (idempotent).
  release_method_map.csv       per 30-deg arc: winning method + its
                               cumulative stats. Arcs with < MIN_N clips
                               fall back to 'deploy' (the strategy-rule
                               chain). Consumed by deploy/release_method.

Run (diamond env):
  cd src\\analysis
  python release_method_update.py                       (holdout 60)
  python release_method_update.py --gt my_new_set.csv --tag set2
    (gt csv columns: clip, nominal_az, true_release; poses must already
     be extracted under data/outputs/<clip>/)
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("..", "../stage2", "../deploy", "../tests"):
    sys.path.insert(0, os.path.join(_HERE, p))

import config
import metrics as M
from measure_auto import load_clip
from real_station_test import detect_arm_2d
from release_offset import (corrected_release_frame, offset_ms, bin_iqr_ms,
                            MAX_BIN_IQR_MS)
from ball_separation_probe import detect_separation

METHODS = ["side", "side_off", "frontal", "deploy", "ballsep"]
EVIDENCE = os.path.join(config.ROOT, "data", "outputs",
                        "release_method_evidence.csv")
MAP_CSV = os.path.join(config.ROOT, "data", "outputs",
                       "release_method_map.csv")
MIN_N = 3            # arcs with fewer labeled clips stay on 'deploy'


def score_clip(name, az):
    """Detected frame per method (None = deferred / no track)."""
    df, raw = load_clip(name, "_rtmp")
    arm = detect_arm_2d(df)
    cap = cv2.VideoCapture(os.path.join(config.ROOT, "data", "videos",
                                        f"{name}.mov"))
    fps = cap.get(cv2.CAP_PROP_FPS) or config.FPS_DEFAULT
    cap.release()

    side = M.release_frame(df, arm, fps, M.JOINTS, view="side", raw_df=raw)
    soff = side
    if bin_iqr_ms(az, 0) <= MAX_BIN_IQR_MS:
        soff = int(np.clip(int(round(side - offset_ms(az, 0) * fps / 1000.0)),
                           0, len(df) - 1))
    front = M.release_frame(df, arm, fps, M.JOINTS, view="frontal",
                            raw_df=raw)
    dep, _, _, _ = corrected_release_frame(df, arm, fps, M.JOINTS, az, 0,
                                           raw_df=raw, vote_share=None)
    seed = dep if dep is not None else side
    sep, _ = detect_separation(name, df, raw, arm, fps, int(seed))
    return {"side": side, "side_off": soff, "frontal": front,
            "deploy": dep, "ballsep": sep}


def rebuild_map(ev):
    """Per-arc winner from the FULL evidence ledger. Rank: <=1f hit rate,
    then median |err|; deploy on ties/short evidence."""
    rows = []
    for az in sorted(ev.true_az.unique()):
        sub = ev[ev.true_az == az]
        best, best_key = "deploy", None
        for m in METHODS:
            e = (sub[m] - sub.true_release).dropna().to_numpy(float)
            if len(e) < MIN_N:
                continue
            key = ((np.abs(e) <= 1).mean(), -np.median(np.abs(e)),
                   m == "deploy")           # deploy wins exact ties
            if best_key is None or key > best_key:
                best, best_key = m, key
        e = (sub[best] - sub.true_release).dropna().to_numpy(float) \
            if best_key is not None else np.array([])
        rows.append({"az": int(az), "method": best,
                     "n": int(len(sub)),
                     "hit1_rate": round(float((np.abs(e) <= 1).mean()), 3)
                     if len(e) else np.nan,
                     "med_abs_err": float(np.median(np.abs(e)))
                     if len(e) else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=os.path.join(
        config.ROOT, "data", "outputs", "holdout60_scoring_inputs.csv"),
        help="labeled set: clip, nominal_az, true_release")
    ap.add_argument("--tag", default=None, help="evidence set label")
    a = ap.parse_args()
    gt = pd.read_csv(a.gt)
    tag = a.tag or os.path.splitext(os.path.basename(a.gt))[0]

    new = []
    for r in gt.itertuples():
        az = int(r.nominal_az)
        det = score_clip(r.clip, az)
        row = {"clip": r.clip, "set": tag, "true_az": az,
               "true_release": int(r.true_release), **det}
        new.append(row)
        errs = "  ".join(
            f"{m}:{'--' if det[m] is None else f'{det[m] - r.true_release:+d}'}"
            for m in METHODS)
        print(f"{r.clip:14s} az{az:<4} GT f{r.true_release}  {errs}",
              flush=True)
    new = pd.DataFrame(new)

    if os.path.exists(EVIDENCE):
        old = pd.read_csv(EVIDENCE)
        old = old[~old.clip.isin(new.clip)]
        ev = pd.concat([old, new], ignore_index=True)
    else:
        ev = new
    ev.to_csv(EVIDENCE, index=False)

    mp = rebuild_map(ev)
    mp.to_csv(MAP_CSV, index=False)
    print(f"\nevidence: {len(ev)} clips -> {EVIDENCE}")
    print(f"map -> {MAP_CSV}")
    print(mp.to_string(index=False))

    # combined score of the CURRENT map over the full ledger
    hits1 = hits2 = n = 0
    for r in ev.itertuples():
        m = mp[mp.az == r.true_az].method.iloc[0]
        v = getattr(r, m)
        if pd.isna(v):
            v = r.deploy                      # detector deferred -> deploy
        if pd.isna(v):
            continue
        e = abs(int(v) - r.true_release)
        n += 1
        hits1 += e <= 1
        hits2 += e <= 2
    print(f"\n[map over full ledger] n={n}  <=1f {hits1}  <=2f {hits2}")


if __name__ == "__main__":
    main()
