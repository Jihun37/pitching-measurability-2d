"""Axis 1 of the event question: how many frames of ANCHOR error can each cell
absorb before it falls off the graded map?

The map is built on GT events. Real deployment detects them, so a cell is only
usable if the metric's tolerance is wider than the detector's error at that
viewpoint (axis 2, event_error_sweep). This file measures the tolerance; the
intersection of the two is the deployable map.

Input: rejected_gt_pairs_offsets.csv.gz, written by
`rejected_gt_full_sweep.py --event-offsets=-3,-2,-1,0,1,2,3`, which reads every
column at its own anchor shifted by k frames in ONE pass (window observables get
both ends shifted). Scoring is the SAME closed-form leave-one-pitcher-out routine
the map uses -- imported from gate_map, not re-implemented.

Definitions, per cell:
  tol_map     largest symmetric k (0..3) such that EVERY offset in [-k, +k] keeps
              the cell on the map (grade != limited). 3 means ">= 3", the sweep
              did not go further.
  tol_strong  the same, but keeping the STRONG grade.
  A cell that is already off the map at offset 0 gets tol = -1 (n/a).

Note on coverage: this reads the SCREENED sweep, i.e. the 36 columns of
rejected_gt_full_sweep. Adopted-only rows (Stride Angle, Release Height, ...)
need the same offsets wired into angle_zone_sweep before their tolerance exists.

Run:  conda activate diamond; cd src\\analysis; python event_tolerance.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config
from gt_landmark_outlier_effect import outlier_pitches
from gate_map import score_cell, ADOPTED_PAIRS

PAIRS = os.path.join(config.OBP_VALIDATION_DIR, "rejected_gt_pairs_offsets.csv.gz")
OUT = os.path.join(config.OBP_VALIDATION_DIR, "event_tolerance.csv")
MODERATE, STRONG = 0.75, 0.80
RANK = {"limited": 0, "moderate": 1, "strong": 2}


def main():
    if not os.path.exists(PAIRS):
        sys.exit("run rejected_gt_full_sweep.py --event-offsets=-3,-2,-1,0,1,2,3")
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    user_of = dict(zip(md.session_pitch, md.user))

    print("loading offset pairs ...")
    d = pd.read_csv(PAIRS, dtype={"metric": "category", "session_pitch": "category",
                                  "az": "int16", "el": "int8", "offset": "int8",
                                  "est": "float32", "truth": "float32"})
    print(f"  {len(d):,} rows, {d.metric.nunique()} metrics, "
          f"{d.offset.nunique()} offsets")

    # same population as the map: gt_clean AND present in the adopted dump, so a
    # tolerance number is comparable with the map cell it belongs to
    keep = set(pd.read_csv(ADOPTED_PAIRS, usecols=["session_pitch"]).session_pitch)
    keep -= outlier_pitches()
    d = d[d.session_pitch.isin(keep)]
    d["user"] = d.session_pitch.map(user_of).astype("category")
    print(f"  scored population: {d.session_pitch.nunique()} pitches\n")

    offsets = sorted(d.offset.unique())
    rows = []
    for metric, gm in d.groupby("metric", observed=True):
        codes = gm.user.cat.codes.to_numpy()
        e = gm.est.to_numpy(np.float64); t = gm.truth.to_numpy(np.float64)
        az = gm.az.to_numpy(); el = gm.el.to_numpy(); of = gm.offset.to_numpy()
        for a in np.unique(az):
            ma = az == a
            for l in np.unique(el[ma]):
                mm = ma & (el == l)
                grades = {}
                for k in offsets:
                    m = mm & (of == k)
                    s = score_cell(e[m], t[m], codes[m], MODERATE, STRONG)
                    grades[k] = (s["grade"], s["ccc"]) if s else ("limited", np.nan)
                g0 = grades[0][0]
                rec = dict(metric=metric, az=int(a), el=int(l), grade0=g0,
                           **{f"ccc{k:+d}": grades[k][1] for k in offsets})
                for name, need in (("tol_map", 1), ("tol_strong", 2)):
                    if RANK[g0] < need:
                        rec[name] = -1
                        continue
                    tol = 0
                    for k in range(1, max(offsets) + 1):
                        if all(RANK[grades[j][0]] >= need
                               for j in (-k, k) if j in grades):
                            tol = k
                        else:
                            break
                    rec[name] = tol
                rows.append(rec)
        print(f"  scored {metric}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, float_format="%.6g")
    print(f"\nsaved -> {OUT}")

    on = out[out.tol_map >= 0]
    print("\n" + "=" * 92)
    print("EVENT TOLERANCE of the cells that are on the map at offset 0")
    print("=" * 92)
    print(f"  cells on the map: {len(on)}")
    for k in range(0, 4):
        n = int((on.tol_map == k).sum())
        lab = f"±{k}" if k < 3 else "±3 or more"
        print(f"    survives {lab:>10} frames : {n:>4}  ({100*n/max(1,len(on)):.0f}%)")
    st = out[out.tol_strong >= 0]
    print(f"  strong cells: {len(st)}, of which "
          f"{int((st.tol_strong == 0).sum())} lose STRONG at ±1 frame")
    print("\n  least tolerant metrics (median tol_map over their map cells):")
    med = on.groupby("metric", observed=True).tol_map.median().sort_values()
    print(med.head(10).to_string())
    print("\n  most tolerant:")
    print(med.tail(6).to_string())


if __name__ == "__main__":
    main()
