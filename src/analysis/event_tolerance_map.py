"""Step 4: one tolerance table over the whole 35-row map.

Two sweeps feed it and neither covers the map alone:
  screened 23 rows  event_tolerance.csv           (rejected_gt_full_sweep offsets)
  adopted  10 rows  adopted_tolerance_pairs.csv.gz (adopted_event_tolerance_sweep)

Both use the SAME tol definition as event_tolerance.py: tol is the largest k in
1..3 such that every offset in [-k, +k] keeps the cell's grade, and -1 marks a cell
that is already off the map at offset 0.

APPLICABILITY. Two adopted rows read no external event anchor at all -- Wrist Speed
takes a whole-clip maximum and Hip-Shoulder Sep locates its own signature anchor --
so a uniform anchor shift is undefined for them, not favourable. They are written
out with applicable=False and are excluded from every denominator. Reporting them
as tolerant would invert their meaning.

Run:  conda activate diamond; cd src\\analysis
      python adopted_event_tolerance_sweep.py --probe
      python adopted_event_tolerance_sweep.py --offsets=-3,-2,-1,0,1,2,3
      python adopted_tolerance_parity.py
      python event_tolerance_map.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)

import numpy as np, pandas as pd
import config
from gate_map import score_cell
from angle_map_2d import CIRCULAR, unwrap_circular
from rejected_gt_full_sweep import CANDS

V = config.OBP_VALIDATION_DIR
OUT = os.path.join(V, "event_tolerance_map.csv")
MODERATE, STRONG = 0.75, 0.80
RANK = {"limited": 0, "moderate": 1, "strong": 2}
WINDOW_OBS = {"elbow_ext_velo_max", "elbow_flex_max", "shoulder_line_velo_max",
              "hip_line_velo_max", "shoulder_line_min", "hz_abd_throw_max",
              "torso_pelvis_timing", "knee_ext_velo_max", "knee_ext_fp_to_br"}


def tol_of(grades, need, offsets):
    """event_tolerance.py's definition, kept identical on purpose."""
    if RANK[grades[0][0]] < need:
        return -1
    tol = 0
    for k in range(1, max(offsets) + 1):
        if all(RANK[grades[j][0]] >= need for j in (-k, k) if j in grades):
            tol = k
        else:
            break
    return tol


def screened_anchor(metric):
    """fp / mer / release, and release+fp where the window is bounded by both."""
    if metric not in CANDS:
        return "", ""
    obs, ev = CANDS[metric]
    if ev == "fp":
        return "fp", "fp"
    if ev == "mer":
        return "mer", "mer"
    if obs in WINDOW_OBS:          # [fp, rel]: the sweep shifts both ends
        return "release+fp", "release+fp"
    return "release", "release"


def score_adopted(pairs, user_of):
    d = pd.read_csv(pairs)
    offsets = sorted(d.offset.unique())
    d["user"] = d.session_pitch.map(user_of)
    rows = []
    for metric, gm in d.groupby("metric", sort=False):
        codes = pd.Categorical(gm.user).codes
        e = gm.est.to_numpy(np.float64); t = gm.truth.to_numpy(np.float64)
        az = gm.az.to_numpy(); el = gm.el.to_numpy(); of = gm.offset.to_numpy()
        # unwrap PER CELL, exactly as gate_map.py:275 does. Unwrapping the whole
        # metric at once walks across cell and offset boundaries and silently
        # rewrites the values: it cost Stride Angle 7 of its 148 graded cells.
        circ = metric.strip() in CIRCULAR
        for a in np.unique(az):
            ma = az == a
            for l in np.unique(el[ma]):
                mm = ma & (el == l)
                grades = {}
                for k in offsets:
                    m = mm & (of == k)
                    e_cell = unwrap_circular(e[m]) if circ else e[m]
                    s = score_cell(e_cell, t[m], codes[m], MODERATE, STRONG)
                    grades[k] = (s["grade"], s["ccc"]) if s else ("limited", np.nan)
                rows.append(dict(
                    metric=metric, az=int(a), el=int(l), grade0=grades[0][0],
                    **{f"ccc{k:+d}": grades[k][1] for k in offsets},
                    tol_map=tol_of(grades, 1, offsets),
                    tol_strong=tol_of(grades, 2, offsets)))
        print(f"  scored {metric}")
    return pd.DataFrame(rows)


def main():
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    user_of = dict(zip(md.session_pitch, md.user))
    ls = pd.read_csv(os.path.join(V, "layer_summary.csv"))
    mapped = ls[ls.map_cells > 0][["metric", "source", "map_cells"]]
    cls = pd.read_csv(os.path.join(V, "adopted_anchor_classes.csv"))

    scr = pd.read_csv(os.path.join(V, "event_tolerance.csv"))
    print(f"screened tolerance: {len(scr):,} cells, {scr.metric.nunique()} rows")
    print("scoring the adopted sweep ...")
    ado = score_adopted(os.path.join(V, "adopted_tolerance_pairs.csv.gz"), user_of)
    print(f"adopted tolerance : {len(ado):,} cells, {ado.metric.nunique()} rows")

    t = pd.concat([scr, ado], ignore_index=True)
    t = t.merge(mapped, on="metric", how="inner")

    ann = []
    for m in t.metric.unique():
        if m in set(cls.metric):
            c = cls.loc[cls.metric == m].iloc[0]
            ann.append(dict(metric=m, applicable=bool(c.applicable),
                            anchor_type=c.anchor_type,
                            shifted_boundaries=str(c.shifted_boundaries or ""),
                            not_applicable_reason=str(c.not_applicable_reason or "")))
        else:
            at, sb = screened_anchor(m)
            ann.append(dict(metric=m, applicable=bool(at), anchor_type=at or "none",
                            shifted_boundaries=sb,
                            not_applicable_reason="" if at else
                            "reads no external event anchor from the context"))
    t = t.merge(pd.DataFrame(ann), on="metric", how="left")

    # the two N/A rows have no sweep, so add them explicitly at cell level
    na = cls[~cls.applicable]
    gm = pd.read_csv(os.path.join(V, "gate_map.csv"))
    extra = gm[gm.metric.isin(set(na.metric)) & gm.gate_pass][
        ["metric", "az", "el", "grade"]].rename(columns={"grade": "grade0"})
    extra = extra.merge(mapped, on="metric", how="left")
    extra["tol_map"] = np.nan; extra["tol_strong"] = np.nan
    extra["applicable"] = False
    extra["anchor_type"] = "none"
    extra["shifted_boundaries"] = ""
    extra["not_applicable_reason"] = na.not_applicable_reason.iloc[0]
    t = pd.concat([t, extra], ignore_index=True)
    t.to_csv(OUT, index=False, float_format="%.6g")

    print(f"\nsaved -> {OUT}  ({len(t):,} rows)")
    print("=" * 78)
    print("TOLERANCE OVER THE 35-ROW MAP")
    print("=" * 78)
    print(f"map rows covered      : {t.metric.nunique()} of {len(mapped)}")
    print(f"  applicable          : {t.loc[t.applicable, 'metric'].nunique()}")
    print(f"  not applicable      : {t.loc[~t.applicable, 'metric'].nunique()} "
          f"({sorted(t.loc[~t.applicable, 'metric'].unique())})")
    ap = t[t.applicable & (t.tol_map >= 0)]
    print(f"\ngraded cells with a shiftable anchor: {len(ap)}")
    for k in range(4):
        n = int((ap.tol_map == k).sum())
        lab = f"+-{k}" if k < 3 else "+-3 or more"
        print(f"  keeps its grade to {lab:>12} : {n:>4}  ({100*n/len(ap):5.1f} %)")
    print(f"\nexcluded as not applicable: "
          f"{int((~t.applicable).sum())} graded cells")
    print("\nleast tolerant rows (median tol_map):")
    med = ap.groupby("metric").tol_map.agg(cells="size", median="median",
                                           worst="min").sort_values("median")
    print(med.head(12).to_string())


if __name__ == "__main__":
    main()
