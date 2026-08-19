"""Offset-0 parity gate for adopted_event_tolerance_sweep.

The tolerance layer is only meaningful if its offset-0 baseline IS the frozen map.
When it was not, 33 cells of `lead_knee_extension_angular_velo_max` looked as
though a one-frame shift had killed them, when in fact the offset dump had been
swept under the pre-2026-07-27 knee window and never shared a baseline with
gate_map.csv at all. So this runs before any +-k sweep is trusted:

  grade   must match gate_map.csv EXACTLY   -> assertion
  CCC     compared with a tolerance, since both sides are stored at %.6g

Run:  conda activate diamond; cd src\\analysis
      python adopted_event_tolerance_sweep.py --offsets 0
      python adopted_tolerance_parity.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)

import numpy as np, pandas as pd
import config
from gate_map import score_cell, grade_of
from angle_map_2d import CIRCULAR, unwrap_circular

V = config.OBP_VALIDATION_DIR
PAIRS = os.path.join(V, "adopted_tolerance_pairs.csv.gz")
CCC_TOL = 2e-6          # both sides are written at %.6g


def main():
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    user_of = dict(zip(md.session_pitch, md.user))
    gm = pd.read_csv(os.path.join(V, "gate_map.csv"))
    cls = pd.read_csv(os.path.join(V, "adopted_anchor_classes.csv"))
    applicable = set(cls.loc[cls.applicable, "metric"])

    d = pd.read_csv(PAIRS)
    d = d[d.offset == 0]
    print(f"{len(d):,} pairs, {d.metric.nunique()} rows, "
          f"{d.session_pitch.nunique()} pitches")

    codes, out = {}, []
    for (metric, az, el), g in d.groupby(["metric", "az", "el"], sort=False):
        e = g.est.to_numpy(float); t = g.truth.to_numpy(float)
        if metric in CIRCULAR:
            e = unwrap_circular(e)
        u = np.array([user_of.get(s, -1) for s in g.session_pitch])
        gu, gi = np.unique(u, return_inverse=True)
        s = score_cell(e, t, gi, 0.75)
        out.append(dict(metric=metric, az=az, el=el,
                        ccc=np.nan if s is None else s["ccc"],
                        grade=("limited" if s is None
                               else grade_of(s["ccc"], 0.80, 0.75))))
    r = pd.DataFrame(out)

    m = r.merge(gm[["metric", "az", "el", "ccc", "grade"]],
                on=["metric", "az", "el"], suffixes=("_new", "_gm"))
    print(f"cells compared: {len(m):,} "
          f"(expect {len(applicable)} rows x 168 = {len(applicable)*168})")

    bad_grade = m[m.grade_new != m.grade_gm]
    dccc = (m.ccc_new - m.ccc_gm).abs()
    print(f"\ngrade mismatches : {len(bad_grade)}")
    print(f"max |dCCC|       : {np.nanmax(dccc):.3g}")
    print(f"cells |dCCC|>tol : {int((dccc > CCC_TOL).sum())}")

    if len(bad_grade):
        print("\nFIRST MISMATCHES")
        print(bad_grade[["metric", "az", "el", "grade_gm", "grade_new",
                         "ccc_gm", "ccc_new"]].head(20).to_string(index=False))
    worst = m.loc[dccc > CCC_TOL]
    if len(worst):
        print("\nCELLS OVER THE CCC TOLERANCE")
        print(worst.assign(d=dccc[dccc > CCC_TOL])
                   .sort_values("d", ascending=False)
                   [["metric", "az", "el", "ccc_gm", "ccc_new", "d"]]
                   .head(20).to_string(index=False))

    assert len(bad_grade) == 0, (
        f"offset-0 grade parity failed on {len(bad_grade)} cells; do NOT run the "
        "+-k sweep until this is zero")
    print("\nPARITY OK -- offset 0 reproduces gate_map.csv grade for grade")


if __name__ == "__main__":
    main()
