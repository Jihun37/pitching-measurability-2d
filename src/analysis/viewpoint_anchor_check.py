"""Layer-1 sub-step D: viewpoint and anchor dependence, on the canonical row set.

TWO AXES ONLY (scope decision, 2026-07-30):
  1. HSS elevation dependence -- the same quantity is unmeasurable from the ground
     and measurable from overhead, which is the paper's cleanest single
     demonstration that measurability is a property of (quantity x viewpoint).
  2. fp_100 vs fp_10 -- the same quantity read against two different definitions of
     the same event, which is the anchor half of the same argument.

⚠ THE fp10 TABLE IS NOT ON THE CANONICAL ROW SET. `gate_map_fp10.csv` was built
2026-07-27, before the dedup, and still carries all 52 rows including the five
duplicate-quantity rows that were removed (Elbow Flex @MER, Glove Sh Abd @MER,
Torso Lat Tilt @MER, Torso Rot @BR, max_pelvis_rotational_velo). Comparing it to
`gate_map.csv` without restriction silently re-admits them and inflates every
fp-dependent total. This script therefore INNER-JOINS both tables onto
`paper_registry.csv` first and asserts the row set afterwards. The fp10 dumps are
left untouched on purpose -- filtering at comparison time is reversible, rewriting
a dump is not, and nothing else reads them.

WHICH ROWS DEPEND ON THE FOOT PLANT. Taken from the registry's `anchor_type`
(`fp` or `release+fp`) and cross-checked against `fp_target_check.fp_dependent_rows()`,
which derives the same set from the code including the window observables whose
event key alone cannot express that they read the fp end. A disagreement is fatal
rather than reconciled silently.

PER-CELL VERDICT. For each fp-dependent row and each of its cells that is graded
under EITHER anchor, the better anchor is the one with the higher LOPO CCC;
`tie` means the two agree to 1e-6. Cells graded under neither are not counted --
they carry no information about the anchor question.

Input:  gate_map.csv, gate_map_fp10.csv, paper_registry.csv
Output: fp_target_cells.csv, fp_target_rows.csv, hss_elevation.csv
Run:  conda activate diamond; cd src\\analysis; python viewpoint_anchor_check.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config

V = config.OBP_VALIDATION_DIR
KEY = ["metric", "source", "az", "el"]
TIE = 1e-6
GRADED = ("strong", "moderate")


def main():
    reg = pd.read_csv(os.path.join(V, "paper_registry.csv"))
    canon = set(reg.metric_id)
    # 47 until the five direct-3D rows were removed on 2026-08-12
    assert len(canon) == 42, len(canon)

    # ---------------------------------------------------- 1. HSS by elevation
    g = pd.read_csv(os.path.join(V, "gate_map.csv"))
    g = g[g.metric.isin(canon)]
    assert g.metric.nunique() == 42

    hss = g[g.metric == "Hip-Shoulder Sep [O]"]
    rec = []
    for el, sub in hss.groupby("el"):
        best = sub.loc[sub.ccc.idxmax()]
        rec.append(dict(el=int(el), best_ccc=best.ccc, best_az=int(best.az),
                        best_r2=best.r2,
                        graded_cells=int(sub.grade.isin(GRADED).sum()),
                        strong_cells=int((sub.grade == "strong").sum())))
    H = pd.DataFrame(rec)
    print("=" * 78)
    print("HSS ELEVATION DEPENDENCE  (Hip-Shoulder Sep [O], 24 azimuths per row)")
    print("=" * 78)
    print(H.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  ground (el 0) best CCC {H.loc[H.el == 0, 'best_ccc'].iloc[0]:.4f}"
          f"   -> overhead best {H.best_ccc.max():.4f}"
          f" at el {int(H.loc[H.best_ccc.idxmax(), 'el'])}")
    print(f"  graded cells: {int(H.graded_cells.sum())} total, "
          f"all at el >= {int(H.el[H.graded_cells > 0].min())}")
    H.to_csv(os.path.join(V, "hss_elevation.csv"), index=False,
             float_format="%.6g")

    # ------------------------------------------------- 2. fp_100 vs fp_10
    p10 = os.path.join(V, "gate_map_fp10.csv")
    if not os.path.exists(p10):
        sys.exit("missing gate_map_fp10.csv -- see fp_target_check.py docstring")
    f10 = pd.read_csv(p10)
    dropped = sorted(set(f10.metric) - canon)
    print("\n" + "=" * 78)
    print("fp_100 vs fp_10  (restricted to the canonical 47 rows)")
    print("=" * 78)
    print(f"  gate_map_fp10.csv carries {f10.metric.nunique()} rows; "
          f"{len(dropped)} are not on the registry and are excluded:")
    for m in dropped:
        print(f"    - {m}")
    f10 = f10[f10.metric.isin(canon)]
    assert f10.metric.nunique() == 42, f10.metric.nunique()

    from fp_target_check import fp_dependent_rows
    # Same documented exception the registry applies: COG Velo @PKH is listed by
    # fp_dependent_rows() only because its estimator falls back to ctx['fp'] when
    # pkh is absent, and under the GT-event convention pkh is never absent, so the
    # branch never runs (measured fp-sensitivity exactly 0.0000). It is pkh-anchored.
    code_set = (fp_dependent_rows() - {"COG Velo @PKH [O]"}) & canon
    reg_set = set(reg.metric_id[reg.anchor_type.isin(["fp", "release+fp"])])
    if code_set != reg_set:
        print(f"\n  registry-only : {sorted(reg_set - code_set)}")
        print(f"  code-only     : {sorted(code_set - reg_set)}")
        sys.exit("fp-dependent row set disagrees between the registry and the code")
    fp_rows = sorted(reg_set)
    print(f"\n  fp-dependent rows: {len(fp_rows)} "
          f"(registry anchor_type and fp_dependent_rows() agree)")

    m = g[g.metric.isin(fp_rows)].merge(
        f10[KEY + ["ccc", "grade"]], on=KEY, suffixes=("_100", "_10"))
    both = m[m.grade_100.isin(GRADED) | m.grade_10.isin(GRADED)].copy()
    d = both.ccc_100 - both.ccc_10
    both["prefers"] = np.where(d.abs() <= TIE, "tie",
                               np.where(d > 0, "fp_100", "fp_10"))
    n100 = int((both.prefers == "fp_100").sum())
    n10 = int((both.prefers == "fp_10").sum())
    ntie = int((both.prefers == "tie").sum())
    print(f"\n  cells graded under either anchor: {len(both)}"
          f"   = {n100} fp_100 + {n10} fp_10 + {ntie} tie")
    assert n100 + n10 + ntie == len(both)

    g100 = int(both.grade_100.isin(GRADED).sum())
    g10 = int(both.grade_10.isin(GRADED).sum())
    print(f"  graded cells over those rows: fp_100 {g100}   fp_10 {g10}"
          f"   (delta {g100 - g10:+d})")

    rows = []
    for mm, sub in both.groupby("metric"):
        rows.append(dict(
            metric_id=mm,
            cells=len(sub),
            graded_fp100=int(sub.grade_100.isin(GRADED).sum()),
            graded_fp10=int(sub.grade_10.isin(GRADED).sum()),
            prefers_fp100=int((sub.prefers == "fp_100").sum()),
            prefers_fp10=int((sub.prefers == "fp_10").sum()),
            ties=int((sub.prefers == "tie").sum())))
    R = pd.DataFrame(rows)
    R["delta"] = R.graded_fp100 - R.graded_fp10
    R = R.sort_values("delta", ascending=False)
    print("\n  per row (graded-cell counts under each anchor):")
    print(R.to_string(index=False))
    assert int(R.cells.sum()) == len(both)
    assert int(R.prefers_fp100.sum()) == n100 and int(R.prefers_fp10.sum()) == n10

    both[KEY + ["ccc_100", "ccc_10", "grade_100", "grade_10", "prefers"]].to_csv(
        os.path.join(V, "fp_target_cells.csv"), index=False, float_format="%.6g")
    R.to_csv(os.path.join(V, "fp_target_rows.csv"), index=False)
    print(f"\nsaved -> {os.path.join(V, 'hss_elevation.csv')}")
    print(f"saved -> {os.path.join(V, 'fp_target_cells.csv')}")
    print(f"saved -> {os.path.join(V, 'fp_target_rows.csv')}")


if __name__ == "__main__":
    main()
