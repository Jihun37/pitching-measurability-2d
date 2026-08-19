"""Layer-1 sub-step F: what happened to every one of the 81 OBP columns.

SCOPE, FIXED 2026-07-30. This closes the coverage question on DISPOSITION only:

    81 columns -> 5 metadata / 34 kinetic / 42 kinematic
    42 kinematic + 5 direct-3D pathways -> 47 evaluated rows
    47 evaluated -> 35 retained + 12 non-retained

⚠ THE A-G FAILURE-REASON SCHEME IS RETIRED. `column_coverage_audit.csv` assigned
each column a cause code (A-kinetics, B-..., through G) and a supporting number.
That scheme is not reproduced here and must not be reintroduced. For the 12
non-retained rows this file reports **best CCC, best r-squared and best view, and
nothing else** -- no cause, no mechanism, no "why". A row can fail to hold a graded
cell for reasons this study did not isolate, and asserting one anyway was the part
of the old audit that could not be defended.

The one causal statement that survives is structural rather than empirical: the 34
kinetic columns are not direct-measurement candidates at all, because they are force
and moment quantities and no kinematic observable contains a force channel. They are
handled by the separate upper-bound inference protocol
(`research/inference_trajectory.py`), which shares only the frozen 394-pitch
population, and they are NOT rows of `paper_registry.csv`.

Input:  poi_metrics.csv, paper_registry.csv, inference_trajectory.csv
Output: coverage_accounting.csv (one row per OBP column, 81 rows)
Run:  conda activate diamond; cd src\\analysis; python coverage_accounting.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import pandas as pd
import config
from paper_registry import METADATA, KINETIC

V = config.OBP_VALIDATION_DIR


def main():
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"),
                      nrows=1)
    cols = list(poi.columns)
    reg = pd.read_csv(os.path.join(V, "paper_registry.csv"))

    # a kinematic column is reached by exactly one registry row
    by_col = {}
    for r in reg.itertuples(index=False):
        if r.truth_source == "obp_column":
            by_col[r.truth_quantity] = r

    rows = []
    for c in cols:
        if c in METADATA:
            cls, path = "metadata", ""
        elif c in KINETIC:
            cls, path = "kinetic", "upper-bound inference protocol"
        else:
            cls = "kinematic"
            path = by_col[c].metric_id if c in by_col else ""
        r = by_col.get(c)
        rows.append(dict(
            column=c, column_class=cls, pathway=path,
            evaluated=bool(r is not None),
            retained=(bool(r.retained) if r is not None else False),
            best_ccc=(r.best_ccc if r is not None else None),
            best_r2=(r.best_r2 if r is not None else None),
            best_view=(f"{int(r.best_az)}/{int(r.best_el)}" if r is not None else ""),
        ))
    C = pd.DataFrame(rows)

    n_meta = int((C.column_class == "metadata").sum())
    n_kin = int((C.column_class == "kinetic").sum())
    n_kine = int((C.column_class == "kinematic").sum())
    print(f"81 OBP columns = {n_meta} metadata + {n_kin} kinetic + {n_kine} kinematic")
    assert (n_meta, n_kin, n_kine) == (5, 34, 42), (n_meta, n_kin, n_kine)
    assert len(C) == 81

    kine_eval = int(C[(C.column_class == "kinematic")].evaluated.sum())
    direct = int((reg.truth_source == "3d_direct").sum())
    print(f"{kine_eval} kinematic-column pathways + {direct} direct-3D-truth rows "
          f"= {kine_eval + direct} evaluated rows")
    # no direct-3D rows since 2026-08-12: every row now takes its truth from a
    # published column, so the evaluated set is fixed by the dataset
    assert kine_eval == 42 and direct == 0

    n_ret = int(reg.retained.sum())
    print(f"{len(reg)} evaluated = {n_ret} retained + {len(reg) - n_ret} non-retained")
    assert n_ret == 30 and len(reg) - n_ret == 12

    # every kinematic column must be evaluated exactly once
    unreached = sorted(C.column[(C.column_class == "kinematic") & ~C.evaluated])
    assert not unreached, unreached
    print("no kinematic column is left unevaluated, and none is evaluated twice")

    C.to_csv(os.path.join(V, "coverage_accounting.csv"), index=False,
             float_format="%.6g")
    print(f"\nsaved -> {os.path.join(V, 'coverage_accounting.csv')}")

    print("\n" + "=" * 92)
    print("THE 12 NON-RETAINED ROWS -- best cell only, no cause is inferred")
    print("=" * 92)
    nr = reg[~reg.retained][["metric_id", "quantity_family", "anchor_type",
                             "best_ccc", "best_r2", "best_az", "best_el"]]
    nr = nr.sort_values("best_ccc", ascending=False)
    print(nr.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nbest CCC among them {nr.best_ccc.max():.3f} "
          f"(moderate contour is 0.75); best r-squared {nr.best_r2.max():.3f} "
          f"(screen is 0.60)")

    # the kinetic side, reported but never merged into the registry
    p = os.path.join(V, "inference_trajectory.csv")
    if os.path.exists(p):
        inf = pd.read_csv(p)
        print(f"\nkinetic targets carried by the inference protocol: {len(inf)}"
              f"   population n = {sorted(inf.n.unique())}")
        assert len(inf) == 34, len(inf)


if __name__ == "__main__":
    main()
