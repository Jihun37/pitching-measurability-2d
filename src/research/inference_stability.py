"""Is a 0.50 crossing in inference_final.py real, or CV noise?

inference_final left three energy-TRANSFER targets just over the R2 0.50 floor.
The ledger already records one such crossing that did not survive
(lead_hip_transfer_fp_br scored 0.544 in an earlier run and failed verification),
so a crossing by 0.02-0.06 has to be shown stable before it is reported as one.

GroupKFold is deterministic, so the spread is produced by varying what is
arbitrary about the protocol and nothing else: the fold count and the learner's
random_state. A target whose score straddles the floor across those settings is
reported as "at the floor", not as a pass.

Run:  conda activate diamond; cd src\\research; python inference_stability.py
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import config  # noqa: E402
from inference_retry_enriched import ANTHRO, FLOOR  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

FINAL = os.path.join(config.OBP_VALIDATION_DIR, "inference_final.csv")
OUT = os.path.join(config.OBP_VALIDATION_DIR, "inference_stability.csv")
SPLITS = (3, 4, 5, 6, 8)
SEEDS = (0, 1, 2)


def cv(df, feats, target, n_splits, seed):
    sub = df.dropna(subset=[target])
    X = sub[feats].to_numpy(float); y = sub[target].to_numpy(float)
    g = sub["user"].to_numpy()
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         GradientBoostingRegressor(random_state=seed))
    pred = cross_val_predict(pipe, X, y, groups=g, cv=GroupKFold(n_splits=n_splits))
    return float(r2_score(y, pred))


def main():
    res = pd.read_csv(FINAL)
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    df = poi.merge(md[["session_pitch", "user"] + ANTHRO], on="session_pitch")

    # everything within 0.10 of the floor, in either direction, is worth testing
    near = res[(res.best_r2 >= FLOOR - 0.10)].sort_values("best_r2", ascending=False)
    print(f"testing {len(near)} targets within 0.10 of the R2 {FLOOR} floor, "
          f"{len(SPLITS)}x{len(SEEDS)} protocol settings each\n")

    # the winning input set for each, rebuilt exactly as inference_final built it
    import inference_final as F  # reuse the set definitions, no second copy
    audit = pd.read_csv(F.AUDIT).set_index("column")
    kinetic = [c for c in poi.columns
               if c in audit.index and audit.loc[c, "reason_code"] == "A-kinetics"]
    meta = [c for c in poi.columns
            if c in audit.index and "metadata" in str(audit.loc[c, "reason"])]
    allkin = sorted(c for c in poi.columns
                    if c != "session_pitch" and c not in kinetic and c not in meta)
    lay = pd.read_csv(F.LAYER)
    from angle_map_2d import adopted_rows, gt_only_rows
    lt = {l.strip(): t for l, _, t in adopted_rows() + gt_only_rows()
          if isinstance(t, str)}
    ok = lay[lay.gate_cells > 0]
    gate = sorted({c for c in ok.metric if c in poi.columns} |
                  {lt[m.strip()] for m in ok.metric if m.strip() in lt})
    SETS = {"anthro": [], "gate": gate, "allkin": allkin}

    rows = []
    for _, r in near.iterrows():
        st = r.best_set if r.best_set in SETS else "allkin"
        feats = ANTHRO + SETS[st]
        vals = [cv(df, feats, r.target, ns, sd) for ns in SPLITS for sd in SEEDS]
        v = np.array(vals)
        rows.append(dict(target=r.target, input_set=st, reported=r.best_r2,
                         mean=v.mean(), sd=v.std(ddof=1), lo=v.min(), hi=v.max(),
                         frac_above=float((v >= FLOOR).mean())))
        print(f"  {r.target:<40}{st:>8}  reported {r.best_r2:.3f}  ->  "
              f"{v.mean():.3f} +- {v.std(ddof=1):.3f}  [{v.min():.3f}, {v.max():.3f}]"
              f"   above floor in {int((v >= FLOOR).sum())}/{len(v)}")

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False, float_format="%.6g")
    print(f"\nsaved -> {OUT}")
    stable = d[d.frac_above == 1.0]
    print(f"\nstable passes (above the floor in EVERY setting): {len(stable)}")
    for _, r in stable.iterrows():
        print(f"  {r.target}  mean {r['mean']:.3f}")
    strad = d[(d.frac_above > 0) & (d.frac_above < 1)]
    print(f"straddling the floor: {len(strad)}")
    for _, r in strad.iterrows():
        print(f"  {r.target}  mean {r['mean']:.3f} ({r.frac_above:.0%} of settings)")


if __name__ == "__main__":
    main()
