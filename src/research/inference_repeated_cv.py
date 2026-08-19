"""A7 -- R2 variability of the kinetic inference under REPEATED grouped CV (audit ST9).

WHY. Sec VII reports "no target reaches R2 = 0.60, the best is 0.597". A 0.003 margin is
not a boundary, and the manuscript presents it as one. The audit asked for the spread
around each target's point estimate so the claim can be scoped honestly.

WHY THE FOLDS HAVE TO BE REBUILT. `GroupKFold` is deterministic -- it takes no
random_state -- so re-running with a different seed varies only the GBM's internal
randomness, not the partition. Genuine repeated grouped CV needs a new pitcher-to-fold
assignment each time, which is what `repeated_folds` builds: pitchers are shuffled and
dealt round-robin into five folds, so no pitcher is ever split across train and test.

SCOPE. Only the PRESPECIFIED input set (`traj+scalar`) is run. Sec VII already reports
that the per-target best-of-four selection is an argmax taken outside the fold, and that
the prespecified set reaches the same conclusion; repeating the best-of-four here would
re-import that optimism into the variance estimate.

Everything else is inherited unchanged from inference_trajectory: the trajectory matrix,
the 34 canonical targets, the gt_clean population, foldwise PCA, and the pipeline.

Output: inference_repeated_cv.csv
Run:  conda activate diamond; cd src\\research; python inference_repeated_cv.py --R 10
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../analysis", "../stage2"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)

import config  # noqa: E402
import inference_trajectory as IT  # noqa: E402
from sklearn.model_selection import cross_val_predict  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.pipeline import make_pipeline, Pipeline  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from gt_landmark_outlier_effect import outlier_pitches  # noqa: E402

V = config.OBP_VALIDATION_DIR


def repeated_folds(groups, n_splits, rng):
    """A fresh pitcher-to-fold assignment. No pitcher spans train and test."""
    uq = np.unique(groups)
    rng.shuffle(uq)
    fold_of = {u: i % n_splits for i, u in enumerate(uq)}
    f = np.array([fold_of[u] for u in groups])
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(n_splits)]


def pipe_for(n_feat, traj_slice, n_comp, seed):
    scal = [i for i in range(n_feat) if i not in traj_slice]
    pre = ColumnTransformer([
        ("traj", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               PCA(n_components=n_comp, random_state=0)),
         list(traj_slice)),
        ("scal", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
         scal)])
    return Pipeline([("pre", pre),
                     ("gbm", GradientBoostingRegressor(random_state=seed))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=10, help="repeats")
    ap.add_argument("--points", type=int, default=25)
    ap.add_argument("--components", type=int, default=30)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("loading full_sig trajectories ...", flush=True)
    ids, T = IT.build_trajectory_matrix(a.points)

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    df = poi.merge(md[["session_pitch", "user"] + IT.ANTHRO], on="session_pitch")
    from paper_registry import KINETIC, METADATA
    kinetic = [c for c in poi.columns if c in KINETIC]
    meta = [c for c in poi.columns if c in METADATA]
    assert len(kinetic) == 34, len(kinetic)
    scalar = sorted(c for c in poi.columns
                    if c != "session_pitch" and c not in kinetic and c not in meta)

    tdf = pd.DataFrame(T, index=ids)
    tdf.index.name = "session_pitch"
    tdf.columns = [f"traj_{i}" for i in range(T.shape[1])]
    d = df.set_index("session_pitch").join(tdf, how="inner").reset_index()
    d = d[~d.session_pitch.isin(outlier_pitches())]
    print(f"  {len(d)} pitches, {d.user.nunique()} pitchers\n", flush=True)

    tcols = list(tdf.columns)
    feats = IT.ANTHRO + scalar + tcols          # the PRESPECIFIED set
    sl = range(len(feats) - len(tcols), len(feats))
    rng = np.random.default_rng(a.seed)

    rows = []
    for ti, t in enumerate(kinetic):
        sub = d.dropna(subset=[t])
        if len(sub) < 60:
            continue
        X = sub[feats].to_numpy(float)
        y = sub[t].to_numpy(float)
        g = sub.user.to_numpy()
        rs = []
        for r in range(a.R):
            folds = repeated_folds(g, a.splits, rng)
            pipe = pipe_for(X.shape[1], sl, a.components, seed=r)
            pred = cross_val_predict(pipe, X, y, cv=folds)
            rs.append(float(r2_score(y, pred)))
        rs = np.array(rs)
        rows.append(dict(target=t, n=len(sub), repeats=a.R,
                         r2_mean=rs.mean(), r2_sd=rs.std(ddof=1),
                         r2_min=rs.min(), r2_max=rs.max(),
                         r2_p2_5=np.percentile(rs, 2.5),
                         r2_p97_5=np.percentile(rs, 97.5),
                         ever_over_060=int((rs >= 0.60).sum())))
        print(f"  [{ti + 1:>2}/34] {t:<42} mean {rs.mean():+.3f} "
              f"sd {rs.std(ddof=1):.3f}  [{rs.min():+.3f}, {rs.max():+.3f}]"
              f"  >=0.60 in {int((rs >= 0.60).sum())}/{a.R}", flush=True)

    R = pd.DataFrame(rows).sort_values("r2_mean", ascending=False)
    R.to_csv(os.path.join(V, "inference_repeated_cv.csv"), index=False)

    print("\n" + "=" * 84)
    print(f"REPEATED GROUPED CV, prespecified set (traj+scalar), R = {a.R}")
    print("=" * 84)
    print(R.head(6).to_string(index=False))
    best = R.iloc[0]
    print(f"\nbest target        : {best.target}")
    print(f"  mean R2          : {best.r2_mean:.3f}  (single-run value in Sec VII: 0.597)")
    print(f"  SD over repeats  : {best.r2_sd:.3f}")
    print(f"  range            : [{best.r2_min:.3f}, {best.r2_max:.3f}]")
    print(f"targets ever reaching 0.60 in any repeat: "
          f"{int((R.ever_over_060 > 0).sum())} of {len(R)}")
    print(f"median of the per-target means          : {R.r2_mean.median():.3f}"
          f"   (single-run median in Sec VII: 0.226)")
    print(f"typical within-target SD                : {R.r2_sd.median():.3f}")
    print("\nwrote inference_repeated_cv.csv")


if __name__ == "__main__":
    main()
