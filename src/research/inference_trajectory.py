"""Back-calculation from full kinematic TRAJECTORIES, not scalar summaries.

Every previous inference run fed ~40 scalar poi columns -- single instants such as
"lead knee angle at release". The kinetic targets are integrals over the delivery
(energy transferred fp->BR, peak moment), so the earlier failures conflated two
different causes: no force channel in the inputs, and an input FORMAT that cannot
express an integral. This run removes the second one.

INPUTS: full_sig joint_angles (45 channels) + joint_velos (39 channels), the
perfect 3D signals, resampled inside two phase windows so both target families are
covered -- [peak knee height -> foot plant] and [foot plant -> ball release]. That
is a CEILING, not a deployable 2D system: a phone reads these imperfectly at best,
so a low number here closes the question for any 2D pipeline.

PCA is fitted INSIDE the CV pipeline. Fitting it on all pitches first would leak
test-fold structure into the components and inflate every score.

Four input sets, so a change can be attributed:
  anthro       height / mass / age only
  scalar       every kinematic poi column (the `allkin` ceiling of inference_final)
  traj         trajectory PCA only
  traj+scalar  both

Floor: 0.60, unified with the measurability map (2026-07-27 decision). 0.50, the
floor the older inference scripts inherited from a superseded convention, is
printed alongside so the two conventions can be compared but not silently mixed.

Run:  conda activate diamond; cd src\\research; python inference_trajectory.py
      python inference_trajectory.py --components 20 --points 25
"""
import os, sys, argparse, zipfile, io
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
sys.path.insert(0, HERE)
import config  # noqa: E402
from inference_retry_enriched import ANTHRO  # noqa: E402
# Same population as every other paper layer (CLAUDE.md: every paper number is
# gt_clean). It matters here twice over: the excluded pitches have an implausible
# foot-plant landmark, which both corrupts the poi truth computed at that landmark
# AND mis-slices the [pkh->fp] / [fp->BR] trajectory windows below.
from gt_landmark_outlier_effect import outlier_pitches  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.pipeline import make_pipeline, Pipeline  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

OUT = os.path.join(config.OBP_VALIDATION_DIR, "inference_trajectory.csv")
FULLSIG = os.path.join(config.OBP_DATA_DIR, "full_sig")
EVENT_COLS = ["pkh_time", "fp_10_time", "fp_100_time", "MER_time", "BR_time",
              "MIR_time"]
FLOOR, OLD_FLOOR = 0.60, 0.50


def read_sig(name):
    with zipfile.ZipFile(os.path.join(FULLSIG, name + ".zip")) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(io.TextIOWrapper(f, "utf-8"))


def resample_pitch(t, Y, t0, t1, n):
    """Y (frames x channels) linearly resampled to n points over [t0, t1]."""
    grid = np.linspace(t0, t1, n)
    return np.stack([np.interp(grid, t, Y[:, j]) for j in range(Y.shape[1])])


def build_trajectory_matrix(n_points):
    """(pitch ids, X, channel names). Two phase windows, so both the pkh->fp and
    the fp->BR target families have their own resolved segment."""
    ang, vel = read_sig("joint_angles"), read_sig("joint_velos")
    ach = [c for c in ang.columns if c not in ["session_pitch", "time"] + EVENT_COLS]
    vch = [c for c in vel.columns if c not in ["session_pitch", "time"] + EVENT_COLS]
    print(f"  channels: {len(ach)} angles + {len(vch)} velocities")
    vel_i = vel.set_index(["session_pitch", "time"])

    n1 = n_points // 2
    n2 = n_points - n1
    ids, rows = [], []
    for sp, g in ang.groupby("session_pitch", sort=False):
        g = g.sort_values("time")
        pkh, fp, br = (float(g[c].iloc[0]) for c in
                       ("pkh_time", "fp_100_time", "BR_time"))
        if not (0 < pkh < fp < br):
            continue
        try:
            v = vel_i.loc[sp].sort_index()
        except KeyError:
            continue
        t = g.time.to_numpy(float)
        A = g[ach].to_numpy(float)
        tv = v.index.to_numpy(float)
        V = v[vch].to_numpy(float)
        seg = [resample_pitch(t, A, pkh, fp, n1), resample_pitch(t, A, fp, br, n2),
               resample_pitch(tv, V, pkh, fp, n1), resample_pitch(tv, V, fp, br, n2)]
        ids.append(sp)
        rows.append(np.concatenate([s.ravel() for s in seg]))
    X = np.vstack(rows)
    print(f"  trajectory matrix: {X.shape[0]} pitches x {X.shape[1]} features")
    return ids, X


def cv_scores(X, y, groups, traj_slice, n_comp, n_splits=5, seed=0):
    """R2 plus the errors in the target's OWN units.

    R2 alone invites the reader to treat 0.45 as "half right". MAE against the
    between-pitcher SD of the truth says whether a prediction could ever separate
    two pitchers: NMAE >= 1 means the typical error is the whole spread of the
    population, i.e. the number carries no information about who is who.
    """
    pred = cv_predict(X, y, groups, traj_slice, n_comp, n_splits, seed)
    mae = float(np.abs(pred - y).mean())
    sd = float(np.std(y, ddof=1))
    return dict(r2=float(r2_score(y, pred)), mae=mae, sd=sd,
                nmae=mae / sd if sd > 0 else np.nan)


def cv_predict(X, y, groups, traj_slice, n_comp, n_splits=5, seed=0):
    """GBM with PCA applied to the trajectory block only, fitted inside the CV."""
    if traj_slice is None:
        pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    else:
        n = X.shape[1]
        scal = [i for i in range(n) if i not in traj_slice]
        pre = ColumnTransformer([
            ("traj", make_pipeline(SimpleImputer(strategy="median"),
                                   StandardScaler(),
                                   PCA(n_components=n_comp, random_state=0)),
             list(traj_slice)),
            ("scal", make_pipeline(SimpleImputer(strategy="median"),
                                   StandardScaler()), scal)])
    pipe = Pipeline([("pre", pre),
                     ("gbm", GradientBoostingRegressor(random_state=seed))])
    return cross_val_predict(pipe, X, y, groups=groups,
                             cv=GroupKFold(n_splits=n_splits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=25, help="resample points/channel")
    ap.add_argument("--components", type=int, default=30, help="PCA components")
    a = ap.parse_args()

    print("loading full_sig trajectories ...")
    ids, T = build_trajectory_matrix(a.points)

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    df = poi.merge(md[["session_pitch", "user"] + ANTHRO], on="session_pitch")

    # TARGET LIST IS CANONICAL, 2026-07-30. This used to select targets by
    # `column_coverage_audit.csv`'s `reason_code == "A-kinetics"`, which tied the
    # kinetic target set to the retired A-G failure-reason scheme and to a file that
    # is no longer an active input. `paper_registry` holds the explicit 34-target and
    # 5-metadata lists instead. The two derivations were verified to select the
    # IDENTICAL columns before the switch, so this changes no result.
    # Note the kinetic targets are deliberately NOT rows of paper_registry.csv --
    # that file is the 47-row direct-measurement registry. Kinetics share only the
    # frozen 394-pitch population.
    from paper_registry import KINETIC, METADATA
    kinetic = [c for c in poi.columns if c in KINETIC]
    meta = [c for c in poi.columns if c in METADATA]
    assert len(kinetic) == 34, len(kinetic)
    assert len(meta) == 5, len(meta)
    scalar = sorted(c for c in poi.columns
                    if c != "session_pitch" and c not in kinetic and c not in meta)

    tdf = pd.DataFrame(T, index=ids)
    tdf.index.name = "session_pitch"      # else the join drops the name
    tdf.columns = [f"traj_{i}" for i in range(T.shape[1])]
    d = df.set_index("session_pitch").join(tdf, how="inner").reset_index()
    bad = outlier_pitches()
    n_before = len(d)
    d = d[~d.session_pitch.isin(bad)]
    print(f"  matched {n_before} pitches -> {len(d)} after the gt_clean filter "
          f"({n_before - len(d)} landmark outliers), {d.user.nunique()} pitchers\n")

    tcols = list(tdf.columns)
    SETS = {
        "anthro":      (ANTHRO, False),
        "scalar":      (ANTHRO + scalar, False),
        "traj":        (ANTHRO + tcols, True),
        "traj+scalar": (ANTHRO + scalar + tcols, True),
    }
    print(f"PCA {a.components} components on the {len(tcols)}-feature trajectory "
          f"block, fitted inside each CV fold")
    print(f"floor {FLOOR} (unified with the map); {OLD_FLOOR} shown for reference\n")

    rows = []
    for t in kinetic:
        sub = d.dropna(subset=[t])
        if len(sub) < 60:
            continue
        y = sub[t].to_numpy(float); g = sub.user.to_numpy()
        line = {"target": t, "n": len(sub)}
        for name, (feats, use_pca) in SETS.items():
            X = sub[feats].to_numpy(float)
            sl = range(len(feats) - len(tcols), len(feats)) if use_pca else None
            s = cv_scores(X, y, g, sl, a.components)
            line[name] = s["r2"]
            line[f"mae_{name}"] = s["mae"]
            line[f"nmae_{name}"] = s["nmae"]
            line["truth_sd"] = s["sd"]
        rows.append(line)
        print(f"  {t:<42}" + "  ".join(f"{k} {line[k]:>6.3f}" for k in SETS))

    res = pd.DataFrame(rows)
    res["best_set"] = res[list(SETS)].idxmax(axis=1)
    res["best_r2"] = res[list(SETS)].max(axis=1)
    res["traj_gain"] = res["traj+scalar"] - res["scalar"]
    res.to_csv(OUT, index=False, float_format="%.6g")

    W = 104
    print("\n" + "=" * W)
    print("TRAJECTORY BACK-CALCULATION  (perfect 3D signals = ceiling; GroupKFold "
          "by pitcher)")
    print("=" * W)
    print(f"{'kinetic target':<40}{'R2 anthro':>10}{'R2 scalar':>10}"
          f"{'R2 traj+sc':>11}{'MAE best':>11}{'truth SD':>10}{'NMAE':>7}")
    print("-" * W)
    for _, r in res.sort_values("best_r2", ascending=False).iterrows():
        bs = r.best_set
        print(f"{r.target:<40}{r.anthro:>10.3f}{r.scalar:>10.3f}"
              f"{r['traj+scalar']:>11.3f}{r[f'mae_{bs}']:>11.3g}"
              f"{r.truth_sd:>10.3g}{r[f'nmae_{bs}']:>7.2f}")
    print("-" * W)
    print(f"clears {FLOOR}: {int((res.best_r2 >= FLOOR).sum())} of {len(res)}   "
          f"clears {OLD_FLOOR}: {int((res.best_r2 >= OLD_FLOOR).sum())} of {len(res)}")
    print(f"median gain from trajectories over scalars: "
          f"{res.traj_gain.median():+.3f}   (max {res.traj_gain.max():+.3f} on "
          f"{res.loc[res.traj_gain.idxmax(), 'target']})")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
