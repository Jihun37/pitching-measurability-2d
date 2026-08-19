"""
Diamond - viewpoint-bin estimator + per-metric ZONE lookup (구성 B successor).

The station classifier hard-codes 3 stations whose boundaries came from the
coarse 45deg map; the el-densified 15deg sweep shows they are wrong in both
directions (lead knee survives to el=75; pelvis rot velo is full-ring usable
already at el=75). Instead of re-drawing class boundaries, this classifier
estimates the (azimuth, elevation) BIN of a clip and the per-metric decision
becomes a lookup into angle_zone_sweep.csv - when the zone table changes, no
retraining is needed.

Pipeline:
  1. kNN (cKDTree, standardized) predicts the (az, el) bin on the same
     15deg x {0,15,30,45,60,75,85} grid the zone sweep measures.
     Features/augmentation reused verbatim from station_classify (event-free
     dimensionless ratios; convention perturbation, aug rows train-only).
  2. Zone lookup for the CLAIMED zones only: the 6 noise-validated
     angle/position metrics, with the normaliser-confounded combinations
     (stride & release height at el>0) counted as OUT of zone. Velocity
     metrics are point recommendations and take no zone decision here.

Leakage guard: GroupKFold by session_pitch, test rows are clean (aug=0).

Reported per metric: false-accept rate on truly-out views (SAFETY - would
output garbage) and false-reject rate on truly-in views (lost coverage);
plus raw bin accuracy, azimuth circular error, elevation error.

Run:  cd src\\analysis
      python viewpoint_zone_classify.py --limit 60    (smoke test)
      python viewpoint_zone_classify.py [--noise 2]
"""
import os, sys, argparse, zlib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import config
import obp_project as O
import metrics as M
from hss_elevation_test import project_cam
from station_classify import features, perturb_convention, FEATS, gkf

AZ = list(range(0, 360, 15))
EL = [0, 15, 30, 45, 60, 75, 85]
FLOOR = 0.60

# claimed zones: noise-validated angle/position metrics only.
ZONE_METRICS = ["Lead Knee Angle [O]", "Arm Slot [O]", "Stride (anchor) [O]",
                "Trunk Tilt (ant) [O]", "Release Height [O]",
                "Hip-Shoulder Sep [O]"]
CONFOUNDED = {"Stride (anchor) [O]", "Release Height [O]"}   # el>0 = disregard


def zone_masks(floor=FLOOR, table="angle_zone_sweep.csv", metrics=None):
    """(metric -> {(az, el): bool}) claimed zone membership at the given
    r2 floor (0.60 = usable tier, 0.80 = reliable tier).
    table: which sweep CSV to look up. Default = detect-once (paper-table
    convention; keeps this script's official CV numbers unchanged).
    Deployment passes the promoted deployment-honest map (measure_auto)."""
    sweep = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR, table))
    masks = {}
    # metrics=None keeps the classifier's six-metric feature set (its published
    # CV numbers depend on it); the deployment driver passes the full adopted
    # list so the map and the driver report the same metrics (2026-07-24).
    for m in (ZONE_METRICS if metrics is None else metrics):
        sub = sweep[sweep.metric == m].set_index(["az", "el"])["r2"]
        d = {}
        for az in AZ:
            for el in EL:
                v = sub.get((az, el), np.nan)
                ok = bool(pd.notna(v) and v >= floor)
                if m in CONFOUNDED and el > 0:
                    ok = False
                d[(az, el)] = ok
        masks[m] = d
    return masks


def build(limit, noise, aug_k=2):
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    rows = []
    done = fail = 0
    for r in md.itertuples(index=False):
        if limit and done >= limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = O.load_c3d_joints(path)
            arm = O.detect_throwing_arm(joints, fps)
        except Exception:
            fail += 1; continue
        for az in AZ:
            for el in EL:
                try:
                    df = project_cam(joints, az, el)
                    if noise > 0:
                        rng = np.random.default_rng(az * 100 + el)
                        for c in df.columns:
                            df[c] = df[c] + rng.normal(0, noise / 300.0, len(df))
                    base = {"session_pitch": r.session_pitch,
                            "az": az, "el": el}
                    rows.append({**features(df, arm), **base, "aug": 0})
                    for a_i in range(aug_k):
                        rng = np.random.default_rng(zlib.crc32(
                            f"{r.session_pitch}|{az}|{el}|{a_i}".encode()))
                        rows.append({**features(perturb_convention(df, rng), arm),
                                     **base, "aug": 1})
                except Exception:
                    pass
        done += 1
        if done % 50 == 0:
            print(f"  ...{done} pitches")
    print(f"processed {done} / {fail} missing / {len(rows)} rows (aug_k={aug_k})")
    return pd.DataFrame(rows)


def knn_bin(Xtr, bins_tr, Xte, k=25):
    """Standardized kNN via cKDTree; majority-vote joint (az, el) bin."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    tree = cKDTree((Xtr - mu) / sd)
    _, idx = tree.query((Xte - mu) / sd, k=k, workers=-1)
    out = np.empty(len(Xte), dtype=int)
    for i, nn in enumerate(idx):
        v, c = np.unique(bins_tr[nn], return_counts=True)
        out[i] = v[np.argmax(c)]
    return out


def circ_err(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="keypoint noise px @ppm=300 (deployment-jitter scale)")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--aug-k", type=int, default=2)
    a = ap.parse_args()

    print(f"[viewpoint-bin -> zone lookup]  noise={a.noise}px  k={a.k}  "
          f"aug_k={a.aug_k}")
    masks = zone_masks()
    data = build(a.limit, a.noise, a.aug_k).dropna(subset=FEATS
                                                   ).reset_index(drop=True)

    bin_id = (data["az"] // 15 * 100 + data["el"]).to_numpy(int)  # az*100/15+el
    X = data[FEATS].to_numpy(float)
    groups = data["session_pitch"].to_numpy()
    clean = (data["aug"] == 0).to_numpy()

    pred = np.full(len(data), -1, dtype=int)
    for tr, te in gkf(groups, k=5):
        te = te[clean[te]]                    # evaluate clean rows only
        pred[te] = knn_bin(X[tr], bin_id[tr], X[te], a.k)

    ev = clean & (pred >= 0)
    paz = (pred[ev] // 100) * 15
    pel = pred[ev] % 100
    taz = data.loc[ev, "az"].to_numpy(int)
    tel = data.loc[ev, "el"].to_numpy(int)

    daz = circ_err(paz, taz)
    del_ = np.abs(pel - tel)
    print("\n" + "=" * 64)
    print(f"[BIN ESTIMATOR]  n_eval={ev.sum()}  (clean rows, GroupKFold)")
    print("=" * 64)
    print(f"exact bin accuracy          : {np.mean((daz == 0) & (del_ == 0)):.3f}")
    print(f"within 1 bin (az&el <=15deg): {np.mean((daz <= 15) & (del_ <= 15)):.3f}")
    print(f"azimuth  median|mean err    : {np.median(daz):.0f} / {daz.mean():.1f} deg"
          f"   (off-by-180: {np.mean(daz >= 165):.3f})")
    print(f"elevation median|mean err   : {np.median(del_):.0f} / {del_.mean():.1f} deg")

    print("\n" + "=" * 64)
    print(f"[ZONE DECISIONS]  claimed usable zones (r2>={FLOOR}, confounded=out)")
    print("=" * 64)
    print(f"{'metric':24s}{'in-zone%':>9s}{'accuracy':>10s}{'f-accept':>10s}"
          f"{'f-reject':>10s}")
    print("-" * 64)
    for m in ZONE_METRICS:
        mk = masks[m]
        t_in = np.array([mk[(az, el)] for az, el in zip(taz, tel)])
        p_in = np.array([mk[(az, el)] for az, el in zip(paz, pel)])
        acc = np.mean(t_in == p_in)
        fa = np.mean(p_in[~t_in]) if (~t_in).any() else np.nan   # SAFETY
        fr = np.mean(~p_in[t_in]) if t_in.any() else np.nan
        print(f"{m:24s}{t_in.mean():>9.2f}{acc:>10.3f}{fa:>10.3f}{fr:>10.3f}")

    out = os.path.join(config.OBP_VALIDATION_DIR, "viewpoint_zone_features.csv")
    data.to_csv(out, index=False)
    print(f"\nfeatures saved -> {out}")


if __name__ == "__main__":
    main()
