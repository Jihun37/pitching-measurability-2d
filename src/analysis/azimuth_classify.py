"""
Diamond - Feasibility test: can a single 2D pitch video self-identify its
camera azimuth from pose geometry alone? (patent element B feasibility)

Each OBP c3d is projected to 7 azimuths (0..90). We extract zoom/distance-
invariant geometric features (all normalized by the vertical torso/stature
size, which is azimuth-invariant) and ask a classifier to recover the azimuth.

Leakage guard: the 7 projections of one pitch share body/individual identity,
so we split by session_pitch (GroupKFold) - a pitch is never in both train
and test. This measures generalization to *unseen pitchers/pitches*, i.e. the
real deployment question.

Reports:
  - 3-class (side / three-quarter / front) accuracy + confusion
  - 7-class exact-azimuth accuracy + confusion
  - azimuth MAE in degrees (kNN regression)

Run:
  conda activate diamond
  cd src\analysis
  python azimuth_classify.py            # all pitches, clean projection
  python azimuth_classify.py --limit 150 --noise 3
"""
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import config
import obp_project as O
import metrics as M

AZIMUTHS = [0, 15, 30, 45, 60, 75, 90]
ELEV = 0.0

# coarse bucket per azimuth
def bucket(az):
    if az <= 15:  return "side"
    if az >= 75:  return "front"
    return "three_quarter"


def compute_features(df, arm, fps):
    """Azimuth-discriminating geometric features from one 2D projection.
    All horizontal quantities normalized by a vertical body size (torso or
    stature) so they are invariant to camera distance / zoom / ppm."""
    J = M.JOINTS
    lead = "left" if arm == "right" else "right"
    rel = M.release_frame(df, arm, fps, J, view="side")
    fp = M.foot_plant_frame(df, lead, fps, J, rel)
    torso = M.body_scale_px(df, J)            # vertical shoulder-hip dist (px)
    stat = M.pixel_stature(df, J)             # vertical stature (px)

    def X(k): return M._xy(df, k, J)[0]
    def Y(k): return M._xy(df, k, J)[1]

    lsx, rsx = X("l_sh"), X("r_sh")
    lhx, rhx = X("l_hip"), X("r_hip")
    lax, rax = X("l_an"), X("r_an")
    wkey = "r_wr" if arm == "right" else "l_wr"
    wx, wy = X(wkey), Y(wkey)
    lead_ax = lax if lead == "left" else rax

    # shoulder / hip projected width (side: line is along depth -> narrow-ish;
    # varies with body rotation, so we take both fp-instant and pitch-median)
    sh_w_fp = abs(lsx[fp] - rsx[fp]) / torso
    hip_w_fp = abs(lhx[fp] - rhx[fp]) / torso
    sh_w_med = float(np.nanmedian(np.abs(lsx - rsx))) / torso
    hip_w_med = float(np.nanmedian(np.abs(lhx - rhx))) / torso
    ratio_fp = sh_w_fp / (hip_w_fp + 1e-6)
    ratio_med = sh_w_med / (hip_w_med + 1e-6)

    # horizontal foot separation at plant (side: large, front: small)
    foot_sep = abs(lax[fp] - rax[fp]) / torso
    # horizontal stride displacement of lead ankle (side: large, front: small)
    stride_u = abs(lead_ax[fp] - lead_ax[0]) / stat
    # throwing-wrist horizontal vs vertical travel over the whole pitch
    wu = (np.nanmax(wx) - np.nanmin(wx)) / stat
    wv = (np.nanmax(wy) - np.nanmin(wy)) / stat
    w_ratio = wu / (wv + 1e-6)

    return {
        "sh_w_fp": sh_w_fp, "hip_w_fp": hip_w_fp,
        "sh_w_med": sh_w_med, "hip_w_med": hip_w_med,
        "ratio_fp": ratio_fp, "ratio_med": ratio_med,
        "foot_sep": foot_sep, "stride_u": stride_u,
        "wu": wu, "wv": wv, "w_ratio": w_ratio,
    }


FEATS = ["sh_w_fp", "hip_w_fp", "sh_w_med", "hip_w_med", "ratio_fp",
         "ratio_med", "foot_sep", "stride_u", "wu", "wv", "w_ratio"]


def build_dataset(limit=None, noise=0.0):
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
        for az in AZIMUTHS:
            try:
                df = O.project_view(joints, azimuth_deg=az, elevation_deg=ELEV,
                                    noise_px=noise, seed=az)
                f = compute_features(df, arm, fps)
                f.update({"session_pitch": r.session_pitch, "az": az})
                rows.append(f)
            except Exception:
                pass
        done += 1
        if done % 100 == 0:
            print(f"  ...{done} pitches")
    print(f"processed {done} pitches / {fail} missing / {len(rows)} projections")
    return pd.DataFrame(rows)


def group_kfold_idx(groups, k=5, seed=0):
    """Yield (train_idx, test_idx) splitting by unique group (session_pitch)."""
    uniq = np.array(sorted(pd.unique(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, k)
    g = np.asarray(groups)
    for i in range(k):
        test_g = set(folds[i].tolist())
        test = np.array([j for j, gv in enumerate(g) if gv in test_g])
        train = np.array([j for j, gv in enumerate(g) if gv not in test_g])
        yield train, test


def knn_predict(Xtr, ytr, Xte, k=15):
    """Standardized kNN; returns majority-vote labels (classification)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    preds = np.empty(len(B), dtype=ytr.dtype)
    for i, b in enumerate(B):
        d = np.sum((A - b) ** 2, axis=1)
        nn = ytr[np.argsort(d)[:k]]
        vals, cnts = np.unique(nn, return_counts=True)
        preds[i] = vals[np.argmax(cnts)]
    return preds


def knn_regress(Xtr, ytr, Xte, k=15):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    out = np.empty(len(B))
    for i, b in enumerate(B):
        d = np.sum((A - b) ** 2, axis=1)
        out[i] = ytr[np.argsort(d)[:k]].mean()
    return out


def confusion(true, pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(true, pred):
        m[idx[t], idx[p]] += 1
    return m


def print_confusion(m, labels, title):
    print(f"\n{title}  (rows=true, cols=pred)")
    w = max(6, max(len(str(l)) for l in labels) + 1)
    print(" " * 14 + "".join(f"{str(l):>{w}}" for l in labels))
    for i, l in enumerate(labels):
        row = "".join(f"{m[i, j]:>{w}d}" for j in range(len(labels)))
        acc = m[i, i] / max(1, m[i].sum())
        print(f"{str(l):>13} {row}   ({acc:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--noise", type=float, default=0.0, help="keypoint noise px")
    ap.add_argument("--k", type=int, default=15)
    a = ap.parse_args()

    print(f"[azimuth self-ID feasibility]  noise={a.noise}px  k={a.k}")
    data = build_dataset(limit=a.limit, noise=a.noise)
    data = data.dropna(subset=FEATS).reset_index(drop=True)

    X = data[FEATS].to_numpy(float)
    az = data["az"].to_numpy(int)
    coarse = np.array([bucket(v) for v in az])
    groups = data["session_pitch"].to_numpy()

    # cross-validated predictions
    pred_c = np.empty(len(data), dtype=coarse.dtype)
    pred_e = np.empty(len(data), dtype=int)
    pred_r = np.empty(len(data))
    for tr, te in group_kfold_idx(groups, k=5):
        pred_c[te] = knn_predict(X[tr], coarse[tr], X[te], a.k)
        pred_e[te] = knn_predict(X[tr], az[tr], X[te], a.k)
        pred_r[te] = knn_regress(X[tr], az[tr], X[te], a.k)

    acc_c = (pred_c == coarse).mean()
    acc_e = (pred_e == az).mean()
    mae = np.abs(pred_r - az).mean()

    print("\n" + "=" * 60)
    print(f"3-class (side/three_quarter/front) accuracy : {acc_c:.3f}")
    print(f"7-class exact-azimuth accuracy              : {acc_e:.3f}")
    print(f"azimuth regression MAE                      : {mae:.1f} deg")
    print("=" * 60)

    print_confusion(confusion(coarse, pred_c, ["side", "three_quarter", "front"]),
                    ["side", "three_quarter", "front"], "3-class confusion")
    print_confusion(confusion(az, pred_e, AZIMUTHS), AZIMUTHS, "7-class confusion")

    out = os.path.join(config.OBP_VALIDATION_DIR, "azimuth_classify_features.csv")
    data.to_csv(out, index=False)
    print(f"\nfeatures saved -> {out}")


if __name__ == "__main__":
    main()
