"""
Diamond - Viewpoint STATION classifier (patent element B + F).

NOTE (2026-07-10): kept as the OFFICIAL §10.3 artifact (paper numbers quote
from PAPER_REVISION_HANDOFF §10.3 only). The station framing itself is
SUPERSEDED by the valid-zone lookup — new work should use
`viewpoint_zone_classify.py` (bin estimation + zone table lookup, §11.4) and
`deploy/measure_auto.py`. The premise below ("only 3 viewpoints unlock
metrics") predates the el-densified zone table, which shows wider regions.

We don't need to regress a precise (azimuth, elevation). Only 3 viewpoints
unlock metrics; everything else is a dead zone. So classify a pitch video into:
  side     : az<=30, el<=30   -> 6 sagittal metrics
  front    : az>=70, el<=30   -> arm slot
  overhead : el>=80           -> hip-shoulder separation
  reject   : dead zone        -> trigger re-shoot guidance (element F)
Station tolerances follow the r2>=0.6 usable regions of angle_map_2d.

Features are event-free, dimensionless RATIOS (zoom/distance-invariant) taken
over the whole clip, using only MediaPipe-available landmarks -> deployable
without a clean reference view.

Leakage guard: 5-fold CV grouped by session_pitch (all viewpoints of one pitch
stay together). Reported: overall accuracy, per-class recall, confusion, and the
SAFETY number = fraction of true-reject predicted as a usable station
(false-accept -> would output garbage).

Run:  cd src\analysis
      python station_classify.py --limit 120
      python station_classify.py [--noise 3]
"""
import os, sys, argparse, zlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import config
import obp_project as O
import metrics as M
from hss_elevation_test import project_cam

AZ = [0, 15, 30, 45, 60, 75, 90]
EL = [0, 15, 30, 45, 60, 75, 85, 90]
CLASSES = ["side", "front", "overhead", "reject"]
USABLE = {"side", "front", "overhead"}


def station(az, el):
    if el >= 80:              return "overhead"
    if el <= 30 and az <= 30: return "side"
    if el <= 30 and az >= 70: return "front"
    return "reject"


def features(df, arm):
    J = M.JOINTS
    def C(k): return M._xy(df, k, J)
    lsx, lsy = C("l_sh"); rsx, rsy = C("r_sh")
    lhx, lhy = C("l_hip"); rhx, rhy = C("r_hip")
    lkx, lky = C("l_kn");  rkx, rky = C("r_kn")
    lax, lay = C("l_an");  rax, ray = C("r_an")
    hdx, hdy = C("head")
    wx, wy = C("r_wr" if arm == "right" else "l_wr")
    eps = 1e-6

    shw = np.abs(rsx - lsx)
    hipw = np.abs(rhx - lhx)
    torsov = np.abs((lsy + rsy) / 2 - (lhy + rhy) / 2)
    footsep = np.abs(lax - rax)
    head_ankle_v = np.maximum(lay, ray) - hdy
    Xs = np.vstack([lsx, rsx, lhx, rhx, lkx, rkx, lax, rax, hdx])
    Ys = np.vstack([lsy, rsy, lhy, rhy, lky, rky, lay, ray, hdy])
    bbox_h = np.nanmax(Ys, 0) - np.nanmin(Ys, 0)
    bbox_w = np.nanmax(Xs, 0) - np.nanmin(Xs, 0)
    lead_ax = lax if arm == "right" else rax     # lead = opposite arm side... (x range only)
    wu = np.nanmax(wx) - np.nanmin(wx)
    wv = np.nanmax(wy) - np.nanmin(wy)

    med = lambda a: float(np.nanmedian(a))
    return {
        "shhip":   med(shw / (hipw + eps)),                 # azimuth
        "footsep": med(footsep / (shw + eps)),              # azimuth (side large)
        "torso":   med(torsov / (shw + eps)),               # elevation (ground large)
        "headv":   med(head_ankle_v / (shw + eps)),         # elevation (ground large)
        "bbox":    med(bbox_h / (bbox_w + eps)),            # elevation (ground tall)
        "wrist":   wu / (wv + eps),                          # azimuth+elevation
        "stride":  (np.nanmax(lead_ax) - np.nanmin(lead_ax)) / (med(shw) + eps),
    }


# shhip (shoulder/hip width ratio) removed 2026-07-04: it is the feature most
# sensitive to landmark-convention differences between backbones (MediaPipe vs
# Halpe26 vs OBP mocap markers) — it alone flipped a real side clip to reject
# on RTMPose coords — and dropping it RAISES clean-OBP CV accuracy
# (0.847 -> 0.852; with convention augmentation 0.877). See
# tests/station_convention_test.py for the experiment matrix.
FEATS = ["footsep", "torso", "headv", "bbox", "wrist", "stride"]


def perturb_convention(df, rng):
    """Systematic landmark-convention offsets used as training augmentation:
    chord-width scaling for shoulders/hips (different backbones place these
    points differently) and a head vertical offset (nose vs head-top style).
    Per-clip constant factors = a convention, not per-frame noise."""
    out = df.copy()
    for l, r in (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip")):
        s = rng.uniform(0.85, 1.15)
        for ax in ("_x", "_y"):
            a, b = out[l + ax], out[r + ax]
            mid = (a + b) / 2
            out[l + ax] = mid + (a - mid) * s
            out[r + ax] = mid + (b - mid) * s
    torso = np.nanmedian(np.abs(
        (out["left_shoulder_y"] + out["right_shoulder_y"]) / 2
        - (out["left_hip_y"] + out["right_hip_y"]) / 2))
    out["head_y"] = out["head_y"] + rng.uniform(-0.15, 0.15) * torso
    return out


def build(limit, noise, aug_k=2):
    """Training features. aug_k adds convention-perturbed copies per
    projection (adopted 2026-07-04: raises CV acc 0.847->0.877, cuts
    false-accept 0.168->0.131, and makes real-video predictions backbone-
    invariant). aug_k=0 reproduces the old clean-only training set.
    Rows carry aug=0/1 so CV can test on clean rows only."""
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
                            df[c] = df[c] + rng.normal(0, noise, len(df))
                    base = {"session_pitch": r.session_pitch,
                            "y": station(az, el)}
                    rows.append({**features(df, arm), **base, "aug": 0})
                    for a_i in range(aug_k):
                        # crc32, NOT hash(): python str hash is salted per
                        # process, which made runs non-reproducible
                        rng = np.random.default_rng(zlib.crc32(
                            f"{r.session_pitch}|{az}|{el}|{a_i}".encode()))
                        rows.append({**features(perturb_convention(df, rng), arm),
                                     **base, "aug": 1})
                except Exception:
                    pass
        done += 1
        if done % 100 == 0:
            print(f"  ...{done} pitches")
    print(f"processed {done} / {fail} missing / {len(rows)} rows"
          f" (aug_k={aug_k})")
    return pd.DataFrame(rows)


def gkf(groups, k=5, seed=0):
    u = np.array(sorted(pd.unique(groups)))
    np.random.default_rng(seed).shuffle(u)
    folds = np.array_split(u, k)
    g = np.asarray(groups)
    for i in range(k):
        tg = set(folds[i].tolist())
        te = np.array([j for j, v in enumerate(g) if v in tg])
        tr = np.array([j for j, v in enumerate(g) if v not in tg])
        yield tr, te


def knn(Xtr, ytr, Xte, k=25):
    """Return (label, vote_share) so a confidence threshold can defer to reject."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    lab = np.empty(len(B), dtype=ytr.dtype)
    shr = np.empty(len(B))
    for i, b in enumerate(B):
        nn = ytr[np.argsort(np.sum((A - b) ** 2, 1))[:k]]
        v, c = np.unique(nn, return_counts=True)
        j = np.argmax(c)
        lab[i], shr[i] = v[j], c[j] / k
    return lab, shr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--aug-k", type=int, default=2,
                    help="convention-perturbed copies per projection (0=legacy)")
    a = ap.parse_args()

    print(f"[station classifier]  noise={a.noise}px  k={a.k}  aug_k={a.aug_k}")
    pool = build(a.limit, a.noise, a.aug_k).dropna(subset=FEATS).reset_index(drop=True)
    # test folds contain ONLY clean rows; augmented copies join the training
    # pool of their own group (never leak into a test fold)
    data = pool[pool.aug == 0].reset_index(drop=True)
    X = data[FEATS].to_numpy(float)
    y = data["y"].to_numpy()
    groups = data["session_pitch"].to_numpy()

    pred = np.empty(len(data), dtype=y.dtype)
    share = np.empty(len(data))
    for tr, te in gkf(groups, 5):
        te_groups = set(groups[te].tolist())
        trp = pool[~pool.session_pitch.isin(te_groups)]
        pred[te], share[te] = knn(trp[FEATS].to_numpy(float),
                                  trp["y"].to_numpy(), X[te], a.k)

    acc = (pred == y).mean()
    print(f"\noverall accuracy: {acc:.3f}\n")

    # persist CV predictions so figures (fig_station_eval) render from the
    # SAME run as the reported numbers instead of re-running their own CV
    pred_csv = os.path.join(config.OBP_VALIDATION_DIR, "station_cv_predictions.csv")
    pd.DataFrame({"y": y, "pred": pred, "share": share,
                  "session_pitch": groups}).to_csv(pred_csv, index=False)
    print(f"cv predictions -> {pred_csv}\n")

    idx = {c: i for i, c in enumerate(CLASSES)}
    cm = np.zeros((4, 4), int)
    for t, p in zip(y, pred):
        cm[idx[t], idx[p]] += 1
    w = 10
    print("confusion  (rows=true, cols=pred)")
    print(" " * 11 + "".join(f"{c:>{w}}" for c in CLASSES) + "   recall")
    for i, c in enumerate(CLASSES):
        rec = cm[i, i] / max(1, cm[i].sum())
        print(f"{c:>10} " + "".join(f"{cm[i, j]:>{w}d}" for j in range(4)) + f"   {rec:.3f}")

    # SAFETY: true reject predicted as a usable station (false accept)
    rej = y == "reject"
    fa = np.isin(pred[rej], list(USABLE)).mean()
    # and the reverse: usable truth sent to reject (over-cautious re-shoot)
    us = np.isin(y, list(USABLE))
    miss = (pred[us] == "reject").mean()
    print(f"\nSAFETY  false-accept (reject->usable): {fa:.3f}"
          f"   over-caution (usable->reject): {miss:.3f}")

    # per-usable-station precision (when we say X, is it really X)
    print("\nusable-station precision (when predicted, how often correct):")
    for c in USABLE:
        pc = pred == c
        prec = (y[pc] == c).mean() if pc.any() else float("nan")
        print(f"  {c:>9}: {prec:.3f}  (predicted {pc.sum()})")

    # confidence threshold: 'when unsure, re-shoot'. Defer to reject if the
    # winning vote share < tau. Trades over-caution for safety (fewer garbage
    # outputs). Shows the deployable tradeoff for element F.
    print("\n[confidence threshold]  predict usable only if vote share >= tau")
    print(f"{'tau':>6}{'acc':>8}{'false-accept':>15}{'over-caution':>15}")
    for tau in [0.0, 0.60, 0.72, 0.84, 0.92]:
        adj = np.where((share >= tau) | (pred == "reject"), pred, "reject")
        fa = np.isin(adj[rej], list(USABLE)).mean()
        mc = (adj[us] == "reject").mean()
        print(f"{tau:>6.2f}{(adj == y).mean():>8.3f}{fa:>15.3f}{mc:>15.3f}")


if __name__ == "__main__":
    main()
