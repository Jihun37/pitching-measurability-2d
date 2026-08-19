"""
Diamond - Can a LEARNED multi-feature model beat the 0.61 single-angle ceiling
for hip-shoulder separation, using ONLY MediaPipe-available 2D landmarks?

The geometric single-angle estimator saturates at r2~0.61 (overhead). Here we
extract several deployable 2D features (shoulder/hip line angles, their rotation
range over the throw, width ratio, arm-cocking geometry, sep at fp/release) from
the overhead projection and fit a grouped-CV regressor. If r2 jumps, the single
angle was leaving information on the table; if it stalls near 0.61, that is the
true single-view ceiling for shoulder+hip 2D landmarks.

Leakage guard: 5-fold CV grouped by session_pitch. Features use only landmarks
MediaPipe provides (shoulders, hips, elbows, wrists) -> deployable.

Run:  cd src\analysis
      python hss_learned_regress.py --limit 250
      python hss_learned_regress.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import config
import obp_project as O
import metrics as M
from hss_elevation_test import project_cam, hss_sep_series

TRUTH = "max_rotation_hip_shoulder_separation"
ELEVATIONS = [75, 90]


def sg(a):
    x = np.asarray(a, float)
    if np.isnan(x).any():
        i = np.arange(len(x)); g = ~np.isnan(x)
        if g.sum() < 5:
            return x
        x = np.interp(i, i[g], x[g])
    w = min(11, len(x) if len(x) % 2 else len(x) - 1)
    return savgol_filter(x, w, 2) if w >= 5 else x


def line_angle(dx, dy):
    return np.degrees(np.arctan2(dy, dx))


def features(df, J, arm, fp, rel):
    """Deployable 2D features (shoulder/hip/elbow/wrist only)."""
    armsign = 1.0 if arm == "right" else -1.0
    lsx, lsy = M._xy(df, "l_sh", J); rsx, rsy = M._xy(df, "r_sh", J)
    lhx, lhy = M._xy(df, "l_hip", J); rhx, rhy = M._xy(df, "r_hip", J)
    wkey = "r_wr" if arm == "right" else "l_wr"
    ekey = "r_el" if arm == "right" else "l_el"
    wx, wy = M._xy(df, wkey, J); ex, ey = M._xy(df, ekey, J)

    sep = sg(hss_sep_series(df, J)) * armsign
    sh_ang = np.unwrap(np.radians(line_angle(rsx - lsx, rsy - lsy)))
    hip_ang = np.unwrap(np.radians(line_angle(rhx - lhx, rhy - lhy)))
    shw = np.hypot(rsx - lsx, rsy - lsy)
    hipw = np.hypot(rhx - lhx, rhy - lhy)
    lo, hi = min(fp, rel), max(fp, rel) + 1
    msx, msy = (lsx + rsx) / 2, (lsy + rsy) / 2

    def rng(a):
        s = a[lo:hi]
        return float(np.nanmax(s) - np.nanmin(s)) if len(s) else np.nan

    return {
        "sep_fp": float(sep[fp]),
        "sep_rel": float(sep[rel]),
        "sep_wmax": float(np.nanmax(sep[lo:hi])) if hi > lo else np.nan,
        "sep_wrange": rng(sep),
        "sh_rot_range": np.degrees(rng(sh_ang)),
        "hip_rot_range": np.degrees(rng(hip_ang)),
        "widthratio_fp": float(hipw[fp] / (shw[fp] + 1e-6)),
        "widthratio_rel": float(hipw[rel] / (shw[rel] + 1e-6)),
        # arm-cocking direction relative to shoulder midpoint at fp
        "arm_ang_fp": float(line_angle(wx[fp] - msx[fp], wy[fp] - msy[fp]) * armsign),
        "elbow_ang_fp": float(line_angle(ex[fp] - msx[fp], ey[fp] - msy[fp]) * armsign),
    }


FEATS = ["sep_fp", "sep_rel", "sep_wmax", "sep_wrange", "sh_rot_range",
         "hip_rot_range", "widthratio_fp", "widthratio_rel", "arm_ang_fp",
         "elbow_ang_fp"]


def group_kfold(groups, k=5, seed=0):
    uniq = np.array(sorted(pd.unique(groups)))
    np.random.default_rng(seed).shuffle(uniq)
    folds = np.array_split(uniq, k)
    g = np.asarray(groups)
    for i in range(k):
        tg = set(folds[i].tolist())
        te = np.array([j for j, v in enumerate(g) if v in tg])
        tr = np.array([j for j, v in enumerate(g) if v not in tg])
        yield tr, te


def knn_reg(Xtr, ytr, Xte, k=15):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    out = np.empty(len(B))
    for i, b in enumerate(B):
        d = np.sum((A - b) ** 2, axis=1)
        out[i] = ytr[np.argsort(d)[:k]].mean()
    return out


def ridge(Xtr, ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    A1 = np.c_[np.ones(len(A)), A]; B1 = np.c_[np.ones(len(B)), B]
    P = A1.T @ A1 + lam * np.eye(A1.shape[1]); P[0, 0] -= lam
    w = np.linalg.solve(P, A1.T @ ytr)
    return B1 @ w


def r2_corr(pred, truth):
    d = pd.DataFrame({"a": pred, "b": truth}).dropna()
    r = d["a"].corr(d["b"])
    return r * r if pd.notna(r) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    J = M.JOINTS

    rows = {el: [] for el in ELEVATIONS}
    done = fail = 0
    for r in md.itertuples(index=False):
        if a.limit and done >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = O.load_c3d_joints(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
            df0 = O.project_view(joints, azimuth_deg=0, elevation_deg=0)
            rel0 = M.release_frame(df0, arm, fps, J, view="side")
            fp0 = M.foot_plant_frame(df0, lead, fps, J, rel0)
        except Exception:
            fail += 1; continue
        for el in ELEVATIONS:
            try:
                df = project_cam(joints, 0, el)
                f = features(df, J, arm, fp0, rel0)
                f["session_pitch"] = r.session_pitch
                rows[el].append(f)
            except Exception:
                pass
        done += 1
        if done % 100 == 0:
            print(f"  ...{done} pitches")
    print(f"processed {done} / {fail} missing\n")

    print("=" * 60)
    print(f"HSS learned regression (grouped 5-fold CV)  vs {TRUTH}")
    print("baseline single-angle geometric r2 ~ 0.61 (el=90)")
    print("-" * 60)
    print(f"{'elevation':<12}{'geom(sep_wmax)':>16}{'kNN':>8}{'ridge':>8}")
    for el in ELEVATIONS:
        d = pd.DataFrame(rows[el]).dropna(subset=FEATS).reset_index(drop=True)
        m = d.merge(poi[["session_pitch", TRUTH]], on="session_pitch", how="inner")
        X = m[FEATS].to_numpy(float); y = m[TRUTH].to_numpy(float)
        g = m["session_pitch"].to_numpy()
        pk = np.empty(len(m)); pr = np.empty(len(m))
        for tr, te in group_kfold(g, 5):
            pk[te] = knn_reg(X[tr], y[tr], X[te])
            pr[te] = ridge(X[tr], y[tr], X[te])
        geom = r2_corr(m["sep_wmax"], y)
        print(f"el={el:<9}{geom:>16.2f}{r2_corr(pk, y):>8.2f}{r2_corr(pr, y):>8.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
