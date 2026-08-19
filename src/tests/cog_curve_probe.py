"""COM forward-velocity CURVE derivative probe (paper-prep, exploratory).

The two adopted COG metrics read only two POINTS of the whole-body COM forward
velocity curve: its peak (COG Fwd Velo) and its value at peak knee height
(COG Velo @PKH). Modern pitching-mechanics practice reads the SHAPE of that
curve instead: how fast the body is still moving when the front foot lands,
how hard it then brakes (linear momentum -> rotation), when the peak occurs
relative to landing, and how far the COM keeps travelling afterwards.

TRUTH SOURCE (important). None of these exist as poi columns. An earlier draft
used our OWN Winter COM as both estimator and truth, which at az0/el0 is a
SELF-REFERENCE -- a side view images the forward axis in-plane, so image-x is
the 3D X axis and the test scored r2~1.0 while measuring nothing (the same trap
as the "Arm Slot CCC=1.000 synthetic identity" caveat in the handoff). Truth is
therefore OBP's OWN full-body model COM from full_sig/landmarks.csv
(`centerofmass_x`), read at OBP's own event times. Validated in
scratch/landmarks_com_validate.py: landmarks rows align frame-for-frame with the
c3d we load, and this signal reproduces the published poi columns at
r=0.9993 (max_cog_velo_x) / r=0.9995 (cog_velo_pkh).

So a score here carries the SAME burden the adopted COG metrics carry: our
Winter COM read from a 2D projection with OUR detected events, versus OBP's
independent model and events (that is why the adopted COG Fwd Velo scores 0.747,
not 1.0).

Reported per candidate: pooled r2 (the usual map number) PLUS the between-/
within-pitcher decomposition, because within_pitcher_agreement.py showed pooled
r2 can be dominated by between-pitcher variance (wrist speed 0.82 pooled -> 0.40
within), so a new candidate must be judged on both from the start.

EXPLORATORY: touches no adopted definition, no official table.
"""
import argparse
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))          # src/
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(HERE, "..", "stage3"))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))

import config                                          # noqa: E402
import metrics as M                                    # noqa: E402
import obp_project as O                                # noqa: E402
from master_angle_table import load_feet               # noqa: E402
from hss_elevation_test import project_cam             # noqa: E402

AZ = [0, 45, 90, 135, 180, 225, 270, 315]
EL = [0, 15, 30]

LM_ZIP = os.path.join(config.OBP_DATA_DIR, "full_sig", "landmarks.zip")
LM_COLS = ["session_pitch", "time", "centerofmass_x",
           "pkh_time", "fp_100_time", "BR_time"]


def sg_deriv(x, fps, order=2, deriv=1):
    """Savitzky-Golay derivative over a ~0.05 s window (same convention as the
    adopted cog_velo_at_pkh: a local polynomial fit, not a 2-frame diff)."""
    n = len(x)
    win = max(5, int(round(0.05 * fps)))
    win += (win % 2 == 0)
    win = min(win, n - (n % 2 == 0))
    if win <= order + 1:
        g = np.gradient(x) * fps
        return g if deriv == 1 else np.gradient(g) * fps
    return M._savgol(x, win, order, deriv=deriv, delta=1.0 / fps, mode="interp")


def curve_feats(fx, fps, fp, rel, pkh):
    """Candidate features from a FORWARD-POSITION series fx (metres). Sign is
    oriented so travel toward the plate is positive, making values comparable
    across camera azimuths (which mirror image-x) and handedness."""
    n = len(fx)
    fp = int(np.clip(fp, 1, n - 2)); rel = int(np.clip(rel, 1, n - 1))
    pkh = int(np.clip(pkh, 0, n - 1))
    sign = np.sign(fx[rel] - fx[0]) or 1.0
    fx = fx * sign
    v = sg_deriv(fx, fps)                       # forward velocity
    a = sg_deriv(v, fps)                        # forward acceleration
    lo, hi = min(fp, rel), max(fp, rel)
    win_a = a[lo:hi + 1]
    vseg = v[:rel + 1]
    pk_i = int(np.nanargmax(vseg)) if vseg.size and not np.all(np.isnan(vseg)) else np.nan
    return {
        "COG Velo @FP":      float(v[fp]),
        "COG Decel FP-BR":   float(v[fp] - v[rel]),
        "COG Peak Decel":    float(-np.nanmin(win_a)) if win_a.size else np.nan,
        "COG PeakT-FP (s)":  (pk_i - fp) / fps if np.isfinite(pk_i) else np.nan,
        "COG Disp FP-BR":    float(fx[rel] - fx[fp]),
        "COG Disp PKH-FP":   float(fx[fp] - fx[pkh]),
    }


CANDS = ["COG Velo @FP", "COG Decel FP-BR", "COG Peak Decel",
         "COG PeakT-FP (s)", "COG Disp FP-BR", "COG Disp PKH-FP"]


def est_feats(df, ctx):
    """2D estimator: Winter COM image-x (metrics.body_com), stature-normalised
    then scaled to metres by the subject's height -- the unit convention of the
    adopted COG metrics. Uses OUR detected events (deployment-honest)."""
    comx, _ = M.body_com(df, M.JOINTS)
    fx = comx / M.pixel_stature(df, M.JOINTS) * ctx["height_m"]
    return curve_feats(fx, ctx["fps"], ctx["fp"], ctx["rel"], ctx["pkh"])


def load_landmarks():
    """OBP full-model COM (x) + event frames per pitch, from full_sig."""
    with zipfile.ZipFile(LM_ZIP) as z:
        with z.open("landmarks.csv") as f:
            lm = pd.read_csv(f, usecols=LM_COLS)
    out, bad = {}, 0
    for sp, g in lm.groupby("session_pitch"):
        g = g.sort_values("time")
        t = g.time.to_numpy(float)
        fps = 1.0 / float(np.median(np.diff(t)))
        times = {c: float(g[c].iloc[0])
                 for c in ("fp_100_time", "BR_time", "pkh_time")}
        # Some pitches carry event times of 0 / NaN (force-plate event missing).
        # argmin(|t - 0|) would silently anchor them at frame 0 and manufacture
        # huge fake errors, so drop them rather than score against absent GT.
        if any((not np.isfinite(v)) or v <= 0 for v in times.values()):
            bad += 1
            continue
        out[sp] = dict(fx=g.centerofmass_x.to_numpy(float), fps=fps,
                       fp=int(np.argmin(np.abs(t - times["fp_100_time"]))),
                       rel=int(np.argmin(np.abs(t - times["BR_time"]))),
                       pkh=int(np.argmin(np.abs(t - times["pkh_time"]))))
    if bad:
        print(f"  ({bad} pitches excluded: OBP event time missing/zero)")
    return out


def r2(e, t):
    e = np.asarray(e, float); t = np.asarray(t, float)
    m = np.isfinite(e) & np.isfinite(t)
    if m.sum() < 3 or e[m].std() < 1e-12 or t[m].std() < 1e-12:
        return np.nan
    return float(np.corrcoef(e[m], t[m])[0, 1] ** 2)


def decompose(d):
    """pooled / between-pitcher / within-pitcher r2 + truth ICC for one cell."""
    d = d.dropna(subset=["est", "truth"])
    if len(d) < 10:
        return dict(pooled=np.nan, between=np.nan, within=np.nan, icc=np.nan)
    per = d.groupby("user").agg(est=("est", "mean"), truth=("truth", "mean"))
    cen_e = d.est - d.groupby("user").est.transform("mean")
    cen_t = d.truth - d.groupby("user").truth.transform("mean")
    cnt = d.groupby("user").size()
    keep = d.user.isin(cnt[cnt >= 2].index)
    grand = d.truth.mean()
    gt = d.groupby("user").truth
    ssb = (gt.count() * (gt.mean() - grand) ** 2).sum()
    ssw = ((d.truth - d.groupby("user").truth.transform("mean")) ** 2).sum()
    return dict(pooled=r2(d.est, d.truth), between=r2(per.est, per.truth),
                within=r2(cen_e[keep], cen_t[keep]),
                icc=ssb / (ssb + ssw) if (ssb + ssw) > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    print("loading OBP landmarks COM (independent truth) ...")
    LM = load_landmarks()
    print(f"  {len(LM)} pitches\n")

    recs, ev = [], []
    done = fail = 0
    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        sp = r.session_pitch
        if sp not in LM:
            fail += 1; continue
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = load_feet(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
            df0 = O.project_view(joints, azimuth_deg=0.0)
            rel = M.release_frame(df0, arm, fps, M.JOINTS)
            fp = M.foot_plant_frame(df0, lead, fps, M.JOINTS, rel)
            if rel <= fp + 1 or fp < 3:
                fail += 1; continue
            pkh = M.peak_knee_height_frame(df0, lead, fp, M.JOINTS)
            ctx = {"arm": arm, "lead": lead, "rel": rel, "fp": fp, "pkh": pkh,
                   "fps": fps, "height_m": float(r.session_height_m)}

            L = LM[sp]
            tru = curve_feats(L["fx"], L["fps"], L["fp"], L["rel"], L["pkh"])
            ev.append((rel - L["rel"], fp - L["fp"], pkh - L["pkh"]))
            # ORACLE-EVENT context: identical 2D COM measurement, but anchored on
            # OBP's own event frames. Comparing the two isolates how much of any
            # shortfall is EVENT-DETECTION error vs the COM/projection itself.
            ctx_or = dict(ctx, fp=L["fp"], rel=L["rel"], pkh=L["pkh"])

            for az in AZ:
                for el in EL:
                    df = project_cam(joints, az, el)
                    try:
                        es = est_feats(df, ctx)
                    except Exception:
                        es = {c: np.nan for c in CANDS}
                    try:
                        es_or = est_feats(df, ctx_or)
                    except Exception:
                        es_or = {c: np.nan for c in CANDS}
                    for c in CANDS:
                        recs.append((c, az, el, int(r.user), sp,
                                     es.get(c, np.nan), es_or.get(c, np.nan),
                                     tru.get(c, np.nan)))
            done += 1
        except Exception:
            fail += 1
        if done and done % 50 == 0:
            print(f"  ...{done} processed")

    print(f"processed {done} / failed {fail}\n")
    ev = np.array(ev, float)
    print("event agreement, OUR detection vs OBP event times (frames @360Hz, "
          "median / mean|.|):")
    for k, nm in enumerate(["release", "foot plant", "peak knee height"]):
        print(f"  {nm:<18} {np.median(ev[:, k]):+7.1f} / {np.mean(np.abs(ev[:, k])):6.1f}")

    df = pd.DataFrame(recs, columns=["cand", "az", "el", "user",
                                     "session_pitch", "est", "est_oracle",
                                     "truth"])
    rows = []
    for c in CANDS:
        g = df[df.cand == c]
        cells, cells_or = [], []
        for (az, el), d in g.groupby(["az", "el"]):
            m = decompose(d); m.update(az=az, el=el); cells.append(m)
            mo = decompose(d.rename(columns={"est": "_e", "est_oracle": "est"}))
            mo.update(az=az, el=el); cells_or.append(mo)
        cells = pd.DataFrame(cells); cells_or = pd.DataFrame(cells_or)
        anch = cells[(cells.az == 0) & (cells.el == 0)].iloc[0]
        anch_or = cells_or[(cells_or.az == 0) & (cells_or.el == 0)].iloc[0]
        best = cells.loc[cells.pooled.idxmax()] if cells.pooled.notna().any() else None
        rows.append({
            "candidate": c,
            "anchor_r2": anch.pooled,
            "anchor_within": anch.within,
            "oracle_r2": anch_or.pooled,          # OBP's own events
            "oracle_within": anch_or.within,
            "truth_ICC": anch.icc,
            "best_cell": f"az{int(best.az)}/el{int(best.el)}" if best is not None else "-",
            "best_r2": best.pooled if best is not None else np.nan,
        })

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 110)
    print("COM FORWARD-VELOCITY CURVE DERIVATIVES   truth = OBP full-model COM"
          f"   (n pitches = {done})")
    print("  anchor = az0/el0 (side).  adoption floor r2 = 0.50")
    print("  within = pitcher-centered r2 (tracks ONE pitcher's changes)")
    print("  oracle = same 2D measurement anchored on OBP's OWN event frames")
    print("           (anchor << oracle  =>  the wall is EVENT DETECTION,")
    print("            anchor ~= oracle  =>  the wall is the COM/projection)")
    print("=" * 110)
    fmt = {c: "{:.3f}".format for c in
           ["anchor_r2", "anchor_within", "oracle_r2", "oracle_within",
            "truth_ICC", "best_r2"]}
    print(out.to_string(index=False, formatters=fmt))

    dst = os.path.join(config.ROOT, "data", "outputs", "obp_validation",
                       "cog_curve_probe.csv")
    df.to_csv(dst.replace(".csv", "_pairs.csv"), index=False)
    out.to_csv(dst, index=False)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
