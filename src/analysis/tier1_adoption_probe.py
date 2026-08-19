"""Formal adoption probe for the GT-event rescreen's Tier-1 candidates (2026-07-24).

rejected_gt_full_sweep screened 28 rejected/untested columns and left six that clear
the usable floor with a real zone and are not colinear shadows. A screen is not an
adoption: this runs each candidate through the SAME gates every adopted metric passed
(LOCO calibration + absolute accuracy + within-pitcher + independence), at a
PRE-SPECIFIED anchor fixed before scoring, using GT events.

Anchors are the zone CENTER (highest 8-neighbour mean among cells >= 0.6), not the raw
argmax -- a robust cell, not a lucky spike. As with every ADOPTED_VIEW anchor, the
viewpoint choice still shares the full-sample selection caveat; the decisive,
non-leaking evidence is the LOCO out-of-fold calibration gain.

Gates (matching the precedent set by wrist / stride / HSS / stride_angle):
  LOCO      leave-one-pitcher-out; a CALIBRATE pass = MAE drops AND pitcher_bias_sd
            shrinks vs raw, reaching CCC >= ~0.80. A DIRECT pass = raw CCC already high.
            A proxy FAILS: MAE barely moves and pitcher_bias_sd does not shrink.
  within    within-pitcher r2 (pitcher-centered) vs truth ICC -- is there signal to
            track, and does the estimate track it?
  independ  partial: r2 of est vs the truth RESIDUAL after regressing out the adopted
            metrics' truths. High => new information, not a re-expression of the map.

Verdict is advisory; the human makes the adoption call. RAW clean-projection: a pass
here is necessary, not sufficient, for deployment (real phone-vs-mocap pending).

Run:  conda activate diamond; cd src\\analysis; python tier1_adoption_probe.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config, metrics as M, obp_project as O
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from obp_gt_events import load_gt_events

# candidate -> (observable, event, truth column, pre-specified (az, el))
CANDS = {
    "elbow_flexion_mer":            ("elbow_flex",  "mer", "elbow_flexion_mer",            (330, 60)),
    "torso_lateral_tilt_mer":       ("trunk_lean",  "mer", "torso_lateral_tilt_mer",       (90, 30)),
    "glove_shoulder_abduction_mer": ("abd_glove",   "mer", "glove_shoulder_abduction_mer", (75, 15)),
    "torso_rotation_br":            ("shoulder_line","rel", "torso_rotation_br",           (90, 85)),
    "torso_rotation_mer":           ("shoulder_line","mer", "torso_rotation_mer",          (90, 85)),
    "torso_lateral_tilt_br":        ("trunk_lean",  "rel", "torso_lateral_tilt_br",        (105, 45)),
}

# 2026-07-27 (user): the full GT re-screen with NO redundancy filter left 12 columns
# outside the adopted list that clear 0.60; two are spikes (<=3 usable cells of 168,
# narrower than our own real-video viewpoint error) and are excluded. The remaining
# ten are probed here. Anchors are zone CENTERS computed from
# rejected_gt_full_grid.csv BEFORE any scoring, same rule as the 2026-07-24 batch.
CANDS.update({
    "arm_slot":                    ("forearm_slot", "rel", "arm_slot",            (135, 0)),
    "torso_anterior_tilt_mer2":    ("trunk_lean",   "mer", "torso_anterior_tilt_mer", (210, 15)),
    "torso_rotation_mer2":         ("shoulder_line", "mer", "torso_rotation_mer",  (135, 75)),
    "max_elbow_flexion":           ("elbow_flex_max", "rel", "max_elbow_flexion",  (300, 45)),
    "knee_ext_fp_to_br":           ("knee_ext_fp_to_br", "rel",
                                    "lead_knee_extension_from_fp_to_br",          (30, 30)),
    "knee_ext_velo_max":           ("knee_ext_velo_max", "rel",
                                    "lead_knee_extension_angular_velo_max",       (165, 30)),
    "torso_anterior_tilt_fp":      ("trunk_lean",   "fp",  "torso_anterior_tilt_fp", (285, 0)),
    "shoulder_hz_abd_fp":          ("hz_abd_throw", "fp",  "shoulder_horizontal_abduction_fp", (255, 85)),
    "shoulder_abd_fp":             ("abd_throw",    "fp",  "shoulder_abduction_fp", (165, 0)),
    "torso_lateral_tilt_br2":      ("trunk_lean",   "rel", "torso_lateral_tilt_br", (105, 45)),
})

# Regressors for the independence test = the truths of the ADOPTED metrics.
# Derived from the live adoption list so a de-adopted metric cannot linger here
# (the previous hardcoded list still carried lead_knee_extension_angular_velo_br,
# de-adopted 2026-07-24). `arm_slot` is kept on top of that list as the only
# available stand-in for the adopted Arm Slot metric, whose truth is 3D-direct and
# has no poi column -- which means the arm_slot CANDIDATE is regressed against a
# proxy of itself and its independence figure reads ~0 BY CONSTRUCTION. Read that
# cell as "same axis as the adopted arm slot", not as a failed test.
from angle_map_2d import adopted_rows as _ar, gt_only_rows as _gr
ADOPTED_TRUTH = sorted({t for _, _, t in _ar() + _gr() if isinstance(t, str)}
                       | {"arm_slot"})
MODELS = ("ratio", "offset", "linear")


def observe_at(df, okey, frame, fps, obs=None, win=None):
    """Scalar 2D observable at `frame`. The three ADOPTED observables delegate to
    the metrics.py primitives (single source of truth); trunk_lean stays local
    because torso lateral tilt is not adopted -- it is only probed here.

    Everything else is taken from rejected_gt_full_sweep.observables (`obs`), the
    SAME code the screen ran, so a probe number can be compared cell-for-cell with
    the screen instead of drifting from it. `win` = (fp, rel) for the window-max
    style observables."""
    # NOTE: the metrics.py branches below take precedence on purpose -- the six
    # 2026-07-24 candidates must keep scoring through the adopted primitives.
    if okey == "elbow_flex":
        return M.elbow_flexion_2d(df, "right", frame, M.JOINTS)
    if okey == "abd_glove":
        return M.shoulder_abduction_2d(df, "glove", frame, M.JOINTS)
    if okey == "shoulder_line":
        return M.torso_rotation_2d(df, frame, M.JOINTS)
    if okey == "trunk_lean":
        def xy(k):
            return (df[f"{M.JOINTS[k]}_x"].to_numpy(float),
                    df[f"{M.JOINTS[k]}_y"].to_numpy(float))
        lsx, lsy = xy("l_sh"); rsx, rsy = xy("r_sh")
        lhx, lhy = xy("l_hip"); rhx, rhy = xy("r_hip")
        msx, msy = (lsx + rsx) / 2, (lsy + rsy) / 2
        mhx, mhy = (lhx + rhx) / 2, (lhy + rhy) / 2
        return float(np.degrees(np.arctan2(msx - mhx, -(msy - mhy)))[int(frame)])
    if obs is not None and okey in obs:
        o = obs[okey]
        if callable(o):
            return float(o(*win)) if win else np.nan
        return float(o[int(frame)])
    raise KeyError(okey)


def sweep_observables(df, fps, lead):
    """rejected_gt_full_sweep.observables + the lead-knee series its main() adds,
    so the probe reads exactly what the screen read."""
    from rejected_gt_full_sweep import observables
    o, _ = observables(df, fps)
    def xy(k):
        return df[f"{k}_x"].to_numpy(float), df[f"{k}_y"].to_numpy(float)
    hx, hy = xy(f"{lead}_hip"); kx, ky = xy(f"{lead}_knee"); ax_, ay = xy(f"{lead}_ankle")
    kang = M._angle(hx, hy, kx, ky, ax_, ay)
    kvel = np.gradient(kang) * fps
    o["knee_ext_velo_max"] = (lambda lo, hi: float(np.nanmax(kvel[max(0, lo):hi + 1]))
                              if hi > lo else np.nan)
    o["knee_ext_velo_at"] = kvel
    o["knee_ext_fp_to_br"] = (lambda lo, hi: float(kang[hi] - kang[max(0, lo)])
                              if 0 <= lo < len(kang) and 0 <= hi < len(kang) else np.nan)
    return o


def fit_apply(model, e_tr, t_tr, e_te):
    if model == "ratio":
        return t_tr.mean() / e_tr.mean() * e_te
    if model == "offset":
        return e_te + (t_tr - e_tr).mean()
    a, b = np.polyfit(e_tr, t_tr, 1)
    return a * e_te + b


def score(pred, t, users):
    d = pred - t
    r = np.corrcoef(pred, t)[0, 1]
    cov = ((pred - pred.mean()) * (t - t.mean())).mean()
    ccc = 2 * cov / (pred.var() + t.var() + (pred.mean() - t.mean()) ** 2)
    per = pd.Series(d).groupby(pd.Series(users)).mean()
    return dict(bias=float(d.mean()), mae=float(np.abs(d).mean()),
                ccc=float(ccc), r2=float(r * r),
                pitcher_bias_sd=float(per.std(ddof=0)))


def r2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or x[m].std() < 1e-9 or y[m].std() < 1e-9:
        return np.nan
    return np.corrcoef(x[m], y[m])[0, 1] ** 2


def main():
    gt = load_gt_events()
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")

    # cache each candidate's est/truth/user, computed once over all pitches
    data = {c: dict(e=[], t=[], u=[], ado=[]) for c in CANDS}
    root = os.path.join(config.OBP_DATA_DIR, "c3d")
    done = 0
    for r in md.itertuples(index=False):
        sp = r.session_pitch
        g = gt.get(sp)
        if sp not in poi.index or not g:
            continue
        path = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            continue
        try:
            joints, fps = load_feet(path)
        except Exception:
            continue
        arm = O.detect_throwing_arm(joints, fps)
        lead = "left" if arm == "right" else "right"
        proj_cache, obs_cache = {}, {}
        win = ((int(g["fp"]), int(g["rel"])) if {"fp", "rel"} <= set(g) else None)
        for c, (okey, ekey, col, (az, el)) in CANDS.items():
            if ekey not in g or col not in poi.columns:
                continue
            f = int(g[ekey])
            if (az, el) not in proj_cache:
                try:
                    proj_cache[(az, el)] = project_cam(joints, az, el)
                except Exception:
                    proj_cache[(az, el)] = None
            df = proj_cache[(az, el)]
            if df is None or not (0 <= f < len(df)):
                continue
            if (az, el) not in obs_cache:
                try:
                    obs_cache[(az, el)] = sweep_observables(df, fps, lead)
                except Exception:
                    obs_cache[(az, el)] = None
            try:
                v = float(observe_at(df, okey, f, fps,
                                     obs=obs_cache[(az, el)], win=win))
            except Exception:
                v = np.nan
            tv = poi.loc[sp, col]
            data[c]["e"].append(v); data[c]["t"].append(tv)
            data[c]["u"].append(int(r.user))
            data[c]["ado"].append([poi.loc[sp, a] if a in poi.columns else np.nan
                                   for a in ADOPTED_TRUTH])
        done += 1
        if done % 100 == 0:
            print(f"  ...{done}")
    print(f"processed {done}\n")

    verdicts = []
    for c, (okey, ekey, col, (az, el)) in CANDS.items():
        e = np.asarray(data[c]["e"], float); t = np.asarray(data[c]["t"], float)
        u = np.asarray(data[c]["u"]); ado = np.asarray(data[c]["ado"], float)
        m = np.isfinite(e) & np.isfinite(t)
        e, t, u, ado = e[m], t[m], u[m], ado[m]

        # LOCO
        preds = {mo: np.full(len(e), np.nan) for mo in MODELS}
        for uu in np.unique(u):
            te = u == uu; tr = ~te
            if tr.sum() < 5:
                continue
            for mo in MODELS:
                preds[mo][te] = fit_apply(mo, e[tr], t[tr], e[te])
        raw = score(e, t, u)
        cal = {mo: score(preds[mo][np.isfinite(preds[mo])],
                         t[np.isfinite(preds[mo])], u[np.isfinite(preds[mo])])
               for mo in MODELS}
        best_mo = max(MODELS, key=lambda mo: cal[mo]["ccc"])
        best = cal[best_mo]

        # within / between pitcher
        dfp = pd.DataFrame({"e": e, "t": t, "u": u})
        gm = dfp.groupby("u")
        bw = r2(gm.e.mean().to_numpy(), gm.t.mean().to_numpy())
        ec = dfp.e - dfp.groupby("u").e.transform("mean")
        tc = dfp.t - dfp.groupby("u").t.transform("mean")
        wi = r2(ec.to_numpy(), tc.to_numpy())
        grand = dfp.t.mean()
        ssb = (gm.t.count() * (gm.t.mean() - grand) ** 2).sum()
        ssw = ((dfp.t - dfp.groupby("u").t.transform("mean")) ** 2).sum()
        icc = ssb / (ssb + ssw) if (ssb + ssw) > 0 else np.nan

        # independence: est vs truth residual after regressing out adopted truths
        A = ado.copy()
        colok = [j for j in range(A.shape[1]) if np.isfinite(A[:, j]).mean() > 0.8]
        A = A[:, colok]
        good = np.isfinite(A).all(axis=1)
        if good.sum() > 20:
            Ag = np.column_stack([np.ones(good.sum()), A[good]])
            coef, *_ = np.linalg.lstsq(Ag, t[good], rcond=None)
            resid = t[good] - Ag @ coef
            indep = r2(e[good], resid)
        else:
            indep = np.nan

        # verdict
        calibrate = (best["ccc"] >= 0.80 and best["mae"] < raw["mae"] and
                     best["pitcher_bias_sd"] <= raw["pitcher_bias_sd"] + 1e-9)
        direct = raw["ccc"] >= 0.80
        verdict = ("DIRECT" if direct else
                   "CALIBRATE" if calibrate else
                   "MARGINAL" if best["ccc"] >= 0.70 else "FAIL")
        verdicts.append(dict(candidate=c, view=f"{az}/{el}", event=ekey,
                             raw_r2=raw["r2"], raw_ccc=raw["ccc"],
                             loco_ccc=best["ccc"], loco_model=best_mo,
                             raw_mae=raw["mae"], loco_mae=best["mae"],
                             pbsd_raw=raw["pitcher_bias_sd"],
                             pbsd_cal=best["pitcher_bias_sd"],
                             within=wi, between=bw, icc=icc,
                             indep_r2=indep, verdict=verdict))

    v = pd.DataFrame(verdicts)
    out = os.path.join(config.OBP_VALIDATION_DIR, "tier1_adoption_probe.csv")
    v.to_csv(out, index=False)

    print("=" * 116)
    print("TIER-1 ADOPTION PROBE  (GT events, pre-specified zone-center anchors)")
    print("=" * 116)
    print(f"{'candidate':<30}{'view':>7}{'raw r2':>8}{'raw CCC':>9}"
          f"{'LOCO CCC':>10}{'mdl':>7}{'MAE raw>cal':>14}{'pbSD raw>cal':>15}"
          f"{'within':>8}{'ICC':>6}{'indep':>7}  verdict")
    print("-" * 116)
    for r in v.itertuples(index=False):
        print(f"{r.candidate:<30}{r.view:>7}{r.raw_r2:>8.3f}{r.raw_ccc:>9.3f}"
              f"{r.loco_ccc:>10.3f}{r.loco_model:>7}"
              f"{f'{r.raw_mae:.2f}>{r.loco_mae:.2f}':>14}"
              f"{f'{r.pbsd_raw:.2f}>{r.pbsd_cal:.2f}':>15}"
              f"{r.within:>8.3f}{r.icc:>6.2f}{r.indep_r2:>7.3f}  {r.verdict}")
    print("\ncolumns: raw = uncalibrated clean projection; LOCO CCC = best out-of-fold "
          "calibrated; within = within-pitcher r2; ICC = truth between-pitcher fraction;")
    print("indep = est vs truth residual after removing the adopted metrics "
          "(new-information test).")
    print("\nverdict key: DIRECT (raw CCC>=.80) / CALIBRATE (LOCO CCC>=.80 + MAE & "
          "pitcher-bias both drop) / MARGINAL (LOCO CCC>=.70) / FAIL.")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
