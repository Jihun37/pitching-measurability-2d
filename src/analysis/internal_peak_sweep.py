"""Diamond - task 5 of the event rebuild: INTERNAL PEAK MIS-SELECTION.

Seven of the map's rows do not read an event at all -- they read an argmax inside a
window. For those the temporal question is not "how many frames is our anchor off"
but "did the 2D argmax land on the SAME extremum the 3D signal has". A projection
can leave the window untouched and still hand the estimator a different hump.

    Wrist Speed [O]                       COG Fwd Velo [O]
    Pelvis Rot Velo [O]                   Hip-Shoulder Sep [O]
    max_pelvis_rotational_velo            max_elbow_flexion
    lead_knee_extension_angular_velo_max                         (266 gate cells)

TRUTH = OBP'S OWN SERIES, NOT A PROXY. The earlier HSS attempt
(hss_anchor_probe.py [C]) scored its frame gap against a marker-line transverse
angle we built ourselves, so a gap could have been the proxy's fault -- the result
was uninterpretable and the handoff said so. obp_gt_peaks.py fixes that: it
recovers the frame at which each `max_*` poi column attains its value by matching
the scalar back onto its own full_sig channel, exactly, on all 411 pitches.

Two reference classes, kept apart on purpose, because they answer different
questions and must never be averaged together:

  poi-exact (5 rows)  the frame OBP's truth peaks at. Answers: does our argmax
                      agree with the ground truth's instant?
  def-3D    (2 rows)  the argmax of the 3D counterpart of OUR OWN definition
                      (3D wrist speed for Wrist Speed -- which IS its adopted
                      truth, so this row is exact too; a 3D Winter COM for COG Fwd
                      Velo, which reproduces max_cog_velo_x at only r2 0.71 and is
                      therefore a DEFINITION reference, not OBP's instant).

Per cell, three estimator variants scored with the map's own gate (gate_map.
score_cell), so the cost of mis-selection is read in CCC, not just in frames:

    argmax   the native estimator -- must reproduce gate_map.csv (control)
    basin    the 2D local peak in the basin containing the GT frame: the argmax we
             WOULD have taken had we not been pulled to a different hump. The
             difference argmax->basin is mis-selection and nothing else.
    at_gt    the 2D value read exactly at the GT frame. Adds instant error on top
             of selection, so at_gt - basin isolates how steep the 2D series is
             around the truth's instant.

A cell where basin >> argmax is losing its grade to peak mis-selection (fixable by
a better peak rule). A cell where basin ~ argmax is losing it to projection
distortion instead, and no selection rule will help.

Outputs: internal_peak_sweep_cells.csv, internal_peak_sweep_pitch.csv.gz
Run:  conda activate diamond
      cd src\\analysis
      python internal_peak_sweep.py --limit 10      (smoke -- NOT a preview:
                                                     LOCO CCC is optimistic near
                                                     the n>=30 floor)
      python internal_peak_sweep.py
"""
import os, sys, argparse, time
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)

import config
import metrics as M
import obp_project as O
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from obp_gt_events import load_gt_events
from obp_gt_peaks import load_gt_peaks
from gate_map import score_cell
from mer_proxy_map import map_population

AZ = list(range(0, 360, 15))
EL = [0, 15, 30, 45, 60, 75, 85]
VARIANTS = ("argmax", "basin", "at_gt", "win_mir", "win_relk")
# MIR sits a nearly fixed lag AFTER release: mir - rel = 12 frames @360 Hz, SD 1.8
# (n=411) -- the same shape as the adopted MER lag (rel - 11, SD 1.1,
# angle_map_2d.MER_LAG_S). So win_relk repeats the win_mir window with a DETECTED
# release plus that lag instead of the GT landmark, which is what decides whether
# a window finding is deployable or another oracle.
MIR_LAG_F360 = 12
# win_mir applies to the three SCREENED rows only (the ones whose window is the
# generic [fp, rel] of rejected_gt_full_sweep). It is a DIAGNOSTIC, not a proposed
# definition: the sweep found that lead_knee_extension_angular_velo_max's truth
# instant falls outside [fp, rel] on 61 % of pitches, so the row has a window
# problem rather than a peak-selection problem, and this measures what the window
# is worth. Every poi peak frame lies strictly before MIR (100 %, n=411), so
# [fp, MIR] is the smallest landmark-bounded window that can contain the truth.
WIDE_ROWS = {"max_pelvis_rotational_velo", "max_elbow_flexion",
             "lead_knee_extension_angular_velo_max"}
J = M.JOINTS

# label -> (source, truth column or ("3d", fn), GT-peak reference)
ROWS = [
    ("Wrist Speed [O]",                     "adopted",  ("3d", "wrist"),  "wrist3d"),
    ("COG Fwd Velo [O]",                    "adopted",  "max_cog_velo_x", "com3d"),
    ("Pelvis Rot Velo [O]",                 "adopted",  "max_pelvis_rotational_velo",
                                                        "max_pelvis_rotational_velo"),
    ("Hip-Shoulder Sep [O]",                "adopted",  "max_rotation_hip_shoulder_separation",
                                                        "max_rotation_hip_shoulder_separation"),
    ("max_pelvis_rotational_velo",          "screened", "max_pelvis_rotational_velo",
                                                        "max_pelvis_rotational_velo"),
    ("max_elbow_flexion",                   "screened", "max_elbow_flexion",
                                                        "max_elbow_flexion"),
    ("lead_knee_extension_angular_velo_max", "screened", "lead_knee_extension_angular_velo_max",
                                                        "lead_knee_extension_angular_velo_max"),
]
# which rows carry OBP's own instant vs the 3D counterpart of our definition
REF_CLASS = {"wrist3d": "def-3D(=truth)", "com3d": "def-3D(proxy)"}


def com3d(j):
    """3D Winter whole-body COM, the direct counterpart of metrics.body_com -- same
    segment table, applied to the c3d joints instead of a projection. Local by the
    same convention master_angle_table uses for its t3_* truths.

    NOTE it is NOT OBP's COM: its peak forward speed reproduces max_cog_velo_x at
    only r2 0.71 (n=150 probe), so it references OUR definition, not the truth."""
    sho = (j["left_shoulder"] + j["right_shoulder"]) / 2
    hip = (j["left_hip"] + j["right_hip"]) / 2
    segs = [(0.497, sho + 0.50 * (hip - sho))]
    for s in ("left", "right"):
        segs.append((0.028, j[f"{s}_shoulder"] + 0.436 * (j[f"{s}_elbow"] - j[f"{s}_shoulder"])))
        segs.append((0.022, j[f"{s}_elbow"] + 0.682 * (j[f"{s}_wrist"] - j[f"{s}_elbow"])))
        segs.append((0.100, j[f"{s}_hip"] + 0.433 * (j[f"{s}_knee"] - j[f"{s}_hip"])))
        segs.append((0.0465, j[f"{s}_knee"] + 0.433 * (j[f"{s}_ankle"] - j[f"{s}_knee"])))
        segs.append((0.0145, j[f"{s}_ankle"]))
    if "head" in j:
        segs.append((0.081, j["head"]))
    w = sum(m for m, _ in segs)
    return sum(m * p for m, p in segs) / w


def t3_wrist_series(j, fps):
    """3D wrist speed on master_angle_table.t3_wrist's definition. _speed() prepends
    a zero so its index i is the step (i-1 -> i); np.diff does not, so the 3D argmax
    is shifted by +1 to land on the same convention."""
    def f(arm):
        w = j[f"{arm}_wrist"]
        return np.concatenate([[0.0],
                               np.linalg.norm(np.diff(w, axis=1), axis=0) * fps])
    return f


def basin_peak(s, lo, hi, g, w=3):
    """Hill-climb from the GT frame to the top of the hump it sits on. The +-w
    neighbourhood makes the climb ignore single-frame noise instead of stalling on
    it; w=3 at 360 Hz is 8 ms, far below any real kinematic feature."""
    if hi <= lo:
        return None
    i = int(np.clip(g, lo, hi))
    for _ in range(hi - lo + 2):
        a, b = max(lo, i - w), min(hi, i + w)
        seg = s[a:b + 1]
        if not np.isfinite(seg).any():
            return None
        k = a + int(np.nanargmax(seg))
        if k == i:
            return i
        i = k
    return i


def series_for(label, df, ctx, j):
    """(series to maximise, lo, hi, value->metric-unit fn) for one row at one view.

    Every expression here is the estimator's own, restated so the ARGMAX FRAME is
    available (the estimators return only the value). The argmax variant is checked
    against gate_map.csv cell by cell, which is what catches any drift."""
    fps, rel, fp = ctx["fps"], ctx["rel"], ctx["fp"]
    n = len(df)
    px = M.pixel_stature(df, J)
    h = ctx["height_m"]

    if label == "Wrist Speed [O]":                      # est_wrist_abs
        a = ctx["arm"]
        wx = df[f"{a}_wrist_x"].to_numpy(float)
        wy = df[f"{a}_wrist_y"].to_numpy(float)
        s = M._speed(wx, wy, fps)
        return s, 0, n - 1, (lambda v: v / px * h)

    if label == "COG Fwd Velo [O]":                     # metrics.cog_fwd_velo
        comx, _ = M.body_com(df, J)
        s = np.abs(np.gradient(comx)) * fps / px
        return s, 0, int(rel), (lambda v: v * h)

    if label == "Pelvis Rot Velo [O]":                  # est_pelvis_rot
        rhx = df["right_hip_x"].to_numpy(float); rhy = df["right_hip_y"].to_numpy(float)
        lhx = df["left_hip_x"].to_numpy(float);  lhy = df["left_hip_y"].to_numpy(float)
        ang = np.unwrap(np.arctan2(rhy - lhy, rhx - lhx))
        win = max(5, int(round(0.05 * fps))); win += (win % 2 == 0)
        win = min(win, n - (n % 2 == 0))
        vel = (np.degrees(savgol_filter(ang, win, 3, deriv=1, delta=1.0 / fps,
                                        mode="interp"))
               if win > 3 else np.degrees(np.gradient(ang) * fps))
        lo = max(0, rel - int(0.40 * fps)); hi = min(n - 1, rel + int(0.05 * fps))
        return np.abs(vel), lo, hi, (lambda v: v)

    if label == "Hip-Shoulder Sep [O]":                 # metrics.hss_peak_overhead
        res = M.hss_peak_overhead(df, fps, J)
        if res is None:
            return None
        sep_f = res["sep_f"]
        # The recipe maximises the COIL direction -sgn*sep_f, and sgn is not in the
        # returned dict. Do NOT infer it from sign(sep_f[peak_f]): that only holds
        # when the extremum is on the far side of zero, and where it is not the
        # inferred sign flips and the series is maximised the wrong way (caught by
        # the gate_map control at |dCCC| 0.0165). Re-derive it with the recipe's own
        # function on the recipe's own filtered series -- deterministic, same args.
        _, sgn = M.hss_transition(sep_f, fps, 0.25)
        lo, hi = res["window"]
        return -sgn * sep_f, int(lo), int(hi), (lambda v: abs(v))

    # ---- screened rows: rejected_gt_full_sweep observables, window [fp, rel] ----
    def xy(k):
        return (df[f"{J[k]}_x"].to_numpy(float), df[f"{J[k]}_y"].to_numpy(float))

    if label == "max_pelvis_rotational_velo":           # hip_line_velo_max
        lhx, lhy = xy("l_hip"); rhx, rhy = xy("r_hip")
        hip_line = np.degrees(np.unwrap(np.arctan2(rhy - lhy, rhx - lhx)))
        return (np.abs(np.gradient(hip_line) * fps), max(0, int(fp)), int(rel),
                (lambda v: v))

    if label == "max_elbow_flexion":                    # elbow_flex_max
        sx, sy = xy("r_sh"); ex, ey = xy("r_el"); wx, wy = xy("r_wr")
        return (180.0 - M._angle(sx, sy, ex, ey, wx, wy), max(0, int(fp)),
                int(rel), (lambda v: v))

    if label == "lead_knee_extension_angular_velo_max":  # knee_ext_velo_max
        # window END carries the adopted derived boundary (+12 f @360Hz). Read it
        # from rejected_gt_full_sweep rather than hardcoding, so this cannot drift
        # from the map again -- it already did once, between this file being written
        # and the window being widened the same day.
        from rejected_gt_full_sweep import WINDOW_END_OFFSET_F360
        dend = int(round(WINDOW_END_OFFSET_F360.get("knee_ext_velo_max", 0)
                         * fps / 360.0))
        ld = ctx["lead"]
        hx, hy = (df[f"{ld}_hip_x"].to_numpy(float), df[f"{ld}_hip_y"].to_numpy(float))
        kx, ky = (df[f"{ld}_knee_x"].to_numpy(float), df[f"{ld}_knee_y"].to_numpy(float))
        ax, ay = (df[f"{ld}_ankle_x"].to_numpy(float), df[f"{ld}_ankle_y"].to_numpy(float))
        kang = M._angle(hx, hy, kx, ky, ax, ay)
        return (np.gradient(kang) * fps, max(0, int(fp)), int(rel) + dend,
                (lambda v: v))

    raise KeyError(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--floor", type=float, default=0.75)
    ap.add_argument("--strong", type=float, default=0.80)
    ap.add_argument("--no-dump", action="store_true")
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    root = os.path.join(config.OBP_DATA_DIR, "c3d")
    gt = load_gt_events()
    gp = load_gt_peaks()
    pop = map_population()
    print(f"population {len(pop)} (frozen map), {len(AZ)}x{len(EL)} cells, "
          f"{len(ROWS)} rows")

    est = {(ri, az, el, v): [] for ri in range(len(ROWS))
           for az in AZ for el in EL for v in VARIANTS}
    tru = {ri: [] for ri in range(len(ROWS))}
    users, recs = [], []
    done = fail = 0
    t0 = time.time()

    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and done >= a.limit:
            break
        sp = r.session_pitch
        if sp not in pop or sp not in poi.index:
            continue
        g = gt.get(sp)
        if not g or not {"fp", "rel", "pkh"} <= set(g):
            fail += 1; continue
        path = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            j, fps = load_feet(path)
            arm = O.detect_throwing_arm(j, fps)
        except Exception:
            fail += 1; continue
        lead = "left" if arm == "right" else "right"
        rel, fp, pkh = int(g["rel"]), int(g["fp"]), int(g["pkh"])
        mir = int(g["mir"]) if "mir" in g else None
        if rel <= fp + 1 or fp < 3:
            fail += 1; continue
        ctx = {"arm": arm, "lead": lead, "rel": rel, "fp": fp, "pkh": pkh,
               "fps": fps, "height_m": float(r.session_height_m)}

        # ---- GT peak frames, per reference class -----------------------------
        ws3 = t3_wrist_series(j, fps)(arm)
        c3 = com3d(j)
        vcom = np.abs(np.gradient(c3[0])) * fps
        gtf = dict(gp.get(sp, {}))
        gtf["wrist3d"] = int(np.nanargmax(ws3))
        gtf["com3d"] = int(np.nanargmax(vcom[:rel + 1]))

        # ---- truths ----------------------------------------------------------
        for ri, (label, src, truth, ref) in enumerate(ROWS):
            if isinstance(truth, tuple):          # 3D-direct
                tv = float(np.nanmax(ws3))
            else:
                tv = float(poi.loc[sp, truth]) if truth in poi.columns else np.nan
            tru[ri].append(tv)
        users.append(r.user)

        for az in AZ:
            for el in EL:
                df = project_cam(j, az, el)
                for ri, (label, src, truth, ref) in enumerate(ROWS):
                    gf = gtf.get(ref, None)
                    try:
                        spec = series_for(label, df, ctx, j)
                    except Exception:
                        spec = None
                    if spec is None or gf is None:
                        for v in VARIANTS:
                            est[(ri, az, el, v)].append(np.nan)
                        continue
                    s, lo, hi, scale = spec
                    lo, hi = int(max(0, lo)), int(min(len(s) - 1, hi))
                    seg = s[lo:hi + 1]
                    if hi <= lo or not np.isfinite(seg).any():
                        for v in VARIANTS:
                            est[(ri, az, el, v)].append(np.nan)
                        continue
                    k_hat = lo + int(np.nanargmax(seg))
                    k_b = basin_peak(s, lo, hi, gf)
                    inw = bool(lo <= gf <= hi)
                    vals = {
                        "argmax": scale(float(s[k_hat])),
                        "basin": scale(float(s[k_b])) if k_b is not None else np.nan,
                        "at_gt": scale(float(s[gf])) if inw else np.nan,
                        "win_mir": np.nan, "win_relk": np.nan}
                    if label in WIDE_ROWS:
                        ends = {"win_mir": mir,
                                "win_relk": rel + int(round(
                                    MIR_LAG_F360 * fps / 360.0))}
                        for vn, e in ends.items():
                            if e is None:
                                continue
                            h2 = int(min(len(s) - 1, e))
                            if h2 > lo and np.isfinite(s[lo:h2 + 1]).any():
                                vals[vn] = scale(float(np.nanmax(s[lo:h2 + 1])))
                    for v in VARIANTS:
                        est[(ri, az, el, v)].append(vals[v])
                    recs.append((label, az, el, sp, r.user, k_hat, k_b, gf,
                                 lo, hi, inw, bool(k_b is not None and k_b == k_hat)))
        done += 1
        if done % 25 == 0:
            el_s = time.time() - t0
            print(f"  ...{done} processed  ({el_s / done:.2f} s/pitch, "
                  f"eta {(len(pop) - done) * el_s / done / 60:.0f} min)")
    print(f"processed {done} / failed {fail}   ({(time.time()-t0)/60:.1f} min)\n")

    P = pd.DataFrame(recs, columns=["metric", "az", "el", "session_pitch", "user",
                                    "k_hat", "k_basin", "k_gt", "lo", "hi",
                                    "gt_in_window", "same_extremum"])
    P["gap"] = P.k_hat - P.k_gt
    P["gap_basin"] = P.k_basin - P.k_gt
    if not a.no_dump:
        pp = os.path.join(config.OBP_VALIDATION_DIR, "internal_peak_sweep_pitch.csv.gz")
        P.to_csv(pp, index=False, compression="gzip")
        print(f"saved -> {pp}  ({len(P):,} rows)")

    # ---- per-cell scoring -------------------------------------------------
    codes = pd.Series(users).astype("category").cat.codes.to_numpy()
    diag = P.groupby(["metric", "az", "el"]).agg(
        gap_med=("gap", "median"),
        gap_absmed=("gap", lambda s: s.abs().median()),
        gap_iqr=("gap", lambda s: s.quantile(.75) - s.quantile(.25)),
        within3=("gap", lambda s: float(np.mean(np.abs(s) <= 3))),
        same_extremum=("same_extremum", "mean"),
        gt_in_window=("gt_in_window", "mean"))

    rows_out = []
    for ri, (label, src, truth, ref) in enumerate(ROWS):
        t = np.asarray(tru[ri], float)
        for az in AZ:
            for el in EL:
                d = diag.loc[(label, az, el)] if (label, az, el) in diag.index else None
                for v in VARIANTS:
                    e = np.asarray(est[(ri, az, el, v)], float)
                    s = score_cell(e, t, codes, a.floor, a.strong)
                    if s is None:
                        continue
                    rec = dict(metric=label, source=src, ref=ref,
                               ref_class=REF_CLASS.get(ref, "poi-exact"),
                               az=az, el=el, variant=v,
                               found=float(np.mean(np.isfinite(e))), **s)
                    if d is not None:
                        rec.update({k: float(d[k]) for k in diag.columns})
                    rows_out.append(rec)
    C = pd.DataFrame(rows_out)
    cp = os.path.join(config.OBP_VALIDATION_DIR, "internal_peak_sweep_cells.csv")
    C.to_csv(cp, index=False, float_format="%.6g")
    print(f"saved -> {cp}\n")
    report(C, P, a)


def report(C, P, a):
    pd.set_option("display.width", 220)
    if C.empty:
        # score_cell needs n >= 30, so a small --limit produces no scored cell at
        # all. Report the frame diagnostics only, and say so -- a smoke run is not
        # a preview of the numbers (handoff trap 4).
        print("no cell reached n>=30 -- SMOKE RUN, no scoring. Frame gaps only:")
        d = P.groupby("metric").agg(
            n=("gap", "size"), gap_med=("gap", "median"),
            same_extremum=("same_extremum", "mean"),
            gt_in_window=("gt_in_window", "mean"))
        print(d.round(3).to_string())
        return
    nat = C[C.variant == "argmax"].set_index(["metric", "az", "el"])

    # ---- control: does the native variant reproduce the frozen map? ----------
    print("=" * 104)
    print("[CONTROL]  argmax variant vs gate_map.csv  (must be ~0 -- otherwise the "
          "restated series drifted)")
    print("=" * 104)
    gm_p = os.path.join(config.OBP_VALIDATION_DIR, "gate_map.csv")
    if os.path.exists(gm_p):
        gm = pd.read_csv(gm_p).set_index(["metric", "az", "el"])
        mg = nat.join(gm[["ccc", "r2"]], rsuffix="_map", how="inner")
        print(f"{'metric':<40}{'cells':>7}{'max |dCCC|':>12}{'max |dr2|':>11}")
        print("-" * 70)
        for m, gsub in mg.groupby(level=0):
            print(f"{m:<40}{len(gsub):>7}"
                  f"{(gsub.ccc - gsub.ccc_map).abs().max():>12.6f}"
                  f"{(gsub.r2 - gsub.r2_map).abs().max():>11.6f}")
    else:
        print("  gate_map.csv not found -- control skipped")

    # ---- the answer ----------------------------------------------------------
    print("\n" + "=" * 104)
    print("[A] DOES THE 2D ARGMAX FIND THE SAME EXTREMUM?   at each row's GATE "
          "cells (grade != limited)")
    print("=" * 104)
    hdr = (f"{'metric':<40}{'ref':>15}{'cells':>7}{'gap med':>9}{'|gap| med':>11}"
           f"{'<=3f':>7}{'same pk':>9}{'gt in win':>11}")
    print(hdr); print("-" * len(hdr))
    for (m, rc), g in nat.groupby(["metric", "ref_class"]):
        ok = g[g.grade != "limited"]
        if not len(ok):
            ok = g
        print(f"{m:<40}{rc:>15}{len(ok):>7}{ok.gap_med.median():>9.1f}"
              f"{ok.gap_absmed.median():>11.1f}{ok.within3.median():>7.2f}"
              f"{ok.same_extremum.median():>9.3f}{ok.gt_in_window.median():>11.3f}")
    print("\n  same pk = share of pitches whose argmax is the peak of the hump the")
    print("  GT frame sits on. gt in win = share whose GT frame is inside the "
          "estimator's own window at all.")

    # ---- what mis-selection costs -------------------------------------------
    print("\n" + "=" * 104)
    print("[B] WHAT DOES IT COST?   CCC of the three variants, and the grade count")
    print("=" * 104)
    hdr2 = (f"{'metric':<40}{'variant':>8}{'strong':>8}{'moder':>7}{'CCC med':>9}"
            f"{'CCC max':>9}{'dCCC vs argmax':>16}")
    print(hdr2); print("-" * len(hdr2))
    for m, g in C.groupby("metric"):
        base = g[g.variant == "argmax"].set_index(["az", "el"]).ccc
        first = True
        for v in VARIANTS:
            s = g[g.variant == v]
            if s.empty:                      # win_mir is screened-rows only
                continue
            d = s.set_index(["az", "el"]).ccc.reindex(base.index) - base
            dtxt = "-" if v == "argmax" else f"{d.median():+.4f}"
            print(f"{m if first else '':<40}{v:>8}"
                  f"{int((s.grade == 'strong').sum()):>8}"
                  f"{int((s.grade == 'moderate').sum()):>7}{s.ccc.median():>9.3f}"
                  f"{s.ccc.max():>9.3f}{dtxt:>16}")
            first = False

    print("\n  basin - argmax  = the cost of MIS-SELECTION alone (same window, same")
    print("  series, only the hump differs). at_gt - basin = how steep the 2D series")
    print("  is at the truth's instant. win_mir = the same argmax over [fp, MIR]")
    print("  instead of [fp, rel] -- a WINDOW diagnostic, screened rows only.")

    print("\n" + "=" * 104)
    print("[B2] WINDOW vs SELECTION   is the truth's instant even inside the "
          "estimator's window?")
    print("=" * 104)
    hdr3 = (f"{'metric':<40}{'gt in win':>11}{'strong':>8}{'[fp,MIR]':>10}"
            f"{'[fp,rel+12]':>13}{'dCCC mir':>10}{'dCCC relk':>11}")
    print(hdr3); print("-" * len(hdr3))
    for m, g in C.groupby("metric"):
        base = g[g.variant == "argmax"]
        wm, wk = g[g.variant == "win_mir"], g[g.variant == "win_relk"]
        if wm.empty:
            continue
        b = base.set_index(["az", "el"]).ccc
        dm = (wm.set_index(["az", "el"]).ccc - b).median()
        dk = (wk.set_index(["az", "el"]).ccc - b).median() if not wk.empty else np.nan
        print(f"{m:<40}{base.gt_in_window.median():>11.3f}"
              f"{int((base.grade == 'strong').sum()):>8}"
              f"{int((wm.grade == 'strong').sum()):>10}"
              f"{int((wk.grade == 'strong').sum()):>13}{dm:>+10.4f}{dk:>+11.4f}")
    print("\n  A row with a low `gt in win` is not mis-SELECTING a peak -- its")
    print("  window does not contain the truth's instant at all. [fp,rel+12] is the")
    print("  DEPLOYABLE form of the same window (no MIR detector); where it matches")
    print("  [fp,MIR] the fix needs no new event. Diagnostic only -- changing a")
    print("  screened row's window is a map decision, not this sweep's.")

    # ---- where it bites ------------------------------------------------------
    print("\n" + "=" * 104)
    print("[C] WHERE IT BITES   cells whose grade IMPROVES when mis-selection is "
          "removed (basin)")
    print("=" * 104)
    ORD = {"limited": 0, "moderate": 1, "strong": 2}
    b = C[C.variant == "basin"].set_index(["metric", "az", "el"])
    j = nat.join(b[["ccc", "grade"]], rsuffix="_b", how="inner")
    j["up"] = j.grade_b.map(ORD) - j.grade.map(ORD)
    n_up = int((j.up > 0).sum())
    print(f"  {n_up} of {len(j)} cells gain a grade; "
          f"{int((j.up < 0).sum())} lose one")
    if n_up:
        top = j[j.up > 0].sort_values("ccc_b", ascending=False).head(15)
        print(f"\n{'metric':<40}{'cell':>10}{'grade':>10}{'->':>4}{'grade_b':>9}"
              f"{'CCC':>8}{'->':>4}{'CCC_b':>8}{'same pk':>9}")
        for (m, az, el), r in top.iterrows():
            print(f"{m:<40}{f'{az}/{el}':>10}{r.grade:>10}{'->':>4}{r.grade_b:>9}"
                  f"{r.ccc:>8.3f}{'->':>4}{r.ccc_b:>8.3f}{r.same_extremum:>9.3f}")

    # ---- elevation structure --------------------------------------------------
    print("\n" + "=" * 104)
    print("[D] BY ELEVATION   same-extremum share (all 168 cells, not just gate)")
    print("=" * 104)
    piv = nat.reset_index().pivot_table(index="metric", columns="el",
                                        values="same_extremum", aggfunc="median")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
