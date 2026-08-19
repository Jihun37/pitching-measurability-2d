"""
Diamond - HSS signature anchor: is t* an event at all? (2026-07-27)

Step 4 of the event-axis rebuild, and it is a VERIFICATION, not a formalisation.
Stride plateau and motion onset both looked like new events and both turned out not
to be, so hss_peak_overhead's t* gets the same test before anything is named.

The recipe is: chord-validity gate -> 90 ms medfilt -> persistence-checked
sustained-swing anchor t* -> windowed |max| over [t* - win_pre, t* + span/4] in
motion time. So t* might be a genuine instant, or just a way to place a search
window. Two questions decide it:

A  WHAT IS t* ?   its offset from every 3D event we can compute: foot plant, peak
   knee height, MER, release, and the pelvis / torso transverse rotation onsets
   (defined the same way motion onset is: |omega| over tau * peak, held `guard`
   frames). A small SD on any one of them means t* is that event plus a constant,
   i.e. a proxy, not a new event.

B  DOES THE ANCHOR MATTER ?   the same |max| taken over five windows:
     native     the recipe's own [t* - win_pre, t* + span/4]
     gt_fp_rel  [foot plant, release]
     gt_mer     [MER - 0.25 s, MER + 0.10 s]
     wide       [peak knee height, release + 0.10 s]
     oracle3d   centred on the TRUE 3D HSS peak (+-0.10 s)
   If the HSS value and its CCC are the same in all five, t* is a peak-search
   heuristic. If they diverge, it is a real anchor and gets formalised.

The two error sources are kept apart from the start, because HSS uses both:
   anchor error   t*_hat - t*_ref, plus window start/end error and gate failures
   peak mis-selection  did the 2D |max| land on the same extremum the 3D series
                       has? measured as the frame gap and the share of pitches
                       that pick a different local extremum

Cells = the 40 the graded map gives Hip-Shoulder Sep (all moderate; el 60-85).
Population = the frozen map's 394. Scoring = gate_map.score_cell.

Outputs: hss_anchor_probe_pitch.csv, hss_anchor_probe_cells.csv
Run:  cd src\\analysis
      python hss_anchor_probe.py --limit 20
      python hss_anchor_probe.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)

import config
import metrics as M
import obp_project as O
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from obp_gt_events import load_gt_events
from gate_map import score_cell
from mer_proxy_map import map_population

HSS_CELLS = [(0, 85), (15, 75), (15, 85), (30, 75), (30, 85), (45, 75), (45, 85),
             (60, 60), (60, 75), (60, 85), (75, 60), (75, 75), (75, 85), (90, 60),
             (90, 75), (90, 85), (105, 60), (105, 75), (105, 85), (120, 75),
             (120, 85), (135, 75), (135, 85), (150, 75), (150, 85), (165, 75),
             (165, 85), (180, 75), (180, 85), (195, 85), (210, 85), (225, 85),
             (240, 85), (255, 85), (270, 85), (285, 85), (300, 85), (315, 85),
             (330, 85), (345, 85)]
WINDOWS = ["native", "gt_fp_rel", "gt_mer", "wide", "oracle3d"]
EVENTS = ["fp", "pkh", "mer", "rel", "pelvis_onset", "torso_onset"]
ONSET_TAU, ONSET_GUARD = 0.20, 5
TRUTH_COL = "max_rotation_hip_shoulder_separation"


def transverse_angle(a, b):
    """Unwrapped image-free transverse orientation (deg) of the b->a line in the
    world XY plane -- the 3D counterpart of the shoulder/hip line the 2D recipe
    reads."""
    return np.degrees(np.unwrap(np.arctan2(a[1] - b[1], a[0] - b[0])))


def onset_frame(ang, fps, lo, hi, tau=ONSET_TAU, guard=ONSET_GUARD):
    """Rotation onset: first frame in [lo, hi] whose |omega| exceeds tau * the peak
    in that span and stays above it for `guard` frames. Same shape as the setup
    quiet-window rule, so the two are comparable."""
    w = np.abs(np.gradient(ang) * fps)
    lo, hi = int(max(1, lo)), int(min(len(w) - 1, hi))
    if hi - lo < guard + 2:
        return np.nan
    pk = np.nanmax(w[lo:hi + 1])
    if not np.isfinite(pk) or pk <= 0:
        return np.nan
    run = 0
    for t in range(lo, hi + 1):
        run = run + 1 if w[t] > tau * pk else 0
        if run >= guard:
            return t - guard + 1
    return np.nan


def win_max(sep, valid, lo, hi):
    """|max| of the filtered separation inside [lo, hi], honouring the recipe's
    own chord-validity mask. Returns (value, frame)."""
    lo, hi = int(max(0, lo)), int(min(len(sep) - 1, hi))
    if hi <= lo:
        return np.nan, np.nan
    s = np.abs(np.asarray(sep, float)[lo:hi + 1]).copy()
    v = np.asarray(valid, bool)[lo:hi + 1]
    s[~v] = np.nan
    if not np.any(np.isfinite(s)):
        return np.nan, np.nan
    k = int(np.nanargmax(s))
    return float(s[k]), lo + k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--floor", type=float, default=0.75)
    ap.add_argument("--strong", type=float, default=0.80)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    root = os.path.join(config.OBP_DATA_DIR, "c3d")
    gt = load_gt_events()
    pop = map_population()

    est = {(w, c): [] for w in WINDOWS for c in HSS_CELLS}
    pkf = {(w, c): [] for w in WINDOWS for c in HSS_CELLS}
    recs, tru, users = [], [], []
    done = fail = 0
    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        sp = r.session_pitch
        if sp not in pop or sp not in poi.index:
            continue
        g = gt.get(sp)
        if not g or not {"fp", "rel", "pkh", "mer"} <= set(g):
            fail += 1; continue
        path = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            j, fps = load_feet(path)
        except Exception:
            fail += 1; continue
        fp, rel, pkh, mer = (int(g["fp"]), int(g["rel"]), int(g["pkh"]),
                             int(g["mer"]))

        # ---- 3D reference series and rotation onsets -------------------------
        hip = transverse_angle(j["right_hip"], j["left_hip"])
        sho = transverse_angle(j["right_shoulder"], j["left_shoulder"])
        sep3 = sho - hip
        ev = {"fp": fp, "pkh": pkh, "mer": mer, "rel": rel,
              "pelvis_onset": onset_frame(hip, fps, pkh, rel),
              "torso_onset": onset_frame(sho, fps, pkh, rel)}
        lo3, hi3 = max(0, pkh), min(len(sep3) - 1, rel)
        seg = np.abs(sep3[lo3:hi3 + 1])
        gt_peak = lo3 + int(np.nanargmax(seg)) if seg.size else np.nan
        gt_hss = float(np.nanmax(seg)) if seg.size else np.nan

        tru.append(float(poi.loc[sp, TRUTH_COL]) if TRUTH_COL in poi.columns
                   else np.nan)
        users.append(r.user)

        for c in HSS_CELLS:
            try:
                df = project_cam(j, c[0], c[1])
                res = M.hss_peak_overhead(df, fps, M.JOINTS)
            except Exception:
                res = None
            if res is None:
                for w in WINDOWS:
                    est[(w, c)].append(np.nan); pkf[(w, c)].append(np.nan)
                recs.append(dict(sp=sp, user=r.user, az=c[0], el=c[1], ok=False))
                continue
            sepf, valid = res["sep_f"], res["valid"]
            t0, lo, hi = res["anchor_f"], res["window"][0], res["window"][1]
            spans = {
                "native": (lo, hi),
                "gt_fp_rel": (fp, rel),
                "gt_mer": (mer - int(0.25 * fps), mer + int(0.10 * fps)),
                "wide": (pkh, rel + int(0.10 * fps)),
                "oracle3d": (gt_peak - int(0.10 * fps), gt_peak + int(0.10 * fps))
                if np.isfinite(gt_peak) else (lo, hi)}
            for w in WINDOWS:
                v, f = win_max(sepf, valid, *spans[w])
                est[(w, c)].append(v); pkf[(w, c)].append(f)
            rec = dict(sp=sp, user=r.user, az=c[0], el=c[1], ok=True,
                       t_star=t0, peak_f=res["peak_f"], win_lo=lo, win_hi=hi,
                       gated_frac=res["gated_frac"], hss=res["hss"],
                       gt_peak=gt_peak, gt_hss=gt_hss, fps=fps,
                       peak_gap=res["peak_f"] - gt_peak
                       if np.isfinite(gt_peak) else np.nan)
            for k, v in ev.items():
                rec[f"d_{k}"] = (t0 - v) if np.isfinite(v) else np.nan
            recs.append(rec)
        done += 1
        if done % 25 == 0:
            print(f"  ...{done} processed")
    print(f"processed {done} / failed {fail}\n")

    P = pd.DataFrame(recs)
    p1 = os.path.join(config.OBP_VALIDATION_DIR, "hss_anchor_probe_pitch.csv")
    P.to_csv(p1, index=False, float_format="%.6g")

    codes = pd.Series(users).astype("category").cat.codes.to_numpy()
    t = np.asarray(tru, float)
    rows = []
    for w in WINDOWS:
        for c in HSS_CELLS:
            e = np.asarray(est[(w, c)], float)
            s = score_cell(e, t, codes, a.floor, a.strong)
            f = np.asarray(pkf[(w, c)], float)
            if s is None:
                continue
            rows.append(dict(window=w, az=c[0], el=c[1],
                             found=float(np.mean(np.isfinite(e))), **s))
    C = pd.DataFrame(rows)
    p2 = os.path.join(config.OBP_VALIDATION_DIR, "hss_anchor_probe_cells.csv")
    C.to_csv(p2, index=False, float_format="%.6g")

    pd.set_option("display.width", 200)
    ok = P[P.ok == True]
    print("=" * 96)
    print(f"HSS ANCHOR PROBE -- {P.sp.nunique()} pitches, {len(HSS_CELLS)} cells")
    print("=" * 96)
    print("\n[A] what is t* ?   t* minus each 3D event, in c3d frames @360Hz")
    hdr = (f"{'event':>14}{'median':>9}{'IQR':>8}{'SD':>8}{'within-SD':>11}"
           f"{'between-SD':>12}{'|d|<=3f':>9}{'n':>7}")
    print(hdr); print("-" * len(hdr))
    for k in EVENTS:
        d = ok[f"d_{k}"].dropna()
        if not len(d):
            continue
        w = ok.groupby("user")[f"d_{k}"].std().median()
        b = ok.groupby("user")[f"d_{k}"].median().std()
        print(f"{k:>14}{d.median():>9.1f}{d.quantile(.75)-d.quantile(.25):>8.1f}"
              f"{d.std():>8.1f}{w:>11.1f}{b:>12.1f}"
              f"{np.mean(np.abs(d - d.median()) <= 3):>9.2f}{len(d):>7}")
    print("\n  A tight SD on one row = t* is that event plus a constant (a PROXY).")
    print("  |d|<=3f is measured around each event's own median offset.")

    print("\n  per-cell view stability of t* (SD of t* across the 40 cells, "
          "within a pitch):")
    vs = ok.groupby("sp").t_star.std()
    print(f"    median {vs.median():.1f} f   p90 {vs.quantile(.9):.1f} f   "
          f"max {vs.max():.1f} f")

    print("\n[B] does the anchor matter ?   |max| over five windows")
    hdr2 = (f"{'window':>11}{'cells':>7}{'strong':>8}{'moder':>7}{'CCC med':>9}"
            f"{'CCC max':>9}{'r2 med':>9}{'found':>8}")
    print(hdr2); print("-" * len(hdr2))
    for w in WINDOWS:
        s = C[C.window == w]
        if not len(s):
            continue
        print(f"{w:>11}{len(s):>7}{int((s.grade=='strong').sum()):>8}"
              f"{int((s.grade=='moderate').sum()):>7}{s.ccc.median():>9.3f}"
              f"{s.ccc.max():>9.3f}{s.r2.median():>9.3f}{s.found.min():>8.3f}")
    base = C[C.window == "native"].set_index(["az", "el"]).ccc
    print("\n  vs the native t* window, median dCCC over cells:")
    for w in WINDOWS:
        if w == "native":
            continue
        s = C[C.window == w].set_index(["az", "el"]).ccc.reindex(base.index)
        print(f"    {w:<11} {(s - base).median():+.4f}  "
              f"(min {(s - base).min():+.4f}, max {(s - base).max():+.4f})")

    print("\n[C] internal peak mis-selection: 2D |max| frame vs the 3D peak")
    gap = ok.peak_gap.dropna()
    print(f"  frame gap: median {gap.median():+.1f}  IQR "
          f"{gap.quantile(.75)-gap.quantile(.25):.1f}  "
          f"|gap|<=3f {np.mean(np.abs(gap)<=3):.2f}  |gap|<=10f "
          f"{np.mean(np.abs(gap)<=10):.2f}")
    print(f"  gated fraction (chord-validity): median "
          f"{ok.gated_frac.median():.3f}  p90 {ok.gated_frac.quantile(.9):.3f}")
    by = ok.groupby("el").agg(gap_med=("peak_gap", "median"),
                              gap_absmed=("peak_gap", lambda s: s.abs().median()),
                              within3=("peak_gap", lambda s: np.mean(np.abs(s) <= 3)),
                              gated=("gated_frac", "median"))
    print("\n  by elevation:")
    print(by.round(3).to_string())
    print(f"\nsaved -> {p1}\nsaved -> {p2}")


if __name__ == "__main__":
    main()
