"""Re-screen EVERY rejected / never-directly-tested kinematic column with GT events,
swept over the full azimuth x elevation grid.

The user's hypothesis, and the reason this is worth a full sweep rather than the
az-only el=0 pass in rejected_gt_rescreen.py:

  1. WRONG EVENT. Most rejected candidates were anchored at OUR foot plant, the
     noisiest event we detect (SD ~15 frames vs the OBP landmark). A rejection there
     can be a detector artefact, not a projection limit. Precedent: the COM-curve
     candidates jumped 0.11-0.47 -> 0.88-0.99 under OBP events (ledger 2-B).
  2. NO EVENT AT ALL. Six columns are anchored at MER (max external rotation) and
     MER is a ROTATION -- 2D pose cannot locate it, so those columns were never even
     attempted (ledger 3, "미검정, 사실상 벽"). GT ships MER_time, so for the first
     time we can read the 2D quantity at the correct instant and ask the projection
     question honestly.
  3. WRONG VIEW. Transverse quantities are degenerate on the ground but revive
     overhead (HSS 0.06 -> 0.63, pelvis rot -> 0.80). Every candidate is therefore
     swept over elevation too, not just azimuth.

For each candidate this finds the best (az, el, event) cell and flags any that
clears the usable floor (0.60) -- i.e. was rejected for event/view reasons, not
geometry. RAW clean-projection r2, pooled, n up to 411. This is a screen: a hit
here means "worth a proper adoption probe (LOCO, absacc, within-pitcher)", not an
adoption.

Run:  conda activate diamond; cd src\\analysis; python rejected_gt_full_sweep.py
      python rejected_gt_full_sweep.py --limit 40     (smoke)
      python rejected_gt_full_sweep.py --dump --detected   (deployment layer)
"""
import os, sys, argparse
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config, metrics as M, obp_project as O
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from obp_gt_events import load_gt_events

AZ = list(range(0, 360, 15))
EL = [0, 15, 30, 45, 60, 75, 85]
FLOOR = 0.60
J = M.JOINTS

# candidate column -> (observable key, event key). event in {fp, fp10, pkh, mer,
# mir, rel}. observable computed once per (df) as a per-frame series, sampled at
# the event frame -- except *_velo which are window maxima.
CANDS = {
    # --- MER-anchored: NEVER directly tested (2D could not locate MER) ---
    "torso_anterior_tilt_mer":       ("trunk_lean",  "mer"),
    "torso_lateral_tilt_mer":        ("trunk_lean",  "mer"),
    "torso_rotation_mer":            ("shoulder_line", "mer"),
    "elbow_flexion_mer":             ("elbow_flex",  "mer"),
    "glove_shoulder_abduction_mer":  ("abd_glove",   "mer"),
    "max_shoulder_external_rotation":("shoulder_line", "mer"),  # ~ occurs at MER
    # --- FP-anchored, rejected on OUR foot plant ---
    "elbow_flexion_fp":              ("elbow_flex",  "fp"),
    "torso_anterior_tilt_fp":        ("trunk_lean",  "fp"),
    "torso_lateral_tilt_fp":         ("trunk_lean",  "fp"),
    "torso_rotation_fp":             ("shoulder_line", "fp"),
    "pelvis_rotation_fp":            ("hip_line",    "fp"),
    "pelvis_lateral_tilt_fp":        ("hip_line",    "fp"),
    "pelvis_anterior_tilt_fp":       ("hip_lean",    "fp"),
    "rotation_hip_shoulder_separation_fp": ("hss_angle", "fp"),
    "shoulder_abduction_fp":         ("abd_throw",   "fp"),
    "glove_shoulder_abduction_fp":   ("abd_glove",   "fp"),
    "shoulder_horizontal_abduction_fp":       ("hz_abd_throw", "fp"),
    "glove_shoulder_horizontal_abduction_fp": ("hz_abd_glove", "fp"),
    "shoulder_external_rotation_fp": ("shoulder_line", "fp"),
    # glove-side external rotation: rotation about the humeral long axis, which has
    # no first-order projection signature (the same reason MER cannot be located).
    # Screened for completeness against the best available proxy (trunk transverse
    # orientation at foot plant), so the coverage audit has zero un-attempted columns.
    "glove_shoulder_external_rotation_fp": ("shoulder_line", "fp"),
    # --- BR-anchored transverse/lateral (adopted anterior tilt is the sibling) ---
    "torso_lateral_tilt_br":         ("trunk_lean",  "rel"),
    "torso_rotation_br":             ("shoulder_line", "rel"),
    # --- velo family (window maxima; event pair implied) ---
    "max_elbow_extension_velo":      ("elbow_ext_velo_max", "rel"),
    "max_elbow_flexion":             ("elbow_flex_max",     "rel"),
    "lead_knee_extension_angular_velo_max": ("knee_ext_velo_max", "rel"),
    "lead_knee_extension_angular_velo_fp":  ("knee_ext_velo_at",  "fp"),
    "max_shoulder_internal_rotational_velo": ("shoulder_line_velo_max", "rel"),
    "max_torso_rotational_velo":     ("shoulder_line_velo_max", "rel"),
    "max_pelvis_rotational_velo":    ("hip_line_velo_max",      "rel"),
    # --- 2026-07-27: the columns that had NO sweep of their own. Their verdicts
    # were inherited from a sibling column or from a definition decision, which is
    # exactly the failure mode this table exists to prevent. Screened here so every
    # non-kinetic column carries its OWN number.
    "torso_rotation_min":            ("shoulder_line_min",   "rel"),
    "max_shoulder_horizontal_abduction": ("hz_abd_throw_max", "rel"),
    "lead_knee_extension_angular_velo_br":  ("knee_ext_velo_at", "rel"),
    "lead_knee_extension_from_fp_to_br":    ("knee_ext_fp_to_br", "rel"),
    "timing_peak_torso_to_peak_pelvis_rot_velo": ("torso_pelvis_timing", "rel"),
    # pronation is rotation about the forearm's own axis: no projection signature
    # exists, so this is screened against the best available proxy (the forearm's
    # image-plane orientation) purely so the column is not left un-attempted.
    "elbow_pronation_fp":            ("forearm_angle",       "fp"),
    # The OBP arm_slot column is a SEPARATE metric from our adopted arm slot, not a
    # duplicate of it: ours is shoulder->wrist scored on 3D-direct truth, this one
    # is forearm-based. Screened with its own matching observable so no column is
    # left un-attempted (2026-07-27, user).
    "arm_slot":                      ("forearm_slot",        "rel"),
}

# ---------------------------------------------------------------------------
# Window-END extension for individual window-max observables, in frames @360 Hz,
# added to the `rel` end of the default [fp, rel] search window.
#
# This is a DERIVED BOUNDARY, NOT AN EVENT DETECTOR and not a new landmark. It is
# an offset from the release anchor the estimator already uses, exactly like the
# adopted MER proxy (angle_map_2d.MER_LAG_S = rel - 11 f). Nothing here detects
# anything; do not describe it as a detector or add it to the event inventory.
#
# knee_ext_velo_max, +12 f (adopted 2026-07-27, user decision):
#   lead_knee_extension_angular_velo_max attains its truth value OUTSIDE [fp, rel]
#   on 61 % of pitches -- there was no peak in the window to find, so the row was
#   failing on its WINDOW, not on projection or peak selection
#   (analysis/internal_peak_sweep.py; docs/legacy_pre_dedup/EVENT_SYSTEM_HANDOFF_2026-07-27.md 6b).
#   Every one of its 411 truth instants lies strictly before MIR, and MIR - BR is
#   12 f at 360 Hz with SD 1.8, so +12 is the smallest release-anchored bound that
#   covers the distribution. Scored both ways before adoption: the GT-landmark
#   window [fp, MIR] and this proxy [fp, rel+12] give the SAME 25 strong cells, and
#   the two other window-max rows move by exactly 0.0000 CCC, so the gain is
#   specific to the row with the window defect rather than a blanket widening.
WINDOW_END_OFFSET_F360 = {"knee_ext_velo_max": 12}


def observables(df, fps):
    """All per-frame image-plane observables, computed once per projected clip.
    Window-max observables return a callable(lo, hi)->scalar instead of a series."""
    def xy(k):
        return df[f"{J[k]}_x"].to_numpy(float), df[f"{J[k]}_y"].to_numpy(float)
    lsx, lsy = xy("l_sh"); rsx, rsy = xy("r_sh")
    lhx, lhy = xy("l_hip"); rhx, rhy = xy("r_hip")
    msx, msy = (lsx + rsx) / 2, (lsy + rsy) / 2
    mhx, mhy = (lhx + rhx) / 2, (lhy + rhy) / 2

    o = {}
    # trunk lean: image angle of hip->shoulder vs vertical. Sagittal or lateral
    # depending on az -- the sweep decides which truth it matches.
    o["trunk_lean"] = np.degrees(np.arctan2(msx - mhx, -(msy - mhy)))
    o["hip_lean"]   = np.degrees(np.arctan2(rhx - lhx, -(rhy - lhy)))  # pelvis obliquity proxy
    # transverse rotation proxies: image-plane orientation of the shoulder / hip
    # line. Degenerate at el=0 (line ~horizontal for all rotations), revives as the
    # camera rises -- the same mechanism that revived HSS / pelvis rot.
    o["shoulder_line"] = np.degrees(np.unwrap(np.arctan2(rsy - lsy, rsx - lsx)))
    o["hip_line"]      = np.degrees(np.unwrap(np.arctan2(rhy - lhy, rhx - lhx)))
    o["hss_angle"]     = o["shoulder_line"] - o["hip_line"]

    for arm, tag in (("r", "throw"), ("l", "glove")):
        sx, sy = xy(f"{arm}_sh"); ex, ey = xy(f"{arm}_el"); wx, wy = xy(f"{arm}_wr")
        hx, hy = xy(f"{arm}_hip")
        if tag == "throw":
            o["elbow_flex"] = 180.0 - M._angle(sx, sy, ex, ey, wx, wy)
            o["elbow_incl"] = M._angle(sx, sy, ex, ey, wx, wy)
        o[f"abd_{tag}"] = M._angle(ex, ey, sx, sy, hx, hy)          # elbow-sh-hip
        # horizontal abduction proxy: upper-arm image angle vs the shoulder line
        ua = np.degrees(np.arctan2(ey - sy, ex - sx))
        o[f"hz_abd_{tag}"] = ua - o["shoulder_line"]
        if tag == "throw":
            # forearm image orientation: the only forearm quantity a 3-point chain
            # carries, used as the pronation proxy (see CANDS note).
            o["forearm_angle"] = np.degrees(np.unwrap(np.arctan2(wy - ey, wx - ex)))
            # FOREARM-based arm slot: the same "angle from vertical" convention as
            # the adopted metrics.arm_slot, but hinged at the ELBOW instead of the
            # shoulder -- because the OBP arm_slot column is the forearm variant.
            # Definition-matched on purpose: CLAUDE.md forbids scoring our
            # shoulder->wrist arm slot against this column.
            o["forearm_slot"] = np.degrees(np.arctan2(np.abs(wx - ex), ey - wy))

    # window-max observables
    def wmax(series):
        def f(lo, hi):
            seg = series[max(0, lo):hi + 1]
            seg = seg[np.isfinite(seg)]
            return float(np.nanmax(seg)) if seg.size else np.nan
        return f
    def wmin(series):
        def f(lo, hi):
            seg = series[max(0, lo):hi + 1]
            seg = seg[np.isfinite(seg)]
            return float(np.nanmin(seg)) if seg.size else np.nan
        return f

    vel = lambda s: np.gradient(s) * fps
    o["shoulder_line_velo_max"] = wmax(np.abs(vel(o["shoulder_line"])))
    o["hip_line_velo_max"]      = wmax(np.abs(vel(o["hip_line"])))
    o["elbow_ext_velo_max"]     = wmax(vel(o["elbow_incl"]))
    o["elbow_flex_max"]         = wmax(o["elbow_flex"])
    o["shoulder_line_min"]      = wmin(o["shoulder_line"])
    o["hz_abd_throw_max"]       = wmax(o["hz_abd_throw"])

    # sequencing interval: peak torso rate minus peak pelvis rate, in seconds.
    sv, hv = np.abs(vel(o["shoulder_line"])), np.abs(vel(o["hip_line"]))
    def timing(lo, hi):
        lo = max(0, lo)
        a, b = sv[lo:hi + 1], hv[lo:hi + 1]
        if not (np.isfinite(a).any() and np.isfinite(b).any()):
            return np.nan
        return float(np.nanargmax(a) - np.nanargmax(b)) / fps
    o["torso_pelvis_timing"] = timing
    return o, (msx, msy, mhx, mhy)


def all_observables(df, fps, lead):
    """observables() PLUS the three lead-knee entries the main loop used to build
    inline. Single entry point so a consumer (event_tolerance_full, deploy_map)
    cannot drift from the definitions the map is built on."""
    o, extra = observables(df, fps)

    def kxy(k):
        return df[f"{k}_x"].to_numpy(float), df[f"{k}_y"].to_numpy(float)
    hx, hy = kxy(f"{lead}_hip"); kx, ky = kxy(f"{lead}_knee")
    axx, ay = kxy(f"{lead}_ankle")
    kang = M._angle(hx, hy, kx, ky, axx, ay)
    kvel = np.gradient(kang) * fps
    o["knee_ext_velo_max"] = (lambda lo, hi:
                              float(np.nanmax(kvel[max(0, lo):hi + 1]))
                              if hi > lo else np.nan)
    o["knee_ext_velo_at"] = kvel
    o["knee_ext_fp_to_br"] = (lambda lo, hi:
                              float(kang[hi] - kang[max(0, lo)])
                              if 0 <= lo < len(kang) and 0 <= hi < len(kang)
                              else np.nan)
    return o, extra


def sample(o, col, ev, fps, d_start=0, d_end=0):
    """Read one candidate column from the observables at the given events, with the
    two window ends shifted INDEPENDENTLY (d_start on fp, d_end on rel). Point
    observables ignore d_end and take d_start on their own anchor event.

    This is the map's own reading rule, factored out of the main loop so the
    tolerance sweep perturbs exactly what the map reads."""
    okey, ekey = CANDS[col]
    obs = o.get(okey)
    if obs is None:
        return np.nan
    dend = int(round(WINDOW_END_OFFSET_F360.get(okey, 0) * fps / 360.0))
    try:
        if callable(obs):
            return obs(int(ev.get("fp", 0)) + d_start,
                       int(ev["rel"]) + d_end + dend)
        f = int(ev.get(ekey, -1)) + d_start
        return float(obs[f]) if 0 <= f < len(obs) else np.nan
    except Exception:
        return np.nan


def r2(e, t):
    e, t = np.asarray(e, float), np.asarray(t, float)
    m = np.isfinite(e) & np.isfinite(t)
    return np.corrcoef(e[m], t[m])[0, 1] ** 2 if m.sum() > 4 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dump", action="store_true",
                    help="also write per-pitch est/truth pairs for every cell "
                         "(rejected_gt_pairs.csv.gz), the input the gate map needs")
    ap.add_argument("--event-offsets", default=None,
                    help="comma list of frame offsets, e.g. '-3,-2,-1,0,1,2,3'. "
                         "Each column is ALSO read with its anchor event shifted "
                         "by k frames, in ONE pass, so the event-TOLERANCE of every "
                         "cell can be scored without re-running the sweep per "
                         "offset. Writes rejected_gt_pairs_offsets.csv.gz with an "
                         "`offset` column. Implies --dump.")
    ap.add_argument("--detected", action="store_true",
                    help="DEPLOYMENT layer: re-detect release and foot plant on "
                         "each projected view (release_view() strategy rule) "
                         "instead of reading the OBP landmarks. MER falls back to "
                         "the adopted rel-11f proxy; fp_10 and MIR have no "
                         "detector, so their columns come back NaN. Pitch "
                         "inclusion still uses the GT gate so the population "
                         "matches the GT run exactly. Outputs get a _detected "
                         "suffix; the official files are untouched.")
    ap.add_argument("--no-window-ext", action="store_true",
                    help="zero WINDOW_END_OFFSET_F360, i.e. reproduce the pre-"
                         "2026-07-27 [fp, rel] window exactly. Exists so the "
                         "adopted change can be A/B'd on identical pitches, the "
                         "same way setup_anchor_regression uses guard=1.")
    ap.add_argument("--out-tag", default="",
                    help="extra tag appended to every output filename, so an A/B "
                         "run cannot overwrite the official tables")
    ap.add_argument("--fp-strategy", choices=["side", "frontal"], default="side",
                    help="which foot-plant detector --detected uses at EVERY view. "
                         "No fp routing rule is adopted, so the two are run "
                         "separately and the rule / oracle are composed per cell "
                         "afterwards (deploy_map.py).")
    ap.add_argument("--fp-target", choices=["fp", "fp10"], default="fp",
                    help="which OBP foot-plant landmark 'fp' resolves to: fp_100 "
                         "(default, the map's convention) or fp_10. Task 6 -- a "
                         "DEFINITION check. Both landmarks already ship in "
                         "landmarks.csv; this adds no event and no detector.")
    a = ap.parse_args()
    if a.fp_target != "fp" and not a.out_tag:
        a.out_tag = f"_{a.fp_target}"
    win_ext = {} if a.no_window_ext else WINDOW_END_OFFSET_F360
    suffix = ("_detected" if a.detected else "") + a.out_tag
    if a.detected and a.fp_strategy != "side":
        suffix += f"_fp{a.fp_strategy}"
    print(f"window-end offsets (f@360Hz): {win_ext or 'none'}")
    release_view = MER_LAG_S = None
    if a.detected:
        from angle_zone_sweep import release_view
        from angle_map_2d import MER_LAG_S

    gt = load_gt_events()
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    root = os.path.join(config.OBP_DATA_DIR, "c3d")

    offsets = ([int(x) for x in a.event_offsets.split(",")]
               if a.event_offsets else [0])
    if a.event_offsets:
        a.dump = True
    print(f"event offsets: {offsets}")
    cols = [c for c in CANDS if c in poi.columns]
    est = {(c, az, el, k): [] for c in cols for az in AZ for el in EL
           for k in offsets}
    tru = {c: [] for c in cols}
    sps = []                      # pitch id per row of est/tru, for --dump
    # adopted-metric truth columns, reported as INFORMATION (is a revived column
    # new, or a colinear shadow of something already measured) -- never as a
    # rejection. Derived from the live adoption list instead of a hand-written one:
    # the previous hardcoded list still carried lead_knee_extension_angular_velo_br,
    # de-adopted 2026-07-24, so columns were being compared against a metric that no
    # longer exists in either map (2026-07-27).
    from angle_map_2d import adopted_rows, gt_only_rows
    ADOPTED_TRUTH = sorted({t for _, _, t in adopted_rows() + gt_only_rows()
                            if isinstance(t, str)})
    print(f"colinearity reference = {len(ADOPTED_TRUTH)} live adopted truth "
          f"columns: {', '.join(ADOPTED_TRUTH)}\n")
    adopted_tru = {c: [] for c in ADOPTED_TRUTH if c in poi.columns}
    done = fail = 0

    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        sp = r.session_pitch
        g = gt.get(sp)
        if sp not in poi.index or not g or not {"fp", "rel"} <= set(g):
            fail += 1; continue
        if a.fp_target != "fp":
            # resolve every 'fp' read -- both the sampled anchors and the window
            # starts -- to the other landmark. Requiring it present keeps the
            # population identical to the fp_100 run rather than silently smaller.
            if a.fp_target not in g:
                fail += 1; continue
            g = {**g, "fp": g[a.fp_target]}
        path = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = load_feet(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
        except Exception:
            fail += 1; continue

        # truth once per pitch
        sps.append(sp)
        for c in cols:
            tru[c].append(poi.loc[sp, c])
        for c in adopted_tru:
            adopted_tru[c].append(poi.loc[sp, c])

        for az in AZ:
            for el in EL:
                try:
                    df = project_cam(joints, az, el)
                    o, _ = all_observables(df, fps, lead)
                    ev = g
                    if a.detected:
                        # deployment events, re-detected on THIS view. MER has no
                        # detector -> the adopted rel-11f proxy; fp_10 and MIR have
                        # neither a detector nor a validated proxy, so they are
                        # simply absent and their columns come back NaN.
                        rel_v = M.release_frame(df, arm, fps, J,
                                                view=release_view(az, el))
                        fp_v = M.foot_plant_frame(df, lead, fps, J, rel_v,
                                                  view=a.fp_strategy)
                        if rel_v <= fp_v + 1 or fp_v < 3:
                            raise ValueError("detection failed at this view")
                        ev = {"rel": rel_v, "fp": fp_v,
                              "mer": int(round(rel_v - MER_LAG_S * fps)),
                              "pkh": M.peak_knee_height_frame(df, lead, fp_v, J)}
                except Exception:
                    for c in cols:
                        for k in offsets:
                            est[(c, az, el, k)].append(np.nan)
                    continue

                for c in cols:
                    okey, ekey = CANDS[c]
                    obs = o.get(okey)
                    # derived window-end extension, scaled from 360 Hz to this clip
                    dend = int(round(win_ext.get(okey, 0) * fps / 360.0))
                    for k in offsets:
                        try:
                            if callable(obs):             # window: [fp, rel], both
                                # the event offset k shifts the ANCHOR; dend is a
                                # fixed boundary measured from that shifted anchor
                                v = obs(int(ev.get("fp", 0)) + k,
                                        int(ev["rel"]) + k + dend)
                            else:
                                f = int(ev.get(ekey, -1)) + k
                                v = float(obs[f]) if 0 <= f < len(obs) else np.nan
                        except Exception:
                            v = np.nan
                        est[(c, az, el, k)].append(v)
        done += 1
        if done % 50 == 0:
            print(f"  ...{done} processed")

    print(f"processed {done} / failed {fail}\n")

    if a.dump:
        # long-format pairs, same schema as angle_zone_pairs_gt.csv.gz so the gate
        # map can read the screened columns and the adopted metrics identically.
        recs = []
        for c in cols:
            tv = tru[c]
            for az in AZ:
                for el in EL:
                    for k in offsets:
                        ev = est[(c, az, el, k)]
                        recs.extend(zip([c] * len(sps), [az] * len(sps),
                                        [el] * len(sps), [k] * len(sps),
                                        sps, ev, tv))
        allp = pd.DataFrame(recs, columns=["metric", "az", "el", "offset",
                                           "session_pitch", "est", "truth"])
        # offset 0 is the map's own dump, written under the canonical name so a
        # tolerance run also refreshes the file the gate map reads
        dp = allp[allp.offset == 0].drop(columns=["offset"])
        if len(offsets) > 1:
            pk = os.path.join(config.OBP_VALIDATION_DIR,
                              f"rejected_gt_pairs_offsets{suffix}.csv.gz")
            allp.to_csv(pk, index=False, compression="gzip", float_format="%.6g")
            print(f"dumped {len(allp):,} pairs over {len(offsets)} offsets -> {pk}")
        outp = os.path.join(config.OBP_VALIDATION_DIR,
                            f"rejected_gt_pairs{suffix}.csv.gz")
        dp.to_csv(outp, index=False, compression="gzip", float_format="%.6g")
        print(f"dumped {len(dp):,} pairs -> {outp}\n")

    # colinearity denominator: max |corr(candidate truth, any adopted truth)|.
    # a candidate whose truth is ~a copy of an adopted truth is not new info.
    def max_adopted_corr(c):
        best = 0.0; who = ""
        tv = np.asarray(tru[c], float)
        for name, series in adopted_tru.items():
            av = np.asarray(series, float)
            m = np.isfinite(tv) & np.isfinite(av)
            if m.sum() > 4:
                rr = abs(np.corrcoef(tv[m], av[m])[0, 1])
                if rr > best:
                    best, who = rr, name
        return best, who

    grid_rows, rows = [], []
    for c in cols:
        grid = np.full((len(EL), len(AZ)), np.nan)
        for ie, el in enumerate(EL):
            for ia, az in enumerate(AZ):
                v = r2(est[(c, az, el, 0)], tru[c])
                grid[ie, ia] = v
                grid_rows.append({"column": c, "az": AZ[ia], "el": EL[ie], "r2": v})
        if np.all(np.isnan(grid)):
            continue
        bi = np.unravel_index(np.nanargmax(grid), grid.shape)
        best_r2 = grid[bi]; best_el, best_az = EL[bi[0]], AZ[bi[1]]
        okey, ekey = CANDS[c]
        cc, who = max_adopted_corr(c)
        rows.append(dict(column=c, obs=okey, event=ekey,
                         best_r2=best_r2, best_az=best_az, best_el=best_el,
                         ground_best=np.nanmax(grid[0]),
                         overhead_best=np.nanmax(grid[-2:]),
                         # how many of the 168 cells clear the floor: a real
                         # measurable zone has many, an optimistic spike has ~1
                         cells_usable=int(np.nansum(grid >= FLOOR)),
                         cells_near_best=int(np.nansum(grid >= best_r2 - 0.10)),
                         adopted_corr=cc, colinear_with=who))
    res = pd.DataFrame(rows).sort_values("best_r2", ascending=False)
    out = os.path.join(config.OBP_VALIDATION_DIR,
                       f"rejected_gt_full_sweep{suffix}.csv")
    res.to_csv(out, index=False)
    pd.DataFrame(grid_rows).to_csv(
        os.path.join(config.OBP_VALIDATION_DIR,
                     f"rejected_gt_full_grid{suffix}.csv"),
        index=False)

    print("=" * 104)
    print("REJECTED / UNTESTED COLUMNS re-screened with GT events over az x el")
    print("  cells = of 168, how many clear 0.60 (zone vs lucky spike);")
    print("  colin = max |corr| of this column's TRUTH with an adopted metric's truth")
    print("=" * 104)
    print(f"{'column':<40}{'evt':>4}{'best r2':>8}{'  view':>11}"
          f"{'grd':>6}{'ovh':>6}{'cells':>6}{'colin':>7}")
    for r in res.itertuples(index=False):
        flag = " *" if r.best_r2 >= FLOOR else "  "
        print(f"{r.column:<40}{r.event:>4}{r.best_r2:>8.3f}"
              f"{f'  {r.best_az}/{r.best_el}':>11}"
              f"{r.ground_best:>6.2f}{r.overhead_best:>6.2f}{r.cells_usable:>6d}"
              f"{r.adopted_corr:>7.2f}{flag}")

    hits = res[res.best_r2 >= FLOOR]
    print(f"\n{len(hits)} / {len(res)} columns clear the 0.60 usable floor under GT events.")
    print("\nrevived by an OVERHEAD view (ground dead, overhead alive) -- the "
          "transverse-revival story:")
    for r in hits.itertuples(index=False):
        if r.overhead_best > r.ground_best + 0.10:
            print(f"  {r.column:<40} ground {r.ground_best:.2f} -> overhead {r.overhead_best:.2f}")
    print("\nCAUTION -- likely a colinear shadow of an already-adopted metric "
          "(truth |corr| > 0.85):")
    for r in hits.itertuples(index=False):
        if r.adopted_corr > 0.85:
            print(f"  {r.column:<40} |corr| {r.adopted_corr:.2f} with {r.colinear_with}")
    print("\nCAUTION -- optimistic spike, not a zone (<=3 of 168 cells clear 0.60):")
    for r in hits.itertuples(index=False):
        if r.cells_usable <= 3:
            print(f"  {r.column:<40} only {r.cells_usable} usable cell(s)")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
