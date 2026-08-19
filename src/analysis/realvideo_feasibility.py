"""Real-video FEASIBILITY: run all 35 canonical retained rows on the 80 eligible clips.

This is NOT an accuracy study. The clips carry no mocap truth, so no CCC, r2, MAE or
strong/moderate grade is computed or reported, and no output here is a graded cell of the
map. What is reported is whether the current code produces a value at all, off which
anchor, and how it fails when it does not.

ROW SET comes from `paper_registry.csv` (retained == True), never from a hardcoded list.
The pilot ran 15 metrics that were the PRE-DEDUP `angle_map_2d.adopted_rows()`; three of
those names are not registry rows at all. See `realvideo_support_matrix.py`.

CLIP SET is `realvideo_clips.csv` (80), built by `analysis/realvideo_clip_table.py`:
60 orbit-sweep clips, 15 real_video_test, 3 overhead and 2 consistency clips, the last
of a LEFT-handed subject.

ANCHORS ARE CONSUMED, NOT DETECTED HERE. The 78 clips carried over from the frozen pilot
keep its anchors verbatim; the two consistency clips were detected when the table was
built, through the same metrics functions and routing.

OUTPUT STATUS per (clip, row), in the order they are decided:
  no_anchor        the row's anchor is absent for this clip
  fp_fallback      the row needs fp and fp was NOT detected -- the frame is a fallback,
                   so a value would be produced from an anchor known to be wrong
  mer_proxy        the row needs MER, which no 2D detector can locate; run off rel-11f
  nan              the estimator ran and returned a non-finite value
  angle_unwrap     an angle past a full turn (|v| > 180): the unwrapped series drifted
  implausible      finite, inside no declared envelope for its family
  wrap_risk        an unwrapped image angle sitting within WRAP_GUARD deg of +-180
  ok               finite, in envelope, no flag

`fp_fallback`, `mer_proxy`, `wrap_risk` and `angle_unwrap` are PRODUCED-BUT-SUSPECT: the
danger in deployment is a plausible wrong number, not a blank. They are counted separately
from `ok` and from hard failures.

Run:  conda activate diamond; cd src\\analysis; python realvideo_feasibility.py
Writes realvideo_feasibility_cells.csv (clip x row) and realvideo_feasibility_rows.csv.
"""
import os, sys, warnings
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3", "../deploy"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import config
import metrics as M
from rejected_gt_full_sweep import CANDS, all_observables, sample, \
    WINDOW_END_OFFSET_F360
from realvideo_support_matrix import ADOPTED

V = config.OBP_VALIDATION_DIR
PILOT = os.path.join(V, "realvideo_clips.csv")
OUT_C = os.path.join(V, "realvideo_feasibility_cells.csv")
OUT_R = os.path.join(V, "realvideo_feasibility_rows.csv")

WRAP_GUARD = 15.0            # deg from +-180 on an unwrapped image angle
# The subject's standing height, needed by the rows reported in absolute units. All 80
# clips are the same subject, who supplied it: 1.72 m.
SUBJECT_HEIGHT_M = 1.72
# Coarse sanity envelope. THIS IS A DECLARED BOUND, NOT A QUALITY FILTER: it exists only
# to separate "produced a number" from "produced a broken number". For an angle read from
# a single atan2 the bound is definitional -- |v| > 180 means an unwrapped series has
# drifted past a full turn, which is the wrap artefact the pilot already documented, so
# those are reported as `angle_unwrap` rather than lumped in with implausible magnitudes.
RANGE = {
    "deg_signed":  (-180.0, 180.0),
    "deg_joint":   (0.0, 200.0),
    "deg_per_s":   (-6000.0, 6000.0),
    "ratio":       (0.0, 3.0),
    "stature_rate":        (0.0, 10.0),     # statures per second, unsigned
    "stature_rate_signed": (-10.0, 10.0),   # ditto, sign carries direction
    "m_per_s":             (0.0, 60.0),     # with the subject height supplied
    "m_per_s_signed":      (-60.0, 60.0),
    "s":           (-1.0, 1.0),
}

# ⚠ The unit of an ADOPTED row is a property of the ROW, not of the observable key it
# shares with other rows. Deriving it from the observable was wrong in three places and
# produced failures that were artefacts of this file:
#   Pelvis Rot Velo   reads the hip line but returns deg/s, and was range-checked as an
#                     angle, so plausible pelvic velocities (365, 459 deg/s on the
#                     overhead clips) were flagged as unwrapped angles.
#   Stride (anchor)   shares `ankle_line` with Stride Angle but returns a stature ratio.
#   COG Velo @PKH     returns SIGNED statures/s without a subject height (metrics.py),
#                     and a zero lower bound rejected every rear-azimuth clip where the
#                     image-x forward direction reverses.
ADOPTED_UNIT = {
    "Arm Slot [O]":         "deg_signed",
    "Release Height [O]":   "ratio",
    "Lead Knee Angle [O]":  "deg_joint",
    "Release Ext [O]":      "ratio",
    "Wrist Speed [O]":      "m_per_s",
    "Stride (anchor) [O]":  "ratio",
    "Stride Angle [O]":     "deg_signed",
    "Trunk Tilt (ant) [O]": "deg_signed",
    "Hip-Shoulder Sep [O]": "deg_signed",
    "Pelvis Rot Velo [O]":  "deg_per_s",
    "COG Fwd Velo [O]":     "m_per_s",
    "COG Velo @PKH [O]":    "m_per_s_signed",
}
OBS_RANGE = {
    "trunk_lean": "deg_signed", "hip_lean": "deg_signed",
    "shoulder_line": "deg_signed", "hip_line": "deg_signed",
    "hss_angle": "deg_signed", "elbow_flex": "deg_joint",
    "elbow_incl": "deg_joint", "abd_throw": "deg_joint",
    "abd_glove": "deg_joint", "hz_abd_throw": "deg_signed",
    "hz_abd_glove": "deg_signed", "forearm_angle": "deg_signed",
    "forearm_slot": "deg_signed", "knee_ext_velo_at": "deg_per_s",
    "knee_ext_velo_max": "deg_per_s", "knee_ext_fp_to_br": "deg_signed",
    "elbow_flex_max": "deg_joint", "shoulder_line_velo_max": "deg_per_s",
    "hip_line_velo_max": "deg_per_s", "elbow_ext_velo_max": "deg_per_s",
    "shoulder_line_min": "deg_signed", "hz_abd_throw_max": "deg_signed",
    "torso_pelvis_timing": "s",
    "arm_slot_sh_wr": "deg_signed", "lead_knee_angle": "deg_joint",
    "ankle_line": "deg_signed", "wrist_point": "ratio",
    "wrist_trail_anchor": "ratio",
    # Rates in stature units. `body_com` is signed because forward COM velocity at peak
    # knee height is legitimately negative on some deliveries; an unsigned bound flagged
    # those as broken output.
    "body_com": "stature_rate_signed", "wrist_speed": "stature_rate",
    # ⚠ These are adopted rows whose value is NOT an image angle. They previously borrowed
    # a screened row's key and inherited its +-180 deg bound: pelvic rotational velocity is
    # deg/s and peaks in the hundreds, so every physically normal reading was flagged as an
    # unwrapped angle. Stride length is a stature ratio; its check was vacuous rather than
    # wrong, since a ratio cannot exceed 180 either.
    "hip_line_velo": "deg_per_s", "ankle_sep_ratio": "ratio",
}
# unwrapped image angles: a value near +-180 is a wrap candidate. A velocity or a ratio
# is not, so hip_line_velo / ankle_sep_ratio / wrist_speed are deliberately absent.
WRAPPY = {"shoulder_line", "hip_line", "hss_angle", "forearm_angle",
          "ankle_line", "shoulder_line_min", "hz_abd_throw", "hz_abd_glove"}


def load_pose(clip):
    """The project's validated best pose for one clip, plus the unsmoothed coords.
    Same convention as deploy/measure_auto: prefer the occlusion-refined sibling, and
    rename RTMPose's `nose` to `head` so metrics.JOINTS resolves."""
    base = os.path.join(config.ROOT, "data", "outputs", clip)
    p = config.refined_or(os.path.join(base, f"{clip}_smoothed_rtmp.csv"))
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if "nose_x" in df.columns and "head_x" not in df.columns:
        df = df.rename(columns={"nose_x": "head_x", "nose_y": "head_y",
                                "nose_v": "head_v"})
    return df


# The adopted estimators are TAKEN FROM THE PROJECT, never reimplemented here: a second
# copy of a metric definition drifting out of sync silently breaks everything downstream
# (CLAUDE.md). angle_map_2d.adopted_rows() is the same table the map is built on, and its
# estimators take (df, ctx) with the contract below.
ADOPTED_EST = None            # {label: estfn}, filled in main()


def adopted_value(m, df, ctx):
    return ADOPTED_EST[m](df, ctx)


def _scalar(v):
    """Estimators return a float, but a few return a 1-element array or a tuple whose
    first entry is the value. Anything else is not a measurement."""
    if v is None:
        return np.nan
    if isinstance(v, (tuple, list)):
        v = v[0] if len(v) else np.nan
    a = np.asarray(v, dtype=float).ravel()
    return float(a[0]) if a.size == 1 else np.nan


def classify(val, obs, need, have, fp_ok, m):
    """Status for one (clip, row). Order matters: anchor problems outrank value
    problems, because a value read off a wrong frame is not evidence the row works."""
    if not have:
        return "no_anchor", ""
    if need == "mer":
        base = "mer_proxy"
    elif not fp_ok:
        return "fp_fallback", "fp not detected; anchor frame is a fallback"
    else:
        base = None
    if val is None or not np.isfinite(val):
        return "nan", "estimator returned non-finite"
    v = float(val)
    # The unit of an adopted row is a property of the ROW and overrides the family its
    # observable key would imply; see ADOPTED_UNIT. Screened rows fall back to the key.
    fam = ADOPTED_UNIT.get(m) or OBS_RANGE.get(obs)
    if fam:
        lo, hi = RANGE[fam]
        if not (lo <= v <= hi):
            if fam.startswith("deg") and abs(v) > 180.0:
                return "angle_unwrap", f"{v:.1f} deg: unwrapped series past a full turn"
            return "implausible", f"{v:.3g} outside {fam} {lo}..{hi}"
    if obs in WRAPPY and abs(abs(v) - 180.0) <= WRAP_GUARD:
        return "wrap_risk", f"{v:.1f} deg within {WRAP_GUARD:g} of +-180"
    return (base or "ok"), ("rel-11f proxy anchor" if base else "")


def main():
    global ADOPTED_EST
    import angle_map_2d as A
    ADOPTED_EST = {lab: fn for lab, fn, _t in A.adopted_rows()}
    missing = set(ADOPTED) - set(ADOPTED_EST)
    assert not missing, f"adopted rows with no project estimator: {missing}"

    reg = pd.read_csv(os.path.join(V, "paper_registry.csv"))
    rows = reg[reg.retained].copy()
    assert len(rows) == 35, len(rows)
    clips = pd.read_csv(PILOT)
    assert len(clips) == 80, len(clips)

    recs = []
    for c in clips.itertuples(index=False):
        clip = c.clip
        df = load_pose(clip)
        arm = str(c.true_arm)
        lead = "left" if arm == "right" else "right"
        fps = float(c.fps)
        anchors = {"release": c.release_f, "fp": c.fp_f, "pkh": c.pkh_f,
                   "mer": c.mer_f, "none": 0}
        fp_ok = bool(c.fp_detected)
        if df is None:
            for r in rows.itertuples(index=False):
                recs.append(dict(clip=clip, metric_id=r.metric_id, status="no_pose",
                                 value=np.nan, note="no extracted pose CSV"))
            continue
        rel = anchors["release"]
        obs = None
        if np.isfinite(rel):
            try:
                obs, _ = all_observables(df, fps, lead)
            except Exception as e:
                obs = None
        for r in rows.itertuples(index=False):
            m, need = r.metric_id, str(r.anchor_type)
            if m in ADOPTED:
                okey = ADOPTED[m][0]
            else:
                okey = CANDS[m][0]
            # which anchor frames this row needs
            keys = ["release", "fp"] if need == "release+fp" else \
                   ([] if need == "none" else [need])
            have = all(np.isfinite(anchors[k]) for k in keys) if keys else True
            if need in ("release+fp", "pkh") or need == "fp":
                have = have and np.isfinite(anchors["release"])
            val, note = np.nan, ""
            if have and obs is not None:
                try:
                    if m in ADOPTED:
                        ctx = dict(arm=arm, lead=lead, fps=fps,
                                   rel=int(anchors["release"]),
                                   fp=int(anchors["fp"]) if np.isfinite(anchors["fp"]) else 0,
                                   pkh=int(anchors["pkh"]) if np.isfinite(anchors["pkh"]) else None,
                                   mer=None,     # no GT MER on real video, ever
                                   height_m=SUBJECT_HEIGHT_M)
                        val = _scalar(adopted_value(m, df, ctx))
                    else:
                        ev = CANDS[m][1]
                        evf = {"fp": anchors["fp"], "rel": anchors["release"],
                               "mer": anchors["mer"], "pkh": anchors["pkh"],
                               "fp10": anchors["fp"], "mir": anchors["release"]}[ev]
                        o2 = dict(obs)
                        d_end = WINDOW_END_OFFSET_F360.get(m, 0)
                        d_end = int(round(d_end * fps / 360.0))
                        val = _scalar(sample(
                            o2, m,
                            {"fp": int(anchors["fp"]) if np.isfinite(anchors["fp"]) else 0,
                             "rel": int(anchors["release"]),
                             "mer": int(evf) if np.isfinite(evf) else 0,
                             "pkh": int(anchors["pkh"]) if np.isfinite(anchors["pkh"]) else 0},
                            fps, 0, d_end))
                except Exception as e:
                    val, note = np.nan, f"{type(e).__name__}: {e}"[:90]
            st, n2 = classify(val, okey, need, have,
                              fp_ok or need in ("release", "none", "mer"), m)
            recs.append(dict(clip=clip, metric_id=m, anchor=need, observable=okey,
                             status=st, value=val, note=note or n2))

    d = pd.DataFrame(recs)
    d.to_csv(OUT_C, index=False)

    ORDER = ["ok", "mer_proxy", "wrap_risk", "angle_unwrap", "fp_fallback",
             "implausible", "nan", "no_anchor", "no_pose"]
    piv = (d.groupby(["metric_id", "status"]).size().unstack(fill_value=0)
           .reindex(columns=ORDER, fill_value=0))
    piv["n"] = piv.sum(axis=1)
    piv["ok_rate"] = piv["ok"] / piv["n"]
    piv = piv.join(reg.set_index("metric_id")[["anchor_type", "metric_grade"]])
    piv.sort_values(["ok_rate", "n"], ascending=False).to_csv(OUT_R)

    tot = len(d)
    print(f"cells {tot} = {len(rows)} rows x {len(clips)} clips")
    print()
    for s in ORDER:
        k = int((d.status == s).sum())
        if k:
            print(f"  {s:<14s} {k:>5}  {100*k/tot:5.1f} %")
    print()
    print("by anchor class:")
    for a, sub in d.groupby("anchor"):
        ok = int((sub.status == "ok").sum())
        print(f"  {a:<11s} n {len(sub):>4}   ok {ok:>4} ({100*ok/len(sub):4.1f} %)")
    print()
    print("rows with NO ok cell at all:")
    dead = piv[piv["ok"] == 0]
    for m, r in dead.iterrows():
        top = max(ORDER, key=lambda s: r.get(s, 0))
        print(f"  {m:<42s} anchor {r.anchor_type:<11s} dominant: {top}")
    print(f"\n  ({len(dead)} of {len(rows)} rows)")
    print(f"\nwrote -> {OUT_C}\n         {OUT_R}")


if __name__ == "__main__":
    main()
