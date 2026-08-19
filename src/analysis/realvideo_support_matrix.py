"""Which canonical retained rows can be RUN on real video, derived from the code.

    real-video runnable rows = canonical 35 retained rows
                               INTERSECT rows the current real-video pipeline supports

This is a STATIC analysis. It executes no estimator on any clip and reports no accuracy
number. It answers one question per row: could the current code produce a value from an
RTMPose clip at all, and if not, exactly what is missing.

WHY THIS FILE EXISTS. The real-video pilot ran a hardcoded list of 15 metrics
(`pilot_metrics_eligible.csv`), which is the PRE-DEDUP `angle_map_2d.adopted_rows()` --
its own source comment says so: "The DEPLOYED map artefacts were deliberately left at
their pre-dedup state (15 rows here, 40 map rows, 991 cells)". Three of those 15 names
(`Glove Sh Abd @MER [O]`, `Torso Lat Tilt @MER [O]`, `Torso Rot @BR [O]`) are not
registry rows at all: the dedup dropped them in favour of the column-named rows that
measure the same quantity. So the 15 is neither a subset nor a superset of the canonical
row set, and it must not be used as the scope of a real-video evaluation.

WHAT MAKES A ROW RUNNABLE. The screened rows are read by
`rejected_gt_full_sweep.all_observables(df, fps, lead)` + `sample(...)`, and
`all_observables` is a PURE FUNCTION of a 2D joint DataFrame, its fps and which side
leads. RTMPose writes that same schema (`stage1/rtmp_extractor.py`). The estimator is
therefore almost never the blocker -- the ANCHOR is. This script separates the two.

ANCHOR AVAILABILITY ON REAL VIDEO, as the pilot actually resolved it:
  release  detected           `deploy/release_offset` + viewpoint routing
  fp       detected OR fallback  `metrics.foot_plant_frame`, routed by `deploy/fp_routing`
  pkh      derived from fp     `metrics.peak_knee_height_frame(df, lead, fp, J)`
                               -- so pkh inherits every fp failure
  mer      PROXY only          rel - 11 f @360 Hz (`angle_map_2d.MER_LAG_S`). No detector
                               exists; a single 2D view cannot locate a rotation instant.
  none     internal            the row anchors itself (HSS signature anchor, whole-clip max)

RUNNABLE IS NOT TRUSTWORTHY. A row anchored on the MER proxy runs, but whether the value
means anything depends on how steep the quantity is in time there -- `angle_map_2d`
records that elbow flexion moves 4.32 deg/frame against a truth SD of 8.07 deg, while
trunk lateral tilt and glove abduction are flat enough to tolerate the jitter. This
script flags that per row; it does not decide it.

TWO THINGS VERIFIED WHILE BUILDING THIS, both recorded in the matrix:

1. RTMPose (Halpe26, `stage1/rtmp_extractor.HALPE26`) writes all twelve limb joints the
   observables need -- shoulders, elbows, wrists, hips, knees, ankles -- plus `nose` and
   the feet. It does NOT write `head`, which `metrics.JOINTS` names. `metrics.body_com`
   already handles that: the head+neck segment (Winter weight 0.081) is dropped and the
   remaining weights renormalised. So the two COM rows run on real video with an
   11-segment COM instead of 12. That is a documented degradation, not a blocker, and it
   is carried in `pose_backbone_caveat`.

2. There is NO WIRED DRIVER for this row set. `deploy/measure_auto.py` runs the legacy 15
   and imports cleanly but dies at RUNTIME on line 115, reading
   `angle_zone_table_offset_rule.csv`, which now exists only in
   `archive_stale_pre_freeze/` -- the freeze's intended breakage. The generic path
   (`all_observables` + `sample`) lives in a script that loads OBP C3D and projects it, so
   feeding it RTMPose CSVs plus detected anchors is new plumbing that does not exist yet.
   This file reports capability, not readiness.

Run:  conda activate diamond; cd src\\analysis; python realvideo_support_matrix.py
Writes realvideo_support_matrix.csv and prints the matrix and the counts.
"""
import os, sys
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config
import metrics as M
from rejected_gt_full_sweep import CANDS, WINDOW_END_OFFSET_F360
import angle_map_2d as A

REG = os.path.join(config.OBP_VALIDATION_DIR, "paper_registry.csv")
OUT = os.path.join(config.OBP_VALIDATION_DIR, "realvideo_support_matrix.csv")
PILOT = os.path.join(config.ROOT, "data", "outputs", "realvideo_pilot",
                     "pilot_metrics_eligible.csv")

# --- joints each image-plane observable reads -------------------------------------
# Transcribed from the formulas in rejected_gt_full_sweep.observables() /
# all_observables() and from the metrics.py functions the adopted rows call. Keys are
# metrics.JOINTS keys and are asserted against it below, so a rename cannot rot this
# table silently. `LEAD`/`THROW`/`GLOVE` are resolved per pitcher, not fixed sides.
OBS_JOINTS = {
    # rejected_gt_full_sweep.observables()
    "trunk_lean":       ["l_sh", "r_sh", "l_hip", "r_hip"],
    "hip_lean":         ["l_hip", "r_hip"],
    "shoulder_line":    ["l_sh", "r_sh"],
    "hip_line":         ["l_hip", "r_hip"],
    "hss_angle":        ["l_sh", "r_sh", "l_hip", "r_hip"],
    "elbow_flex":       ["THROW_sh", "THROW_el", "THROW_wr"],
    "elbow_incl":       ["THROW_sh", "THROW_el", "THROW_wr"],
    "abd_throw":        ["THROW_el", "THROW_sh", "THROW_hip"],
    "abd_glove":        ["GLOVE_el", "GLOVE_sh", "GLOVE_hip"],
    "hz_abd_throw":     ["THROW_sh", "THROW_el", "l_sh", "r_sh"],
    "hz_abd_glove":     ["GLOVE_sh", "GLOVE_el", "l_sh", "r_sh"],
    "forearm_angle":    ["THROW_el", "THROW_wr"],
    "forearm_slot":     ["THROW_el", "THROW_wr"],
    "knee_ext_velo_at": ["LEAD_hip", "LEAD_kn", "LEAD_an"],
    "knee_ext_velo_max": ["LEAD_hip", "LEAD_kn", "LEAD_an"],
    "knee_ext_fp_to_br": ["LEAD_hip", "LEAD_kn", "LEAD_an"],
    "torso_pelvis_timing": ["l_sh", "r_sh", "l_hip", "r_hip"],
    # adopted-row estimators in metrics.py / angle_map_2d.py.
    # ⚠ An adopted row gets its OWN key even when it reads the same joints as a screened
    # row, because the key also selects the plausibility range downstream. Reusing
    # `hip_line` for pelvic rotational VELOCITY put a deg/s quantity under a +-180 deg
    # bound and flagged every physically normal reading as a wrap artefact.
    "arm_slot_sh_wr":   ["THROW_sh", "THROW_wr"],
    "wrist_point":      ["THROW_wr"],
    "wrist_speed":      ["THROW_wr"],
    "lead_knee_angle":  ["LEAD_hip", "LEAD_kn", "LEAD_an"],
    "ankle_line":       ["l_an", "r_an"],
    "ankle_sep_ratio":  ["l_an", "r_an"],
    "hip_line_velo":    ["l_hip", "r_hip"],
    "wrist_trail_anchor": ["THROW_wr", "TRAIL_an"],
    "body_com":         ["ALL 12 limb joints + head (head optional)"],
}
# derived observables inherit their base's joints
for _d, _b in (("shoulder_line_velo_max", "shoulder_line"),
               ("hip_line_velo_max", "hip_line"),
               ("elbow_ext_velo_max", "elbow_incl"),
               ("elbow_flex_max", "elbow_flex"),
               ("shoulder_line_min", "shoulder_line"),
               ("hz_abd_throw_max", "hz_abd_throw")):
    OBS_JOINTS[_d] = OBS_JOINTS[_b]

# --- the adopted rows: observable + what scale information they need --------------
# `scale` values: none (pure image angle) | stature (pixel stature, no camera needed)
# | height_m (the subject's standing height, entered once by tape measure)
ADOPTED = {
    "Arm Slot [O]":         ("arm_slot_sh_wr", "none"),
    "Release Height [O]":   ("wrist_point", "stature"),
    "Lead Knee Angle [O]":  ("lead_knee_angle", "none"),
    "Release Ext [O]":      ("wrist_trail_anchor", "stature"),
    "Wrist Speed [O]":      ("wrist_speed", "height_m"),
    "Stride (anchor) [O]":  ("ankle_sep_ratio", "stature"),
    "Stride Angle [O]":     ("ankle_line", "none"),
    "Trunk Tilt (ant) [O]": ("trunk_lean", "none"),
    "Hip-Shoulder Sep [O]": ("hss_angle", "none"),
    "Pelvis Rot Velo [O]":  ("hip_line_velo", "none"),
    "COG Fwd Velo [O]":     ("body_com", "height_m"),
    "COG Velo @PKH [O]":    ("body_com", "height_m"),
}

# Anchors, and how the real-video pipeline obtains each one.
ANCHOR_SOURCE = {
    "release":    ("detected", "deploy/release_offset + viewpoint routing"),
    "fp":         ("detected or fallback", "metrics.foot_plant_frame via deploy/fp_routing"),
    "release+fp": ("detected or fallback", "both ends; inherits every fp failure"),
    "pkh":        ("derived from fp", "metrics.peak_knee_height_frame(df, lead, fp, J)"),
    "mer":        ("PROXY ONLY", "rel - 11 f @360 Hz (angle_map_2d.MER_LAG_S); no detector"),
    "none":       ("internal", "the row anchors itself inside the clip"),
}

# Rows whose quantity is steep in time at MER, so the release-lag proxy is a
# QUALITY caveat rather than a support blocker. Recorded in angle_map_2d's MER_LAG_S
# note with its measured slope; listed here so the matrix can carry the flag.
MER_STEEP = {"elbow_flexion_mer"}


def main():
    reg = pd.read_csv(REG)
    ret = reg[reg.retained].copy()
    assert len(ret) == 35, len(ret)

    for keys in OBS_JOINTS.values():
        for k in keys:
            base = k.split("_", 1)[-1] if k.split("_")[0] in ("THROW", "GLOVE", "LEAD",
                                                              "TRAIL") else k
            if base.startswith("ALL"):
                continue
            assert base in M.JOINTS or f"l_{base}" in M.JOINTS or base in (
                "sh", "el", "wr", "hip", "kn", "an"), k

    adopted_labels = {lab for lab, _f, _t in A.adopted_rows()}
    pilot = set(pd.read_csv(PILOT).metric.unique()) if os.path.exists(PILOT) else set()

    recs = []
    for r in ret.itertuples(index=False):
        m = r.metric_id
        anchor = str(r.anchor_type)
        if m in ADOPTED:
            obs, scale = ADOPTED[m]
            path = "angle_map_2d.adopted_rows"
            impl = m in adopted_labels
        elif m in CANDS:
            obs, ev = CANDS[m]
            scale = "none"                    # every screened row is an image angle
            path = "rejected_gt_full_sweep.all_observables + sample"
            impl = True
            assert ev in ("fp", "fp10", "pkh", "mer", "mir", "rel"), ev
        else:
            obs, scale, path, impl = "", "none", "NONE", False

        src, how = ANCHOR_SOURCE.get(anchor, ("UNKNOWN", ""))
        joints = OBS_JOINTS.get(obs, ["?"])

        if not impl:
            verdict, why = "not runnable", "no real-video estimator for this row"
        elif anchor == "mer":
            verdict = "runnable via proxy anchor"
            why = ("MER cannot be located from one 2D view; runs off rel - 11 f. "
                   + ("STEEP in time here, so the value is not trustworthy"
                      if m in MER_STEEP else "flat enough in time to tolerate the jitter"))
        elif anchor in ("fp", "release+fp", "pkh"):
            verdict = "runnable, fp-conditional"
            why = "produced only on clips where foot plant is detected, not fallback"
        else:
            verdict, why = "runnable", ""

        # RTMPose writes no `head`; metrics.body_com drops the head+neck segment and
        # renormalises, so the COM rows run on an 11-segment COM on real video.
        caveat = ("11-segment COM: RTMPose has no head joint, head+neck (w 0.081) "
                  "dropped and weights renormalised" if obs == "body_com" else "")

        recs.append(dict(
            metric_id=m, row_class=r.row_class, grade=r.metric_grade,
            observable=obs, joints="+".join(joints), anchor=anchor,
            anchor_source=src, anchor_how=how, scale_needed=scale,
            estimator_path=path, estimator_exists=impl,
            window_offset_f360=WINDOW_END_OFFSET_F360.get(m, 0),
            pose_backbone_caveat=caveat,
            in_pilot_15=m in pilot, verdict=verdict, note=why))

    d = pd.DataFrame(recs)
    d.to_csv(OUT, index=False)

    # ---- report -------------------------------------------------------------
    runnable = d[d.estimator_exists]
    print(f"canonical retained rows                 {len(d)}")
    print(f"  estimator exists for real video       {len(runnable)}")
    print(f"  no real-video estimator               {len(d) - len(runnable)}")
    print()
    for v, sub in d.groupby("verdict"):
        print(f"{v:28s} {len(sub):>3}")
    print()
    print("by anchor:")
    for a, sub in d.groupby("anchor"):
        src = ANCHOR_SOURCE.get(a, ("?", ""))[0]
        print(f"  {a:<12s} {len(sub):>3}   anchor on real video: {src}")
    print()
    print(f"rows the pilot's hardcoded 15 actually ran: {int(d.in_pilot_15.sum())} "
          f"of {len(d)} canonical retained rows")
    only_pilot = sorted(pilot - set(d.metric_id))
    print(f"pilot names that are NOT canonical rows ({len(only_pilot)}): {only_pilot}")
    missed = sorted(d[~d.in_pilot_15 & d.estimator_exists].metric_id)
    print(f"canonical rows runnable but NEVER run on real video ({len(missed)}):")
    for m in missed:
        print("   ", m)
    print()
    print("scale information needed:")
    for s, sub in d.groupby("scale_needed"):
        print(f"  {s:<10s} {len(sub):>3}   {', '.join(sorted(sub.metric_id))[:88]}")
    cav = d[d.pose_backbone_caveat != ""]
    print(f"\npose-backbone caveat on {len(cav)} rows: "
          f"{', '.join(sorted(cav.metric_id))}")
    print("READINESS, separate from capability: no wired driver runs this row set on "
          "real video.\n  deploy/measure_auto.py runs the legacy 15 and dies at runtime "
          "on a retired arc table.")
    print(f"\nwrote -> {OUT}")


if __name__ == "__main__":
    main()
