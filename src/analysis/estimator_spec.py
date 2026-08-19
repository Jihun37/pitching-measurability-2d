"""THE CANONICAL ESTIMATOR SPECIFICATION for the 47 evaluated rows.

WHY THIS IS A SCRIPT AND NOT A DOCUMENT. The paper's contribution sentence is "47
measurement pathways evaluated exhaustively", so a reader has to be able to see what each
row actually computes. Writing that as a table in a document creates the largest possible
drift surface in a project whose entire freeze exists because documents drifted from code.
So the specification is a module that ASSERTS itself against the live sources and emits the
table. If a registry row has no entry, or an entry names a row that is not in the registry,
this script fails.

DESTINATION (decision D4, 2026-08-07). The paper carries NO supplement and NO appendix.
This specification ships with the RELEASED CODE and the article cites it. The body carries
the common operators, the five direct-3D truth definitions and the implementation
parameters; it does not reproduce the 47 rows.

WHAT IS ASSERTED
  * every registry metric_id has exactly one SPEC entry and vice versa
  * each entry's `anchor` matches the registry's `anchor_type`
  * each entry's `circular` matches the registry's `circular`
  * every joint an entry names exists in obp_project.MARKER_MAP
  * `truth` matches the registry's truth_quantity

TWO SOURCES OF ESTIMATE. The 12 adopted rows are computed by
`metrics.compute_candidates`; the 35 screened rows by
`rejected_gt_full_sweep.observables`. An entry records which, in `impl`.

⚠ INCOMPLETE. Rows are being filled from `metrics.py` in passes. `main()` fails while any
registry row is missing, which is deliberate -- a half-filled specification must not ship
looking complete. Progress prints on every run.

Run:  conda activate diamond; cd src\\analysis; python estimator_spec.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config

V = config.OBP_VALIDATION_DIR

# ---------------------------------------------------------------------------
# COMMON OPERATORS. These are what the body of the paper carries (equation E-2);
# every row below is expressed in terms of them.
# ---------------------------------------------------------------------------
OPERATORS = {
    "angle3": dict(
        source="metrics._angle",
        formula="theta(a,b,c) = degrees(arccos(clip(((a-b).(c-b)) / "
                "(|a-b||c-b| + 1e-6), -1, 1)))",
        range="[0, 180] degrees",
        signed=False,
        note="UNSIGNED -- an arccos of a normalised dot product cannot distinguish "
             "the direction of flexion. ⚠ The 1e-6 in the denominator is a SILENT "
             "FALLBACK, not a refusal: if a segment has zero projected length the "
             "quotient is 0 and the operator returns 90 degrees rather than NaN."),
    "orient2": dict(
        source="numpy.arctan2 on an image-plane segment",
        formula="phi(p,q) = degrees(arctan2(p_y - q_y, p_x - q_x))",
        range="(-180, 180] degrees, or unbounded where the series is unwrapped",
        signed=True,
        note="Image-frame orientation. Image y increases DOWNWARD, so the sign is an "
             "image convention, not an anatomical one -- which is why several rows "
             "differ from their OBP column by a near-constant offset and calibrate "
             "rather than agree raw."),
    "speed2": dict(
        source="metrics._speed",
        formula="v[i] = hypot(x[i]-x[i-1], y[i]-y[i-1]) * fps,  v[0] = 0",
        range="pixels/second",
        signed=False,
        note="⚠ A PLAIN FIRST DIFFERENCE, not Savitzky-Golay. The SG derivative is "
             "used by the COM and angular-velocity estimators only; the two paths "
             "must not be conflated in the parameter table."),
    "body_scale": dict(
        source="metrics.body_scale_px",
        formula="median over frames of |mean(shoulder_y) - mean(hip_y)|, floored at 1",
        range="pixels",
        signed=False,
        note="Normalises lengths and speeds against camera distance and zoom."),
    "stature_px": dict(
        source="metrics.pixel_stature",
        formula="95th percentile over frames of (max(ankle_y) - head_y), floored at 1; "
                "without a head joint, (max(ankle_y) - shoulder_mid_y) * 1.25",
        range="pixels",
        signed=False,
        note="Basis of the stature-normalised distances (stride length, release "
             "height)."),
}

# ---------------------------------------------------------------------------
# ROW SPECIFICATIONS. Keyed by paper_registry.metric_id.
# ---------------------------------------------------------------------------
S = dict


def row(observable, joints, formula, unit, anchor, extremum=None, sign=None,
        wrap=None, refusal=None, truth=None, impl="compute_candidates", note=None):
    return dict(observable=observable, joints=joints, formula=formula, unit=unit,
                anchor=anchor, extremum=extremum, sign=sign, wrap=wrap,
                refusal=refusal, truth=truth, impl=impl, note=note)


SPEC = {
    # ---------------- adopted, verified from metrics.py 2026-08-08 -------------
    "Stride Angle [O]": row(
        observable="orientation of the lead-to-trail ankle line",
        joints=["l_an", "r_an"],
        formula="orient2(lead_ankle, trail_ankle) at the foot-plant frame "
                "(metrics.stride_angle_2d)",
        unit="degrees",
        anchor="fp",
        sign="signed, image convention; differs from the OBP column by a near-constant "
             "~90 deg offset, so the row is CALIBRATE and never DIRECT",
        wrap="circular; unwrapped per cell by gate_map before scoring",
        refusal="none beyond a missing foot-plant frame",
        truth="stride_angle",
        note="Handedness-relative. Measurable only because the loader reflects every "
             "LHP into an equivalent RHP; pooling raw handedness cancels it."),

    "Stride (anchor) [O]": row(
        observable="lead-ankle settled position against the trail foot's pre-motion anchor",
        joints=["l_an", "r_an"],
        formula="|median(lead_ankle_x over [rel - 0.08 s, rel]) - "
                "trail_anchor_x(trail_ankle_x, rel, fps)| / stature_px "
                "(metrics.stride_settled_2d)",
        unit="statures",
        anchor="release",
        extremum="median over an 80 ms window ending AT release -- not a read at foot "
                 "plant. The lead ankle keeps travelling for >=80 ms past 100 % load as "
                 "the shank rotates over the planted foot; reading the settled position "
                 "scores 0.844 against 0.823 and removes the foot-plant dependency. Do "
                 "not widen the window back toward foot plant.",
        sign="unsigned (absolute distance)",
        wrap="none",
        refusal="NaN if the window is empty or all-NaN",
        truth="stride_length"),

    "Release Ext [O]": row(
        observable="wrist-to-trail-anchor forward distance",
        joints=["r_wr", "r_an"],
        formula="|wrist_x[rel] - trail_anchor_x(trail_ankle_x, fp, fps)| / stature_px, "
                "times height_m for metres (metrics.release_extension)",
        unit="statures, or metres when stature is supplied",
        anchor="release+fp",
        sign="unsigned (absolute distance)",
        wrap="none",
        refusal="⚠ THE ONLY EXPLICIT REFUSAL AMONG THE ADOPTED ROWS. Returns NaN when "
                "trail_quiet_len(...) < 0.10 s, i.e. when the delivery has no still "
                "setup (walk-in, crow-hop) and the pre-motion anchor would degrade "
                "silently. This is why Release Ext is the only row with material "
                "missingness -- n = 358 of 394, the sole row below n = 380 among graded "
                "cells. Substituting a trail-ankle-at-release reference is NOT a valid "
                "fallback: it is a different quantity (r = 0.71).",
        truth="t3_release_ext",
        note="Needs no rubber and works on flat ground: the reference is where the "
             "pitcher was standing, not the rubber."),

    "COG Fwd Velo [O]": row(
        observable="whole-body COM forward speed, peak",
        joints=["l_sh", "r_sh", "l_hip", "r_hip", "l_el", "r_el", "l_wr", "r_wr",
                "l_kn", "r_kn", "l_an", "r_an", "head"],
        formula="max over [0, rel] of |gradient(COM_x)| * fps / stature_px, times "
                "height_m for m/s (metrics.cog_fwd_velo)",
        unit="statures/s, or m/s when stature is supplied",
        anchor="release",
        extremum="window maximum over [0, release]",
        sign="unsigned -- the absolute value is taken before the maximum",
        wrap="none",
        refusal="NaN if the segment is empty or all-NaN",
        truth="max_cog_velo_x",
        note="⚠ Differentiated by np.gradient (central difference), NOT by the "
             "Savitzky-Golay path and NOT by metrics._speed. Three differentiation "
             "routes exist in the estimator layer and must be kept apart."),

    "COG Velo @PKH [O]": row(
        observable="whole-body COM forward velocity at peak knee height",
        joints=["l_sh", "r_sh", "l_hip", "r_hip", "l_el", "r_el", "l_wr", "r_wr",
                "l_kn", "r_kn", "l_an", "r_an", "head"],
        formula="Savitzky-Golay first derivative of COM_x evaluated at the pkh frame, "
                "/ stature_px, times height_m for m/s (metrics.cog_velo_at_pkh)",
        unit="statures/s, or m/s when stature is supplied",
        anchor="pkh",
        extremum="point read at the pkh frame; pkh itself is argmin of the lead-knee "
                 "image y over [0, fp]",
        sign="SIGNED -- unlike COG Fwd Velo, no absolute value is taken",
        wrap="none",
        refusal="falls back to np.gradient * fps when the SG window would be <= 3 frames",
        truth="cog_velo_pkh",
        note="SG parameters: window = max(5, round(0.05 s * fps)) forced odd and capped "
             "at the series length, polyorder 2, deriv 1, delta 1/fps, mode 'interp'. A "
             "local linear fit rather than a two-frame gradient; it lifts the clean r2 "
             "from 0.80 to 0.82."),

    "Hip-Shoulder Sep [O]": row(
        observable="signed angle between the shoulder chord and the hip chord",
        joints=["l_sh", "r_sh", "l_hip", "r_hip"],
        formula="sep = degrees(arctan2(sh x hp, sh . hp)) with sh = r_sh - l_sh and "
                "hp = r_hip - l_hip; gated, median-filtered, then read at a "
                "signature-anchored extremum (metrics.hss_peak_overhead)",
        unit="degrees",
        anchor="none",
        extremum="the full recipe, in order: (1) chord-validity gate -- both chords must "
                 "be >= 0.45 of their clip median length, else the frame is voided; "
                 "(2) median filter, kernel max(3, int(0.09 s * fps) rounded to odd); "
                 "(3) signature anchor t* = the largest SUSTAINED swing over 0.25 s that "
                 "PERSISTS (up to 8 candidates >= one span apart; accepted when the mean "
                 "over [i+w, i+2w] moves the same way by >= 0.5 |swing|; else the largest "
                 "raw swing); (4) value = |sep| at the COIL-DIRECTION extremum, "
                 "argmax(-sign * sep), over [t* - 0.45 s, t* + span/4]",
        sign="signed series; the reported value is its absolute magnitude",
        wrap="none",
        refusal="returns None -- hence NaN -- when no anchor is found",
        truth="max_rotation_hip_shoulder_separation",
        note="⚠ NEVER a whole-clip |max| and never a release-anchored window: the "
             "follow-through rebound beats the real dip, and wrist-speed release anchors "
             "are backbone-unstable. Both are VERIFIED failure modes. ⚠ The 2D chord "
             "angle reads about TWICE the 3D truth on OBP, so raw values must never be "
             "compared with 3D literature numbers -- the row is moderate-only and "
             "calibrates rather than agrees raw."),

    "Arm Slot [O]": row(
        observable="shoulder-to-wrist line orientation against the vertical",
        joints=["r_sh", "r_wr"],
        formula="degrees(arctan2(|wrist_x[rel] - shoulder_x[rel]|, "
                "shoulder_y[rel] - wrist_y[rel]))",
        unit="degrees",
        anchor="release",
        sign="unsigned in the horizontal component; overhand is SMALL (0-40 deg), "
             "three-quarter 50-60, sidearm above 70",
        wrap="none",
        refusal="none",
        truth="t3_armslot",
        note="⚠ A CORONAL quantity: most accurate from the front (az~90) and not "
             "measurable from a pure side view. ⚠ Deliberately NOT scored against the "
             "OBP arm_slot column, which is forearm-based; that column is evaluated "
             "separately as the screened row `arm_slot` with its own forearm_slot "
             "observable."),

    "Release Height [O]": row(
        observable="throwing-wrist height above the lower ankle",
        joints=["r_wr", "l_an", "r_an"],
        formula="(max(all ankle_y) - wrist_y[rel]) / body_scale_px",
        unit="body-scale units (torso lengths)",
        anchor="release",
        sign="positive upward; image y increases downward, hence the subtraction order",
        wrap="none",
        refusal="none",
        truth="t3_relh",
        note="⚠ DISCREPANCY TO RESOLVE: the registry's image_plane_observable calls this "
             "'wrist height / stature', but the code divides by body_scale_px (the "
             "shoulder-to-hip torso length), not by pixel_stature. The code is "
             "authority; the registry description should be corrected."),

    "Lead Knee Angle [O]": row(
        observable="hip-knee-ankle angle on the lead leg",
        joints=["l_hip", "l_kn", "l_an"],
        formula="angle3(lead_hip, lead_knee, lead_ankle) at the release frame",
        unit="degrees",
        anchor="release",
        sign="unsigned",
        wrap="none",
        refusal="inherits the angle3 silent fallback",
        truth="t3_knee_abs",
        note="⚠ THE ADOPTED DEFINITION IS THE RELEASE ANGLE, not the foot-plant-to-"
             "release difference. The absolute angle scores far above the extension "
             "difference; the difference survives only as the separate screened row "
             "lead_knee_extension_from_fp_to_br."),

    "Trunk Tilt (ant) [O]": row(
        observable="trunk-line orientation against the image vertical",
        joints=["l_sh", "r_sh", "l_hip", "r_hip"],
        formula="degrees(arctan2(sh_mid_x - hip_mid_x, -(sh_mid_y - hip_mid_y))) at "
                "release, multiplied by a direction sign",
        unit="degrees",
        anchor="release",
        sign="signed, and multiplied by pitch_dir so the tilt reads consistently for "
             "either delivery direction",
        wrap="none",
        refusal="none",
        truth="torso_anterior_tilt_br",
        note="Same observable as the screened trunk_lean rows. Which anatomical lean it "
             "captures is set by the viewpoint, not the formula: sagittal (anterior) "
             "from the side, coronal (lateral) from the front."),

    "Pelvis Rot Velo [O]": row(
        observable="hip-line yaw rate, seen face-on from overhead",
        joints=["l_hip", "r_hip", "l_sh", "r_sh"],
        formula="hip-line angle resolved for 180 deg flips by continuity against the "
                "shoulder line, interpolated across occluded frames, then a "
                "Savitzky-Golay derivative, |.|, and a ~42 ms median filter; the row "
                "takes the window maximum near release",
        unit="degrees/s",
        anchor="release",
        extremum="window maximum near release",
        sign="unsigned -- the absolute value is taken before the maximum",
        wrap="180 deg ambiguity resolved by CONTINUITY, re-anchored to the shoulder "
             "line, so only true ~180 jumps flip and noise does not",
        refusal="frames are gated as occluded when the hip chord collapses below 0.5 of "
                "its clip median, or when per-joint confidence is below 0.4 where "
                "confidence columns exist; gated frames are INTERPOLATED, not dropped. "
                "OBP projections carry no confidence columns, so only the chord test "
                "applies there",
        truth="max_pelvis_rotational_velo",
        note="⚠ OVERHEAD ONLY -- degenerate from side or front, where the hip line does "
             "not sweep in-plane. ⚠ On real video the absolute scale is inflated about "
             "2x by pose jitter (verified as jitter, not geometry); shape, timing and "
             "threshold crossings are correct, absolute values are not."),

    "Wrist Speed [O]": row(
        observable="throwing-wrist point speed, whole-clip peak",
        joints=["r_wr"],
        formula="max over the whole clip of speed2(wrist_x, wrist_y) / pixel_stature, "
                "times height_m",
        unit="metres/second",
        anchor="none",
        extremum="WHOLE-CLIP maximum -- reads no external event frame, which is why "
                 "this row is out of scope for the anchor-shift sweep",
        sign="unsigned",
        wrap="none",
        refusal="⚠ RETURNS NaN WHEN height_m IS NOT SUPPLIED. The adopted output is "
                "absolute m/s and needs the subject's height; the /stature and "
                "/body-scale variants are kept only as backward-compatible aliases and "
                "are not the evaluated row.",
        truth="t3_wrist",
        note="⚠ Differentiated by speed2, the plain first difference -- not by "
             "np.gradient and not by Savitzky-Golay. ⚠ Validated against 3D-DIRECT "
             "wrist speed, never against max_elbow_extension_velo, which is angular "
             "and tracks it poorly."),

    "elbow_flexion_mer": row(
        observable="elbow flexion, 0 = straight",
        joints=["r_sh", "r_el", "r_wr"],
        formula="180 - angle3(shoulder, elbow, wrist) at the MER frame "
                "(metrics.elbow_flexion_2d)",
        unit="degrees",
        anchor="mer",
        sign="unsigned",
        wrap="none",
        refusal="inherits the angle3 silent fallback at zero segment length",
        truth="elbow_flexion_mer",
        impl="observables"),

    "glove_shoulder_abduction_mer": row(
        observable="glove-side shoulder abduction",
        joints=["l_el", "l_sh", "l_hip"],
        formula="angle3(elbow, shoulder, hip) on the GLOVE side at the MER frame "
                "(metrics.shoulder_abduction_2d)",
        unit="degrees",
        anchor="mer",
        sign="unsigned",
        wrap="none",
        refusal="inherits the angle3 silent fallback",
        truth="glove_shoulder_abduction_mer",
        impl="observables"),

    "torso_rotation_mer": row(
        observable="shoulder-line orientation, a transverse-plane proxy",
        joints=["l_sh", "r_sh"],
        formula="degrees(unwrap(arctan2(r_sh_y - l_sh_y, r_sh_x - l_sh_x))) at the MER "
                "frame (metrics.torso_rotation_2d)",
        unit="degrees",
        anchor="mer",
        sign="signed, image convention",
        wrap="UNWRAPPED INSIDE the estimator, then treated as circular by gate_map -- "
             "check for double handling before changing either",
        refusal="none",
        truth="torso_rotation_mer",
        impl="observables",
        note="Degenerate at ground level: the shoulder line is near-horizontal for "
             "every rotation. Recovered from overhead."),

    "torso_lateral_tilt_mer": row(
        observable="trunk-line orientation against the image vertical",
        joints=["l_sh", "r_sh", "l_hip", "r_hip"],
        formula="degrees(arctan2(sh_mid_x - hip_mid_x, -(sh_mid_y - hip_mid_y))) at the "
                "MER frame (metrics.trunk_lean_2d)",
        unit="degrees",
        anchor="mer",
        sign="signed; the negation of the y term puts 'up' positive against the "
             "downward image axis",
        wrap="none",
        refusal="none",
        truth="torso_lateral_tilt_mer",
        impl="observables",
        note="⚠ ONE OBSERVABLE, TWO FAMILIES. trunk_lean_2d backs BOTH the anterior-tilt "
             "and the lateral-tilt rows; which anatomical lean it captures is decided by "
             "the viewpoint (sagittal from the side, coronal from the front), not by the "
             "formula."),
}


# ---------------------------------------------------------------------------
# SCREENED ROWS. All 35 are (observable, anchor) pairs over the same per-frame
# observable set, so they are specified as data rather than as 35 hand-written
# entries -- which is how `rejected_gt_full_sweep` itself is written.
# ---------------------------------------------------------------------------
OBS = {
    "trunk_lean": ("degrees(arctan2(sh_mid_x - hip_mid_x, -(sh_mid_y - hip_mid_y)))",
                   ["l_sh", "r_sh", "l_hip", "r_hip"], "degrees", "signed", None),
    "hip_lean": ("degrees(arctan2(r_hip_x - l_hip_x, -(r_hip_y - l_hip_y)))",
                 ["l_hip", "r_hip"], "degrees", "signed", "pelvis obliquity proxy"),
    "shoulder_line": ("degrees(unwrap(arctan2(r_sh_y - l_sh_y, r_sh_x - l_sh_x)))",
                      ["l_sh", "r_sh"], "degrees", "signed, unwrapped",
                      "degenerate at el 0 -- the line is near-horizontal for every "
                      "rotation; revives as the camera rises"),
    "hip_line": ("degrees(unwrap(arctan2(r_hip_y - l_hip_y, r_hip_x - l_hip_x)))",
                 ["l_hip", "r_hip"], "degrees", "signed, unwrapped", None),
    "hss_angle": ("shoulder_line - hip_line", ["l_sh", "r_sh", "l_hip", "r_hip"],
                  "degrees", "signed",
                  "the SCREENED hip-shoulder separation, read at foot plant. NOT the "
                  "adopted Hip-Shoulder Sep [O] row, which uses the gated, "
                  "median-filtered, signature-anchored recipe."),
    "elbow_flex": ("180 - angle3(shoulder, elbow, wrist) on the throwing arm",
                   ["r_sh", "r_el", "r_wr"], "degrees", "unsigned, 0 = straight", None),
    "elbow_incl": ("angle3(shoulder, elbow, wrist) on the throwing arm",
                   ["r_sh", "r_el", "r_wr"], "degrees", "unsigned",
                   "the included angle; elbow_flex is its complement"),
    "abd_throw": ("angle3(elbow, shoulder, hip) on the throwing side",
                  ["r_el", "r_sh", "r_hip"], "degrees", "unsigned", None),
    "abd_glove": ("angle3(elbow, shoulder, hip) on the glove side",
                  ["l_el", "l_sh", "l_hip"], "degrees", "unsigned", None),
    "hz_abd_throw": ("degrees(arctan2(r_el_y - r_sh_y, r_el_x - r_sh_x)) - shoulder_line",
                     ["r_sh", "r_el", "l_sh"], "degrees", "signed",
                     "upper-arm image angle measured against the shoulder line"),
    "hz_abd_glove": ("degrees(arctan2(l_el_y - l_sh_y, l_el_x - l_sh_x)) - shoulder_line",
                     ["l_sh", "l_el", "r_sh"], "degrees", "signed", None),
    "forearm_angle": ("degrees(unwrap(arctan2(r_wr_y - r_el_y, r_wr_x - r_el_x)))",
                      ["r_el", "r_wr"], "degrees", "signed, unwrapped",
                      "⚠ PRONATION HAS NO PROJECTION SIGNATURE -- rotation about the "
                      "forearm's own axis. Screened against this proxy purely so the "
                      "column is not left un-attempted."),
    "forearm_slot": ("degrees(arctan2(|r_wr_x - r_el_x|, r_el_y - r_wr_y))",
                     ["r_el", "r_wr"], "degrees", "unsigned",
                     "angle from vertical hinged at the ELBOW. Definition-matched to "
                     "the OBP arm_slot column, which is forearm-based -- deliberately "
                     "NOT the adopted shoulder-to-wrist arm slot."),
    "elbow_ext_velo_max": ("window max of gradient(elbow_incl) * fps",
                           ["r_sh", "r_el", "r_wr"], "degrees/s",
                           "SIGNED -- no absolute value before the maximum", None),
    "elbow_flex_max": ("window max of elbow_flex", ["r_sh", "r_el", "r_wr"],
                       "degrees", "unsigned", None),
    "shoulder_line_velo_max": ("window max of |gradient(shoulder_line)| * fps",
                               ["l_sh", "r_sh"], "degrees/s", "unsigned", None),
    "hip_line_velo_max": ("window max of |gradient(hip_line)| * fps",
                          ["l_hip", "r_hip"], "degrees/s", "unsigned", None),
    "shoulder_line_min": ("window MINIMUM of shoulder_line", ["l_sh", "r_sh"],
                          "degrees", "signed", None),
    "hz_abd_throw_max": ("window max of hz_abd_throw", ["r_sh", "r_el", "l_sh"],
                         "degrees", "signed", None),
    "knee_ext_velo_max": ("window max of gradient(angle3(lead_hip, lead_knee, "
                          "lead_ankle)) * fps",
                          ["l_hip", "l_kn", "l_an"], "degrees/s",
                          "SIGNED -- no absolute value before the maximum", None),
    "knee_ext_velo_at": ("gradient(angle3(lead_hip, lead_knee, lead_ankle)) * fps, "
                         "read at the anchor frame",
                         ["l_hip", "l_kn", "l_an"], "degrees/s", "signed", None),
    "knee_ext_fp_to_br": ("angle3(...)[hi] - angle3(...)[lo], the change in lead-knee "
                          "angle between the two window ends",
                          ["l_hip", "l_kn", "l_an"], "degrees", "signed",
                          "a DIFFERENCE BETWEEN TWO EVENTS, not a point read or an "
                          "extremum"),
    "torso_pelvis_timing": ("(argmax|gradient(shoulder_line)| - "
                            "argmax|gradient(hip_line)|) / fps over the window",
                            ["l_sh", "r_sh", "l_hip", "r_hip"], "seconds", "signed",
                            "an interval between two window maxima, not a point read"),
}

# row -> (observable, anchor key) exactly as rejected_gt_full_sweep.CANDS holds it
SCREENED = {
    "torso_anterior_tilt_mer": ("trunk_lean", "mer"),
    "torso_lateral_tilt_mer": ("trunk_lean", "mer"),
    "torso_rotation_mer": ("shoulder_line", "mer"),
    "elbow_flexion_mer": ("elbow_flex", "mer"),
    "glove_shoulder_abduction_mer": ("abd_glove", "mer"),
    "max_shoulder_external_rotation": ("shoulder_line", "mer"),
    "elbow_flexion_fp": ("elbow_flex", "fp"),
    "torso_anterior_tilt_fp": ("trunk_lean", "fp"),
    "torso_lateral_tilt_fp": ("trunk_lean", "fp"),
    "torso_rotation_fp": ("shoulder_line", "fp"),
    "pelvis_rotation_fp": ("hip_line", "fp"),
    "pelvis_lateral_tilt_fp": ("hip_line", "fp"),
    "pelvis_anterior_tilt_fp": ("hip_lean", "fp"),
    "rotation_hip_shoulder_separation_fp": ("hss_angle", "fp"),
    "shoulder_abduction_fp": ("abd_throw", "fp"),
    "glove_shoulder_abduction_fp": ("abd_glove", "fp"),
    "shoulder_horizontal_abduction_fp": ("hz_abd_throw", "fp"),
    "glove_shoulder_horizontal_abduction_fp": ("hz_abd_glove", "fp"),
    "shoulder_external_rotation_fp": ("shoulder_line", "fp"),
    "glove_shoulder_external_rotation_fp": ("shoulder_line", "fp"),
    "torso_lateral_tilt_br": ("trunk_lean", "rel"),
    "torso_rotation_br": ("shoulder_line", "rel"),
    "max_elbow_extension_velo": ("elbow_ext_velo_max", "rel"),
    "max_elbow_flexion": ("elbow_flex_max", "rel"),
    "max_shoulder_internal_rotational_velo": ("shoulder_line_velo_max", "rel"),
    "max_torso_rotational_velo": ("shoulder_line_velo_max", "rel"),
    "torso_rotation_min": ("shoulder_line_min", "rel"),
    "max_shoulder_horizontal_abduction": ("hz_abd_throw_max", "rel"),
    "timing_peak_torso_to_peak_pelvis_rot_velo": ("torso_pelvis_timing", "rel"),
    "elbow_pronation_fp": ("forearm_angle", "fp"),
    "arm_slot": ("forearm_slot", "rel"),
    # knee observables are built in rejected_gt_full_sweep AFTER observables() returns,
    # on the LEAD leg, so they are keyed on the lead-side joints
    "lead_knee_extension_angular_velo_max": ("knee_ext_velo_max", "rel"),
    "lead_knee_extension_angular_velo_fp": ("knee_ext_velo_at", "fp"),
    "lead_knee_extension_angular_velo_br": ("knee_ext_velo_at", "rel"),
    "lead_knee_extension_from_fp_to_br": ("knee_ext_fp_to_br", "rel"),
}

# rows the code itself flags as having NO first-order projection signature: screened
# against the best available proxy purely so no column is left un-attempted.
NO_SIGNATURE = {
    "elbow_pronation_fp": "rotation about the forearm's own axis",
    "glove_shoulder_external_rotation_fp": "rotation about the humeral long axis",
    "shoulder_external_rotation_fp": "rotation about the humeral long axis",
    "max_shoulder_external_rotation": "rotation about the humeral long axis",
}

WINDOWED = {"elbow_ext_velo_max", "elbow_flex_max", "shoulder_line_velo_max",
            "hip_line_velo_max", "shoulder_line_min", "hz_abd_throw_max",
            "torso_pelvis_timing", "knee_ext_velo_max", "knee_ext_fp_to_br"}

ANCHOR_OF = {"mer": "mer", "fp": "fp", "rel": "release"}
WINDOW_END_OFFSET_F360 = {"knee_ext_velo_max": 12}


def _expand_screened():
    """Turn the (observable, anchor) pairs into full SPEC entries."""
    for m, (obs, akey) in SCREENED.items():
        if m in SPEC:
            continue
        formula, joints, unit, sign, obs_note = OBS[obs]
        # Which observables are CALLABLES taking (lo, hi) in the source, hence read the
        # whole [fp, release] window and depend on BOTH events. Listed explicitly: a
        # suffix rule missed knee_ext_fp_to_br, which reads kang[hi] - kang[lo] and is
        # therefore fp-dependent despite being named for its release end. The registry's
        # anchor_type caught it -- keep the list in step with the source, not with names.
        windowed = obs in WINDOWED
        anchor = "release+fp" if windowed else ANCHOR_OF[akey]
        note = obs_note
        if m in NO_SIGNATURE:
            note = (f"⚠ NO PROJECTION SIGNATURE: {NO_SIGNATURE[m]}. Screened against "
                    f"the best available proxy so that the census leaves no column "
                    f"un-attempted. A failure here is not evidence that 2D measurement "
                    f"of this quantity was attempted and lost; there is no 2D "
                    f"observable for it. " + (obs_note or ""))
        SPEC[m] = row(
            observable=obs,
            joints=joints,
            formula=f"{obs} = {formula}",
            unit=unit,
            anchor=anchor,
            extremum=("window read over [fp, release]"
                      + (f", release end extended by "
                         f"{WINDOW_END_OFFSET_F360[obs]} frames at 360 Hz"
                         if obs in WINDOW_END_OFFSET_F360 else "")
                      + "; nanmax/nanmin over finite values, NaN if the window is empty"
                      if windowed else f"point read at the {akey} frame"),
            sign=sign,
            wrap="unwrapped inside the observable" if "unwrap" in formula else "none",
            refusal="NaN when the window holds no finite value" if windowed
                    else "inherits the angle3 silent fallback where angle3 is used",
            truth=m,
            impl="observables",
            note=note)


def main():
    _expand_screened()
    reg = pd.read_csv(os.path.join(V, "paper_registry.csv"))
    reg_rows = list(reg.metric_id)
    assert len(reg_rows) == 47, len(reg_rows)

    missing = [m for m in reg_rows if m not in SPEC]
    extra = [m for m in SPEC if m not in reg_rows]
    print(f"registry rows {len(reg_rows)}   specified {len(SPEC)}   "
          f"missing {len(missing)}   not in registry {len(extra)}")

    if extra:
        raise AssertionError(f"SPEC names rows absent from the registry: {extra}")

    # cross-checks on what IS specified
    idx = reg.set_index("metric_id")
    # The joint chain is TWO links, and the specification has to validate both:
    #   metrics.JOINTS   l_sh -> "left_shoulder"      (the 2D column prefix)
    #   MARKER_MAP       left_shoulder -> ["LSHO"]    (the c3d markers, centroided)
    # That chain IS the 15-point mapping the body table needs (edit M3). The two heels
    # are not in MARKER_MAP; they are loaded separately by obp_project.load_feet.
    from obp_project import MARKER_MAP
    from metrics import JOINTS
    bad = []
    for m, s in SPEC.items():
        r = idx.loc[m]
        if s["anchor"] != r.anchor_type:
            bad.append(f"{m}: anchor {s['anchor']!r} != registry {r.anchor_type!r}")
        if s["truth"] != r.truth_quantity:
            bad.append(f"{m}: truth {s['truth']!r} != registry {r.truth_quantity!r}")
        for j in s["joints"]:
            if j not in JOINTS:
                bad.append(f"{m}: joint key {j!r} not in metrics.JOINTS")
            elif JOINTS[j] not in MARKER_MAP:
                bad.append(f"{m}: {j!r} -> {JOINTS[j]!r} not in MARKER_MAP")
    chain = pd.DataFrame([dict(joint_key=k, column_prefix=v,
                               source_markers=" ".join(MARKER_MAP.get(v, [])),
                               construction="centroid of the listed markers")
                          for k, v in JOINTS.items()])
    chain.to_csv(os.path.join(V, "estimator_joint_chain.csv"), index=False)
    if bad:
        print("\nCROSS-CHECK FAILURES:")
        for b in bad:
            print("   " + b)
        raise AssertionError(f"{len(bad)} cross-check failures")
    print("cross-checks passed on the specified rows "
          "(anchor, truth, joint names)")

    out = []
    for m, s in SPEC.items():
        r = idx.loc[m]
        out.append(dict(metric_id=m, paper_name=r.paper_name, row_class=r.row_class,
                        **{k: v for k, v in s.items() if k != "joints"},
                        joints=" ".join(s["joints"])))
    pd.DataFrame(out).to_csv(os.path.join(V, "estimator_spec.csv"), index=False)
    pd.DataFrame(OPERATORS).T.to_csv(os.path.join(V, "estimator_operators.csv"))
    print(f"wrote estimator_spec.csv ({len(out)} rows) and estimator_operators.csv")

    if missing:
        print(f"\nSTILL TO SPECIFY ({len(missing)}):")
        for m in missing:
            print(f"   {m}")
        raise SystemExit("INCOMPLETE -- see the list above. This is deliberate: a "
                         "partial specification must not exit 0.")


if __name__ == "__main__":
    main()
