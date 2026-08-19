"""M7 -- specification of the Sec VIII EVENT DETECTORS, and of `p2f`.

WHY SEPARATELY FROM estimator_spec. An estimator answers "what is the value at this
frame"; a detector answers "which frame". Sec VIII compares detectors, so Table IV's
numbers are only reproducible if the detectors themselves are written down. Sec VIII
reports no cell counts (decision of 2026-08-06) but it DOES report quantitative detector
error, and that has to be reproducible.

DESTINATION (decision D4): Sec VIII prose. No supplement, no appendix.

SOURCES, all live and none touched by the freeze:
    metrics.release_frame                     side and frontal release
    metrics.foot_plant_frame                  side foot plant, and the frontal dispatch
    metrics.foot_plant_frame_front_speed      adopted frontal foot plant
    metrics.foot_plant_frame_front            y-floor frontal foot plant, ROLLBACK ONLY
    metrics._is_frontal                       the spelling test
    analysis.event_error_sweep                Table IV and the p1f/p2f/p3f definitions

Run:  conda activate diamond; cd src\\analysis; python detector_spec.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../stage2"):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config

V = config.OBP_VALIDATION_DIR
FPS_C3D = 360.0
PHONE_FPS = 120.0

DETECTORS = {
    "release / side": dict(
        source="metrics.release_frame(view='side')",
        signal="throwing-wrist point speed, speed2 = hypot(dx, dy) * fps "
               "(PLAIN FIRST DIFFERENCE, not Savitzky-Golay)",
        joints="throwing wrist",
        smoothing="none on the projected map. On real video an optional two-stage "
                  "refinement is available: the SMOOTHED argmax locates the peak "
                  "spike-robustly, then the RAW series refines within +/-0.03 s. OBP "
                  "projections are passed without raw_df, so every published number "
                  "here is single-stage.",
        candidates="none -- a single whole-clip argmax",
        selection="frame of the maximum wrist speed over the entire clip",
        thresholds="none",
        fallback="none",
        note="Ball velocity is in-plane laterally, so the peak is slot-invariant."),

    "release / frontal": dict(
        source="metrics.release_frame(view='frontal')",
        signal="arm extension, hypot(wrist - shoulder), searched backward from the "
               "SIDE detector's answer",
        joints="throwing wrist, throwing shoulder",
        smoothing="as above",
        candidates="the window [pk - 0.20 s, pk], where pk is the side detector's "
                   "wrist-speed peak",
        selection="frame of maximum arm extension inside that window",
        thresholds="win_s = 0.20 s",
        fallback="returns pk unchanged when the window is entirely NaN",
        note="⚠ NOT AN INDEPENDENT METHOD. The frontal detector STARTS from the side "
             "detector's output and moves it backward. Its failure at an open-side "
             "station is therefore a failure of that displacement, not of a separate "
             "algorithm -- which sharpens rather than weakens the transfer claim. "
             "Release precedes the frontal speed peak by about 30 ms, so a 200 ms "
             "window contains it while excluding earlier extended-arm phases."),

    "foot plant / side": dict(
        source="metrics.foot_plant_frame(view='side')",
        signal="lead-ankle forward travel and vertical settling",
        joints="lead ankle",
        smoothing="none; vertical speed is |diff(y)| * fps with a zero prepended",
        candidates="frames f in [3, release) meeting ALL THREE of: forward travel "
                   "> 0.70 of its own maximum, normalised height ynorm > 0.97 "
                   "(i.e. within 3 % of the lowest the ankle reaches), and vertical "
                   "speed < 0.15 of its own peak",
        selection="the FIRST such frame",
        thresholds="0.70 forward travel, 0.97 ynorm, 0.15 vertical speed",
        fallback="release - 0.13 s when no candidate qualifies, or when the ankle "
                 "shows no forward travel at all (max |fwd| < 1e-6)",
        note="The sign of forward travel is inferred per clip: if the negative "
             "excursion dominates, the axis is flipped."),

    "foot plant / frontal": dict(
        source="metrics.foot_plant_frame_front_speed  (adopted 2026-07-26)",
        signal="total lead-ankle image speed |v| = hypot(vx, vy), both components a "
               "Savitzky-Golay first derivative",
        joints="lead ankle",
        smoothing="SG derivative, window = max(5, round(0.03 s * fps)) forced odd and "
                  "capped at the series length, POLYORDER 3, deriv 1, delta 1/fps, "
                  "mode 'interp'",
        candidates="the scan window [max(1, pkh, release - 0.5 s), release - 3 frames]",
        selection="scan BACKWARD for the last frame whose |v| exceeds the threshold; "
                  "the plant is the frame immediately after it",
        thresholds="tau = 0.25 of the window's own peak |v|; guard = 3 frames; "
                   "back_s = 0.5 s",
        fallback="release - 0.13 s when peak knee height cannot be found, the window "
                 "is shorter than 4 frames, or the segment is all-NaN; the window "
                 "start when the peak is ~0 or nothing exceeds the threshold",
        note="⚠ A planted foot has near-zero image velocity regardless of leg-lift "
             "height, so unlike the retired y-floor detector this makes no assumption "
             "about the windup fraction -- which is why it wins on set/stretch "
             "deliveries. ⚠ IT REQUIRES A CORRECT RELEASE: the guard is "
             "release-anchored, so a mis-detected early release corrupts the plant. "
             "That is a release-detector limitation, not a foot-plant one."),

    "foot plant / frontal (retired)": dict(
        source="metrics.foot_plant_frame_front  -- ROLLBACK REFERENCE ONLY",
        signal="pkh-anchored y-floor",
        joints="lead ankle",
        smoothing="n/a",
        candidates="a window at k = 0.80 of the peak-knee-height-to-release interval",
        selection="y-floor crossing",
        thresholds="k = 0.80, wfrac = 0.10",
        fallback="n/a",
        note="NOT ADOPTED. Retained for rollback. It assumes a windup leg-lift "
             "fraction and lands 6-10 frames late on set-position deliveries."),
}

VIEW_GATE = dict(
    source="angle_zone_sweep.release_view(az, el)",
    rule="frontal if 60 <= az <= 120 AND el <= 15, side everywhere else -- including "
         "the entire rear half and every elevation at or above 30 degrees",
    history="The elevation bound was tightened from el <= 60 to el <= 15 on 2026-07-24. "
            "Scored against the released landmark rather than our own az0 detection, "
            "the frontal detector COLLAPSES above el 15: at az 90-120 x el 30-60 its "
            "release IQR is about 200 ms where side is 3 ms, and the older rule routed "
            "exactly those ten cells to frontal. Measured cost of the old rule: torso "
            "lateral tilt at MER read r2 0.007 with frontal against 0.798 with side at "
            "its az90/el30 anchor.",
    open_exception="NOT ADOPTED: at el 0 the rear sector az 255-285 is an exact u-flip "
                   "mirror of the front and genuinely prefers frontal, but it is kept "
                   "out to keep the rule simple.",
    stations="The gate assigns S1 (az0/el0) -> side, S2 (az90/el0) -> frontal, "
             "S3 (az0/el85) -> side. Table IV shows the gate picking the LOWER-error "
             "detector at all three. That is consistent with Sec VIII's own reasoning: "
             "the existential claim is that no FIXED choice is right everywhere, and "
             "the pipeline's answer to that is precisely a viewpoint gate -- which is "
             "why a count taken over the grid would report the dispatcher's coverage "
             "rather than a property of the problem.")

SHARED = {
    "_is_frontal": dict(
        rule="a view string is frontal when it lower-cases to 'front' or 'frontal'",
        note="⚠ A REPRODUCTION TRAP. release_view() returns 'frontal' while "
             "foot_plant_frame once accepted only 'front', so the frontal branch was "
             "silently bypassed and the SIDE foot-plant detector ran at frontal "
             "stations. One spelling test now serves every view-aware detector."),
    "quiet window": dict(
        rule="QUIET_THR = 0.20 of the trail ankle's own peak speed; the crossing must "
             "HOLD for QUIET_GUARD = 5 frames at the clip's own rate",
        note="ONE definition shared by trail_anchor_x and trail_quiet_len so they "
             "cannot drift; each carried its own literal 0.20 until 2026-07-27. This "
             "is what the Release Ext refusal (min_quiet_s = 0.10 s) tests, so the "
             "estimator's only explicit NaN rule and the detector layer share a "
             "constant."),
}

METRICS = {
    "MAE": "mean absolute detector error against the released ground-truth landmark, "
           "reported in milliseconds",
    "p1f/p2f/p3f": "the share of pitches whose absolute error is within 1, 2 or 3 "
                   "PHONE frames, one phone frame = 1000/120 = 8.333... ms = three "
                   "c3d frames at the 360 Hz reference. So the boundaries are "
                   "p1f 8.333 ms (3 frames), p2f 16.666... ms (6 frames), "
                   "p3f 25.0 ms (9 frames). ⚠ Write the p2f boundary as the exact "
                   "fraction. err_ms is a multiple of 1000/360, and TWO phone frames "
                   "is exactly six c3d frames = 16.666... ms whose next decimal is a "
                   "6, so ANY finite decimal rounding rounds UP and lands above the "
                   "threshold -- %.4g gives 16.67 and even %.10g gives 16.66666667. "
                   "Every pitch sitting exactly on the boundary then drops out of a "
                   "p2f recomputed from the dump, while p1f (8.333) and p3f (25.0) "
                   "round the safe way and hide the defect. The dump is written with "
                   "no float_format for this reason.",
    "p_within_33ms": "a separate column using a FLAT 33 ms. It is not p2f and not any "
                     "pNf boundary (four phone frames would be 33.333... ms).",
}


def main():
    rows = []
    for name, d in DETECTORS.items():
        rows.append(dict(detector=name, **d))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(V, "detector_spec.csv"), index=False)

    print("=" * 78)
    print("SEC VIII DETECTOR SPECIFICATION")
    print("=" * 78)
    for name, d in DETECTORS.items():
        print(f"\n{name}\n  source     : {d['source']}")
        for k in ("signal", "candidates", "selection", "thresholds", "fallback"):
            print(f"  {k:<11}: {d[k]}")
    # the console here is cp949; keep stdout ASCII and leave the marks in the CSVs
    def p(s):
        print(str(s).replace("⚠", "!!"))

    p("\n" + "=" * 78)
    p("VIEW GATE")
    for k, v in VIEW_GATE.items():
        p(f"  {k:<15}: {v}")
    p("\nSHARED RULES")
    for k, v in SHARED.items():
        p(f"  {k}: {v['rule']}")
    p("\nERROR METRICS")
    for k, v in METRICS.items():
        p(f"  {k}: {v}")
    print(f"\none phone frame = {FPS_C3D / PHONE_FPS:.0f} c3d frames = "
          f"{1000.0 / PHONE_FPS:.6f} ms   (p2f boundary = "
          f"{2000.0 / PHONE_FPS:.6f} ms)")

    tbl = os.path.join(V, "event_error_map.csv")
    if os.path.exists(tbl):
        t = pd.read_csv(tbl)
        print(f"\nevent_error_map.csv present: {len(t)} rows, "
              f"columns {list(t.columns)[:10]}")
    p("\nwrote detector_spec.csv")


if __name__ == "__main__":
    main()
