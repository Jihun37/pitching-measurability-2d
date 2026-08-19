"""
⚠ SUPERSEDED 2026-07-28 — SUPPLEMENT ONLY, NOT A BODY FIGURE.

This draws the 16-metric r2 ZONE map. The paper's map is now GRADED and has
**35 rows** (29 strong-capable + 6 moderate-only); the r2 layer is only a
SCREENING pre-filter. ORDER below is frozen at the 2026-07-24 16-metric
convention and therefore CANNOT draw the current map.

Body figure = viz/fig_graded_map.py (fig_graded_map.png / fig_graded_strip.png).
Old renders archived: data/outputs/obp_validation/archive_stale_20260728/.
Authority for counts: docs/legacy_pre_dedup/FINAL_GATE_MAP_394.md.
"""
"""Single source of the angle-map figure layout: which metrics, in what order,
from which sweep file.

Both map figures (`angle_zone_fig.py` = zone outlines, `angle_zone_map_numbers.py`
= numbers-in-boxes) used to carry their own hardcoded metric list, and both
silently dropped every metric adopted after they were written. `check_coverage`
now makes that impossible: it compares this list against the sweep CSV and fails
loudly if the map contains a metric the figure does not draw.

Convention (2026-07-24): the paper map is the GT-event, gt_clean population
(n=394, 16 metrics). The detected-event sweep is the deployment layer and is NOT
what these figures render.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402

# the published map: GT landmark events, broken-foot-plant pitches dropped
SWEEP_CSV = "angle_zone_sweep_gt_clean.csv"


def sweep_path():
    return os.path.join(config.OBP_VALIDATION_DIR, SWEEP_CSV)


# (csv key, display name, stature-CONFOUNDED at el>0, region)
#
# CONFOUNDED = the metric is normalised by stature, so the normalisation breaks
# once the camera is raised: every el>0 cell is an artefact regardless of its r2
# (LEDGER section 1, caveat 4). Angle metrics are fine at any elevation.
#
# region drives the label colour: ground / front / elevated / overhead.
ORDER = [
    # --- sagittal, ground-level side view ---
    ("Lead Knee Angle [O]",     "Lead knee angle",    False, "ground"),
    ("Stride (anchor) [O]",     "Stride length",      True,  "ground"),
    ("Trunk Tilt (ant) [O]",    "Trunk tilt (ant.)",  False, "ground"),
    ("Release Height [O]",      "Release height",     True,  "ground"),
    ("Release Ext [O]",         "Release ext.",       True,  "ground"),
    ("Wrist Speed [O]",         "Wrist speed",        True,  "ground"),
    ("COG Fwd Velo [O]",        "COG fwd velo",       True,  "ground"),
    ("COG Velo @PKH [O]",       "COG velo @PKH",      True,  "ground"),
    # --- front, ground level ---
    ("Arm Slot [O]",            "Arm slot",           False, "front"),
    ("Stride Angle [O]",        "Stride angle",       False, "front"),
    # --- intermediate elevations (all adopted 2026-07-24, all at MER) ---
    ("Glove Sh Abd @MER [O]",   "Glove sh. abd @MER", False, "elevated"),
    ("Torso Lat Tilt @MER [O]", "Torso lat tilt @MER", False, "elevated"),
    ("Elbow Flex @MER [O]",     "Elbow flex @MER",    False, "elevated"),
    # --- overhead ---
    ("Hip-Shoulder Sep [O]",    "Hip-shoulder sep",   False, "overhead"),
    ("Pelvis Rot Velo [O]",     "Pelvis rot. velo",   False, "overhead"),
    ("Torso Rot @BR [O]",       "Torso rot @BR",      False, "overhead"),
]

# Velocity metrics get a single validated viewpoint, not a zone: a derivative
# amplifies jitter, so their zones did not survive the deployment-chain noise
# probe (zone_edge_noise_velo). Values are the pre-specified paper anchors
# (absacc_table.ADOPTED_VIEW), as (el, az).
POINT_ONLY = {
    "Wrist Speed [O]":     [(0, 0)],
    "COG Fwd Velo [O]":    [(0, 0)],
    "COG Velo @PKH [O]":   [(0, 0)],
    "Pelvis Rot Velo [O]": [(85, 0)],
}

LABEL_COLOR = {"ground": "#2C2C2A", "front": "#2C2C2A",
               "elevated": "#8A6D2F", "overhead": "#B07D18"}


def check_coverage(df):
    """Fail loudly if the sweep carries a metric this figure would not draw.

    Re-rendering a figure does NOT update a hardcoded metric list -- this is the
    guard that turns that silent drop into an error (CLAUDE.md warning)."""
    drawn = {k for k, *_ in ORDER}
    present = set(df["metric"].unique())
    missing = present - drawn
    absent = drawn - present
    if missing:
        raise SystemExit(
            f"{SWEEP_CSV} has {len(missing)} metric(s) not in map_metrics.ORDER: "
            f"{sorted(missing)}\nAdd them to ORDER (and POINT_ONLY if velocity) "
            f"before re-rendering.")
    if absent:
        raise SystemExit(
            f"map_metrics.ORDER lists {len(absent)} metric(s) absent from "
            f"{SWEEP_CSV}: {sorted(absent)}\nDrop them or regenerate the sweep.")
