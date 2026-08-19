"""Which of the 16 adopted metrics are actually deployable, BY NAME.

Written because a count was quoted three different ways (16 measurable, "12 -> 14
deployable", "only Elbow Flex is GT-only") and those cannot all be true: 16 - 1
is 15. Counts hide which metric moved, so this prints the NAME SETS and their
differences, and attributes every gap to one of three causes:

  WIRING    the estimator exists but the deployment driver never calls it
  EVENT     the anchor event cannot be detected/proxied well enough in 2D
  REALVIDEO clean projection is fine but real-video transfer is unvalidated

Sources of truth (no hardcoded metric lists):
  paper map      data/outputs/obp_validation/angle_zone_sweep_gt_clean.csv
  detected map   data/outputs/obp_validation/angle_zone_sweep.csv
  driver         deploy/measure_auto -> viewpoint_zone_classify.ZONE_METRICS
                 plus the point metrics measure_auto reports separately
  estimators     stage2/metrics.compute_candidates keys

Run:  conda activate diamond; cd src\\analysis; python deployability_audit.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../deploy"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import io, re
import pandas as pd
import config
from viewpoint_zone_classify import ZONE_METRICS

def driver_metrics():
    """Every metric label measure_auto actually prints, parsed FROM ITS SOURCE.

    Hardcoding this list is what produced the wrong "8" on 2026-07-24: the
    ZONE_METRICS loop is only the first of three reporting blocks (the velocity
    points and the COG/release-extension block follow it)."""
    src = io.open(os.path.join(_HERE, "..", "deploy", "measure_auto.py"),
                  encoding="utf-8").read()
    return set(re.findall(r'"([A-Z][^"]*\[O\])"', src))


# 2026-07-24 findings, one line each (see GT_EVENT_MAP_HANDOFF 6e-6h).
# After the routing fix + wiring the only surviving gap is Elbow Flex; the rest
# are kept as history so a future regression is recognisable.
NOTES = {
    "Elbow Flex @MER": ("EVENT",
        "4.32 deg/frame at MER vs truth SD 8.07 deg: no anchor at 120 fps can "
        "hold it (best 0.64). Per-pitch deployment impossible; session mean 0.786."),
    # --- resolved 2026-07-24, listed so a reappearance is obviously a regression
    "Torso Lat Tilt @MER": ("RESOLVED",
        "release-30.6ms proxy + side strategy (LOCO CCC 0.884); wired."),
    "Glove Sh Abd @MER": ("RESOLVED",
        "release-30.6ms proxy + side strategy (LOCO CCC 0.891); wired."),
    "Torso Rot @BR": ("RESOLVED",
        "release-anchored all along; promoted out of gt_only_rows and wired."),
    "Stride Angle": ("RESOLVED",
        "wired with the frontal foot plant (compute_candidates uses "
        "foot_plant_frame_front when the view is frontal)."),
}


def names(csv):
    p = os.path.join(config.OBP_VALIDATION_DIR, csv)
    return set(pd.read_csv(p).metric.unique())


def show(title, s):
    print(f"\n{title}  ({len(s)})")
    for m in sorted(s):
        print(f"    {m}")


def main():
    paper = names("angle_zone_sweep_gt_clean.csv")
    detected = names("angle_zone_sweep.csv")
    driver = driver_metrics()
    cand = driver

    show("A. PAPER map, GT-clean (measurable)", paper)
    show("B. DETECTED map (detect-once deployment table)", detected)
    show("C. DRIVER: metrics measure_auto actually reports", driver)

    print("\n" + "=" * 74)
    print("DIFFERENCES")
    print("=" * 74)
    for lab, s in (("A - B  (in the paper map, not in the detected map)",
                    paper - detected),
                   ("A - C  (measurable but the driver does not report)",
                    paper - driver),
                   ("B - C  (in the detected map but not reported)",
                    detected - driver)):
        print(f"\n{lab}: {len(s)}")
        for m in sorted(s):
            key = m.replace(" [O]", "")
            cause, why = NOTES.get(key, ("?", "unclassified"))
            print(f"    {m:<24} {cause:<18} {why}")

    print("\n" + "=" * 74)
    print("SCOREBOARD (names, not counts)")
    print("=" * 74)
    blocked_event = {m for m in paper
                     if NOTES.get(m.replace(" [O]", ""), ("", ""))[0] == "EVENT"}
    print(f"  measurable (paper map)                 : {len(paper)}")
    print(f"  blocked by EVENT precision, per pitch  : {len(blocked_event)} "
          f"-> {sorted(blocked_event)}")
    print(f"  everything else is a WIRING or REALVIDEO gap, not a measurement one:"
          f" {len(paper) - len(blocked_event)}")
    print("\n  => the honest deployment ceiling after the routing fix AND wiring is")
    print(f"     {len(paper) - len(blocked_event)} of {len(paper)}, "
          f"NOT the 14 previously reported.")
    print("     Today's driver actually reports "
          f"{len(driver & paper)}: {sorted(driver & paper)}")


if __name__ == "__main__":
    main()
