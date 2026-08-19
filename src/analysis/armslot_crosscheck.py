"""Arm Slot independent cross-check against OBP's forearm-based arm_slot column.

WHY THIS EXISTS. The adopted Arm Slot metric (shoulder->wrist) is validated
against 3D-direct truth and scores r2 = 1.000 at az90 -- but that is a CLEAN-
PROJECTION SYNTHETIC IDENTITY (at az90 the 2D coronal definition equals the
3D-direct definition), so it is a self-consistency check and cannot be presented
as evidence of real measurement capability. That made arm slot the weakest-
evidenced row in the adopted table.

The fix does NOT require a new metric. OBP's `arm_slot` column is an
INDEPENDENT truth (forearm-based, from OBP's own model, a different definition),
and the ALREADY-ADOPTED estimator recovers it at r2 ~ 0.83 from the front. That
is genuine evidence the identity cannot supply.

CLAUDE.md's rule still stands and is not violated here: the adopted metric's
TRUTH MAPPING remains 3D-direct (we do not validate a shoulder->wrist definition
against a forearm column). This script produces a clearly-labelled SECONDARY
cross-check, reported separately, never as the metric's headline number.

Not added as a 13th metric, deliberately: a definition-matched forearm estimator
recovers the same column slightly WORSE (0.803), correlates with the adopted
estimator at r = +0.956, and adds r2 = 0.004 of incremental explanatory power --
pure duplication (scratch/armslot_redundancy_check.py).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(HERE, "..", "stage3"))

import config                                    # noqa: E402
import metrics as M                              # noqa: E402
import obp_project as O                          # noqa: E402
from master_angle_table import load_feet         # noqa: E402
from hss_elevation_test import project_cam       # noqa: E402

ANCHOR_AZ, ANCHOR_EL = 90, 0
AZ = [0, 45, 90, 135, 180, 225, 270, 315]


def est_arm_slot(df, arm, rel):
    """The ADOPTED estimator, unchanged: shoulder->wrist vs vertical (deg)."""
    a = arm[0]
    sx, sy = M._xy(df, f"{a}_sh", M.JOINTS)
    wx, wy = M._xy(df, f"{a}_wr", M.JOINTS)
    r = int(rel)
    return float(np.degrees(np.arctan2(abs(wx[r] - sx[r]), (sy[r] - wy[r]))))


def main():
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    rows = []
    for r in md.itertuples(index=False):
        sp = r.session_pitch
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path) or sp not in poi.index:
            continue
        try:
            joints, fps = load_feet(path)
            arm = O.detect_throwing_arm(joints, fps)
            df0 = O.project_view(joints, azimuth_deg=0.0)
            rel = M.release_frame(df0, arm, fps, M.JOINTS)
            truth = float(poi.loc[sp, "arm_slot"])
            for az in AZ:
                df = project_cam(joints, az, 0)
                rows.append((az, sp, est_arm_slot(df, arm, rel), truth))
        except Exception:
            continue

    d = pd.DataFrame(rows, columns=["az", "session_pitch", "est", "truth"]).dropna()
    print("=" * 78)
    print("ARM SLOT - independent cross-check vs OBP forearm-based `arm_slot`")
    print("  (SECONDARY evidence; the adopted truth mapping stays 3D-direct)")
    print("=" * 78)
    print(f"  {'az':>5}{'r':>10}{'r2':>9}{'n':>7}")
    for az, g in d.groupby("az"):
        r = np.corrcoef(g.est, g.truth)[0, 1]
        mark = "   <-- adopted anchor" if az == ANCHOR_AZ else ""
        print(f"  {az:>5}{r:>+10.4f}{r*r:>9.3f}{len(g):>7}{mark}")

    a = d[d.az == ANCHOR_AZ]
    r = np.corrcoef(a.est, a.truth)[0, 1]
    print(f"\n  ANCHOR az{ANCHOR_AZ}/el{ANCHOR_EL}:  r = {r:+.4f}   r2 = {r*r:.3f}"
          f"   n = {len(a)}")
    print(f"    our estimate   {a.est.mean():6.2f} +- {a.est.std():5.2f} deg")
    print(f"    OBP arm_slot   {a.truth.mean():6.2f} +- {a.truth.std():5.2f} deg")
    print("\n  The negative sign is a REFERENCE-AXIS CONVENTION difference between the")
    print("  two definitions, not a disagreement; the magnitude of the association")
    print("  is what this cross-check reports.")

    dst = os.path.join(config.ROOT, "data", "outputs", "obp_validation",
                       "armslot_crosscheck.csv")
    d.to_csv(dst, index=False)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
