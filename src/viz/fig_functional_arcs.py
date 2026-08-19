"""Figure (Act 2, section 5.4): operational resolution is arc-level, not 15 deg.

The 15-degree map is scientifically real but a single clip cannot place itself to
that precision (stride parallax, +-1 bin, uncorrectable, section 5.3). Intersecting
the map with what is identifiable leaves functional ARCS. Each arc is a run of
azimuth bins that share the same guaranteed metric menu (adopted metrics with
r2 >= 0.6 at el=0 in the deployment-honest grid). The arcs are not a chosen UI,
they are what map INTERSECT identifiability leaves.

Data: arc_partition.csv (scratch/arc_partition.py). Azimuth convention is
HANDEDNESS-RELATIVE (2026-07-23): az0 = the OPEN (throwing-arm) side, az90 =
front, az180 = glove/closed side, az270 = behind the pitcher. LHP are reflected
to RHP in the loader, so az0 is 3B for a RHP and 1B for a LHP.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

INK = "#0E1B33"
FAM_C = {"SIDE": "#0FA3B1", "AXIS": "#7C3AED", "BORDER": "#9Fb2C4", "DEAD": "#E1E5EA"}
NAME = {"SIDE0": "SIDE-OPEN", "SIDE4": "SIDE-CLOSED", "AXIS2": "FRONT", "AXIS6": "REAR"}


def short(menu):
    if not isinstance(menu, str) or not menu:
        return ""
    m = [x.replace(" [O]", "").replace("Lead Knee Angle", "Lead Knee")
          .replace("Release Height", "Rel Height").replace("Wrist Speed", "Wrist Spd")
         for x in menu.split("|")]
    return " · ".join(m)


def main():
    d = pd.read_csv(os.path.join(os.path.dirname(config.OBP_VALIDATION_DIR),
                                 "arc_partition.csv"))

    fig, ax = plt.subplots(figsize=(12.6, 3.6))
    y0, h = 0.0, 1.0
    for _, r in d.iterrows():
        c = FAM_C[r.family]
        lo, hi, w = r.az_lo, r.az_hi, r.width_deg
        center = (lo + hi) / 2 if hi >= lo else ((lo + hi + 360) / 2) % 360
        segs = []
        a, b = center - w / 2, center + w / 2   # tile span
        if a < 0:
            segs = [(a + 360, 360), (0, b)]
        elif b > 360:
            segs = [(a, 360), (0, b - 360)]
        else:
            segs = [(a, b)]
        for s0, s1 in segs:
            ax.add_patch(Rectangle((s0, y0), s1 - s0, h, facecolor=c,
                                   edgecolor="white", lw=1.4))
        # arc name inside wide bands, at the centre of the widest visible segment
        if r.family in ("SIDE", "AXIS"):
            s0, s1 = max(segs, key=lambda s: s[1] - s[0])
            lx = (s0 + s1) / 2
            ax.text(lx, y0 + h * (0.5 if len(segs) > 1 else 0.62),
                    NAME.get(r.arc, r.family), ha="center", va="center",
                    fontsize=10, weight="bold", color="white")
            if len(segs) == 1:  # skip menu subtext on the wrapped arc (legend covers it)
                ax.text(lx, y0 + h * 0.26, short(r.menu_guaranteed), ha="center",
                        va="center", fontsize=7.6, color="white")

    ax.set_xlim(0, 360); ax.set_ylim(-0.02, h + 0.02)
    ax.set_yticks([])
    ax.set_xticks([0, 90, 180, 270, 360])
    # Handedness-relative labels (2026-07-23): LHP are reflected to RHP, so az0
    # is the OPEN (throwing-arm) side for every pitcher -- 3B for a RHP, 1B for
    # a LHP. Field-base names would be wrong for half the bank.
    ax.set_xticklabels(["az0\nopen side", "az90\nfront", "az180\nglove side",
                        "az270\nbehind", "az360\nopen side"], fontsize=9)
    ax.set_xlabel("camera azimuth around the pitcher", fontsize=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=FAM_C["SIDE"], markersize=11,
               label="SIDE  (Lead Knee, Rel Height, Wrist Speed)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=FAM_C["AXIS"], markersize=11,
               label="FRONT / REAR  (Arm Slot, Rel Height)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=FAM_C["BORDER"], markersize=11,
               label="BORDER  (Rel Height only)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=FAM_C["DEAD"], markersize=11,
               markeredgecolor="0.7", label="DEAD  (no trustworthy metric)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.8, loc="upper center",
              bbox_to_anchor=(0.5, -0.32), ncol=4, handletextpad=0.4, columnspacing=1.2)

    ax.set_title("Operational resolution is arc-level, not 15° (map $\\cap$ identifiability)",
                 fontsize=12.5, color=INK, weight="bold", pad=12)

    out = os.path.join(config.OBP_VALIDATION_DIR, "fig_functional_arcs.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); print("saved ->", out)


if __name__ == "__main__":
    main()
