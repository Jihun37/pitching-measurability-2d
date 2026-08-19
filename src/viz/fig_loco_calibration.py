"""Figure (Act 3, sections 6.1-6.2): absolute accuracy and LOCO calibration.

r2 measures trend, not agreement. Concordance (CCC) exposes systematic
projection bias, and a leave-one-cluster-out (LOCO) calibration asks whether that
bias generalises to unseen pitchers. Groups: DIRECT metrics are already accurate
(calibration is a no-op); CALIBRATE metrics carry a multiplicative or convention
bias that LOCO recovers (HSS 0.12 -> 0.76, stride 0.41 -> 0.92, torso rotation
0.02 -> 0.89); trunk tilt is PARTIAL. Arm slot is drawn as an IDENTITY (its 1.00
is definitional, not a measurement). The old PROXY group is empty since lead-knee
extension velocity was de-adopted on 2026-07-24.

Data: loco_calibration_gt_clean.csv (loco_calibration.py --suffix _gt --clean,
the GT-event paper convention). raw = uncalibrated CCC,
loco = best of ratio/offset/linear under leave-one-cluster-out. Clean-projection.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
import config

INK = "#0E1B33"; TEAL = "#0FA3B1"; VIOLET = "#7C3AED"; PROXY_C = "#E4572E"
PARTIAL_C = "#D69A16"; GREY = "#9AA3B2"; IDENT_C = "#9AA3B2"

# 2026-07-24: GT-clean convention, 16 adopted metrics. The PROXY group is now
# EMPTY -- its only member (Knee Ext Velo BR) was de-adopted rather than kept as
# a proxy. Colour kept so the legend still explains the retired category.
GROUP = {
    "Release Height": "DIRECT", "Lead Knee Angle": "DIRECT",
    "Pelvis Rot Velo": "DIRECT", "COG Velo @PKH": "DIRECT",
    "Elbow Flex @MER": "DIRECT", "Torso Lat Tilt @MER": "DIRECT",
    "Release Ext": "CALIBRATE", "Stride (anchor)": "CALIBRATE",
    "Wrist Speed": "CALIBRATE", "COG Fwd Velo": "CALIBRATE",
    "Hip-Shoulder Sep": "CALIBRATE", "Stride Angle": "CALIBRATE",
    "Glove Sh Abd @MER": "CALIBRATE", "Torso Rot @BR": "CALIBRATE",
    "Trunk Tilt (ant)": "PARTIAL",
    # kept in the figure, but flagged: at az90 the 2D coronal definition IS the
    # 3D-direct definition, so 1.00 is a synthetic identity (LEDGER caveat 1).
    "Arm Slot": "IDENTITY",
}
GC = {"DIRECT": TEAL, "CALIBRATE": VIOLET, "PARTIAL": PARTIAL_C,
      "PROXY": PROXY_C, "IDENTITY": IDENT_C}


def main():
    d = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR,
                                 "loco_calibration_gt_clean.csv"))
    d["metric"] = d.metric.str.replace(r"\s*\[O\]$", "", regex=True)

    rec = []
    for m, g in d.groupby("metric"):
        raw = g[g.model == "raw"].ccc.iloc[0]
        loco = g[g.model != "raw"].ccc.max()
        rec.append((m, raw, loco, GROUP.get(m, "DIRECT")))
    df = pd.DataFrame(rec, columns=["metric", "raw", "loco", "grp"]).sort_values("loco").reset_index(drop=True)

    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    for i, r in df.iterrows():
        c = GC[r.grp]
        ident = r.grp == "IDENTITY"
        ax.plot([r.raw, r.loco], [i, i], color=c, lw=2.4, alpha=0.5, zorder=1)
        ax.scatter(r.raw, i, s=46, facecolor="white", edgecolor=GREY, lw=1.6, zorder=2)
        ax.scatter(r.loco, i, s=64, color=c, zorder=3,
                   marker="D" if ident else "o",
                   edgecolor=IDENT_C if ident else "none",
                   facecolor="white" if ident else c, linewidth=1.6 if ident else 0)
        ax.text(r.loco + 0.012, i, f"{r.loco:.2f}", va="center", fontsize=8, color=INK)
        if r.loco - r.raw > 0.05:
            # a near-zero raw CCC would push its label off-axis into the metric
            # name, so flip it to the right of the marker instead
            if r.raw < 0.10:
                ax.text(r.raw + 0.014, i + 0.22, f"{r.raw:.2f}", va="center",
                        ha="left", fontsize=7.6, color=GREY)
            else:
                ax.text(r.raw - 0.012, i, f"{r.raw:.2f}", va="center",
                        ha="right", fontsize=7.6, color=GREY)

    ax.axvline(0.9, ls="--", color="0.5", lw=1.1)
    ax.text(0.905, 0.15, "CCC 0.9", fontsize=8.6, color="0.4", ha="left", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}  (identity)" if g == "IDENTITY" else m
                        for m, g in zip(df.metric, df.grp)], fontsize=9)
    for i, tick in enumerate(ax.get_yticklabels()):
        tick.set_color(GC[df.grp.iloc[i]])
    ax.set_xlim(0, 1.06); ax.set_xlabel("concordance (CCC)")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # legend by group
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor="white",
               markeredgecolor=GREY, markersize=8, label="raw (uncalibrated)"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor=TEAL, markersize=9,
               label="DIRECT (calibration no-op)"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor=VIOLET, markersize=9,
               label="CALIBRATE (LOCO recovers bias)"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor=PARTIAL_C, markersize=9,
               label="PARTIAL (already close, small gain)"),
        Line2D([0], [0], marker="D", color="white", markerfacecolor="white",
               markeredgecolor=IDENT_C, markeredgewidth=1.6, markersize=8,
               label="IDENTITY (2D def. = 3D def. at az90, not skill)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.6, loc="upper left")
    ax.set_title("r$^2$ is not agreement. Calibration recovers bias for some, not all",
                 fontsize=12, color=INK, weight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out = os.path.join(config.OBP_VALIDATION_DIR, "fig_loco_calibration_gt.png")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved ->", out)


if __name__ == "__main__":
    main()
