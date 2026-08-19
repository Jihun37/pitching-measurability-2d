"""Figure (Act 1, section 4.5): the event-precision wall.

Panel A, the controlled contrast. The same COM and the same kind of read (an
instantaneous forward speed) score 0.889 when read at peak knee height (event
error 0.2 frames) and 0.366 when read at foot plant (event error 6.3 frames).
Panel B, six shape features of the COM forward-velocity curve. With our detected
events they fail the 0.6 floor, with OBP's own event frames five of six jump to
0.88-0.99, so what fails is knowing WHEN to read, not the 2D COM itself.

Data: cog_curve_probe.csv (tests/cog_curve_probe.py); @PKH value from
within_pitcher_agreement.csv (COG Velo @PKH pooled r2). Clean-projection ceilings.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
import config

INK = "#0E1B33"; TEAL = "#0FA3B1"; VIOLET = "#7C3AED"; GREY = "#B8C0CC"

SHORT = {  # tidy display labels
    "COG Velo @FP": "velocity @FP",
    "COG Decel FP-BR": "decel FP→BR",
    "COG Peak Decel": "peak decel",
    "COG PeakT-FP (s)": "peak-time − FP",
    "COG Disp FP-BR": "displacement FP→BR",
    "COG Disp PKH-FP": "displacement PKH→FP",
}


def main():
    df = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR, "cog_curve_probe.csv"))
    df = df.sort_values("oracle_r2").reset_index(drop=True)

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(12.4, 5.0), gridspec_kw={"width_ratios": [1.0, 2.15]})

    # --- Panel A, controlled contrast (same COM, same read, different event) ---
    # HARDCODED headline pair -- re-render does NOT refresh these. Source:
    # angle_zone_sweep.csv (COG Velo @PKH anchor az0/el0) and
    # cog_curve_probe.csv (COG Velo @FP). Update on every regeneration.
    pkh_r2, fp_r2 = 0.889, 0.366
    xs = [0, 1]
    axA.bar(xs, [pkh_r2, fp_r2], width=0.62, color=[TEAL, VIOLET])
    axA.axhline(0.6, ls="--", color="0.45", lw=1.2)
    axA.text(-0.42, 0.62, "$r^2=0.6$", fontsize=9, color="0.35", va="bottom")
    for x, v, ev in ((0, pkh_r2, "0.2 f err"), (1, fp_r2, "6.3 f err")):
        axA.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=10.5, color=INK, weight="bold")
        axA.text(x, 0.03, ev, ha="center", fontsize=9, color="white", weight="bold")
    axA.set_xticks(xs); axA.set_xticklabels(["@ peak knee\nheight", "@ foot\nplant"], fontsize=9.5)
    axA.set_ylim(0, 1.0); axA.set_ylabel("$r^2$")
    axA.set_title("Same COM, same read,\ndifferent event precision", fontsize=11, color=INK, weight="bold")
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    # --- Panel B, our event vs OBP (perfect) event for six curve features ------
    y = np.arange(len(df)); h = 0.36
    axB.barh(y + h/2, df.oracle_r2, h, color=GREY, label="with OBP's own events (perfect timing)")
    axB.barh(y - h/2, df.anchor_r2, h, color=TEAL, label="with our detected events")
    axB.axvline(0.6, ls="--", color="0.45", lw=1.2)
    axB.text(0.615, 0.15, "$r^2=0.6$", fontsize=9, color="0.35", ha="left", va="center")
    for i, r in df.iterrows():
        axB.text(r.oracle_r2 + 0.008, i + h/2, f"{r.oracle_r2:.2f}", va="center", fontsize=8, color="0.4")
        axB.text(r.anchor_r2 + 0.008, i - h/2, f"{r.anchor_r2:.2f}", va="center", fontsize=8, color=INK)
    axB.set_yticks(y); axB.set_yticklabels([SHORT.get(c, c) for c in df.candidate], fontsize=9)
    axB.set_xlim(0, 1.05); axB.set_xlabel("$r^2$")
    axB.set_title("COM curve features are event-limited, not measurement-limited",
                  fontsize=11, color=INK, weight="bold")
    axB.legend(frameon=False, fontsize=8.6, loc="lower right")
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    fig.suptitle("The event-precision wall", fontsize=13.5, color=INK, weight="bold", y=1.02)
    out = os.path.join(config.OBP_VALIDATION_DIR, "fig_event_precision_wall.png")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved ->", out)


if __name__ == "__main__":
    main()
