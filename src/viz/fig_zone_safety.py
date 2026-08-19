"""
Figure (patent Fig. 8): confidence-gated deferral makes zone measurement safe (element F).

Successor to the station-classifier confusion figure. The viewpoint-zone system
decides PER METRIC from the kNN vote share (p_in = fraction of neighbours whose
(az, el) bin lies in the metric's valid zone) and measures only when p_in >= t,
deferring the rest with re-shoot guidance. Sweeping t trades coverage for
safety:
  - false-accept = out-of-zone cases wrongly measured (would output garbage)
  - coverage     = in-zone cases actually measured

Left: the per-metric safety-coverage trade-off as t sweeps 0.2 -> 1.0, with the
deployment operating point t=0.6 marked; the dashed line is the prior station
classifier's false-accept (0.131). Right: false-accept per metric at t=0.6, all
below the prior classifier (coverage annotated - narrow zones defer more, the
conservative direction). Data: zone_vote_curve.csv (n_eval=68544, GroupKFold).

Run:
  conda activate diamond
  cd src\\viz
  python fig_zone_safety.py
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
import config

INK = "#0E1B33"; GRAY = "#64748B"
PRIOR_FA = 0.131          # station classifier false-accept (§10.3, aug retrain)
OP = 0.6                  # deployment operating threshold

NAME = {
    "Lead Knee Angle [O]":  ("lead knee",       "#0FA3B1"),
    "Arm Slot [O]":         ("arm slot",        "#F2A900"),
    "Stride (anchor) [O]":  ("stride",          "#7C3AED"),
    "Trunk Tilt (ant) [O]": ("trunk tilt",      "#E0533D"),
    "Release Height [O]":   ("release height",  "#2E9E5B"),
    "Hip-Shoulder Sep [O]": ("hip-shoulder sep","#3B6EA5"),
}


def main():
    df = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR, "zone_vote_curve.csv"))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.6),
                                   gridspec_kw={"width_ratios": [1.32, 1.0]})

    # ---- Panel A: safety-coverage trade-off ----
    for m, (short, col) in NAME.items():
        s = df[df["metric"] == m].sort_values("t")
        axA.plot(s["coverage"], s["f_accept"], color=col, lw=1.9, marker="o",
                 ms=4, alpha=0.9, zorder=3, label=short)
        op = s[np.isclose(s["t"], OP)]
        axA.scatter(op["coverage"], op["f_accept"], s=95, color=col,
                    edgecolors="white", linewidths=1.4, zorder=5)
    axA.axhline(PRIOR_FA, ls="--", lw=1.2, color=GRAY, zorder=2)
    axA.text(0.015, PRIOR_FA + 0.006, f"prior station classifier  {PRIOR_FA:.3f}",
             fontsize=9, color=GRAY, va="bottom")
    axA.set_xlabel("coverage   (in-zone cases measured)  →", fontsize=10.5)
    axA.set_ylabel("false-accept   (out-of-zone wrongly measured)", fontsize=10.5)
    axA.set_title("safety–coverage trade-off   (threshold t: 0.2 → 1.0)",
                  fontsize=11, color=INK)
    axA.set_xlim(0, 1.02); axA.set_ylim(-0.01, 0.34)
    axA.grid(True, color="#E3E1D8", lw=0.7)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)
    axA.legend(loc="upper left", frameon=False, fontsize=9.5, ncol=2,
               title="●  operating point  t = 0.6", title_fontsize=9.5)

    # ---- Panel B: false-accept at the operating point ----
    op = df[np.isclose(df["t"], OP)].copy()
    op["short"] = op["metric"].map(lambda m: NAME[m][0])
    op["col"] = op["metric"].map(lambda m: NAME[m][1])
    op = op.sort_values("f_accept")
    y = np.arange(len(op))
    axB.barh(y, op["f_accept"], color=op["col"], height=0.62, zorder=3)
    for yi, (_, r) in zip(y, op.iterrows()):
        axB.text(r["f_accept"] + 0.003, yi,
                 f"{r['f_accept']:.3f}   (cov {r['coverage']*100:.0f}%)",
                 va="center", ha="left", fontsize=9, color=INK)
    axB.axvline(PRIOR_FA, ls="--", lw=1.2, color=GRAY, zorder=2)
    axB.text(PRIOR_FA, len(op) - 0.35, f"prior  {PRIOR_FA:.3f}", rotation=90,
             fontsize=8.5, color=GRAY, va="top", ha="right")
    axB.set_yticks(y); axB.set_yticklabels(op["short"], fontsize=10)
    axB.set_xlabel("false-accept at  t = 0.6", fontsize=10.5)
    axB.set_title("every metric safer than the prior classifier", fontsize=11,
                  color=INK)
    axB.set_xlim(0, 0.16)
    axB.grid(True, axis="x", color="#E3E1D8", lw=0.7)
    for sp in ("top", "right", "left"):
        axB.spines[sp].set_visible(False)

    fig.suptitle("Confidence-gated deferral keeps per-metric measurement safe "
                 "(re-shoot safety handle)", fontsize=12.5, fontweight="medium",
                 color=INK, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(config.OBP_VALIDATION_DIR, "fig_zone_safety.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
