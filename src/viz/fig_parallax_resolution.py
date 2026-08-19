"""Figure (Act 2, section 5.3): the stride-parallax resolution limit.

A pitcher strides ~1.5 m toward home before release, so a real (pinhole) camera
sees the pitch from an effectively rotated bearing. The shift is a full 15-degree
bin at phone distance, and its size depends on BOTH camera distance and framing
(mound-centred vs release-centred), neither known at deployment, so a fixed
un-shift over/under-corrects. The 15-degree azimuth output is therefore ±1 bin in
principle, not by tuning.

Fraction of 20 OBP RHP pitches (camera at az0, el~0) whose recovered azimuth lands
one bin off az0. The orthographic control (no perspective) is the noise floor.
Source: scratch/perspective_stride_probe.py printed histogram (n=20).
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
import config

INK = "#0E1B33"; TEAL = "#0FA3B1"; VIOLET = "#7C3AED"

N = 20
# shifted-off-az0 counts from perspective_stride_probe.py (az0 hits below)
#   ortho 0:16  mound_d4 0:10  release_d4 0:3  mound_d6 0:14  release_d6 0:9
#   mound_d8 0:16  release_d8 0:12
DIST = [4, 6, 8]
MOUND = [(N-10)/N, (N-14)/N, (N-16)/N]     # mound-framed
RELEASE = [(N-3)/N, (N-9)/N, (N-12)/N]     # release-framed
ORTHO = (N-16)/N                            # orthographic control (perspective off)


def main():
    x = np.arange(len(DIST)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.bar(x - w/2, MOUND, w, color=TEAL, label="mound-framed")
    ax.bar(x + w/2, RELEASE, w, color=VIOLET, label="release-framed")

    ax.axhline(ORTHO, ls="--", color="0.45", lw=1.3)
    ax.text(1.0, 0.905, "dashed line = orthographic control (20%, no perspective)",
            fontsize=8.8, color="0.35", ha="center", va="center")

    for xi, m, r in zip(x, MOUND, RELEASE):
        ax.text(xi - w/2, m + 0.015, f"{m*100:.0f}%", ha="center", fontsize=9, color=INK)
        ax.text(xi + w/2, r + 0.015, f"{r*100:.0f}%", ha="center", fontsize=9, color=INK)

    ax.set_xticks(x); ax.set_xticklabels([f"{d} m\n(phone)" if d == 4 else f"{d} m"
                                          for d in DIST])
    ax.set_xlabel("camera distance")
    ax.set_ylabel("pitches reading one bin off az0")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
    ax.set_title("Stride parallax shifts the 15° bin, and the shift is uncorrectable",
                 fontsize=12, color=INK, weight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out = os.path.join(config.OBP_VALIDATION_DIR, "fig_parallax_resolution.png")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved ->", out)


if __name__ == "__main__":
    main()
