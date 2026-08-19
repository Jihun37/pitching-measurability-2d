"""Patent 도6 composite: the six metric-overlay panels tiled into ONE image.

Reads the PNGs already written by visualize_3d_2d.py and fig_metrics_patent.py
(it renders nothing itself, so the panels and the composite can never disagree)
and lays them out 3 per row (3 cols x 3 rows) with (a)-(i) captions:

  (a) side     / foot plant : stride
  (b) side     / release    : lead knee, trunk tilt, release height, wrist speed,
                              release extension  (knee ext. velo DE-ADOPTED 2026-07-24)
  (c) front    / release    : arm slot
  (d) front    / foot plant : stride angle
  (e) front    / MER        : torso lateral tilt @MER       (new 2026-07-24)
  (f) front    / MER        : glove-arm abduction @MER      (new 2026-07-24)
  (g) overhead              : hip-shoulder separation, pelvis rotation velocity
  (h) overhead / release    : torso transverse rotation @BR (new 2026-07-24)
  (i) side     / COM        : COG fwd velo (peak), COG velo @PKH

= the 15 deployment metrics. All source panels share one figure aspect, so the
grid needs no per-panel scaling.

Run (after the two renderers):
  conda activate diamond
  cd src\\viz
  python visualize_3d_2d.py
  python fig_metrics_patent.py
  python fig_metrics_grid_patent.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

INK = "#0E1B33"
SUB = "#5F5E5A"

VIZ = os.path.join(config.ROOT, "data", "outputs", "viz")

PANELS = [
    ("fig_metrics_footplant.png",      "(a) Side · foot plant",
     "stride"),
    ("fig_metrics_release_patent.png", "(b) Side · release",
     "lead knee · trunk tilt · release height ·\nwrist speed · release ext."),
    ("fig_metrics_front.png",          "(c) Front · release",
     "arm slot"),
    ("fig_metrics_stride_angle.png",   "(d) Front · foot plant",
     "stride angle"),
    ("fig_metrics_torso_lat_tilt.png", "(e) Front · MER",
     "torso lateral tilt @MER"),
    ("fig_metrics_glove_sh_abd.png",   "(f) Front · MER",
     "glove-arm abduction @MER"),
    ("fig_metrics_overhead.png",       "(g) Overhead",
     "hip-shoulder separation · pelvis rot. velo"),
    ("fig_metrics_torso_rot.png",      "(h) Overhead · release",
     "torso transverse rotation @BR"),
    ("fig_metrics_cog_patent.png",     "(i) Side · whole-body COM",
     "COG fwd velo (peak) · COG velo @PKH"),
]


def crop_white(img, pad=8):
    """Trim the uniform white border each panel is saved with, so every cell of
    the grid is filled by drawn content instead of margin (the panels are saved
    without bbox_inches='tight' to keep a common aspect, which leaves a lot of
    white around the narrow poses)."""
    a = img[..., :3] if img.ndim == 3 else img
    ink = (a < 0.99).any(axis=2) if a.ndim == 3 else (a < 0.99)
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return img
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, img.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, img.shape[1])
    return img[r0:r1, c0:c1]


def main():
    missing = [p for p, _, _ in PANELS if not os.path.exists(os.path.join(VIZ, p))]
    if missing:
        raise SystemExit("missing panel PNG(s) -- run visualize_3d_2d.py and "
                         "fig_metrics_patent.py first:\n  " + "\n  ".join(missing))

    fig, axes = plt.subplots(3, 3, figsize=(15.6, 17.1))
    for ax, (png, title, sub) in zip(axes.ravel(), PANELS):
        ax.imshow(crop_white(mpimg.imread(os.path.join(VIZ, png))))
        ax.axis("off")
        # imshow's equal aspect shrinks the axes box to the image aspect; anchor
        # it to the top of its grid cell so every panel title aligns on its row
        # (wide/short panels like the overhead views otherwise drop their title).
        ax.set_anchor("N")
        ax.set_title(title, fontsize=14, weight="bold", color=INK, pad=13)
        ax.text(0.5, -0.015, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=10.5, color=SUB)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.955, bottom=0.035,
                        wspace=0.02, hspace=0.16)
    out = os.path.join(VIZ, "fig_metrics_grid_patent.png")
    fig.savefig(out, dpi=150, facecolor="white")
    print("saved ->", out)


if __name__ == "__main__":
    main()
