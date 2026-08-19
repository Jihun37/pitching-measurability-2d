"""Patent 도7 composite: the seven real-video metric-overlay stills tiled into ONE
image (3 cols x 3 rows; the last two cells are blank). Reads PNGs already written
by validate_overlay.py / hss_overhead_real.py (unchanged panels a,c,d) and
fig_realvideo_new_metrics_patent.py (b without knee-velo, plus e,f,g), so it
renders nothing itself.

  (a) side     / foot plant : stride                        [angle00_00]
  (b) side     / release    : lead knee, trunk tilt, wrist speed, release height
                              (knee ext. velo DE-ADOPTED 2026-07-24)   [angle00_00]
  (c) front    / release    : arm slot                      [angle03_04]
  (d) overhead / HSS anchor : hip-shoulder separation (pelvis line = manual GT)
                                                             [pitching_overhead_02]
  (e) front    / MER        : torso lateral tilt @MER       [angle03_04]  (new)
  (f) front    / MER        : glove-arm abduction @MER      [angle03_04]  (new)
  (g) overhead / release    : torso transverse rotation     [pitching_overhead_02]  (new)

Run (after the still generators):
  conda activate diamond
  cd src\\tests
  python fig_realvideo_new_metrics_patent.py
  python fig_realvideo_grid_patent.py
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

INK = "#0E1B33"; SUB = "#5F5E5A"
OUT = os.path.join(config.ROOT, "data", "outputs")


def p(clip, name):
    return os.path.join(OUT, clip, name)


PANELS = [
    (p("angle00_00", "angle00_00_rtmp_footplant_sq.png"),
     "(a) Side · foot plant", "stride"),
    (p("angle00_00", "angle00_00_rtmp_patent_release_sq.png"),
     "(b) Side · release", "lead knee · trunk tilt · wrist speed · release height · release ext"),
    (p("angle00_00", "angle00_00_rtmp_cog_sq.png"),
     "(c) Side · whole-body COM", "COG fwd velo · COG velo @PKH"),
    (p("angle03_04", "angle03_04_rtmp_armslot_sq.png"),
     "(d) Front · release", "arm slot"),
    (p("angle03_04", "angle03_04_rtmp_strideangle_sq.png"),
     "(e) Front · foot plant", "stride angle"),
    (p("angle03_04", "angle03_04_rtmp_mer_sq.png"),
     "(f) Front · MER", "torso lateral tilt · glove-arm abduction @MER"),
    (p("pitching_overhead_02", "pitching_overhead_02_hss_manual_sq.png"),
     "(g) Overhead · HSS anchor", "hip-shoulder separation"),
    (p("pitching_overhead_02", "pitching_overhead_02_pelvisrot_sq.png"),
     "(h) Overhead · rot velo", "pelvis rotational velocity"),
    (p("pitching_overhead_02", "pitching_overhead_02_torsorot_sq.png"),
     "(i) Overhead · release", "torso transverse rotation @BR"),
]


def main():
    missing = [q for q, _, _ in PANELS if not os.path.exists(q)]
    if missing:
        raise SystemExit("missing still(s) -- run the generators first:\n  "
                         + "\n  ".join(missing))

    fig, axes = plt.subplots(3, 3, figsize=(16.5, 17.5))
    flat = axes.ravel()
    for ax, (png, title, sub) in zip(flat, PANELS):
        ax.imshow(mpimg.imread(png))
        ax.axis("off"); ax.set_anchor("N")
        ax.set_title(title, fontsize=15, weight="bold", color=INK, pad=11)
        ax.text(0.5, -0.02, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=11, color=SUB)
    for ax in flat[len(PANELS):]:
        ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.02,
                        wspace=0.03, hspace=0.14)
    out = os.path.join(OUT, "fig_realvideo_grid_patent.png")
    fig.savefig(out, dpi=140, facecolor="white")
    print("saved ->", out)


if __name__ == "__main__":
    main()
