"""F1 -- camera geometry in one figure: where the camera is put, and what a view can
know about where it is.

WHY THE MERGE (round 4 of the 2026-08-07 review). The identifiability panel was a full
body figure carrying NO quantitative claim: Sec III-C calls it "a premise of the map
rather than a result of it" and states outright that neither sign bit is measured anywhere
in the paper. A standalone figure for that is out of proportion, and decision D4 removed
the appendix it might otherwise have moved to. It is not cut, though -- the four-fold
ambiguity does not survive prose alone -- so it becomes panel (b) of the figure it belongs
with. Both panels answer the same question, which is what the camera index means.

  (a) the swept dome: 24 azimuths x 7 elevations = 168 viewpoints, handedness-relative
  (b) one pose, its four-fold azimuth family, and the two image-plane sign bits that
      separate them

BOTH HALVES ARE DRAWN BY THEIR OWN MODULES -- `fig_camera_setup_methods.draw` and
`fig_symmetry_signbits.draw_pose`, over `fig_symmetry_signbits.prepare` -- so this figure
and the two standalones cannot disagree.

Authored at BODY_W, saved without a tight bbox.

Run:  conda activate diamond; cd src\\viz; python fig_camera_geometry.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
from fig_graded_map import INK, BODY_W
import fig_camera_setup_methods as CAM
import fig_symmetry_signbits as SYM

OUT = os.path.join(config.ROOT, "data", "outputs", "viz")
FIG_H = 3.55
FS_PANEL = 7.6


def main():
    os.makedirs(OUT, exist_ok=True)
    projs, rel, lead, trail, span = SYM.prepare()

    fig = plt.figure(figsize=(BODY_W, FIG_H))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.06, 1.0],
                  left=0.0, right=0.995, top=0.940, bottom=0.045, wspace=0.02)

    # ---- (a) the swept dome ------------------------------------------------
    axa = fig.add_subplot(gs[0, 0], projection="3d")
    CAM.draw(axa, label_fs=6.4, note_fs=6.0, axis_fs=6.6)
    # A 3D axes places its axis labels and tick numbers OUTSIDE the subplot box, and at
    # half width they were clipped at the bottom while the z-label floated into the gap
    # between the panels. They are dropped rather than shrunk: the panel already names
    # the four cardinal azimuths, the overhead row and the pitch direction on the dome
    # itself, which says more than "X (pitch direction)" does.
    axa.set_xlabel(""); axa.set_ylabel(""); axa.set_zlabel("")
    axa.set_xticks([]); axa.set_yticks([]); axa.set_zticks([])
    axa.text2D(0.02, 1.045, "(a)  camera placement", transform=axa.transAxes,
               fontsize=FS_PANEL, color=INK, weight="bold", ha="left", va="top")

    # ---- (b) the four-fold family and the two sign bits --------------------
    sub = gs[0, 1].subgridspec(2, 2, wspace=0.04, hspace=0.16)
    axb = [fig.add_subplot(sub[i, j]) for i in range(2) for j in range(2)]
    signs = {}
    for ax, az in zip(axb, SYM.FAMILY):
        signs[az] = SYM.draw_pose(ax, projs[az], rel, lead, trail, span)
        # NO TICKS. These are projected pixel coordinates and the panel is about pose
        # chirality and the two signs, not about position -- at this size the numbers
        # were both uninformative and large enough to collide across panels.
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#CBD5E1")
    fig.text(0.523, 0.988, "(b)  what one view determines about its own azimuth",
             fontsize=FS_PANEL, color=INK, weight="bold", ha="left", va="top")

    out = os.path.join(OUT, "fig_camera_geometry.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved -> {out}   {BODY_W:.2f} x {FIG_H:.2f} in")
    print("sign bits per azimuth (stride direction, shoulder chirality):")
    for az, s in signs.items():
        print(f"   az {az:>3}: {s}")


if __name__ == "__main__":
    main()
