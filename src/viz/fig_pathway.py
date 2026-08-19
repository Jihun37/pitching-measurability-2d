"""F2 -- the measurement pathway: how one 3D pitch becomes one cell of the map.

WHY THIS FIGURE EXISTS. Sec IV-A is titled "The Measurement Pathway" and says every result
follows the same one, but nothing showed it. Fig 1 carries the argument, Fig 2 the camera
geometry; neither shows how a pitch turns into a graded cell.

FOUR THINGS IT HAS TO SHOW, and three of them are easy to leave out:

  1. WHERE TRUTH COMES FROM. Every row is scored against its own point-of-interest
     column, so the evaluated set IS the kinematic column list. Until 2026-08-12 five
     further rows were compared against a value computed directly on the 3D markers,
     and this lane forked to show it; those rows are gone and the fork with them.

  2. THE AGGREGATION BOUNDARY. The pathway runs once per (pitch, quantity, viewpoint), but
     the correction, the agreement statistics and the grade are computed PER CELL, ACROSS
     PITCHERS. A single-file flow chart hides where per-pitch ends and per-cell begins,
     which is exactly where a reader misreads what n = 394 attaches to.

  3. HANDEDNESS REFLECTION HAPPENS AT LOAD TIME, before projection -- so azimuth is
     handedness-relative, not field-relative, everywhere downstream.

  4. THE FOLD STRUCTURE. The correction is fitted on OTHER pitchers and applied to the
     held-out one; the model family is chosen from per-fold votes that are themselves
     blind to the held-out pitcher.

Authored at BODY_W and saved without a tight bbox, so the authored width is the placed
width. Nothing is drawn outside the axes.

Run:  conda activate diamond; cd src\\viz; python fig_pathway.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", ".."):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
from fig_graded_map import INK, MUTE, BODY_W

OUT = os.path.join(config.ROOT, "data", "outputs", "viz")

C_EST = "#0C447C"        # estimate lane
C_TRUTH = "#0FA3B1"      # truth lane
C_CELL = "#334155"       # the per-cell stage
BAND_A = "#F1F5F9"
BAND_B = "#E2E8F0"

FS_BOX, FS_BAND, FS_NOTE = 5.9, 6.2, 5.2
FIG_H = 3.30


def box(ax, x, y, w, h, text, fc="white", ec=INK, lw=0.7, fs=FS_BOX, tc=INK,
        weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.01",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4, linespacing=1.35, weight=weight)
    return (x, y, w, h)


def arrow(ax, p, q, color=MUTE, lw=0.8, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=6,
                                 linewidth=lw, color=color, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(BODY_W, FIG_H))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---------------- bands -------------------------------------------------
    # Band captions sit INSIDE their band. Drawn outside the axes they were clipped,
    # and this repository's figures must place nothing beyond the axes: the saved PNG
    # is used at its authored width with no tight bbox, so anything outside is simply
    # lost rather than expanding the canvas.
    ax.add_patch(Rectangle((0.015, 0.435), 0.975, 0.545, facecolor=BAND_A,
                           edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.015, 0.120), 0.975, 0.290, facecolor=BAND_B,
                           edgecolor="none", zorder=0))
    ax.text(0.026, 0.958, "FOR EACH OF THE 394 PITCHES, AT ONE VIEWPOINT",
            ha="left", va="top", fontsize=FS_BAND, color=C_EST, weight="bold")
    ax.text(0.026, 0.395, "FOR EACH CELL — THE 394 PITCHES POOLED ACROSS 98 PITCHERS",
            ha="left", va="top", fontsize=FS_BAND, color=C_CELL, weight="bold")

    # ---------------- band A: two lanes ------------------------------------
    yE, yT, h = 0.760, 0.520, 0.130
    src = box(ax, 0.030, 0.610, 0.105, 0.155,
              "3D marker\ntrajectories\n(one pitch)", fc="white", ec=INK, lw=0.9)

    e1 = box(ax, 0.172, yE, 0.130, h,
             "reflect LHP → RHP\nAT LOAD TIME\nazimuth becomes\nhandedness-relative",
             ec=C_EST, fs=5.2)
    e2 = box(ax, 0.325, yE, 0.118, h,
             "orthographic\nprojection at\n(azimuth, elevation)", ec=C_EST)
    e3 = box(ax, 0.466, yE, 0.098, h, "15 image\npoints", ec=C_EST)
    e4 = box(ax, 0.587, yE, 0.140, h,
             "estimator read at\nits temporal anchor\nor window", ec=C_EST)
    e5 = box(ax, 0.750, yE, 0.135, h, "2D measurement", ec=C_EST,
             fc="#EAF1F9", weight="bold")

    # One arm, so the box states the count rather than a note downstream summing two.
    # Narrower than the fork it replaces, and back at the shared box font: the 5.2 pt
    # was there to fit three lines that no longer exist.
    t1 = box(ax, 0.172, yT, 0.235, h,
             "its point-of-interest column\n(42 rows)", ec=C_TRUTH)
    t2 = box(ax, 0.750, yT, 0.135, h, "matched\nreference value", ec=C_TRUTH,
             fc="#E6F6F8", weight="bold")

    for a, b in ((e1, e2), (e2, e3), (e3, e4), (e4, e5)):
        arrow(ax, (a[0] + a[2], yE + h / 2), (b[0], yE + h / 2), C_EST)
    arrow(ax, (src[0] + src[2], 0.715), (e1[0], yE + h / 2), C_EST)
    arrow(ax, (src[0] + src[2], 0.650), (t1[0], yT + h / 2), C_TRUTH)
    arrow(ax, (t1[0] + t1[2], yT + h / 2), (t2[0], yT + h / 2), C_TRUTH)

    # ---------------- band A -> band B -------------------------------------
    # the two lanes merge on the right, then the flow reverses into band B
    ax.text(0.805, 0.462, "one (estimate, truth) pair per pitch",
            ha="right", va="center", fontsize=FS_NOTE, color=MUTE, style="italic")
    arrow(ax, (0.8175, yE), (0.8175, 0.470), C_EST)
    arrow(ax, (0.8175, yT), (0.8175, 0.470), C_TRUTH)
    arrow(ax, (0.8175, 0.462), (0.8175, 0.365), C_CELL, lw=1.0)

    # ---------------- band B, flowing right to left ------------------------
    yC, hC = 0.170, 0.140
    c1 = box(ax, 0.612, yC, 0.205, hC,
             "leave-one-pitcher-out correction\n"
             "offset / ratio / linear,\nfitted on the OTHER pitchers", ec=C_CELL,
             fs=5.3)
    c2 = box(ax, 0.430, yC, 0.150, hC,
             "CCC, MAE,\nper-pitcher bias", ec=C_CELL)
    c3 = box(ax, 0.262, yC, 0.136, hC,
             "strong / moderate\n/ limited", ec=C_CELL)
    c4 = box(ax, 0.118, yC, 0.112, hC, "ONE cell", ec=C_CELL, fc="#DCE3EA",
             weight="bold")
    arrow(ax, (0.8175, 0.365), (0.8175, yC + hC / 2), C_CELL, style="-")
    arrow(ax, (0.8175, yC + hC / 2), (c1[0] + c1[2], yC + hC / 2), C_CELL)
    for a, b in ((c1, c2), (c2, c3), (c3, c4)):
        arrow(ax, (a[0], yC + hC / 2), (b[0] + b[2], yC + hC / 2), C_CELL)
    ax.text(0.714, yC - 0.012,
            "the model family is chosen from per-fold votes,\n"
            "each blind to its own pitcher",
            ha="center", va="top", fontsize=FS_NOTE, color=MUTE, style="italic",
            linespacing=1.3)

    # ---------------- band C ------------------------------------------------
    box(ax, 0.015, 0.014, 0.975, 0.082,
        "42 rows  ×  168 viewpoints  =  7,056 evaluated cells      →      "
        "the graded measurability map",
        ec=INK, lw=0.9, fs=6.4, weight="bold")
    arrow(ax, (0.174, yC), (0.174, 0.096), C_CELL, lw=1.0)

    out = os.path.join(OUT, "fig_pathway.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved -> {out}   {BODY_W:.2f} x {FIG_H:.2f} in")
    print("CAPTION material: 42 POI-column rows, one per kinematic column; "
          "per-pitch stages above the boundary, per-cell below; "
          "reflection precedes projection; correction is leave-one-pitcher-out.")


if __name__ == "__main__":
    main()
