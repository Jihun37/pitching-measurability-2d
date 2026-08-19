"""Figure (III-B): the 15 joint centres the 2D pipeline consumes, drawn on one
real pitch projected to a single virtual camera.

Raw C3D markers are grey, the 13 joint centres teal, and the two heels amber
(they enter only the stride and foot-plant computations). One label per
anatomical name; every name except the head is a bilateral pair, so 7 pairs
plus the head give 15 points.

Markers and centres are projected together through obp_project.project_view so
both live in the same image frame as the measurements themselves.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(HERE, "..", "stage3"))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import config
import obp_project as O
import ezc3d
from master_angle_table import HEEL
from obp_gt_events import load_gt_events

INK = "#0E1B33"; TEAL = "#0FA3B1"; AMBER = "#F2A900"; GREY = "0.72"

AZ, EL = 35.0, 0.0          # slightly off the open side so L/R separate
SP = "1346_1"               # a control pitch, healthy landmarks (fp_outlier_check)
COL_IN = 3.3                # IEEE single-column width; the \includegraphics width

BONES = [("head", "left_shoulder"), ("head", "right_shoulder"),
         ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
         ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
         ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
         ("left_ankle", "left_heel"), ("right_ankle", "right_heel")]

# label, anchor joint, text position as (u, v) in the joint bounding box
# (u 0 = left, v 0 = top), and the horizontal alignment of the text. Positions
# are hand-placed into the empty regions of this pose; re-tune them if SP, AZ,
# EL or the frame changes.
LABELS = [("head",     "head",           0.68, -0.06, "left"),
          ("shoulder", "left_shoulder",  0.84,  0.04, "left"),
          ("elbow",    "right_elbow",    0.26,  0.25, "right"),
          ("wrist",    "right_wrist",    0.20,  0.05, "right"),
          ("hip",      "right_hip",      0.28,  0.44, "right"),
          ("knee",     "right_knee",     0.42,  0.72, "left"),
          ("heel",     "right_heel",     0.30,  0.90, "left"),
          ("ankle",    "right_ankle",    0.30,  1.03, "left")]


def load_raw(path):
    """joint centres (13 + 2 heels) and every raw marker, from one c3d."""
    c = ezc3d.c3d(path)
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    fps = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    pts = c["data"]["points"][:3]
    idx = {l: i for i, l in enumerate(labels)}

    def col(mk):
        if mk in idx:
            return idx[mk]
        hits = [l for l in labels if l.endswith("_" + mk) or l.endswith(":" + mk)]
        return idx[hits[0]] if len(hits) == 1 else None

    centres, used = {}, set()
    for name, mks in {**O.MARKER_MAP, **HEEL}.items():
        cs = [col(m) for m in mks]
        cs = [ci for ci in cs if ci is not None]
        if not cs:
            continue
        centres[name] = np.nanmean([pts[:, ci, :] for ci in cs], axis=0)
        used.update(cs)
    markers = {f"mk{ci}": pts[:, ci, :] for ci in sorted(used)}
    return centres, markers, fps


def main():
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    md = md.set_index("session_pitch")
    r = md.loc[SP]
    path = os.path.join(config.OBP_DATA_DIR, "c3d",
                        f"{int(r.user):06d}", r.filename_new)
    centres, markers, fps = load_raw(path)

    ev = load_gt_events().get(SP, {})
    f = int(ev.get("fp", 0)) or int(0.5 * centres["head"].shape[1])

    # project markers and centres together so they share one image frame
    df = O.project_view({**centres, **markers}, azimuth_deg=AZ, elevation_deg=EL)
    XY = {n: (float(df[f"{n}_x"][f]), float(df[f"{n}_y"][f]))
          for n in list(centres) + list(markers)}

    # Size the figure from the data so the axes exactly fills it. savefig must
    # NOT use bbox_inches="tight": cropping would shrink the PNG and LaTeX
    # would then scale it back up to 3.3in, enlarging every font with it.
    jx = [XY[n][0] for n in centres]; jy = [XY[n][1] for n in centres]
    x0, y0 = min(jx), min(jy)
    w, h = max(jx) - x0, max(jy) - y0
    xlo, xhi = x0 - 0.14 * w, x0 + 1.14 * w
    ylo, yhi = y0 - 0.14 * h, y0 + 1.16 * h
    fig = plt.figure(figsize=(COL_IN, COL_IN * (yhi - ylo) / (xhi - xlo)))
    ax = fig.add_axes([0, 0, 1, 1])

    mk = np.array([XY[n] for n in markers])
    ax.scatter(mk[:, 0], mk[:, 1], s=7, color=GREY, zorder=2,
               label="C3D marker")

    for a, b in BONES:
        if a in XY and b in XY:
            ax.plot([XY[a][0], XY[b][0]], [XY[a][1], XY[b][1]],
                    color=INK, lw=1.7, solid_capstyle="round", zorder=3)

    core = [n for n in centres if not n.endswith("heel")]
    cp = np.array([XY[n] for n in core])
    ax.scatter(cp[:, 0], cp[:, 1], s=26, color=TEAL, zorder=4,
               edgecolors="white", linewidths=0.6,
               label="joint centre (13)")
    hp = np.array([XY[n] for n in centres if n.endswith("heel")])
    if len(hp):
        ax.scatter(hp[:, 0], hp[:, 1], s=26, color=AMBER, zorder=5,
                   edgecolors="white", linewidths=0.6,
                   label="heel (2)")

    for text, anchor, u, v, ha in LABELS:
        if anchor not in XY:
            continue
        ax.annotate(text, XY[anchor], xytext=(x0 + u * w, y0 + v * h),
                    fontsize=7.5, color=INK, va="center", ha=ha, zorder=6,
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5,
                                    shrinkA=2, shrinkB=3))

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(yhi, ylo)                       # image y increases downward
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=6.8, frameon=False, ncol=3,
              handletextpad=0.4, borderpad=0.2, columnspacing=1.4)

    out = os.path.join(config.OBP_VALIDATION_DIR, "fig_joint_centres.png")
    fig.savefig(out, dpi=300)
    print(f"saved -> {out}   (pitch {SP}, frame {f}, az {AZ:g} el {EL:g})")


if __name__ == "__main__":
    main()
