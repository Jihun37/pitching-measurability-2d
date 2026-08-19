"""Fig. 2 of section VIII: the release detector does not transfer across viewpoints.

One delivery of the frozen population, projected to the two ground-level stations. At
each station BOTH detectors are run. Dark = the pose at the ground-truth release frame;
coloured = the pose at the frame that detector returned. The diagonal coincides, the
off-diagonal does not.

COLOUR IS THE ROUTING RULE, NOT THE OUTCOME. Green marks the detector Table A1 selects
at that station (`is_ruled`), red the one transferred in from the other station, so the
figure and the table say the same thing by construction. The millisecond figure carries
the accuracy on its own; a green panel is not a claim of a small error.

THE PITCH IS NOT A FREE CHOICE. `3055_3` was SELECTED and the search is settled
(2026-08-07) -- do not run it again. Of the 394, the 35 whose two ruled cells sit at or
one frame off the 5.6 ms floor were scored by PROJECTED joint displacement between the
ground-truth and the detected frame, because that, not the millisecond figure, is what
the eye reads in a stick figure. `3055_3` won at 0.118 m of worst-panel separation.
`1779_3`, the only pitch of the 394 whose two ruled cells are BOTH at the floor, reads
0.130 m and was the incumbent until the visual comparison was made; the ms values are
printed on the panels either way, so the tighter drawing wins.

WHY THE GREEN PANELS STILL DO NOT COINCIDE EXACTLY -- and no pitch makes them:

  * the residual is the THROWING WRIST and nothing else. Max displacement equals wrist
    displacement in every one of the 35, while the mean over the 13 drawn joints is
    0.018-0.033 m: torso, legs and glove arm already coincide;
  * there is a floor and it is not noise. At BOTH ruled cells all 394 errors are
    NEGATIVE and none is under 5.6 ms: the ruled detector is always two or more frames
    early, a definitional offset (median -11.1 ms, range -5.6 to -19.4). The wrist
    travels about 35 m/s at release, so two frames is ~0.19 m of true motion and
    ~0.12 m once the side camera projects it. A perfectly coincident green panel does
    not exist in this population, and the figure should not be asked for one.

DO NOT go looking for a pitch that makes the S2 side cell visibly wrong either: the
population says it is not. Its MAE is 16.7 ms against the ruled detector's 10.8, the
1.5x asymmetry section VIII-A ¶3 asserts, and the pitches where it exceeds 50 ms number
ONE in 394. The figure shows the catastrophic direction only; the reversal itself is
Table A1's to carry.

LAYOUT. Every panel is drawn at ONE common scale (m per inch), so a pose that looks
bigger IS bigger -- per-panel autoscaling made the S2 poses look half the size of the
S1 poses for no reason.

SINGLE_COL decides the rest, because it decides where the label goes; see the constants.
  True  -> 3.50 x 4.29 in, a `figure` in one column, labels above the panels.
  False -> 7.16 x 5.14 in, a `figure*` across both, labels in a strip beside each pose,
           where `check_no_overlap` asserts in display coordinates that no label touches
           a bone. Only an in-panel label can collide, so the check is a no-op at column
           width.
The full-width form was itself the fix for an autoscaled 2 x 2 that ran 7.16 x 7.0.

Run:  conda activate diamond
      cd src\\viz
      python fig_release_transfer.py
"""
import os, sys

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../stage2", "../stage3", "../analysis"):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
import obp_project as O
import metrics as M
from master_angle_table import load_feet
from angle_zone_sweep import project_cam
from obp_gt_events import load_gt_events
from fig_graded_map import INK, BODY_W

V = config.OBP_VALIDATION_DIR
OUT = os.path.join(config.ROOT, "data", "outputs", "viz")

SP = "3055_3"
STATIONS = [(0, 0, "S1  open side (az 0$^\\circ$, el 0$^\\circ$)"),
            (90, 0, "S2  frontal (az 90$^\\circ$, el 0$^\\circ$)")]
DETS = ["side", "frontal"]
RULED = {(0, 0): "side", (90, 0): "frontal"}   # Table A1's bold cells

GT_C, OWN_C, XFER_C = "#1a1a1a", "#1a7a1a", "#c0392b"

# WHERE THE LABEL GOES IS WHAT DECIDES THE WIDTH THIS FIGURE CAN LIVE AT.
# A strip beside the pose costs width and buys height, which is the right trade at
# \textwidth: 1.15 in out of 6.81 leaves the poses plenty. At \columnwidth there are
# only 3.15 in of axes to share between two panels, so two strips would take three
# quarters of it and the poses would come out under half an inch. Above the pose the
# label costs height instead -- 0.30 in a row -- and the poses keep the full width.
SINGLE_COL = True
FIG_W = 3.50 if SINGLE_COL else BODY_W
BAND_IN = 0.0 if SINGLE_COL else 1.15   # label strip beside the pose
HEAD_IN = 0.20 if SINGLE_COL else 0.0   # label band above the pose: the
                      # title sits just above its axes, so anything past the
                      # line height is dead space between the rows
PAD_M = 0.06          # margin around the poses, metres, so no line touches an edge
LEFT_IN, RIGHT_IN = 0.30, 0.05
LEGEND_IN, BOTTOM_IN = (0.50, 0.02) if SINGLE_COL else (0.30, 0.02)
LW = FIG_W / BODY_W   # bones are drawn in points, so they must shrink with the figure
                      # or a half-width figure gets a skeleton twice as heavy

BONES = [("head", "left_shoulder"), ("head", "right_shoulder"),
         ("left_shoulder", "right_shoulder"),
         ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
         ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
         ("left_hip", "right_hip"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle")]
NODES = sorted({j for b in BONES for j in b})


def pose_xy(df, f):
    r = df.iloc[int(f)]
    return (np.array([float(r[j + "_x"]) for j in NODES]),
            np.array([float(r[j + "_y"]) for j in NODES]))


def skeleton(ax, df, f, col, lw, z):
    r = df.iloc[int(f)]
    for a, b in BONES:
        ax.plot([r[a + "_x"], r[b + "_x"]], [r[a + "_y"], r[b + "_y"]], "-",
                color=col, lw=lw, zorder=z, solid_capstyle="round")


def panels():
    """The four cells: geometry, frames and errors, with the errors checked against the
    dump Table A1 is computed from, so the figure cannot drift away from the table."""
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    row = md[md.session_pitch == SP].iloc[0]
    frozen = pd.read_csv(os.path.join(V, "population_frozen.csv"))
    assert SP in set(frozen.session_pitch.astype(str)), "%s is not in the frozen 394" % SP
    joints, fps = load_feet(os.path.join(config.OBP_DATA_DIR, "c3d",
                                         "%06d" % int(row.user), row.filename_new))
    arm = O.detect_throwing_arm(joints, fps)
    rel_gt = int(load_gt_events()[SP]["rel"])
    print("%s  fps=%g  arm=%s  GT release f%d" % (SP, fps, arm, rel_gt))

    dump = pd.read_csv(os.path.join(V, "event_error_map_pairs.csv.gz"))
    dump = dump[(dump["event"] == "rel") & (dump["sp"] == SP)]

    out = []
    for i, (az, el, sname) in enumerate(STATIONS):
        df = project_cam(joints, az, el)
        for j, d in enumerate(DETS):
            rv = int(M.release_frame(df, arm, fps, M.JOINTS, view=d))
            err = (rv - rel_gt) / fps * 1000.0
            ref = dump[(dump["az"] == az) & (dump["el"] == el) &
                       (dump["detector"] == d)]["err_ms"]
            assert len(ref) == 1 and abs(float(ref.iloc[0]) - err) < 0.05, \
                "cell az%d %s: %.1f ms here, %.1f in the dump" % (az, d, err, ref.iloc[0])
            gx, gy = pose_xy(df, rel_gt)
            dx, dy = pose_xy(df, rv)
            box = (min(gx.min(), dx.min()) - PAD_M, max(gx.max(), dx.max()) + PAD_M,
                   min(gy.min(), dy.min()) - PAD_M, max(gy.max(), dy.max()) + PAD_M)
            out.append(dict(i=i, j=j, az=az, el=el, det=d, df=df, gt=rel_gt, det_f=rv,
                            err=err, box=box, ruled=(RULED[(az, el)] == d)))
            print("  %-30s %-8s det f%-4d  %+7.1f ms  %s"
                  % (sname.split("(")[0].strip(), d, rv, err,
                     "ruled" if out[-1]["ruled"] else "transferred"))
    return out


def check_no_overlap(fig, texts, poses):
    """Assert no label touches a skeleton, in rendered display coordinates."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for t, ax in texts:
        tb = t.get_window_extent(renderer=r)
        for a, line in poses:
            if a is not ax:
                continue
            lb = line.get_window_extent(renderer=r) if hasattr(line, "get_window_extent") \
                else None
            if lb is not None and tb.overlaps(lb):
                raise SystemExit("label %r overlaps a bone -- widen BAND_IN" % t.get_text())


def main():
    os.makedirs(OUT, exist_ok=True)
    P = panels()
    ncol, nrow = len(DETS), len(STATIONS)

    # one scale for every panel: the columns share out the width left over once each
    # panel has taken its label strip
    col_w = [max(p["box"][1] - p["box"][0] for p in P if p["j"] == j) for j in range(ncol)]
    row_h = [max(p["box"][3] - p["box"][2] for p in P if p["i"] == i) for i in range(nrow)]
    axes_w = FIG_W - LEFT_IN - RIGHT_IN
    s = sum(col_w) / (axes_w - ncol * BAND_IN)          # metres per inch
    col_in = [w / s + BAND_IN for w in col_w]
    row_in = [h / s for h in row_h]
    fig_h = sum(row_in) + nrow * HEAD_IN + LEGEND_IN + BOTTOM_IN
    print("\nscale %.3f m/in   cols %s in   rows %s in" %
          (s, [round(c, 2) for c in col_in], [round(r, 2) for r in row_in]))

    fig = plt.figure(figsize=(FIG_W, fig_h))
    texts, bones = [], []
    for p in P:
        i, j = p["i"], p["j"]
        x0 = (LEFT_IN + sum(col_in[:j])) / FIG_W
        y0 = (BOTTOM_IN + sum(row_in[i + 1:]) + (nrow - 1 - i) * HEAD_IN) / fig_h
        ax = fig.add_axes([x0, y0, col_in[j] / FIG_W, row_in[i] / fig_h])
        # the poses are centred in what is left of the panel once the label strip is
        # taken, so the strip stays empty without the narrow panels stranding their
        # poses against the right edge
        xa, xb, ya, yb = p["box"]
        free = (col_in[j] - BAND_IN) * s
        x0m = xa - (free - (xb - xa)) / 2.0 - BAND_IN * s
        ax.set_xlim(x0m, x0m + col_in[j] * s)
        mid = (ya + yb) / 2.0
        ax.set_ylim(mid + row_in[i] * s / 2.0, mid - row_in[i] * s / 2.0)   # y inverted
        ax.set_aspect("equal"); ax.axis("off")
        col = OWN_C if p["ruled"] else XFER_C
        skeleton(ax, p["df"], p["det_f"], col, 2.6 * LW, 3)
        skeleton(ax, p["df"], p["gt"], GT_C, 1.5 * LW, 4)
        bones += [(ax, ln) for ln in ax.lines]
        if HEAD_IN:
            ax.set_title("%s detector   %+.1f ms" % (p["det"], p["err"]),
                         fontsize=7.0, color=col, weight="bold", pad=2.6)
        else:
            t = ax.text(0.015, 0.985, "%s detector\n%+.1f ms" % (p["det"], p["err"]),
                        transform=ax.transAxes, va="top", ha="left", fontsize=8,
                        color=col, fontweight="bold", linespacing=1.35)
            texts.append((t, ax))     # only an in-panel label can collide with a bone
        if j == 0:
            fig.text((LEFT_IN - 0.20) / FIG_W,
                     (BOTTOM_IN + sum(row_in[i + 1:]) + (nrow - 1 - i) * HEAD_IN
                      + row_in[i] / 2.0) / fig_h,
                     STATIONS[i][2], rotation=90, va="center", ha="center",
                     fontsize=8 * (1 if not SINGLE_COL else 0.85), color=INK)

    fs = 8 if not SINGLE_COL else 6.6
    fig.legend(handles=[plt.Line2D([], [], color=GT_C, lw=1.5,
                                   label="ground-truth release frame"),
                        plt.Line2D([], [], color=OWN_C, lw=2.6,
                                   label="detected — detector's own station"),
                        plt.Line2D([], [], color=XFER_C, lw=2.6,
                                   label="detected — transferred from the other station")],
               loc="lower center", bbox_to_anchor=(0.5, 1 - LEGEND_IN / fig_h),
               # three entries do not fit across 3.5 in; stack them at column width
               ncol=1 if SINGLE_COL else 3, fontsize=fs, frameon=False,
               handlelength=1.8, columnspacing=1.6, handletextpad=0.5,
               labelspacing=0.35)

    check_no_overlap(fig, texts, bones)
    out = os.path.join(OUT, "fig_release_transfer.png")
    fig.savefig(out, dpi=300)          # never bbox_inches="tight"
    print("-> %s   %.2f x %.2f in" % (out, FIG_W, fig_h))


if __name__ == "__main__":
    main()
