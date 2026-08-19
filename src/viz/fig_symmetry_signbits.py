"""Figure (도3 lower panel): 4-fold viewpoint symmetry and its sign-bit
disambiguation.

The scale-free viewpoint features are blind to the pose symmetry family
{az, -az, 180-az, 180+az}: all four project to feature-identical silhouettes,
so no scale-invariant descriptor can tell them apart. Two image-plane SIGN bits break the
tie:
  stride direction  ~ sign(cos az)  : lead ankle vs trail ankle in image x.
  shoulder chirality ~ sign(sin az)  : right shoulder vs left shoulder in image x.
The four family members carry the four distinct (+/-, +/-) sign combinations,
so the pair uniquely identifies the member.

Shown on one OBP release pose projected at the fold-30 family {30,150,210,330}.

2026-08-03: the two sign labels overlapped the skeleton in all four panels. They were
placed at fixed axes corners while the pose auto-scaled to fill the axes, so a collision
was guaranteed, not incidental -- nudging the corners would only have moved which limb
they crossed. Fixed by reserving an empty band inside the data range at each end (see
BAND_F) and putting one label in each, so no placement tuning is involved. Two changes
came with it: the four panels now share ONE data box, so the family is drawn at a single
scale and the foreshortening difference between {30,210} and {150,330} is visible rather
than normalised away by per-panel autoscale; and "(cos)"/"(sin)" moved off the panels
into the note, which had the width for them. The four sign pairs are unchanged --
30:(+,-) 150:(-,-) 210:(-,+) 330:(+,+), still the four distinct combinations.

Then trimmed to 3.58 x 3.40 in: the in-figure note is GONE (it duplicated the LaTeX
caption and cost ~0.3 in), and the two signs share ONE band under the pose instead of a
band at each end, stacked on two lines so they keep "(cos)" and "(sin)". Widening the
band rather than the figure is nearly free -- the figure height is fixed, so the second
line buys back the trigonometric labels at the price of a ~7 % smaller pose and no page
space at all.

Both failure modes are now asserted rather than eyeballed: each label must fit inside
its panel horizontally and inside the reserved band vertically, checked against the real
renderer before the save. Lengthen a label or shrink BAND_F and the script fails here
instead of in print.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(HERE, "..", "stage3"))
import config
import obp_project as O
import metrics as M

INK = "#0E1B33"; TEAL = "#0FA3B1"; AMBER = "#F2A900"; RED = "#E0533D"
GREEN = "#16A34A"; PURPLE = "#7C3AED"; GRAY = "#64748B"

CONNECT = [("head", "left_shoulder"), ("head", "right_shoulder"),
           ("left_shoulder", "right_shoulder"),
           ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
           ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
           ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
           ("left_hip", "right_hip"),
           ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
           ("right_hip", "right_knee"), ("right_knee", "right_ankle")]

FAMILY = [30, 150, 210, 330]
NAMES = sorted({c for pair in CONNECT for c in pair})

# The sign labels used to be dropped at fixed axes corners (0.03, 0.97) and (0.03, 0.06)
# while the pose auto-scaled to fill the axes, so they landed ON the skeleton in all four
# panels. They now sit in BANDS RESERVED INSIDE THE DATA RANGE -- the y limits are opened
# by BAND_F * (pose height) at each end and nothing is drawn there, so the collision is
# impossible by construction rather than by tuning. Both bands are a fraction of the
# COMMON pose box, so the four panels share one scale and one box shape.
BAND_F = 0.27      # reserved band BELOW the pose, as a fraction of the common pose height
PAD_F = 0.04       # breathing room around the pose itself
# ONE band, TWO lines. A band at each end cost 15 % of panel height to carry one short
# word; a single one-line band was cheaper still but only fitted "stride +" / "shoulder -"
# side by side -- the pair WITH "(cos)"/"(sin)" is ~1.6 in against a ~1.0 in panel box.
# Widening the band instead of the figure is nearly free: the figure height is fixed, so
# the second line costs a slightly smaller pose and NO page space, and the panels keep
# saying which sign bit is which trigonometric function.
# 5.4 / 5.0 pt was unreadable at the 3.58 in single-column placement; the labels are the
# whole point of the figure, so they are set at a size a reader can actually resolve.
FS_TITLE, FS_SIGN = 7.0, 6.0


def xy(df, name, f):
    return float(df[f"{name}_x"].iloc[f]), float(df[f"{name}_y"].iloc[f])


def pose_box(df, f):
    xs = [xy(df, n, f)[0] for n in NAMES]
    ys = [xy(df, n, f)[1] for n in NAMES]
    return min(xs), max(xs), min(ys), max(ys)


def draw_pose(ax, df, f, lead, trail, span):
    for a, b in CONNECT:
        x1, y1 = xy(df, a, f); x2, y2 = xy(df, b, f)
        ax.plot([x1, x2], [y1, y2], color=GRAY, lw=1.0, zorder=2)
    for name in [c for pair in CONNECT for c in pair]:
        x, y = xy(df, name, f)
        ax.plot(x, y, "o", color=INK, ms=1.6, zorder=3)

    # stride-direction bit: trail ankle -> lead ankle in image x (actual points)
    lax, lay = xy(df, f"{lead}_ankle", f)
    tax, tay = xy(df, f"{trail}_ankle", f)
    s_stride = int(np.sign(lax - tax))
    ax.annotate("", xy=(lax, lay), xytext=(tax, tay),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.6, zorder=5))

    # shoulder-chirality bit: left shoulder -> right shoulder in image x
    rsx, rsy = xy(df, "right_shoulder", f)
    lsx, lsy = xy(df, "left_shoulder", f)
    s_sho = int(np.sign(rsx - lsx))
    ax.annotate("", xy=(rsx, rsy), xytext=(lsx, lsy),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.6, zorder=5))

    # Limits: the common pose box, centred on THIS pose, opened by one reserved band
    # below. Every panel therefore gets an identical box, and the band is empty.
    sx, sy, band, pad = span
    x0, x1, y0, y1 = pose_box(df, f)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_x = sx / 2.0 + pad
    half_y = sy / 2.0 + pad
    ax.set_xlim(cx - half_x, cx + half_x)
    ax.set_ylim(cy + half_y + band, cy - half_y)      # inverted: image y grows down
    # A pose drawn into a wide axes box is stretched unless the aspect is pinned.
    # `datalim` keeps the box the gridspec gave and widens the data range instead,
    # so the extra width becomes margin and the figure never distorts anatomy.
    ax.set_aspect("equal", adjustable="datalim")

    # the two signs stacked in the band, in the order the reader meets them: the stride
    # arrow is the wide one across the ankles, the shoulder arrow the short one up top
    frac = band / (2.0 * half_y + band)
    ax.text(0.5, frac * 0.66, f"stride (cos)  {'+' if s_stride > 0 else '−'}",
            color=AMBER, fontsize=FS_SIGN, ha="center", va="center", weight="bold",
            transform=ax.transAxes)
    ax.text(0.5, frac * 0.22, f"shoulder (sin)  {'+' if s_sho > 0 else '−'}",
            color=PURPLE, fontsize=FS_SIGN, ha="center", va="center", weight="bold",
            transform=ax.transAxes)
    return s_stride, s_sho


def prepare():
    """Load one pitch, project its four-fold azimuth family, and size a single box for
    all four panels. Split out of main() 2026-08-08 so the merged camera-geometry figure
    draws from the same preparation rather than a copy of it.
    Returns (projs, rel, lead, trail, span)."""
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    r = md.iloc[0]
    path = os.path.join(config.OBP_DATA_DIR, "c3d",
                        f"{int(r.user):06d}", r.filename_new)
    joints, fps = O.load_c3d_joints(path)
    arm = O.detect_throwing_arm(joints, fps)
    lead = "left" if arm == "right" else "right"
    trail = "right" if lead == "left" else "left"
    df0 = O.project_view(joints, azimuth_deg=0)
    wk = "r_wr" if arm == "right" else "l_wr"
    wx, wy = M._xy(df0, wk, M.JOINTS)
    rel = int(np.nanargmax(M._speed(wx, wy, fps)))
    projs = {az: O.project_view(joints, azimuth_deg=az) for az in FAMILY}
    boxes = [pose_box(df, rel) for df in projs.values()]
    sx = max(b[1] - b[0] for b in boxes)
    sy = max(b[3] - b[2] for b in boxes)
    span = (sx, sy, BAND_F * sy, PAD_F * sy)
    return projs, rel, lead, trail, span


def main():
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    r = md.iloc[0]
    path = os.path.join(config.OBP_DATA_DIR, "c3d",
                        f"{int(r.user):06d}", r.filename_new)
    joints, fps = O.load_c3d_joints(path)
    arm = O.detect_throwing_arm(joints, fps)
    lead = "left" if arm == "right" else "right"
    trail = "right" if lead == "left" else "left"

    # release frame from the side (az=0) projection
    df0 = O.project_view(joints, azimuth_deg=0)
    wk = "r_wr" if arm == "right" else "l_wr"
    wx, wy = M._xy(df0, wk, M.JOINTS)
    rel = int(np.nanargmax(M._speed(wx, wy, fps)))

    # SINGLE COLUMN (2026-08-01). Was authored at 13.5 in for a slide-width deployment
    # figure; at \columnwidth that is a 3.8x shrink and every label collapses. Now a
    # 2x2 grid at the width it is placed at, and the sign table below the poses is gone:
    # each panel already carries its own two signs, so the table only repeated them.
    COL_W = 3.58
    projs = {az: O.project_view(joints, azimuth_deg=az) for az in FAMILY}

    # one box for all four panels: the widest and the tallest pose of the family, so no
    # panel clips and the four are drawn at a single scale
    boxes = [pose_box(df, rel) for df in projs.values()]
    sx = max(b[1] - b[0] for b in boxes)
    sy = max(b[3] - b[2] for b in boxes)
    band, pad = BAND_F * sy, PAD_F * sy
    span = (sx, sy, band, pad)

    # HEIGHT IS THE PAGE COST, so it is chosen, not derived. A pitching pose is ~1.5x
    # taller than wide, so with aspect="equal" each panel's box is narrower than its
    # slot and there is horizontal white space no layout can recover -- deriving the
    # height to close that gap only inflates the figure (it produced 5.86 in). Every
    # extra inch here buys larger poses, not more information, and this is a supplement.
    # Raise HEIGHT_F only if the joints stop being resolvable in print.
    L, R, TOP, BOT = 0.012, 0.988, 0.945, 0.010
    WSP, HSP = 0.04, 0.16          # HSP also carries row 2's title
    HEIGHT_F = 0.95

    fig, axes = plt.subplots(2, 2, figsize=(COL_W, COL_W * HEIGHT_F))
    axes = axes.ravel()
    signs = {}
    for ax, az in zip(axes, FAMILY):
        ss, sh = draw_pose(ax, projs[az], rel, lead, trail, span)
        signs[az] = (ss, sh)
        ax.set_title(f"azimuth {az}°", fontsize=FS_TITLE, color=INK, weight="bold",
                     pad=2.4)
        # limits are set inside draw_pose, so the aspect must adapt the BOX, never the
        # limits -- 'datalim' would silently eat the reserved bands
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    # No in-figure note. It said what the LaTeX caption says, and two lines of it cost
    # ~0.3 in on a figure whose whole job is to fit in one column.
    fig.subplots_adjust(left=L, right=R, top=TOP, bottom=BOT, wspace=WSP, hspace=HSP)

    # The labels have collided with the pose once and nearly overflowed the panel once.
    # Both are now checked geometrically instead of by eye: every sign label must sit
    # inside its panel horizontally, and inside the reserved band vertically. A future
    # edit that lengthens a label or shrinks the band fails here rather than in print.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    band_frac = band / (sy + 2 * pad + band)
    for ax, az in zip(axes, FAMILY):
        ab = ax.get_window_extent(rend)
        for t in ax.texts:
            if not t.get_text().strip():
                continue                      # the two arrows are empty-string annotates
            tb = t.get_window_extent(rend)
            assert tb.x0 >= ab.x0 and tb.x1 <= ab.x1, \
                f"az{az}: '{t.get_text()}' overflows the panel width"
            assert tb.y1 <= ab.y0 + band_frac * ab.height, \
                f"az{az}: '{t.get_text()}' reaches out of the band and into the pose"
    print(f"label fit OK; band = {band_frac:.3f} of panel height")

    out = os.path.join(config.OBP_VALIDATION_DIR, "fig_symmetry_signbits.png")
    fig.savefig(out, dpi=300)
    print("saved ->", out)
    print("signs:", signs)


if __name__ == "__main__":
    main()
