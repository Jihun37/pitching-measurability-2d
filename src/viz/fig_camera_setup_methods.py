"""Fig. 1 (paper, Methods): virtual camera setup over azimuth x elevation around
the pitcher. This is the CLEAN methods version -- it shows only the camera
parameterization (dome grid, camera markers, pitcher, pitch direction, axes and
the az0/az90 reference points). It deliberately carries NO usable-zone bands and
NO SIDE/FRONT/OVERHEAD anchors: those belong to the measurability map (Fig. 2),
and the station/zone framing was retired for the paper. For the schematic-zone
variant see fig_camera_setup_3d.py (kept for other uses).

Azimuth is HANDEDNESS-RELATIVE (2026-07-23): az0 = the OPEN (throwing-arm) side,
az90 = front (home), az180 = glove side, az270 = behind. LHP are reflected to RHP
in the loader (obp_project.reflect_to_rhp), so the skeleton drawn here is already
in the RHP frame and az0 means 3B for a RHP but 1B for a LHP. Do NOT relabel these
axes with field-base names.

The actual sweep is 24 azimuths (0-345, step 15) x 7 elevations {0,15,30,45,60,
75,85} = 168 viewpoints; this figure draws a representative quarter dome so the
parameterization stays legible. The exact grid and counts live in the caption/text.

The drawn elevations stop at 85 (EL_MAX) and the top marker is labelled
"el 85 (overhead)". Do NOT restore a 90-degree pole: el=90 is the image-basis
singularity, it is never swept, and drawing it contradicts Sec. III-A."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.lines import Line2D

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(HERE, "..", "stage3"))
import config
import obp_project as O
import metrics as M

INK = "#0E1B33"; TEAL = "#0FA3B1"; GREY = "0.55"
BODY_W = 7.16       # IEEE Access double-column body width; see fig_graded_map
R = 1.0
AZ_STEP = 15        # the sweep's azimuth spacing (analysis/angle_zone_sweep.AZ_STEP)
BONES = [("head", "left_shoulder"), ("head", "right_shoulder"),
         ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
         ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
         ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")]


def pos(az, el):
    """Camera position on the dome for (az, el), in the reflected RHP world frame.

    VERIFIED against the data, do not "simplify" the Y sign away: after
    obp_project.reflect_to_rhp the throwing side is -Y and home is +X on every
    pitch (checked on 4 RHP and 4 LHP, throwing wrist minus pelvis in Y is
    negative in all 8). So az0, the OPEN side, must sit at -Y and az180, the
    glove side, at +Y. An earlier version of this figure dropped the minus and
    drew the open and glove sides swapped."""
    a, e = np.radians(az), np.radians(el)
    return np.array([np.sin(a)*np.cos(e), -np.cos(a)*np.cos(e), np.sin(e)]) * R


def pitcher_skeleton(height=0.6):
    """A real OBP pitch pose at release, centered and scaled to sit at the dome
    center (X=pitch dir, Y=side, Z=up -- same axes as the camera positions)."""
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    r = md.iloc[0]
    path = os.path.join(config.OBP_DATA_DIR, "c3d", f"{int(r.user):06d}", r.filename_new)
    joints, fps = O.load_c3d_joints(path)
    arm = O.detect_throwing_arm(joints, fps)
    f = M.release_frame(O.project_view(joints, azimuth_deg=0), arm, fps, M.JOINTS)
    P = {n: joints[n][:, f].astype(float) for n in joints}
    A = np.array(list(P.values()))
    anchor = ((P["left_ankle"] + P["right_ankle"]) / 2.0).astype(float).copy()
    anchor[2] = (A[:, 2].min() + A[:, 2].max()) / 2.0
    s = height / (np.ptp(A[:, 2]) + 1e-9)
    return {n: (P[n] - anchor) * s for n in P}


def draw(ax, label_fs=8.5, note_fs=8.0, axis_fs=9.0):
    """Draw the camera dome onto a 3D axes. Split out of main() 2026-08-08 so the
    merged camera-geometry figure and this standalone one cannot drift apart."""
    # EL matches the real sweep exactly and stops at 85; el=90 is the image-basis
    # singularity and is never swept, so the dome must not draw or label a pole.
    EL_MAX = 85
    # The FULL sweep, drawn in full: 24 azimuths over the whole 360 deg orbit and
    # 7 elevations. A quarter dome used to be drawn here and read as though only a
    # 90 deg wedge was measured, which is wrong.
    AZ = list(range(0, 360, AZ_STEP))
    EL = [0, 15, 30, 45, 60, 75, EL_MAX]
    # mesh only, kept sparse so 168 nodes stay legible against the skeleton
    MERIDIAN_AZ = list(range(0, 360, 30))
    # z=0 sits at the pitcher's mid-torso (chest height); the skeleton's feet fall
    # below zero, so an el=0 camera reads as a side view aimed at the body.
    ZC = 0.0
    C = np.array([0.0, 0.0, ZC])
    # faint dome mesh (meridians + parallels)
    for az in MERIDIAN_AZ:
        pts = np.array([pos(az, e) + C for e in np.linspace(0, EL_MAX, 40)])
        ax.plot(*pts.T, color="0.86", lw=0.6, zorder=1)
    for el in EL:
        pts = np.array([pos(a, el) + C for a in np.linspace(0, 360, 160)])
        ax.plot(*pts.T, color="0.86", lw=0.6, zorder=1)
    # camera markers at every (az, el) node
    for az in AZ:
        for el in EL:
            p = pos(az, el) + C
            ax.scatter(*p, color=GREY, s=8, zorder=2)

    # pitcher skeleton at the dome center + pitch-direction arrow (toward home, +X)
    Q = pitcher_skeleton(1.45)   # sized against the FULL dome, not the old quarter
    for a, b in BONES:
        if a in Q and b in Q:
            ax.plot(*[[Q[a][k], Q[b][k]] for k in range(3)], color=INK, lw=2.0, zorder=5)
    pts = np.array(list(Q.values()))
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=TEAL, s=13, zorder=6)
    zf = pts[:, 2].min()
    # the skeleton needs no caption: nothing else in the panel is a body
    ax.quiver(0, 0, zf, 0.98, 0, 0, color="0.4", lw=1.4, arrow_length_ratio=0.11)
    # above the arrow, not beside it: the floor pane edge falls away to the left
    # and clipped the tail of the word at any height that read as "beside".
    ax.text(0.86, 0, zf + 0.34, "to home", fontsize=8, ha="left", color="0.4")

    # All four cardinal azimuths at el=0 plus the top row of the sweep, so the
    # handedness-relative convention of Sec. III-A reads straight off the figure.
    # az0 = open side (-Y, see pos()), az90 = front (+X), az180 = glove side,
    # az270 = rear, up = overhead. `ha` grows each label AWAY from the dome; at
    # view azim -45 the four land lower-left, lower-right, upper-right, upper-left.
    # Two lines, not one. At this font the long labels are wider than the gap
    # between the dome and the wall pane, so on one line they either crossed the
    # pane or, pulled inward, sat on their own marker. Breaking after the angle
    # halves the width and keeps all four at the same radius.
    pending = []               # drawn flat at the end, on top of the node grid
    for az, label, ha in [(0,   "az 0$^\\circ$\n(open side)",    "right"),
                          (90,  "az 90$^\\circ$\n(front)",       "left"),
                          (180, "az 180$^\\circ$\n(glove side)", "left"),
                          (270, "az 270$^\\circ$\n(rear)",       "right")]:
        p = pos(az, 0) + C
        ax.scatter(*p, color=INK, s=34, zorder=6)
        t = pos(az, 0) * 1.05 + C
        pending.append((t[0], t[1], t[2] + 0.05, label, ha))
    ptop = pos(0, EL_MAX) + C   # top row of the sweep (85), not the 90 pole
    pending_dot = ptop          # crowded top row buries it in the depth sort
    pending.append((ptop[0], ptop[1], ptop[2] + 0.16, "el 85$^\\circ$ (overhead)",
                    "center"))

    ax.text2D(0.0, 0.99,
              "grey nodes = camera viewpoints (az $\\times$ el grid)\n"
              "sweep: 24 az (step 15$^\\circ$) $\\times$ 7 el = 168 views",
              transform=ax.transAxes, fontsize=note_fs, color="0.35",
              ha="left", va="top", linespacing=1.4)

    ax.set_xlabel("X  (pitch direction)", fontsize=axis_fs)
    # handedness-relative: -Y is the OPEN (throwing-arm) side and +Y the glove side
    # for every pitcher, since LHP are reflected to RHP in the loader (verified in
    # pos()). Never label this "1B" or "3B".
    ax.set_ylabel("Y  (glove side)", fontsize=axis_fs)
    ax.set_zlabel("Z  (up)", fontsize=axis_fs)
    L = 1.15
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(zf, L)
    try:                       # zoom trims the dead space a wide, short dome leaves
        ax.set_box_aspect((2 * L, 2 * L, L - zf), zoom=1.15)
    except TypeError:          # older matplotlib has no zoom kwarg
        ax.set_box_aspect((2 * L, 2 * L, L - zf))
    # azim puts az0 (-Y, open side) on the left and az90 (+X, front) on the right,
    # both on the near face. With the old azim=52 the Y-sign correction in pos()
    # dragged both onto the same side of the frame.
    ax.view_init(elev=26, azim=-45)
    try:
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([0, 1])
    except Exception:
        pass

    # Only now is the projection final -- box aspect, zoom, limits and view all
    # feed it, so projecting any earlier would place the labels against a matrix
    # that later changed underneath them.
    proj = ax.get_proj()
    u, v, _ = proj3d.proj_transform(*pending_dot, proj)
    # s=34 in the 3D scatter is an area in points squared; Line2D wants a diameter
    ax.add_artist(Line2D([u], [v], marker="o", ms=34 ** 0.5, color=INK,
                         ls="none", transform=ax.transData, zorder=1e6))
    for x, y, z, label, ha in pending:
        u, v, _ = proj3d.proj_transform(x, y, z, proj)
        # sit ON the anchor horizontally but ABOVE it vertically, which is what
        # the Text3D baseline used to give: centred on it, each label lands back
        # on its own marker.
        ax.annotate(label, (u, v), xycoords=ax.transData, textcoords="offset points",
                    xytext=(0, 5), fontsize=label_fs, color=INK, weight="bold",
                    ha=ha, va="bottom", linespacing=1.25, zorder=1e6)


def main():
    # 2026-08-08: authored at BODY_W and saved WITHOUT a tight bbox. It used to be
    # figsize (7, 5.6) with tight_layout() and bbox_inches="tight" at dpi 200 -- the
    # exact practice this repository forbids for a body figure, because a tight bbox
    # crops or expands the PNG to its content and \includegraphics[width=7.16in] then
    # rescales every font by whatever that crop happened to be.
    fig = plt.figure(figsize=(BODY_W, BODY_W * 0.80))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    draw(ax)
    out = os.path.join(config.OBP_VALIDATION_DIR, "fig_camera_setup_methods.png")
    fig.savefig(out, dpi=300)
    print(f"saved -> {out}   {BODY_W:.2f} x {BODY_W * 0.80:.2f} in")


if __name__ == "__main__":
    main()
