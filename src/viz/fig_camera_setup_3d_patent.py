"""Figure (patent Fig. 2, PATENT COPY of fig_camera_setup_3d): virtual camera setup over
azimuth x elevation around the pitcher. Each metric is trustworthy over a REGION
of the dome (a valid zone), not a single point; the classic viewpoints remain as
the optimal ANCHORS inside their zones. The shaded bands are schematic - the
measured per-metric zones are the quantitative zone map (patent Fig. 4 in the patent).
3D quarter dome.

This is a PATENT-ONLY copy so the paper figure (fig_camera_setup_3d.py) is left
untouched. It differs from the paper version in exactly two ways:
  - it draws the DEPLOYMENT set of 15 metrics (see anchor contents below), and
  - it writes fig_camera_setup_3d_patent.png (never overwrites the paper png).

Azimuth is HANDEDNESS-RELATIVE (2026-07-23): az0 = the OPEN (throwing-arm) side,
az90 = front, az180 = glove side, az270 = behind. LHP are reflected to RHP in the
loader (obp_project.reflect_to_rhp), so the skeleton drawn here is already in the
RHP frame and az0 means 3B for a RHP but 1B for a LHP. Do NOT relabel these axes
with field-base names.

Deployment set is 15 (2026-07-24). The 16th measurable metric, elbow flexion
@MER, is frame-rate-excluded from per-pitch deployment (best 0.64 at 120 fps) so
it is NOT drawn here. Anchor contents:
  SIDE     (8) = the sagittal group: lead knee angle, stride length, trunk
                 anterior tilt, release height, wrist speed, release extension,
                 forward COG velocity, COG velocity @ peak knee height.
  FRONT    (4) = arm slot, stride angle, glove-arm abduction @MER,
                 torso lateral tilt @MER.
  OVERHEAD (3) = HSS, pelvis rotational velocity, torso transverse rotation
                 @release.
Stride angle also sharpens with elevation (el85 r2 0.966, azimuth-independent),
so the overhead band is not exclusively the transverse metrics -- the ground
front anchor is simply the deployable one for it."""
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

INK = "#0E1B33"; TEAL = "#0FA3B1"; AMBER = "#F2A900"; VIOLET = "#7C3AED"; RED = "#E0533D"
R = 1.0
BONES = [("head", "left_shoulder"), ("head", "right_shoulder"),
         ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
         ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
         ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")]


def pos(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.array([np.sin(a)*np.cos(e), np.cos(a)*np.cos(e), np.sin(e)]) * R


def dome_patch(ax, az0, az1, el0, el1, color, zc=0.0):
    """Translucent band on the dome surface = a metric's usable viewpoint zone
    (schematic). Sits just outside the mesh radius to avoid z-fighting. zc lifts
    the band with the dome center (cameras orbit the torso, not the feet)."""
    aa = np.radians(np.linspace(az0, az1, 24))
    ee = np.radians(np.linspace(el0, el1, 24))
    A, E = np.meshgrid(aa, ee)
    X = np.sin(A) * np.cos(E) * R * 1.003
    Y = np.cos(A) * np.cos(E) * R * 1.003
    Z = np.sin(E) * R * 1.003 + zc
    ax.plot_surface(X, Y, Z, color=color, alpha=0.20, shade=False,
                    linewidth=0, antialiased=True, zorder=3)


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
    # Keep the pose at its natural (un-rotated) angle; anchor the ankle-midpoint
    # stance in X/Y, but put the body's MID-HEIGHT at z=0 so the origin lands at
    # chest level (dome center) and the feet fall below zero.
    A = np.array(list(P.values()))
    anchor = ((P["left_ankle"] + P["right_ankle"]) / 2.0).astype(float).copy()
    anchor[2] = (A[:, 2].min() + A[:, 2].max()) / 2.0
    s = height / (np.ptp(A[:, 2]) + 1e-9)
    return {n: (P[n] - anchor) * s for n in P}


def main():
    fig = plt.figure(figsize=(7, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    AZ = [0, 15, 30, 45, 60, 75, 90]; EL = [0, 15, 30, 45, 60, 75, 90]
    # The coordinate origin (z=0) sits at the pitcher's mid-torso: the dome is
    # centered there and the skeleton hangs its feet BELOW zero (see
    # pitcher_skeleton). So an el=0 camera sits at z=0 = chest height and reads
    # as a side view aimed at the body, not a shot from the ground.
    ZC = 0.0
    C = np.array([0.0, 0.0, ZC])
    # faint dome mesh
    for az in AZ:
        pts = np.array([pos(az, e) + C for e in np.linspace(0, 90, 30)])
        ax.plot(*pts.T, color="0.8", lw=0.6, zorder=1)
    for el in EL:
        pts = np.array([pos(a, el) + C for a in np.linspace(0, 90, 30)])
        ax.plot(*pts.T, color="0.8", lw=0.6, zorder=1)
    # camera markers
    for az in AZ:
        for el in EL:
            p = pos(az, el) + C
            ax.scatter(*p, color="0.55", s=9, zorder=2)

    # pitcher skeleton at the dome center + pitch-direction arrow (toward home, +X)
    Q = pitcher_skeleton(0.85)
    for a, b in BONES:
        if a in Q and b in Q:
            ax.plot(*[[Q[a][k], Q[b][k]] for k in range(3)], color=INK, lw=2.0, zorder=5)
    pts = np.array(list(Q.values()))
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=TEAL, s=13, zorder=6)
    zf = pts[:, 2].min()  # foot level; the plot's bottom pane is pinned here (see
                          # set_zlim) so the pitcher stands ON the axes floor while
                          # z=0 stays at chest height -- no separate floor drawn.
    ax.text(0, 0, zf+0.04, "pitcher", fontsize=9, ha="center", color=INK)
    ax.quiver(0, 0, zf, 0.6, 0, 0, color="0.4", lw=1.4, arrow_length_ratio=0.14)
    ax.text(0.82, 0, zf+0.01, "to home", fontsize=8, ha="left", color="0.4")

    # per-metric usable ZONES (schematic bands) with the optimal viewpoint
    # ANCHOR inside each. Band extents are illustrative; the measured zones
    # (15deg x 7-elevation, r2>=0.6) are the quantitative map (patent Fig. 4).
    # (az0,az1,el0,el1) per band. Deployment set = 15 metrics.
    for (az, el, col, name, off, zone) in [
        (0, 0, TEAL, "SIDE anchor\n(sagittal zone)", (0.05, -0.42, 0.02),
         (0, 40, 0, 45)),
        (90, 0, AMBER, "FRONT anchor\n(arm slot, stride angle,\nglove abd, torso lat tilt)",
         (0.05, 0.05, 0.12), (40, 90, 0, 30)),
        (0, 85, VIOLET, "OVERHEAD anchor\n(HSS, pelvis, torso rot)", (0.02, 0.02, 0.13),
         (0, 90, 60, 90))]:
        dome_patch(ax, *zone, col, zc=ZC)
        u = pos(az, el)
        p = u + C
        ax.scatter(*p, color=col, s=90, edgecolors="white", linewidths=1.2, zorder=6)
        d = -u / R
        ax.quiver(*p, *(d*0.3), color=col, lw=1.6, arrow_length_ratio=0.25, zorder=5)
        ax.text(p[0]+off[0], p[1]+off[1], p[2]+off[2], name, fontsize=9,
                color=col, weight="bold")

    ax.text2D(0.0, 0.99, "shaded = usable viewpoint zone\n"
              "(schematic; measured map = Fig. 4)\n● optimal anchor",
              transform=ax.transAxes, fontsize=8, color="0.35",
              ha="left", va="top", linespacing=1.4)

    ax.set_xlabel("X  (pitch direction)", fontsize=9)
    # handedness-relative: +Y is the OPEN (throwing-arm) side for every pitcher,
    # since LHP are reflected to RHP in the loader. Never label this "3B".
    ax.set_ylabel("Y  (open side)", fontsize=9)
    ax.set_zlabel("Z  (up)", fontsize=9)
    ax.set_xlim(0, 1.1); ax.set_ylim(0, 1.1); ax.set_zlim(zf, 1.1)
    ax.set_box_aspect((1.1, 1.1, 1.1 - zf))
    ax.view_init(elev=22, azim=52)
    try:
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_zticks([0, 0.5, 1])
    except Exception:
        pass
    out = os.path.join(config.OBP_VALIDATION_DIR, "fig_camera_setup_3d_patent.png")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); print("saved ->", out)


if __name__ == "__main__":
    main()
