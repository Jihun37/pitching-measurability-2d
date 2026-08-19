"""Figure (도6, left panel): a real bent-over release-pose skeleton on a
transparent background for the multi-view capture-geometry diagram. Reuses the
OBP release pose and the fig_camera_setup_3d bone set; the TikZ figure overlays
the three cameras, sightlines and occlusion on top of this PNG. Centered on the
hip so the TikZ camera rays can aim at the image center."""
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
from fig_camera_setup_3d import BONES

INK = "#0E1B33"; TEAL = "#0FA3B1"; AMBER = "#F2A900"


def main():
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    r = md.iloc[0]
    path = os.path.join(config.OBP_DATA_DIR, "c3d", f"{int(r.user):06d}", r.filename_new)
    joints, fps = O.load_c3d_joints(path)
    arm = O.detect_throwing_arm(joints, fps)
    f = M.release_frame(O.project_view(joints, azimuth_deg=0), arm, fps, M.JOINTS)
    P = {n: joints[n][:, f].astype(float) for n in joints}
    hip = (P["left_hip"] + P["right_hip"]) / 2.0
    P = {n: P[n] - hip for n in P}                      # center on hip

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")                          # origin stays centered
    for a, b in BONES:
        if a in P and b in P:
            ax.plot(*[[P[a][k], P[b][k]] for k in range(3)],
                    color=INK, lw=3.2, solid_capstyle="round", zorder=3)
    pts = np.array(list(P.values()))
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=TEAL, s=18, zorder=4)
    ax.scatter(0, 0, 0, color=AMBER, s=120, edgecolors="white",
               linewidths=1.3, zorder=5)                # hip marker at origin

    m = float(np.nanmax(np.abs(pts))) * 1.05            # symmetric limits -> hip centered
    ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(-m, m)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=6, azim=-78)
    ax.set_axis_off()
    fig.patch.set_alpha(0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)   # full-frame, no crop
    out = os.path.join(config.OBP_VALIDATION_DIR, "fig_capture_skeleton.png")
    fig.savefig(out, dpi=220, transparent=True)
    print("saved ->", out)


if __name__ == "__main__":
    main()
