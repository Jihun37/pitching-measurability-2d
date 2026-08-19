import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(HERE, "..", "stage3"))

import config
import metrics as M
import obp_project as O

CONNECT = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
]

def pt(df, joint, f):
    return df[f"{joint}_x"].iloc[f], df[f"{joint}_y"].iloc[f]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="metadata row index")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    r = md.iloc[args.index]
    path = os.path.join(
        config.OBP_DATA_DIR,
        "c3d",
        f"{int(r.user):06d}",
        r.filename_new
    )

    joints, fps = O.load_c3d_joints(path)
    arm = O.detect_throwing_arm(joints, fps)
    lead = "left" if arm == "right" else "right"

    df = O.project_view(joints, azimuth_deg=0)
    cand = M.compute_candidates(df, fps=fps, arm=arm, view="side")
    vals = {k: v for k, (v, _) in cand.items()}

    rel = M.release_frame(df, arm, fps, M.JOINTS, view="side")
    fp = M.foot_plant_frame(df, lead, fps, M.JOINTS, rel)

    fig, ax = plt.subplots(figsize=(6, 7))

    # skeleton at release
    for a, b in CONNECT:
        try:
            ax.plot(
                [df[f"{a}_x"].iloc[rel], df[f"{b}_x"].iloc[rel]],
                [df[f"{a}_y"].iloc[rel], df[f"{b}_y"].iloc[rel]],
                linewidth=2
            )
        except Exception:
            pass

    # key points
    for j in [
        "left_shoulder", "right_shoulder", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_wrist", "right_wrist"
    ]:
        try:
            x, y = pt(df, j, rel)
            ax.scatter(x, y, s=25)
        except Exception:
            pass

    # knee angle
    hip = pt(df, f"{lead}_hip", rel)
    knee = pt(df, f"{lead}_knee", rel)
    ankle = pt(df, f"{lead}_ankle", rel)

    ax.plot([hip[0], knee[0]], [hip[1], knee[1]], linewidth=3)
    ax.plot([knee[0], ankle[0]], [knee[1], ankle[1]], linewidth=3)
    ax.text(
        knee[0], knee[1],
        f"knee {vals['lead_knee_angle']:.0f}°",
        fontsize=10,
        weight="bold"
    )

    # trunk anterior tilt
    lh = pt(df, "left_hip", rel)
    rh = pt(df, "right_hip", rel)
    ls = pt(df, "left_shoulder", rel)
    rs = pt(df, "right_shoulder", rel)

    mid_hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    mid_sh = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)

    ax.plot(
        [mid_hip[0], mid_sh[0]],
        [mid_hip[1], mid_sh[1]],
        linewidth=3
    )
    ax.text(
        mid_hip[0], mid_hip[1],
        f"trunk {vals['trunk_anterior_tilt']:.0f}°",
        fontsize=10,
        weight="bold"
    )

    # stride line at foot plant
    la = pt(df, "left_ankle", fp)
    ra = pt(df, "right_ankle", fp)

    ax.plot([la[0], ra[0]], [la[1], ra[1]], linewidth=3)
    ax.text(
        (la[0] + ra[0]) / 2,
        (la[1] + ra[1]) / 2,
        f"stride {vals['stride_pct_height']:.2f}×H",
        fontsize=10,
        weight="bold"
    )

    # knee extension velocity
    ax.text(
        knee[0],
        knee[1] + 0.06,
        f"knee velo {vals['knee_ext_velo_br']:.0f}°/s",
        fontsize=10,
        weight="bold"
    )

    # wrist speed arrow
    wrist = f"{arm}_wrist"
    wx = df[f"{wrist}_x"].to_numpy(float)
    wy = df[f"{wrist}_y"].to_numpy(float)

    dx = wx[rel] - wx[max(rel - 2, 0)]
    dy = wy[rel] - wy[max(rel - 2, 0)]

    ax.arrow(
        wx[rel], wy[rel],
        dx * 3, dy * 3,
        head_width=0.03,
        length_includes_head=True
    )
    ax.text(
        wx[rel], wy[rel],
        f"wrist spd {vals['wrist_speed']:.1f}",
        fontsize=10,
        weight="bold"
    )

    ax.set_title("Lateral 2D Projection (0°) with Five Metrics")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    out = args.out or os.path.join(
        config.OBP_VALIDATION_DIR,
        f"lateral_metrics_index{args.index}.png"
    )
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()