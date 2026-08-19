"""
Video and plots side by side.

The joint-overlay video on the left, three plots on the right with the current
frame marked. Paths come from config.py.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # render without a display
import matplotlib.pyplot as plt
import os, sys, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
from calculator import angle_series, joint_speed, hip_shoulder_separation, peak_frame, motion_start_frame
from skeleton import load_smoothed, draw_skeleton, find_video, backbone_suffix


def make_graph_frame(elbow, wrist_spd, hss, cur_frame, release, start, h_px):
    """The plots as a numpy array, with a rule at the current frame."""
    fig, axes = plt.subplots(3, 1, figsize=(6, h_px/100), sharex=True, dpi=100)

    axes[0].plot(wrist_spd, color="tab:blue")
    axes[0].axvline(release, color="red", ls="--", lw=1)
    axes[0].axvline(start, color="green", ls="--", lw=1)
    axes[0].set_ylabel("Wrist spd")

    axes[1].plot(elbow, color="tab:orange")
    axes[1].axvline(release, color="red", ls="--", lw=1)
    axes[1].set_ylabel("Elbow")

    axes[2].plot(hss, color="tab:green")
    axes[2].axvline(release, color="red", ls="--", lw=1)
    axes[2].set_ylabel("Hip-sh sep")
    axes[2].set_xlabel("Frame")

    # the current frame, as a heavy black rule
    for ax in axes:
        ax.axvline(cur_frame, color="black", lw=2, alpha=0.7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="video name (no ext); default config.VIDEO_NAME")
    ap.add_argument("--backbone", default="rtmp", choices=["rtmp", "mediapipe"])
    a = ap.parse_args()
    name = a.video or config.VIDEO_NAME

    # Smoothed coordinates and the quantities from them, the same coordinates
    # the overlay renders
    df = load_smoothed(name, a.backbone)
    out_path = os.path.join(config.ROOT, "data", "outputs", name,
                            f"{name}_sidebyside{backbone_suffix(a.backbone)}.mp4")
    arm = config.THROWING_ARM
    elbow = angle_series(df, f"{arm}_shoulder", f"{arm}_elbow", f"{arm}_wrist")
    wrist_spd = joint_speed(df, f"{arm}_wrist", config.FPS_DEFAULT)
    hss = hip_shoulder_separation(df)
    release = peak_frame(wrist_spd)
    start = motion_start_frame(wrist_spd)

    cap = cv2.VideoCapture(find_video(name))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = width + 600   # 600 px of plots beside the video
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, height))

    frame_idx = 0
    while cap.isOpened() and frame_idx < len(df):
        ret, frame = cap.read()
        if not ret:
            break

        draw_skeleton(frame, df, frame_idx)

        # render the plots and set them beside the frame
        graph_img = make_graph_frame(elbow, wrist_spd, hss, frame_idx,
                                     release, start, height)
        graph_img = cv2.resize(graph_img, (600, height))
        combined = np.hstack([frame, graph_img])
        out.write(combined)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{total} ({frame_idx/total*100:.0f}%)")

    cap.release()
    out.release()
    print(f"done, written to {out_path}")


if __name__ == "__main__":
    main()