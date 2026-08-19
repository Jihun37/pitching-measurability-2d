"""
Overlay the quantities onto the video, as small live panels.

The joint-overlay video with three mini plots in the corner. Paths come from
config.py.
"""

import cv2
import numpy as np
import os, sys, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
from calculator import angle_series, joint_speed, hip_shoulder_separation, peak_frame, motion_start_frame
from skeleton import load_smoothed, draw_skeleton, find_video, backbone_suffix


def draw_mini_graph(frame, series, cur, x, y, w, h, color, label, vmin, vmax):
    """Draw one small plot at (x, y) over the frame."""
    # translucent backing
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # the label
    cv2.putText(frame, label, (x+5, y+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # the trace
    n = len(series)
    pts = []
    for i in range(n):
        px = x + int(i / n * w)
        val = (series[i] - vmin) / (vmax - vmin + 1e-6)
        val = np.clip(val, 0, 1)
        py = y + h - int(val * (h - 25)) - 5
        pts.append((px, py))
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i-1], pts[i], color, 1)

    # current position: a dot and a vertical rule
    if cur < len(pts):
        cv2.line(frame, (pts[cur][0], y+22), (pts[cur][0], y+h-3), (255,255,255), 1)
        cv2.circle(frame, pts[cur], 4, (255,255,255), -1)
        # the current value
        cv2.putText(frame, f"{series[cur]:.0f}", (x+w-55, y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="video name (no ext); default config.VIDEO_NAME")
    ap.add_argument("--backbone", default="rtmp", choices=["rtmp", "mediapipe"])
    a = ap.parse_args()
    name = a.video or config.VIDEO_NAME

    df = load_smoothed(name, a.backbone)     # same coords the pipeline measures
    out_path = os.path.join(config.ROOT, "data", "outputs", name,
                            f"{name}_overlay{backbone_suffix(a.backbone)}.mp4")
    arm = config.THROWING_ARM
    elbow = angle_series(df, f"{arm}_shoulder", f"{arm}_elbow", f"{arm}_wrist")
    wrist_spd = joint_speed(df, f"{arm}_wrist", config.FPS_DEFAULT)
    hss = hip_shoulder_separation(df)
    release = peak_frame(wrist_spd)

    cap = cv2.VideoCapture(find_video(name))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # size and position of the mini plots: three stacked, top right
    gw, gh = 320, 90
    gx = width - gw - 20

    frame_idx = 0
    while cap.isOpened() and frame_idx < len(df):
        ret, frame = cap.read()
        if not ret:
            break

        draw_skeleton(frame, df, frame_idx)

        # the three mini plots, overlaid
        draw_mini_graph(frame, wrist_spd, frame_idx, gx, 20, gw, gh,
                        (255,180,0), "Wrist speed", 0, np.nanmax(wrist_spd))
        draw_mini_graph(frame, elbow, frame_idx, gx, 120, gw, gh,
                        (0,180,255), "Elbow angle", 0, 180)
        draw_mini_graph(frame, hss, frame_idx, gx, 220, gw, gh,
                        (0,255,0), "Hip-shoulder sep", np.nanmin(hss), np.nanmax(hss))

        # mark the release frame
        if frame_idx == release:
            cv2.putText(frame, "RELEASE!", (gx, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{total} ({frame_idx/total*100:.0f}%)")

    cap.release()
    out.release()
    print(f"done, written to {out_path}")


if __name__ == "__main__":
    main()