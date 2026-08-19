"""
Diamond - shared coords loading + skeleton drawing for viz scripts.

Rendering is backbone-agnostic: it draws whatever coords a stage1 engine
produced (<name>_coords.csv = MediaPipe, <name>_coords_rtmp.csv = RTMPose).
Viz scripts no longer run pose inference inline — they load the cached CSV
(extracting once through the chosen stage1 engine if it doesn't exist yet),
so rendering matches the analysis pipeline exactly and is much faster.
"""
import os, sys
import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "stage1"))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
import config

CONNECTIONS = [
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]
_JOINTS = sorted(set(j for pair in CONNECTIONS for j in pair))


def backbone_suffix(backbone):
    return "_rtmp" if backbone == "rtmp" else ""


def csv_path(name, backbone, kind):
    """kind = 'coords' | 'smoothed'."""
    d = os.path.join(config.ROOT, "data", "outputs", name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{name}_{kind}{backbone_suffix(backbone)}.csv")


def find_video(name):
    for ext in (".mov", ".mp4", ".MOV", ".MP4"):
        p = os.path.join(config.ROOT, "data", "videos", name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no video found for '{name}' in data/videos")


def load_coords(name=None, backbone="rtmp"):
    """Cached raw coords, extracting via the chosen stage1 engine if missing."""
    name = name or config.VIDEO_NAME
    p = csv_path(name, backbone, "coords")
    if os.path.exists(p):
        return pd.read_csv(p)
    video = find_video(name)
    if backbone == "rtmp":
        from rtmp_extractor import extract_pose_rtmp
        df, _ = extract_pose_rtmp(video, save_csv=p)
    else:
        from pose_extractor import extract_pose
        df, _ = extract_pose(video, config.MODEL_PATH, save_csv=p)
    return df


def load_smoothed(name=None, backbone="rtmp"):
    """Cached smoothed coords (stage2), building from raw coords if missing."""
    name = name or config.VIDEO_NAME
    p = csv_path(name, backbone, "smoothed")
    if os.path.exists(p):
        return pd.read_csv(p)
    from smoother import smooth_coordinates
    sm = smooth_coordinates(load_coords(name, backbone))
    sm.to_csv(p, index=False)
    return sm


def square_crop(img, df, f, size=1200, pad=0.18):
    """Person-centered 1:1 crop -> size x size (paper figures need equal
    aspect). Call AFTER drawing joint-anchored geometry (full-res coords)
    and BEFORE drawing banner text (which should scale with the output)."""
    H, W = img.shape[:2]
    xs, ys = [], []
    for j in _JOINTS:
        try:
            x, y = df[f"{j}_x"].iloc[f], df[f"{j}_y"].iloc[f]
            if np.isfinite([x, y]).all():
                xs.append(float(x)); ys.append(float(y))
        except Exception:
            pass
    if not xs:
        side = min(W, H)
        ox, oy = (W - side) / 2, (H - side) / 2
    else:
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        side = min(max(x1 - x0, y1 - y0) * (1 + 2 * pad), W, H)
        ox = np.clip((x0 + x1) / 2 - side / 2, 0, W - side)
        oy = np.clip((y0 + y1) / 2 - side / 2, 0, H - side)
    crop = img[int(oy):int(oy + side), int(ox):int(ox + side)]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def draw_skeleton(frame, df, f, line_color=(0, 200, 255),
                  dot_color=(0, 255, 0), thickness=2, radius=6):
    """Draw the joint skeleton for frame index f from a coords DataFrame."""
    if f >= len(df):
        return frame
    for a, b in CONNECTIONS:
        xa, ya = df[f"{a}_x"].iloc[f], df[f"{a}_y"].iloc[f]
        xb, yb = df[f"{b}_x"].iloc[f], df[f"{b}_y"].iloc[f]
        if np.isfinite(xa) and np.isfinite(xb):
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                     line_color, thickness)
    for j in _JOINTS:
        x, y = df[f"{j}_x"].iloc[f], df[f"{j}_y"].iloc[f]
        if np.isfinite(x):
            cv2.circle(frame, (int(x), int(y)), radius, dot_color, -1)
    return frame
