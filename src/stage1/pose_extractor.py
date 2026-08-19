"""
Joint extraction from video, with MediaPipe.

The first stage of the video path. RTMPose in stage1/rtmp_extractor.py is the
official backbone; this one is kept as the legacy alternative and writes the same
CSV schema.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerOptions, RunningMode


# Joint indices as MediaPipe numbers them.
JOINTS = {
    "nose":           0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
    "left_foot":      31,
    "right_foot":     32,
}


def extract_pose(video_path, model_path, save_csv=None):
    """
    Extract per-frame joint coordinates from a clip.

    Parameters
    ----------
    video_path : str   the clip to read
    model_path : str   MediaPipe pose_landmarker.task
    save_csv   : str   where to write the result; None writes nothing

    Returns
    -------
    df   : DataFrame   joint coordinates per frame, in pixels
    meta : dict        fps, width, height, total_frames
    """
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open the clip: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    meta = {"fps": fps, "width": width, "height": height, "total_frames": total}
    print(f"clip: {width}x{height}, {fps:.1f} fps, {total} frames")

    rows = []
    frame_idx = 0
    detected  = 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            timestamp_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            row = {"frame": frame_idx}
            if result.pose_landmarks:
                detected += 1
                lm = result.pose_landmarks[0]
                for name, idx in JOINTS.items():
                    row[f"{name}_x"] = lm[idx].x * width
                    row[f"{name}_y"] = lm[idx].y * height
                    row[f"{name}_v"] = lm[idx].visibility   # confidence
            else:
                # NaN where the detector found nothing
                for name in JOINTS:
                    row[f"{name}_x"] = np.nan
                    row[f"{name}_y"] = np.nan
                    row[f"{name}_v"] = np.nan

            rows.append(row)
            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"  {frame_idx}/{total} ({frame_idx/total*100:.0f}%)")

    cap.release()

    df = pd.DataFrame(rows)
    print(f"joints detected in {detected}/{total} frames ({detected/total*100:.1f}%)")

    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"csv written: {save_csv}")

    return df, meta


if __name__ == "__main__":
    # paths come from config.py, one level up in src/
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config

    df, meta = extract_pose(config.VIDEO_PATH, config.MODEL_PATH,
                            save_csv=config.COORDS_CSV)
    print(df.head())