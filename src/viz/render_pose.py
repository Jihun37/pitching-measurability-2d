"""
Diamond - 관절 시각화 영상 생성
render_pose.py
영상에 관절·연결선·각도를 그려서 출력 영상을 만든다. (검증용)

Renders from the cached coords CSV (backbone-agnostic) instead of running
pose inference inline — what you see is exactly what the analysis pipeline
measured. Extracts once via the chosen stage1 engine if no cache exists.

Run:  cd src\viz
      python render_pose.py                     # config.VIDEO_NAME, RTMPose
      python render_pose.py --backbone mediapipe
      python render_pose.py --video pitching_lateral_02
"""
import os, sys, argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from skeleton import load_coords, draw_skeleton, find_video, backbone_suffix


def calc_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def render(name, backbone, output_path=None, arm=None):
    df = load_coords(name, backbone)
    video_path = find_video(name)
    arm = arm or config.THROWING_ARM
    output_path = output_path or os.path.join(
        config.ROOT, "data", "outputs", name,
        f"{name}_output{backbone_suffix(backbone)}.mp4")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {width}x{height}, {fps:.1f}fps, {total} frames  ({backbone})")

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (width, height))
    f = 0
    while f < len(df):
        ret, frame = cap.read()
        if not ret:
            break
        draw_skeleton(frame, df, f)

        ex = df[f"{arm}_elbow_x"].iloc[f]
        if np.isfinite(ex):
            ang = calc_angle(
                (df[f"{arm}_shoulder_x"].iloc[f], df[f"{arm}_shoulder_y"].iloc[f]),
                (df[f"{arm}_elbow_x"].iloc[f], df[f"{arm}_elbow_y"].iloc[f]),
                (df[f"{arm}_wrist_x"].iloc[f], df[f"{arm}_wrist_y"].iloc[f]))
            cv2.putText(frame, f"{arm} elbow: {ang:.0f}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.putText(frame, f"Frame: {f}/{total}",
                    (width - 220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1)
        out.write(frame)
        f += 1
        if f % 60 == 0:
            print(f"  rendering: {f}/{total} ({f/total*100:.0f}%)")

    cap.release()
    out.release()
    print(f"output video: {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="video name (no ext); default config.VIDEO_NAME")
    ap.add_argument("--backbone", default="rtmp", choices=["rtmp", "mediapipe"])
    ap.add_argument("--arm", default=None, help="right/left; default config.THROWING_ARM")
    a = ap.parse_args()
    render(a.video or config.VIDEO_NAME, a.backbone, arm=a.arm)
