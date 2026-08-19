"""
Diamond - engine 1b
rtmp_extractor.py
Video -> joint coords via RTMPose (rtmlib BodyWithFeet, Halpe26, ONNX).

Official backbone for the overhead pipeline since 2026-07-04 (MediaPipe
pose_landmarker dies on top-down video: full detection loss during the arm
whip, shadow tracking — see backbone_diag_rtmp.csv). Same CSV schema and
(df, meta) interface as pose_extractor.extract_pose, so stage2 and every
downstream consumer work unchanged.

Speed: ~0.8 s/frame on CPU at 4K (offline processing, not real-time).
Models are auto-downloaded by rtmlib on first use (~500 MB cache).
"""
import cv2
import numpy as np
import pandas as pd

# Halpe26 indices -> our joint names (same names/order as pose_extractor.JOINTS;
# foot = big toe, the closest Halpe point to MediaPipe's foot_index)
HALPE26 = {
    "nose": 0,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
    "left_foot": 20, "right_foot": 21,
}
CORE = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]

# person-selection (select="track"): temporal-continuity gate. Verified need
# 2026-07-11: per-frame conf argmax hijacks onto BACKGROUND people (18/60
# holdout clips had hip teleports up to ~1900 px / 3 statures; clean clips
# 0 teleports, max 28 px), which is what produced the az0/az30/az330 release
# defers and part of the az270 late bias. The gate radius must admit the
# fastest real between-frame body motion (~30 px at 120 fps / 600 px
# stature) while rejecting cross-person jumps (>500 px in those clips).
TRACK_RADIUS = 0.45          # x person bbox diagonal, per missed frame +50%
TRACK_RADIUS_MAX = 1.5       # growth cap (x diagonal)
TRACK_MIN_CONF = 0.3         # detections below this never (re)seed the track


def _core_geom(kpts, i, core_idx):
    """(center, bbox diagonal) of person i from all keypoints."""
    c = kpts[i, core_idx].mean(axis=0)
    dx = kpts[i, :, 0].max() - kpts[i, :, 0].min()
    dy = kpts[i, :, 1].max() - kpts[i, :, 1].min()
    return c, float(np.hypot(dx, dy))


def extract_pose_rtmp(video_path, save_csv=None, mode="performance",
                      select="track"):
    """
    Extract per-frame joint coordinates with RTMPose.

    Parameters
    ----------
    video_path : str   video to analyze
    save_csv   : str   CSV output path (None = don't save)
    mode       : str   rtmlib preset: performance / balanced / lightweight
    select     : str   multi-person selection:
                       "track"  = seed on the largest confident person (the
                         subject is nearest the camera), then follow by
                         center continuity; frames with no candidate inside
                         the gate stay NaN instead of jumping to a
                         background person. Default since 2026-07-11.
                       "argmax" = legacy per-frame max core confidence
                         (matches every extraction made before 2026-07-11).

    Returns
    -------
    df   : DataFrame   per-frame joint coords in px (joint_x/_y, joint_v=score)
    meta : dict        fps, width, height, total_frames
    """
    from rtmlib import BodyWithFeet
    model = BodyWithFeet(mode=mode, backend="onnxruntime", device="cpu")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = {"fps": fps, "width": width, "height": height, "total_frames": total}
    print(f"video: {width}x{height}, {fps:.1f}fps, {total} frames  (rtmpose {mode})")

    core_idx = [HALPE26[j] for j in CORE]
    rows, f, det = [], 0, 0
    prev_c, prev_diag, miss, gated = None, None, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        kpts, scores = model(frame)
        row = {"frame": f}
        i = None
        if len(kpts):
            conf = scores[:, core_idx].mean(axis=1)
            if select != "track":
                i = int(np.argmax(conf))              # legacy main person
            elif prev_c is None:
                # seed: largest confident person = the subject
                cand = np.where(conf >= TRACK_MIN_CONF)[0]
                if len(cand):
                    i = int(cand[np.argmax([_core_geom(kpts, j, core_idx)[1]
                                            for j in cand])])
            else:
                cents = np.array([_core_geom(kpts, j, core_idx)[0]
                                  for j in range(len(kpts))])
                d = np.hypot(*(cents - prev_c).T)
                r = prev_diag * min(TRACK_RADIUS * 1.5 ** miss,
                                    TRACK_RADIUS_MAX)
                near = np.where(d <= r)[0]
                if len(near):
                    i = int(near[np.argmin(d[near])])
                else:
                    gated += 1                        # NaN beats wrong person
        if i is not None:
            det += 1
            if select == "track":
                prev_c, prev_diag = _core_geom(kpts, i, core_idx)
                miss = 0
            for name, idx in HALPE26.items():
                row[f"{name}_x"] = float(kpts[i, idx, 0])
                row[f"{name}_y"] = float(kpts[i, idx, 1])
                row[f"{name}_v"] = float(scores[i, idx])
        else:
            miss += 1
            for name in HALPE26:
                row[f"{name}_x"] = np.nan
                row[f"{name}_y"] = np.nan
                row[f"{name}_v"] = np.nan
        rows.append(row)
        f += 1
        if f % 60 == 0:
            print(f"  rtmpose: {f}/{total} ({f/total*100:.0f}%)", flush=True)
    cap.release()

    df = pd.DataFrame(rows)
    print(f"rtmpose detection: {det}/{total} frames ({det/max(total,1)*100:.1f}%)"
          + (f"  [track: {gated} frames gated to NaN]"
             if select == "track" and gated else ""))
    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"CSV saved: {save_csv}")
    return df, meta


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config
    out = config.COORDS_CSV.replace(".csv", "_rtmp.csv")
    df, meta = extract_pose_rtmp(config.VIDEO_PATH, save_csv=out)
    print(df.head())
