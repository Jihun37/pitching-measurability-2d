"""Geometry helpers over a table of joint coordinates.

Small, general operations on points and on per-frame series: angles, distances, speeds
and the frame at which a series turns. The measured quantities themselves live in
metrics.py and are defined once there; anything here is a building block, so a change
to a definition belongs there rather than in this file.
"""

import numpy as np
import pandas as pd


def angle(p1, p2, p3):
    """Angle in degrees at p2, between p2->p1 and p2->p3.

    Unsigned and on [0, 180], so it cannot tell a flexion from its mirror. The guard in
    the denominator keeps a collapsed segment returning 90 rather than a division by
    zero, which matters because a projected limb can foreshorten to nothing.
    """
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    v1 = p1 - p2
    v2 = p3 - p2
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def distance(p1, p2):
    """Distance between two image points, in pixels."""
    p1, p2 = np.array(p1), np.array(p2)
    return np.linalg.norm(p1 - p2)


def segment_angle(p1, p2):
    """Orientation of the segment p1->p2 against the image horizontal, in degrees.

    Signed, unlike angle(), and measured against a downward image vertical, so its sign
    is an image convention and not an anatomical one.
    """
    p1, p2 = np.array(p1), np.array(p2)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx))


def get_point(df, joint, frame):
    """The (x, y) of one joint at one frame."""
    return (df.loc[frame, f"{joint}_x"], df.loc[frame, f"{joint}_y"])


def joint_speed(df, joint, fps):
    """Per-frame speed of a joint, in pixels per second.

    A plain first difference scaled by the frame rate. The first frame has no
    predecessor and is reported as zero rather than dropped, so the series stays the
    same length as the table and a frame index means the same thing in both.
    """
    x = df[f"{joint}_x"].values
    y = df[f"{joint}_y"].values
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx**2 + dy**2) * fps
    return np.concatenate([[0], speed])


def angle_series(df, j1, j2, j3):
    """angle() at every frame, vertex at j2."""
    out = []
    for f in range(len(df)):
        out.append(angle(
            get_point(df, j1, f),
            get_point(df, j2, f),
            get_point(df, j3, f),
        ))
    return np.array(out)


def normalize_angle(deg):
    """Fold an angle onto (-180, 180], so 352 reads as -8."""
    return (deg + 180) % 360 - 180


def hip_shoulder_separation(df):
    """Per-frame angle between the shoulder line and the hip line.

    Unwrapped in radians before being returned in degrees. Without that, a rotation
    passing the +/-180 boundary registers as a 360 jump, and any peak read off the
    series afterwards lands on the discontinuity rather than on the rotation.
    """
    raw = []
    for f in range(len(df)):
        hip_ang = segment_angle(
            get_point(df, "left_hip", f),
            get_point(df, "right_hip", f),
        )
        sh_ang = segment_angle(
            get_point(df, "left_shoulder", f),
            get_point(df, "right_shoulder", f),
        )
        raw.append(sh_ang - hip_ang)

    unwrapped = np.degrees(np.unwrap(np.radians(raw)))
    return unwrapped


def peak_frame(series):
    """Frame of the maximum, ignoring NaN."""
    return int(np.nanargmax(series))


def min_frame(series):
    """Frame of the minimum, ignoring NaN."""
    return int(np.nanargmin(series))


def motion_start_frame(speed_series, threshold_ratio=0.2):
    """First frame whose speed exceeds a fraction of the clip's peak speed.

    Relative to the peak rather than absolute, so it does not depend on the pixel scale
    of the recording. It marks where motion begins, which is not one of the paper's
    temporal anchors.
    """
    peak = np.nanmax(speed_series)
    thr = peak * threshold_ratio
    for i, s in enumerate(speed_series):
        if s > thr:
            return i
    return 0


if __name__ == "__main__":
    import pandas as pd
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config

    df = pd.read_csv(config.SMOOTHED_CSV)
    fps = config.FPS_DEFAULT
    arm = config.THROWING_ARM

    elbow = angle_series(df, f"{arm}_shoulder", f"{arm}_elbow", f"{arm}_wrist")
    print(f"elbow angle    : min {np.nanmin(elbow):.0f} deg, max {np.nanmax(elbow):.0f} deg")

    wrist_spd = joint_speed(df, f"{arm}_wrist", fps)
    print(f"wrist speed    : peak {np.nanmax(wrist_spd):.0f} px/s at frame "
          f"{peak_frame(wrist_spd)}")

    hss = hip_shoulder_separation(df)
    print(f"hip-shoulder   : max |separation| {np.nanmax(np.abs(hss)):.0f} deg")
