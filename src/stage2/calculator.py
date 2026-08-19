"""
Diamond - 공통 엔진 3
calculator.py
보정된 관절 좌표로 각도·속도·거리 등을 계산하는 함수 모음.
모든 드릴별 모듈이 여기서 함수를 가져다 쓴다.
"""

import numpy as np
import pandas as pd


# ── 기본 계산 함수 ──────────────────────────────────────

def angle(p1, p2, p3):
    """
    세 점 p1-p2-p3 가 이루는 각도(도)를 반환. p2가 꼭짓점.
    예: angle(어깨, 팔꿈치, 손목) = 팔꿈치 각도
    """
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    v1 = p1 - p2
    v2 = p3 - p2
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def distance(p1, p2):
    """두 점 사이의 거리(픽셀)."""
    p1, p2 = np.array(p1), np.array(p2)
    return np.linalg.norm(p1 - p2)


def segment_angle(p1, p2):
    """
    두 점을 잇는 선분이 수평선과 이루는 각도(도).
    예: 양 골반을 이으면 골반의 회전(기울기)을 알 수 있다.
    """
    p1, p2 = np.array(p1), np.array(p2)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx))


# ── DataFrame 기반 계산 ─────────────────────────────────

def get_point(df, joint, frame):
    """특정 프레임의 관절 좌표 (x, y) 반환."""
    return (df.loc[frame, f"{joint}_x"], df.loc[frame, f"{joint}_y"])


def joint_speed(df, joint, fps):
    """
    관절의 프레임별 이동 속도(픽셀/초) 시계열을 반환.
    손목 속도 → 스윙·투구 파워 추정에 사용.
    """
    x = df[f"{joint}_x"].values
    y = df[f"{joint}_y"].values
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx**2 + dy**2) * fps
    return np.concatenate([[0], speed])   # 첫 프레임은 0


def angle_series(df, j1, j2, j3):
    """
    세 관절로 이루는 각도의 시계열(프레임별)을 반환.
    예: angle_series(df, 'right_shoulder', 'right_elbow', 'right_wrist')
        → 매 프레임의 오른쪽 팔꿈치 각도
    """
    out = []
    for f in range(len(df)):
        out.append(angle(
            get_point(df, j1, f),
            get_point(df, j2, f),
            get_point(df, j3, f),
        ))
    return np.array(out)


def normalize_angle(deg):
    """
    각도를 -180 ~ 180도 범위로 정규화.
    352도 같은 값을 -8도로 바꿔준다.
    """
    return (deg + 180) % 360 - 180


def hip_shoulder_separation(df):
    """
    힙-어깨 분리(hip-shoulder separation) 시계열을 반환.
    스윙·투구의 핵심 지표. 골반 회전과 어깨 회전의 각도 차이.

    np.unwrap으로 ±180도 경계에서 생기는 점프를 제거해
    연속적인 회전 흐름으로 만든다.
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

    # 라디안으로 바꿔 unwrap 후 다시 도(degree)로
    unwrapped = np.degrees(np.unwrap(np.radians(raw)))
    return unwrapped


# ── 이벤트(순간) 감지 헬퍼 ──────────────────────────────

def peak_frame(series):
    """시계열에서 최댓값이 나오는 프레임 번호."""
    return int(np.nanargmax(series))


def min_frame(series):
    """시계열에서 최솟값이 나오는 프레임 번호."""
    return int(np.nanargmin(series))


def motion_start_frame(speed_series, threshold_ratio=0.2):
    """
    움직임이 시작되는 프레임 감지.
    속도가 (최고속도 × threshold_ratio)를 처음 넘는 지점.
    드릴 출발·동작 시작 감지에 사용.
    """
    peak = np.nanmax(speed_series)
    thr  = peak * threshold_ratio
    for i, s in enumerate(speed_series):
        if s > thr:
            return i
    return 0


if __name__ == "__main__":
    # 사용 예시 (config 기반)
    import pandas as pd
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config

    df = pd.read_csv(config.SMOOTHED_CSV)
    fps = config.FPS_DEFAULT
    arm = config.THROWING_ARM

    # 팔꿈치 각도 시계열
    elbow = angle_series(df, f"{arm}_shoulder", f"{arm}_elbow", f"{arm}_wrist")
    print(f"팔꿈치 각도: 최소 {np.nanmin(elbow):.0f}도, 최대 {np.nanmax(elbow):.0f}도")

    # 손목 속도
    wrist_spd = joint_speed(df, f"{arm}_wrist", fps)
    print(f"손목 최고 속도: {np.nanmax(wrist_spd):.0f} 픽셀/초")
    print(f"속도 피크 프레임: {peak_frame(wrist_spd)}")

    # 힙-어깨 분리
    hss = hip_shoulder_separation(df)
    print(f"힙-어깨 분리: 최대 {np.nanmax(np.abs(hss)):.0f}도")