"""
Diamond - 공통 엔진 2
smoother.py
MediaPipe가 뽑은 관절 좌표의 노이즈(흔들림)를 제거한다.
SciPy의 Savitzky-Golay 필터 사용.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def smooth_coordinates(df, window=7, polyorder=2, visibility_threshold=0.5):
    """
    모든 관절 좌표 컬럼(_x, _y)에 Savitzky-Golay 필터를 적용해 노이즈를 제거한다.

    Parameters
    ----------
    df                   : DataFrame   pose_extractor가 만든 좌표 데이터
    window               : int         필터 윈도우 크기 (홀수, 클수록 부드러움)
    polyorder            : int         다항식 차수 (보통 2)
    visibility_threshold : float       신뢰도가 이 값 미만인 좌표는 튄 값으로 보고
                                       NaN 처리 후 보간 (0~1)

    Returns
    -------
    smoothed : DataFrame   노이즈가 제거된 좌표 데이터
    """
    smoothed = df.copy()

    # 1) 신뢰도(_v) 낮은 프레임의 좌표를 NaN으로 (튀는 값 제거)
    joint_names = set(c[:-2] for c in df.columns if c.endswith("_x"))
    for joint in joint_names:
        vcol = f"{joint}_v"
        if vcol in df.columns:
            bad = df[vcol] < visibility_threshold
            smoothed.loc[bad, f"{joint}_x"] = np.nan
            smoothed.loc[bad, f"{joint}_y"] = np.nan

    # 2) _x, _y 컬럼에 필터 적용
    coord_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]

    for col in coord_cols:
        series = smoothed[col].values

        # NaN(감지 실패·저신뢰)을 선형 보간으로 채움
        if np.isnan(series).any():
            series = pd.Series(series).interpolate(
                method="linear", limit_direction="both"
            ).values

        win = min(window, len(series) if len(series) % 2 == 1 else len(series) - 1)
        if win < polyorder + 2:
            smoothed[col] = series
            continue

        smoothed[col] = savgol_filter(series, window_length=win, polyorder=polyorder)

    return smoothed


def smoothing_report(df_raw, df_smooth, joint="right_wrist"):
    """
    보정 전/후 흔들림(노이즈)이 얼마나 줄었는지 비교 출력.
    """
    for axis in ["x", "y"]:
        col = f"{joint}_{axis}"
        raw_jitter    = np.nanstd(np.diff(df_raw[col].values))
        smooth_jitter = np.nanstd(np.diff(df_smooth[col].values))
        print(f"{col}: 흔들림 {raw_jitter:.2f} → {smooth_jitter:.2f} "
              f"({(1-smooth_jitter/raw_jitter)*100:.0f}% 감소)")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config

    df_raw = pd.read_csv(config.COORDS_CSV)
    df_smooth = smooth_coordinates(df_raw)
    df_smooth.to_csv(config.SMOOTHED_CSV, index=False)

    print("노이즈 제거 완료")
    smoothing_report(df_raw, df_smooth)