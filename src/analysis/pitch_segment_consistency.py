"""
Diamond - 한 영상 멀티투구 분할 + 반복 일관성
pitch_segment_consistency.py

한 영상에 여러 투구(예: 5구)가 들어있을 때:
  1) 손목 속도 시계열에서 투구 피크들을 자동 검출 (scipy find_peaks)
  2) 각 피크 기준 [앞 pre초 ~ 뒤 post초] 구간으로 좌표를 분할
     (앞 3초 기본: trail anchor 가 와인드업 전 정지구간을 필요로 함)
  3) 구간별로 metrics.compute_candidates -> 투구별 지표
  4) 투구간 mean / SD / CV + 무릎각 중심 일관성 readout (OBP §8 검증)

전제: 카메라는 전체 영상 동안 삼각대 고정 (투구 간 카메라 변동 없음).
      -> 클립 분할 방식보다 오히려 깨끗한 일관성 데이터.

사용 (좌표 추출·스무딩 후):
  python pitch_segment_consistency.py \
    --coords ../../data/outputs/<name>/<name>_smoothed.csv \
    --fps 120 --pitches 5
  # 피크 검출이 빗나가면: --min-gap (투구 최소 간격 초) 조정, --plot 으로 확인
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(_HERE, "..", "stage3"))

import config
import metrics as M

GRADE = {
    "lead_knee_angle":     "A (SD r2=0.96)",
    "knee_ext_velo_br":    "B (SD r2=0.44)",
    "release_height":      "B (SD r2=0.38)",
    "stride_pct_height":   "참고 (SD r2=0.37)",
    "wrist_speed":         "참고 (SD r2=0.20)",
    "trunk_anterior_tilt": "참고 (SD r2=0.16)",
}
SHOW = list(GRADE.keys())


def detect_arm(df):
    def pk(j):
        x = df[f"{j}_x"].to_numpy(float); y = df[f"{j}_y"].to_numpy(float)
        return np.nanmax(np.hypot(np.diff(x), np.diff(y)))
    return "right" if pk("right_wrist") >= pk("left_wrist") else "left"


def wrist_speed_series(df, arm, fps):
    x = df[f"{arm}_wrist_x"].to_numpy(float)
    y = df[f"{arm}_wrist_y"].to_numpy(float)
    s = np.hypot(np.diff(x), np.diff(y)) * fps
    s = np.concatenate([[0.0], s])
    return pd.Series(s).interpolate(limit_direction="both").to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", required=True, help="스무딩된 좌표 CSV (영상 전체)")
    ap.add_argument("--fps", type=float, default=config.FPS_DEFAULT)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--pitches", type=int, default=5, help="기대 투구 수")
    ap.add_argument("--min-gap", type=float, default=4.0,
                    help="투구 간 최소 간격(초). 피크가 덜/더 잡히면 조정")
    ap.add_argument("--pre", type=float, default=3.0,
                    help="피크 앞 구간(초). trail anchor 용 정지구간 포함")
    ap.add_argument("--post", type=float, default=1.0, help="피크 뒤 구간(초)")
    ap.add_argument("--plot", action="store_true",
                    help="손목속도+피크 그래프 PNG 저장(검출 확인용)")
    a = ap.parse_args()

    df = pd.read_csv(a.coords)
    if "nose_x" in df.columns and "head_x" not in df.columns:
        df = df.rename(columns={"nose_x": "head_x", "nose_y": "head_y",
                                "nose_v": "head_v"})
    arm = a.arm or detect_arm(df)
    spd = wrist_speed_series(df, arm, a.fps)

    # ── 1) 투구 피크 검출 ──
    dist = max(int(a.min_gap * a.fps), 1)
    height = 0.45 * np.nanmax(spd)            # 전역 최고속의 45% 이상만 투구로
    peaks, props = find_peaks(spd, distance=dist, height=height)
    # 기대 투구 수보다 많이 잡히면 높은 순으로 상위 n개
    if len(peaks) > a.pitches:
        order = np.argsort(props["peak_heights"])[::-1][:a.pitches]
        peaks = np.sort(peaks[order])
    print(f"arm={arm}  frames={len(df)}  검출된 투구 피크: {len(peaks)}개 "
          f"(기대 {a.pitches})")
    for i, p in enumerate(peaks):
        print(f"  pitch {i+1}: frame {p}  (t={p/a.fps:.1f}s, "
              f"speed={spd[p]:.0f}px/s)")
    if len(peaks) < 2:
        print("\n[중단] 피크 2개 미만. --min-gap 줄이거나 --plot 으로 확인.")
        return
    if len(peaks) != a.pitches:
        print(f"\n[주의] 기대 {a.pitches}개와 다름. --plot 으로 검출 확인 권장.")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4))
        t = np.arange(len(spd)) / a.fps
        ax.plot(t, spd, lw=0.8)
        ax.plot(peaks / a.fps, spd[peaks], "rv", ms=10)
        ax.axhline(height, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("time (s)"); ax.set_ylabel("wrist speed (px/s)")
        ax.set_title(f"pitch peaks: {len(peaks)}")
        png = a.coords.replace(".csv", "_peaks.png")
        fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)
        print(f"피크 그래프 -> {png}")

    # ── 2) 구간 분할 + 3) 투구별 지표 ──
    pre = int(a.pre * a.fps); post = int(a.post * a.fps)
    rows = []
    for i, p in enumerate(peaks):
        lo, hi = max(0, p - pre), min(len(df), p + post)
        seg = df.iloc[lo:hi].reset_index(drop=True)
        try:
            cand = {k: v for k, (v, _) in
                    M.compute_candidates(seg, fps=a.fps, arm=arm,
                                         view="side").items()}
        except Exception as e:
            print(f"  pitch {i+1}: 계산 실패 ({e})")
            continue
        row = {"pitch": i + 1, "frame_lo": lo, "frame_hi": hi,
               "peak_frame": int(p)}
        row.update({m: cand.get(m, np.nan) for m in SHOW})
        rows.append(row)

    res = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("[투구별 측정값]")
    print("=" * 78)
    print(res.to_string(index=False))

    # ── 4) 일관성 ──
    print("\n" + "=" * 78)
    print("[투구간 일관성]  mean / SD / CV%  (일관성 신뢰등급: OBP §8)")
    print("=" * 78)
    print(f"{'지표':22s}{'mean':>9s}{'SD':>9s}{'CV%':>8s}   일관성등급")
    print("-" * 78)
    for m in SHOW:
        v = res[m].to_numpy(float); v = v[np.isfinite(v)]
        if len(v) < 2:
            continue
        mean, sd = v.mean(), v.std(ddof=1)
        cv = abs(sd / mean * 100) if mean != 0 else np.nan
        print(f"{m:22s}{mean:>9.2f}{sd:>9.3f}{cv:>8.1f}   {GRADE[m]}")

    ka = res["lead_knee_angle"].to_numpy(float); ka = ka[np.isfinite(ka)]
    if len(ka) >= 2:
        sd = ka.std(ddof=1)
        print("\n" + "=" * 78)
        print("[Release consistency — lead knee angle (검증된 일관성 지표)]")
        print("=" * 78)
        print(f"  무릎각 투구간 SD = {sd:.2f} deg  (n={len(ka)})")
        print(f"  참고: OBP 엘리트 평균 truSD = 4.3 deg.")
        print(f"  -> SD 작을수록 앞다리 블록이 매 투구 일정 (제구 토대).")

    out = a.coords.replace("_smoothed.csv", "_pitches.csv") \
                  .replace("_coords.csv", "_pitches.csv")
    res.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()