"""
Diamond - 4단계: 실제 영상 도메인 체크
realvideo_check.py

stage1/stage2가 뽑은 좌표 CSV(우리 폰 영상)를 metrics로 돌려,
신뢰 3종 지표가 OBP-투영 분포(candidate_features_obp.csv) 안에 떨어지는지 확인.
떨어지면 = 우리 촬영이 OBP 검증과 같은 도메인 = 지표 신뢰 가능.

사용:
    python realvideo_check.py --coords ../../data/outputs/test_03/test_03_smoothed.csv --fps 30
"""
import os, sys, argparse
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
import metrics as M
from smoother import smooth_coordinates

REF = os.path.join(config.OBP_VALIDATION_DIR, "candidate_features_obp.csv")
ARMSLOT_REF = os.path.join(config.OBP_VALIDATION_DIR, "armslot_ref.csv")
# 새로 검증된 측면(0°) 5종 -> OBP-투영 분포 참조 컬럼.
# 주의: 참조 CSV(candidate_features_obp.csv)는 새 metrics 정의로 재생성돼 있어야 함
#       (먼저: python obp_project.py 배치 재실행). knee_ext_velo_br·wrist_speed 는
#       구버전 CSV엔 없으므로 재생성 전에는 '참조 없음'으로 표시됨.
CORE = {
    "lead_knee_angle":     "lead_knee_at_release",   # 절대각 (참조는 별칭 컬럼)
    "stride_pct_height":   "stride_pct_height",
    "trunk_anterior_tilt": "lateral_trunk_tilt",     # 값 동일, 참조는 별칭 컬럼
    "knee_ext_velo_br":    "knee_ext_velo_br",
    "wrist_speed":         "wrist_speed",
}


def detect_arm(df, fps):
    """던지는 팔 자동검출 (노이즈 강건판).
    단일 최고속 프레임(max)은 raw 좌표의 스파이크 1개에 속음 -> 95퍼센타일 사용.
    좌투 영상에서 글러브 손 스파이크로 오판된 사례의 수정."""
    def pk(j):
        x = df[f"{j}_x"].to_numpy(float); y = df[f"{j}_y"].to_numpy(float)
        s = np.hypot(np.diff(x), np.diff(y))
        s = s[np.isfinite(s)]
        return np.percentile(s, 95) if len(s) else 0.0
    return "right" if pk("right_wrist") >= pk("left_wrist") else "left"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", required=True, help="stage1/stage2 좌표 CSV")
    ap.add_argument("--fps", type=float, default=config.FPS_DEFAULT)
    ap.add_argument("--arm", default=None, help="right/left (생략시 자동검출)")
    ap.add_argument("--smooth", type=int, default=0,
                    help="SG 윈도우(프레임). 0=이미 smoothed.csv면 생략")
    ap.add_argument("--view", default="side", choices=["side", "frontal"],
                    help="side=측면 3종(trunk·knee·stride), frontal=정면 arm slot")
    ap.add_argument("--release", type=int, default=None,
                    help="릴리즈 프레임 수동 지정(공 없이 폼만 찍어 자동검출이 틀릴 때)")
    a = ap.parse_args()

    df = pd.read_csv(a.coords)
    if "_coords" in os.path.basename(a.coords) and not a.smooth:
        print("[경고] raw 좌표(_coords.csv) 사용 중. 노이즈 스파이크로 팔/릴리즈")
        print("       검출이 틀릴 수 있음 -> _smoothed.csv 사용 또는 --smooth 7 권장.\n")
    # 우리 pose_extractor 는 머리를 'nose'로 저장 -> metrics 는 'head' 사용
    if "nose_x" in df.columns and "head_x" not in df.columns:
        df = df.rename(columns={"nose_x": "head_x", "nose_y": "head_y", "nose_v": "head_v"})
    if a.smooth and a.smooth >= 5:
        df = smooth_coordinates(df, window=a.smooth)

    arm = a.arm or detect_arm(df, a.fps)
    if not a.arm:
        print(f"[참고] 팔 자동검출: {arm}. 틀리면 --arm left/right 로 수동 지정.")
    # raw 좌표가 옆에 있으면 릴리즈 프레임 정밀화에 사용 (release_frame 참조)
    raw_path = a.coords.replace("_smoothed", "_coords")
    raw_df = None
    if raw_path != a.coords and os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        if "nose_x" in raw_df.columns and "head_x" not in raw_df.columns:
            raw_df = raw_df.rename(columns={"nose_x": "head_x", "nose_y": "head_y"})
    cand = M.compute_candidates(df, fps=a.fps, arm=arm, view=a.view, raw_df=raw_df)
    vals = {k: v for k, (v, _) in cand.items()}

    print(f"던지는 팔: {arm} / fps: {a.fps:.0f} / 프레임: {len(df)} / view: {a.view}\n")

    if a.view == "frontal":
        # 정면: arm slot (shoulder->hand vs 수직) 을 OBP 3D 정면 정답 분포와 대조
        ours = vals.get("arm_slot", np.nan)
        if a.release is not None:                 # 릴리즈 수동 지정
            rel = int(a.release)
            sx, sy = df[f"{arm}_shoulder_x"].iloc[rel], df[f"{arm}_shoulder_y"].iloc[rel]
            wx, wy = df[f"{arm}_wrist_x"].iloc[rel],   df[f"{arm}_wrist_y"].iloc[rel]
            ours = float(np.degrees(np.arctan2(abs(wx-sx), (sy-wy))))
            print(f"[릴리즈 수동지정] frame{rel}  arm_slot 재계산 = {ours:.1f}\n")
        print("[도메인 체크 · 정면]  arm slot  vs  OBP 3D 정면 정답 분포")
        print(f"{'지표':14s}{'우리값':>9s}{'OBP평균':>9s}{'OBP std':>9s}{'z':>7s}  판정")
        print("-" * 56)
        if os.path.exists(ARMSLOT_REF):
            ref = pd.read_csv(ARMSLOT_REF)["arm_slot_truth"]
            mu, sd = ref.mean(), ref.std()
            z = (ours - mu) / sd if sd > 0 else np.nan
            ok = "정상범위" if abs(z) <= 2 else "범위밖(촬영/검출 의심)"
            print(f"{'arm_slot':14s}{ours:>9.2f}{mu:>9.2f}{sd:>9.2f}{z:>7.2f}  {ok}")
        else:
            print(f"{'arm_slot':14s}{ours:>9.2f}   (armslot_ref.csv 없음 -> armslot_validate.py --batch 먼저)")
        return

    # 측면(기본): 새 검증 5종
    ref = pd.read_csv(REF) if os.path.exists(REF) else None
    print("[도메인 체크]  우리 영상  vs  OBP-투영 분포")
    print(f"{'지표':22s}{'우리값':>10s}{'OBP평균':>10s}{'OBP std':>9s}{'z':>7s}  판정")
    print("-" * 70)
    for ours, refcol in CORE.items():
        v = vals.get(ours, np.nan)
        if ref is not None and refcol in ref.columns:
            mu, sd = ref[refcol].mean(), ref[refcol].std()
            z = (v - mu) / sd if sd > 0 else np.nan
            ok = "정상범위" if abs(z) <= 2 else "범위밖(촬영/검출 의심)"
            print(f"{ours:22s}{v:>10.2f}{mu:>10.2f}{sd:>9.2f}{z:>7.2f}  {ok}")
        else:
            tag = "(참조 컬럼 없음 - obp_project 재생성 필요)" if ref is not None else "(참조 CSV 없음)"
            print(f"{ours:22s}{v:>10.2f}   {tag}")

    print("\n[참고] 전체 후보 지표:")
    for k, v in vals.items():
        print(f"  {k:24s} {v:8.2f}")


if __name__ == "__main__":
    main()