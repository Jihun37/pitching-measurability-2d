"""
Diamond - 1단계: OBP 3D 지표로 구속 예측 '상한선' 잡기
phase1_velocity_model.py

목적
  - CV(영상→포즈) 단계 전에, OBP가 제공하는 '정답 품질' 3D 지표로
    pitch_speed_mph 가 얼마나 설명되는지 측정 = 우리 파이프라인의 천장.
  - 피처를 4그룹으로 나눠, '측면 2D로 측정 가능한 것만' 썼을 때의 R²를 따로 본다.

핵심 주의
  - 한 투수가 여러 투구를 가짐 -> GroupKFold(by user)로 투수 누수 차단.
    (안 그러면 R²가 '투수 암기'로 부풀려짐)
  - max_*_velo(팔/몸통 각속도)는 구속의 '출력'이라 거의 동어반복 -> 별도 그룹.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

ROOT = r"D:\project\diamond\data\datasets\OBP\openbiomechanics\baseball_pitching\data"
META = os.path.join(ROOT, "metadata.csv")
POI  = os.path.join(ROOT, "poi", "poi_metrics.csv")

# ── 피처 그룹 ───────────────────────────────────────────
ANTHRO = ["session_height_m", "session_mass_kg", "age_yrs"]

# 측면 2D 시상면에서 복원 가능성이 있는 운동학(=우리의 현실 목표)
TWO_D = [
    "stride_length", "stride_angle", "arm_slot",
    "torso_anterior_tilt_fp", "torso_anterior_tilt_br",
    "elbow_flexion_fp", "max_elbow_flexion",
    "lead_knee_extension_angular_velo_fp",
    "lead_knee_extension_angular_velo_br",
    "lead_knee_extension_angular_velo_max",
    "lead_knee_extension_from_fp_to_br",
]
# 회전(횡단면): 측면에서도 못 잡음 -> 빼면 얼마 손해인지 확인용
ROTATION = [
    "rotation_hip_shoulder_separation_fp", "max_rotation_hip_shoulder_separation",
    "torso_rotation_fp", "torso_rotation_br", "pelvis_rotation_fp",
]
# 출력 각속도: 구속과 거의 동어반복 -> 상한선 참고용
OUTPUT_VELO = [
    "max_shoulder_internal_rotational_velo", "max_elbow_extension_velo",
    "max_torso_rotational_velo", "max_pelvis_rotational_velo",
]
TARGET = "pitch_speed_mph"


def load():
    md = pd.read_csv(META)
    poi = pd.read_csv(POI)
    df = poi.merge(md[["session_pitch", "user"] + ANTHRO],
                   on="session_pitch", how="inner")
    if TARGET + "_x" in df:  # 머지 중복 정리
        df[TARGET] = df[TARGET + "_x"]
    df = df.dropna(subset=[TARGET, "user"])
    return df


def evaluate(df, feats, label, model="gbm"):
    """투수단위 GroupKFold 교차검증 R²/RMSE."""
    feats = [f for f in feats if f in df.columns]
    X = df[feats].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    groups = df["user"].to_numpy()
    if model == "linear":
        est = make_pipeline(SimpleImputer(strategy="median"),
                            StandardScaler(), LinearRegression())
    else:
        est = make_pipeline(SimpleImputer(strategy="median"),
                            GradientBoostingRegressor(random_state=0,
                                n_estimators=300, max_depth=3, learning_rate=0.03))
    cv = GroupKFold(n_splits=5)
    pred = cross_val_predict(est, X, y, groups=groups, cv=cv)
    r2 = r2_score(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    print(f"  {label:36s} n_feat={len(feats):2d}  CV R²={r2:5.3f}  RMSE={rmse:4.2f} mph")
    return r2, rmse


def single_feature_corr(df):
    feats = ANTHRO + TWO_D + ROTATION
    rows = []
    for f in feats:
        if f in df.columns:
            rows.append((f, df[f].corr(df[TARGET])))
    print("\n[단일 피처 × 구속 상관 (절댓값 순)]")
    for f, r in sorted(rows, key=lambda t: -abs(t[1])):
        tag = ("anthro" if f in ANTHRO else "2D" if f in TWO_D else "rotation")
        print(f"  {f:42s} r={r:+.3f}   [{tag}]")


def gbm_importance(df, feats, label):
    feats = [f for f in feats if f in df.columns]
    X = df[feats].to_numpy(float); y = df[TARGET].to_numpy(float)
    est = make_pipeline(SimpleImputer(strategy="median"),
                        GradientBoostingRegressor(random_state=0,
                            n_estimators=300, max_depth=3, learning_rate=0.03))
    est.fit(X, y)
    imp = est[-1].feature_importances_
    print(f"\n[{label} - GBM 피처 중요도 상위]")
    for f, w in sorted(zip(feats, imp), key=lambda t: -t[1])[:8]:
        print(f"  {f:42s} {w:.3f}")


def main():
    df = load()
    print(f"투구 {len(df)}개 / 투수 {df['user'].nunique()}명 / 구속 "
          f"{df[TARGET].mean():.1f}±{df[TARGET].std():.1f} mph "
          f"({df[TARGET].min():.0f}~{df[TARGET].max():.0f})")

    print("\n[교차검증 R² / RMSE]  (투수단위 GroupKFold)")
    evaluate(df, ANTHRO, "M0 신장·체중·나이만 (baseline)")
    evaluate(df, ANTHRO + TWO_D, "M1 +측면2D 운동학 (현실 목표)")
    evaluate(df, ANTHRO + TWO_D + ROTATION, "M2 +회전지표 (측면에선 불가)")
    evaluate(df, ANTHRO + TWO_D + ROTATION + OUTPUT_VELO, "M3 +출력각속도 (동어반복 상한)")
    evaluate(df, ANTHRO + TWO_D, "M1 (선형회귀 버전)", model="linear")

    single_feature_corr(df)
    gbm_importance(df, ANTHRO + TWO_D, "M1 현실 목표")


if __name__ == "__main__":
    main()