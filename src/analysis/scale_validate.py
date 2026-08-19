"""
Diamond - 3단계 스케일 표준화 검증
scale_validate.py

우리 stride_pct_height (stride_px / 픽셀신장) vs OBP stride_length(%body height).
순위(r²)뿐 아니라 '절대 일치'를 본다: 편향(bias)·RMSE, 보정상수 산출.
(각도 지표는 무차원이라 스케일 불필요 — 스트라이드만 대상)
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

FEATURES = os.path.join(config.OBP_VALIDATION_DIR, "candidate_features_obp.csv")
OBP_DATA = config.OBP_DATA_DIR


def main():
    feat = pd.read_csv(FEATURES)
    poi = pd.read_csv(os.path.join(OBP_DATA, "poi", "poi_metrics.csv"))
    df = feat.merge(
        poi[["session_pitch", "stride_length"]].rename(columns={"stride_length": "obp_stride"}),
        on="session_pitch", how="inner")
    d = df[["stride_pct_height", "obp_stride"]].dropna()
    our = d["stride_pct_height"].to_numpy()
    obp = d["obp_stride"].to_numpy()
    print(f"매칭 {len(d)}개\n")

    r = np.corrcoef(our, obp)[0, 1]
    print("[스케일 검증]  우리 stride_pct_height  vs  OBP stride_length(%height)")
    print("-" * 60)
    print(f"  상관 r = {r:.3f}   r² = {r*r:.3f}")
    print(f"  우리:  평균 {our.mean():.3f}  std {our.std():.3f}")
    print(f"  OBP:   평균 {obp.mean():.3f}  std {obp.std():.3f}")
    print(f"  보정 전 편향(우리-OBP) = {(our-obp).mean():+.3f}")
    print(f"  보정 전 RMSE          = {np.sqrt(((our-obp)**2).mean()):.3f}")

    # 보정: 단순 비율 k = OBP/우리 (편향 흡수)
    k = obp.mean() / our.mean()
    corr = our * k
    print(f"\n  보정상수 k = {k:.3f}  (stride_pct_height × k = 신장정규화 스트라이드)")
    print(f"  보정 후 편향 = {(corr-obp).mean():+.3f}")
    print(f"  보정 후 RMSE = {np.sqrt(((corr-obp)**2).mean()):.3f}")
    print(f"  -> 보정 후, 신장의 {corr.mean()*100:.0f}% 평균, "
          f"오차 ±{np.sqrt(((corr-obp)**2).mean())*100:.1f}%p (1σ)")

    # 선형 적합(slope/intercept)도 참고
    A = np.vstack([our, np.ones_like(our)]).T
    slope, icpt = np.linalg.lstsq(A, obp, rcond=None)[0]
    print(f"\n  (참고) 선형적합: OBP ≈ {slope:.3f}×우리 + {icpt:+.3f}")


if __name__ == "__main__":
    main()