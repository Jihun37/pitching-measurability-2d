"""
Diamond - 5단계 Level-A 검증
level_a_validate.py

질문: "우리 측면 2D 추정값이 OBP 3D 정답값을 얼마나 복원하는가?"
  -> 지표별 r(부호) / r²(일치도). 임시 등급(clean/depth/degenerate)을 측정값으로 교체.
부산물: 각 2D 지표 × 실제 구속(pitch_speed_mph) 상관 = Level-B 진단.
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

OBP_DATA = config.OBP_DATA_DIR
FEATURES = os.path.join(config.OBP_VALIDATION_DIR, "candidate_features_obp.csv")  # obp_project.py 배치 출력

# 우리 2D 지표 -> OBP 3D 정답 컬럼 (clean=직접대응 / loose=느슨)
MAP = {
    "arm_slot":            ("arm_slot",                          "clean"),
    # stride: the adopted 0.82 definition is the /stature version
    # (stride_pct_height); our "stride_length" (/body_scale) reads only 0.59.
    "stride_pct_height":   ("stride_length",                     "clean"),
    "lateral_trunk_tilt":  ("torso_anterior_tilt_br",            "clean"),
    "lead_knee_extension": ("lead_knee_extension_from_fp_to_br", "loose"),
    "wrist_peak_speed":    ("max_elbow_extension_velo",          "loose"),
}


def main():
    feat = pd.read_csv(FEATURES)
    # 우리 피처와 OBP 정답이 같은 이름(arm_slot, stride_length)이라 merge 충돌 -> 접두어
    feat = feat.rename(columns={c: f"our_{c}" for c in feat.columns
                                if c != "session_pitch"})
    poi = pd.read_csv(os.path.join(OBP_DATA, "poi", "poi_metrics.csv"))
    df = feat.merge(poi, on="session_pitch", how="inner")
    print(f"매칭된 투구: {len(df)}개\n")

    print("=" * 64)
    print("[Level-A] 우리 2D 추정값  vs  OBP 3D 정답값")
    print("=" * 64)
    print(f"{'our metric':22s}{'OBP truth':28s}{'r':>7s}{'r²':>7s}  type")
    print("-" * 64)
    rows = []
    for ours, (truth, kind) in MAP.items():
        oc = f"our_{ours}"
        if oc in df and truth in df:
            d = df[[oc, truth]].dropna()
            r = d[oc].corr(d[truth]) if len(d) > 2 else np.nan
            rows.append((ours, truth, r, kind))
    for ours, truth, r, kind in sorted(rows, key=lambda t: -abs(t[2])):
        verdict = ("신뢰" if abs(r) >= 0.7 else "보통" if abs(r) >= 0.4 else "낮음")
        print(f"{ours:22s}{truth:28s}{r:>7.3f}{r*r:>7.3f}  {kind:5s} [{verdict}]")

    print("\n" + "=" * 64)
    print("[Level-B 진단] 우리 2D 추정값 × 실제 구속")
    print("=" * 64)
    sp = "pitch_speed_mph"
    diag = []
    for ours in MAP:
        oc = f"our_{ours}"
        if oc in df:
            d = df[[oc, sp]].dropna()
            diag.append((ours, d[oc].corr(d[sp])))
    for ours, r in sorted(diag, key=lambda t: -abs(t[1])):
        print(f"  {ours:22s} r={r:+.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=FEATURES, help="obp_project 배치 출력 csv")
    a = ap.parse_args()
    FEATURES = a.features
    main()