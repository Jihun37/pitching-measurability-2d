"""
Diamond - match_analyze.py  (수정판)
변경점:
  (1) shoulder/hip 의 .max() 포화 지표 제거
  (2) HSS 선분각 단일지표 -> 후보군 전체(metrics.compute_candidates)로 확장
  (3) 좌표에서 직접 계산 -> 우리 영상과 '같은 함수'로 검증 (교집합)
  (4) 모든 후보 × release_speed 상관을 한 표로 랭킹

※ 데이터셋 좌표 컬럼명이 우리와 다르면 아래 DATASET_JOINTS 만 채우면 됨.
   (inspect_dataset.py 의 컬럼 리스트로 매핑)
"""
import os, sys
import pandas as pd, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
import metrics as M   # stage2/metrics.py

DATASET_DIR  = r"D:\project\diamond\data\datasets\pitcher_motion"
MOTION_CSV   = os.path.join(DATASET_DIR, "Pitcher_Motion_Data.csv")
STATCAST_CSV = os.path.join(DATASET_DIR, "Pitcher_Motion_Data_Statcast_Companion.csv")
OUT_DIR      = r"D:\project\diamond\data\outputs\dataset_analysis"
os.makedirs(OUT_DIR, exist_ok=True)
FPS = 60

# 데이터셋 컬럼명 -> 우리 표준 이름. inspect_dataset.py 결과로 채울 것.
# 예시(데이터셋이 'shoulder_L_x' 식이면): {"left_shoulder":"shoulder_L", ...}
DATASET_JOINTS = None   # None 이면 metrics.JOINTS(우리 이름) 그대로 사용

def main():
    print("Loading motion data...")
    motion = pd.read_csv(MOTION_CSV)
    if {"no_missing_frames","smooth_CoM_flag"}.issubset(motion.columns):
        motion = motion[(motion.no_missing_frames==1) & (motion.smooth_CoM_flag==1)]
    print(f"  rows after filter: {len(motion):,}")

    if DATASET_JOINTS:                      # 컬럼명이 다르면 rename
        ren = {}
        for std, ds in DATASET_JOINTS.items():
            ren[f"{ds}_x"] = f"{std}_x"; ren[f"{ds}_y"] = f"{std}_y"
        motion = motion.rename(columns=ren)

    p_throws = (pd.read_csv(STATCAST_CSV)[["pitcher","p_throws"]]
                .drop_duplicates("pitcher").set_index("pitcher")["p_throws"].to_dict())

    rows = []
    keys = ["pitcher","pitch_type","pitch_id"]
    for (pid, ptype, pno), g in motion.groupby(keys):
        g = g.reset_index(drop=True)
        arm = "left" if str(p_throws.get(pid,"R")).upper().startswith("L") else "right"
        try:
            c = M.compute_candidates(g, fps=FPS, arm=arm)
        except KeyError:
            continue   # 필요한 관절 컬럼이 없는 투구는 건너뜀
        row = {"pitcher":pid, "pitch_type":ptype, "pitch_id":pno, "n_frames":len(g)}
        for name,(val,_) in c.items(): row[name] = val
        rows.append(row)
    feat = pd.DataFrame(rows)
    print(f"  pitches with features: {len(feat)}")

    sc = pd.read_csv(STATCAST_CSV)[["pitcher","pitch_type","pitch_id",
                                    "release_speed","release_spin_rate"]]
    df = feat.merge(sc, on=keys, how="inner").dropna(subset=["release_speed"])
    print(f"  matched: {len(df)}")

    cand_names = [k for k in feat.columns if k not in keys + ["n_frames"]]
    grade = M.CANDIDATE_GRADES

    print("\n" + "="*58)
    print(f"{'metric':24s}{'r(speed)':>10s}{'r(spin)':>10s}   grade")
    print("="*58)
    table = []
    for n in cand_names:
        rs = df[n].corr(df["release_speed"])
        rp = df[n].corr(df["release_spin_rate"]) if "release_spin_rate" in df else np.nan
        table.append((n, rs, rp, grade.get(n,"?")))
    for n,rs,rp,gr in sorted(table, key=lambda t:-abs(t[1])):
        print(f"{n:24s}{rs:>10.3f}{rp:>10.3f}   {gr}")

    df.to_csv(os.path.join(OUT_DIR,"candidate_features.csv"), index=False)
    pd.DataFrame(table, columns=["metric","r_speed","r_spin","grade"]
                 ).to_csv(os.path.join(OUT_DIR,"correlation_ranking.csv"), index=False)
    print(f"\nsaved -> {OUT_DIR}\\correlation_ranking.csv")

if __name__ == "__main__":
    main()