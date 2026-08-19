"""
Candidate quantities against release speed, on an external dataset.

Revised from the first version in four ways:
  1. the saturating shoulder and hip .max() quantities are dropped;
  2. hip-shoulder separation as a single segment angle is replaced by the whole
     candidate set from metrics.compute_candidates;
  3. the quantities are computed by the SAME functions our own clips go through,
     rather than reimplemented from the coordinates here, which is what makes the
     comparison mean anything;
  4. every candidate is correlated against release_speed and ranked in one table.

Where the dataset names its coordinate columns differently, only DATASET_JOINTS
below needs filling, from the column list inspect_dataset.py prints.
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

# Dataset column names -> our standard names. Fill from what inspect_dataset.py
# reports. For a dataset naming joints "shoulder_L_x", that is
# {"left_shoulder": "shoulder_L", ...}.
DATASET_JOINTS = None   # None uses metrics.JOINTS unchanged

def main():
    print("Loading motion data...")
    motion = pd.read_csv(MOTION_CSV)
    if {"no_missing_frames","smooth_CoM_flag"}.issubset(motion.columns):
        motion = motion[(motion.no_missing_frames==1) & (motion.smooth_CoM_flag==1)]
    print(f"  rows after filter: {len(motion):,}")

    if DATASET_JOINTS:                      # rename where the columns differ
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
            continue   # a pitch missing a joint column is skipped
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