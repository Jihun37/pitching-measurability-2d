"""
Diamond - 뷰 스윕: 카메라 각도(azimuth) × 지표별 Level-A r²
view_sweep.py

각 c3d 를 한 번만 로드 -> 여러 azimuth 로 메모리에서 투영 -> metrics 계산
-> azimuth 별로 OBP 3D 정답과 r² -> 지표별 최고 뷰 표.
(0°=측면, 90°=정면/포수, 사이=대각선)
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import config
import obp_project as O
import metrics as M

OBP_DATA = config.OBP_DATA_DIR
AZIMUTHS = [0, 15, 30, 45, 60, 75, 90]
ELEV = 0.0

# 우리 지표 -> OBP 3D 정답 컬럼
#  주의: arm_slot 은 정의가 바뀌어(shoulder->hand vs 수직) OBP 의 'arm_slot'(전완 투영각)
#        컬럼과 다르므로 여기서 제외. arm slot 검증은 armslot_validate.py 가 전담.
MAP = {
    "stride_length":       "stride_length",
    "lateral_trunk_tilt":  "torso_anterior_tilt_br",
    "lead_knee_extension": "lead_knee_extension_from_fp_to_br",
    "wrist_peak_speed":    "max_elbow_extension_velo",
}


def main():
    md = pd.read_csv(os.path.join(OBP_DATA, "metadata.csv"))
    poi = pd.read_csv(os.path.join(OBP_DATA, "poi", "poi_metrics.csv"))
    c3d_root = os.path.join(OBP_DATA, "c3d")

    # az -> list of feature rows
    acc = {az: [] for az in AZIMUTHS}
    done = fail = 0
    for r in md.itertuples(index=False):
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = O.load_c3d_joints(path)
            arm = O.detect_throwing_arm(joints, fps)
        except Exception:
            fail += 1; continue
        for az in AZIMUTHS:
            try:
                df = O.project_view(joints, azimuth_deg=az, elevation_deg=ELEV)
                c = M.compute_candidates(df, fps=fps, arm=arm)
                row = {"session_pitch": r.session_pitch}
                row.update({k: v for k, (v, _) in c.items()})
                acc[az].append(row)
            except Exception:
                pass
        done += 1
        if done % 100 == 0:
            print(f"  ...{done} 처리")
    print(f"처리 {done} / 실패 {fail}\n")

    # az 별 r²
    table = {m: {} for m in MAP}
    for az in AZIMUTHS:
        feat = pd.DataFrame(acc[az])
        df = feat.merge(poi, on="session_pitch", how="inner", suffixes=("_our", ""))
        for ours, truth in MAP.items():
            oc = ours + "_our" if (ours + "_our") in df.columns else ours
            d = df[[oc, truth]].dropna()
            r = d[oc].corr(d[truth]) if len(d) > 2 else np.nan
            table[ours][az] = r * r if pd.notna(r) else np.nan

    # 출력
    print("=" * (16 + 8 * len(AZIMUTHS) + 14))
    hdr = "metric".ljust(20) + "".join(f"{az:>7d}°" for az in AZIMUTHS) + "   best"
    print("[Level-A r²  by camera azimuth]   (0=측면, 90=정면)")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for ours in MAP:
        vals = table[ours]
        best_az = max(AZIMUTHS, key=lambda a: (vals[a] if pd.notna(vals[a]) else -1))
        line = ours.ljust(20) + "".join(
            f"{(vals[a] if pd.notna(vals[a]) else float('nan')):>8.2f}" for a in AZIMUTHS)
        line += f"   {best_az}° ({vals[best_az]:.2f})"
        print(line)
        rows.append({"metric": ours, "best_az": best_az,
                     "best_r2": vals[best_az], **{f"r2_{a}": vals[a] for a in AZIMUTHS}})

    _out = os.path.join(config.OBP_VALIDATION_DIR, "view_sweep_results.csv")
    pd.DataFrame(rows).to_csv(_out, index=False)
    print(f"\nsaved -> {_out}")


if __name__ == "__main__":
    main()