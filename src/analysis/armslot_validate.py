"""
Diamond — arm slot 전용 검증 (새 정의: shoulder->hand vs 수직, Escamilla & Fleisig)
armslot_validate.py

OBP 의 arm_slot 컬럼은 '전완(forearm) 투영각'이라 우리 정의(shoulder->hand vs 수직)와 다르다.
그래서 정답을 OBP 컬럼이 아니라 3D 좌표에서 직접 만든다:
    3D 정면평면(Y-Z) 정답 = angle(  (hand-shoulder) 의 Y,Z 성분,  수직 Z )
그런 다음 각 카메라 각도(azimuth)에서 우리 2D 추정을 이 정답과 대조 -> r².
(예상: 정면 90°에서 최고, 측면 0°에서 붕괴.)

사용:
  # 전체 OBP (r² 표)
  python armslot_validate.py --batch
  # 단일 c3d 점검(추정 vs 정답)
  python armslot_validate.py --c3d <경로.c3d>
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import obp_project as O
import metrics as M

AZIMUTHS = [0, 15, 30, 45, 60, 75, 90]


def truth_3d_frontal(joints, arm, rel):
    """3D 정면평면 arm slot 정답: (hand-shoulder) 의 [수평=Y, 수직=Z] 로 수직과 이루는 각."""
    S = joints[f"{arm}_shoulder"][:, rel]
    W = joints[f"{arm}_wrist"][:, rel]
    vec = W - S                       # (X,Y,Z); X=투구방향, Y=좌우, Z=수직
    run = abs(vec[1])                 # 정면평면 수평 성분 = Y
    rise = vec[2]                     # 수직 성분 = Z
    return float(np.degrees(np.arctan2(run, rise)))


def estimate_2d(joints, arm, rel, az):
    df = O.project_view(joints, azimuth_deg=az)
    sx, sy = df[f"{arm}_shoulder_x"].iloc[rel], df[f"{arm}_shoulder_y"].iloc[rel]
    wx, wy = df[f"{arm}_wrist_x"].iloc[rel],   df[f"{arm}_wrist_y"].iloc[rel]
    return float(np.degrees(np.arctan2(abs(wx-sx), (sy-wy))))


def release_of(joints, arm, fps):
    df0 = O.project_view(joints, azimuth_deg=0.0)
    return M.release_frame(df0, arm, fps, M.JOINTS)


def one(path):
    joints, fps = O.load_c3d_joints(path)
    arm = O.detect_throwing_arm(joints, fps)
    rel = release_of(joints, arm, fps)
    truth = truth_3d_frontal(joints, arm, rel)
    est = {az: estimate_2d(joints, arm, rel, az) for az in AZIMUTHS}
    return truth, est


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = np.nansum((y-yhat)**2); ss_tot = np.nansum((y-np.nanmean(y))**2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3d", default=None)
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    if a.c3d and not a.batch:
        truth, est = one(a.c3d)
        print(f"3D 정면 정답 arm slot = {truth:.1f}deg")
        for az in AZIMUTHS:
            print(f"  az={az:2d}deg  2D추정 {est[az]:5.1f}  (오차 {est[az]-truth:+.1f})")
        return

    import config
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    truths, ests = [], {az: [] for az in AZIMUTHS}
    n = fail = 0
    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit: break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path): fail += 1; continue
        try:
            t, e = one(path)
            truths.append(t)
            for az in AZIMUTHS: ests[az].append(e[az])
            n += 1
        except Exception:
            fail += 1
    print(f"처리 {n}개 / 실패 {fail}개\n")
    # arm slot 정답 분포 저장 (정면 실영상 z-score 비교용)
    try:
        import config as _cfg
        os.makedirs(_cfg.OBP_VALIDATION_DIR, exist_ok=True)
        pd.DataFrame({"arm_slot_truth": truths}).to_csv(
            os.path.join(_cfg.OBP_VALIDATION_DIR, "armslot_ref.csv"), index=False)
        print(f"arm slot 정답 분포 저장 -> {os.path.join(_cfg.OBP_VALIDATION_DIR,'armslot_ref.csv')}"
              f"  (평균 {np.mean(truths):.1f} / std {np.std(truths):.1f})\n")
    except Exception as _e:
        pass
    print(f"arm slot 정의 = shoulder->hand vs 수직 (정답=3D 정면평면)")
    print(f"{'azimuth':>8} {'r2':>8} {'MAE':>8}")
    for az in AZIMUTHS:
        e = np.array(ests[az]); t = np.array(truths)
        print(f"{az:6d}° {r2(t,e):8.3f} {np.nanmean(np.abs(e-t)):8.1f}")
    print("\n→ 정면(90°)에 가까울수록 r² 최고 · 측면(0°)에서 붕괴 예상")


if __name__ == "__main__":
    main()