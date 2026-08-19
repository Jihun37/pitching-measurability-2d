"""
Diamond - two extra figures for the paper/patent set, in the same visual
language as the render_station skeleton images:

  1) OVERHEAD hip-shoulder separation (HSS) - synthetic OBP overhead projection
     (no real overhead footage exists). Top-down skeleton with the shoulder line
     and pelvis line highlighted and the separation angle annotated.
  2) RELEASE HEIGHT - on the real lateral video: vertical ground->wrist line at
     release with the value. Saved as a separate file (does not touch the
     existing _station.png).

Run:  cd src\viz
      python render_station_extras.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import cv2

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(HERE, "..", "stage3"))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import config
import metrics as M
import obp_project as O
from hss_elevation_test import project_cam, hss_sep_series
from render_station import (load_coords, detect_arm, draw_skeleton,
                            frame_from_video, CONNECT)

OUT_ROOT = os.path.join(config.ROOT, "data", "outputs")
VIDEO_DIR = os.path.join(config.ROOT, "data", "videos")


def dashed_line(img, p1, p2, color, thick, dash=14):
    p1 = np.array(p1, float); p2 = np.array(p2, float)
    dist = np.linalg.norm(p2 - p1)
    n = max(1, int(dist / dash))
    for i in range(n):
        if i % 2 == 0:
            a = p1 + (p2 - p1) * (i / n)
            b = p1 + (p2 - p1) * ((i + 1) / n)
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thick, cv2.LINE_AA)


def banner(img, title, sub, usable=True):
    h, w = img.shape[:2]
    color = (60, 170, 60) if usable else (40, 120, 230)
    bh = int(96 * max(1.0, w / 1000.0))
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (w, bh), (30, 30, 30), -1)
    cv2.rectangle(ov, (0, 0), (int(w * 0.012), bh), color, -1)
    cv2.addWeighted(ov, 0.72, img, 0.28, 0, img)
    fs = max(0.7, w / 1500.0)
    cv2.putText(img, title, (int(w * 0.03), int(bh * 0.42)),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 1.05, color, 2, cv2.LINE_AA)
    cv2.putText(img, sub, (int(w * 0.03), int(bh * 0.82)),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.9, (235, 235, 235), 2, cv2.LINE_AA)


# ---------------------------------------------------------------- overhead HSS
def render_hss_overhead(index=0, size=900):
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    r = md.iloc[index]
    path = os.path.join(config.OBP_DATA_DIR, "c3d", f"{int(r.user):06d}", r.filename_new)
    joints, fps = O.load_c3d_joints(path)
    arm = O.detect_throwing_arm(joints, fps)
    lead = "left" if arm == "right" else "right"

    df0 = O.project_view(joints, azimuth_deg=0)
    rel = M.release_frame(df0, arm, fps, M.JOINTS)
    fp = M.foot_plant_frame(df0, lead, fps, M.JOINTS, rel)

    df = project_cam(joints, 0, 90)                      # true bird's-eye
    sep = hss_sep_series(df, M.JOINTS)
    lo, hi = min(fp, rel), max(fp, rel) + 1
    fr = lo + int(np.nanargmax(np.abs(sep[lo:hi])))      # frame of max separation
    hss_val = float(abs(sep[fr]))

    joints_draw = ["left_shoulder", "right_shoulder", "left_hip", "right_hip",
                   "left_knee", "right_knee", "left_ankle", "right_ankle",
                   "left_elbow", "right_elbow", "left_wrist", "right_wrist", "head"]
    P = {j: (df[f"{j}_x"].iloc[fr], df[f"{j}_y"].iloc[fr]) for j in joints_draw}

    # frame on the TORSO (shoulders+hips), equal aspect, so the two lines are
    # large and centered (the overhead full body is a long thin streak).
    tor = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    txs = [P[j][0] for j in tor]; tys = [P[j][1] for j in tor]
    cx = (min(txs) + max(txs)) / 2; cy = (min(tys) + max(tys)) / 2
    span = max(max(txs) - min(txs), max(tys) - min(tys)) + 1e-6
    scale = (size * 0.34) / span

    def px(j):
        x, y = P[j]
        return (int((x - cx) * scale + size / 2), int((y - cy) * scale + size / 2 + 40))

    img = np.full((size, size, 3), 38, np.uint8)
    for a, b in CONNECT:                                  # faint context skeleton
        cv2.line(img, px(a), px(b), (70, 120, 70), 2, cv2.LINE_AA)
    for j in joints_draw:
        cv2.circle(img, px(j), 5, (110, 110, 110), -1, cv2.LINE_AA)

    lsh, rsh = px("left_shoulder"), px("right_shoulder")
    lh, rh = px("left_hip"), px("right_hip")
    cv2.line(img, lsh, rsh, (255, 210, 40), 7, cv2.LINE_AA)      # shoulder line (cyan)
    cv2.line(img, lh, rh, (40, 200, 255), 7, cv2.LINE_AA)        # pelvis line (orange)
    for p, col in [(lsh, (255, 210, 40)), (rsh, (255, 210, 40)),
                   (lh, (40, 200, 255)), (rh, (40, 200, 255))]:
        cv2.circle(img, p, 8, col, -1, cv2.LINE_AA)

    # separation angle at the hip midpoint: translate the shoulder direction
    # there (dashed) and draw the wedge between it and the pelvis line.
    hipmid = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
    sh = np.array([rsh[0] - lsh[0], rsh[1] - lsh[1]], float); sh /= np.linalg.norm(sh) + 1e-9
    hp = np.array([rh[0] - lh[0], rh[1] - lh[1]], float); hp /= np.linalg.norm(hp) + 1e-9
    L = int(size * 0.17)
    dashed_line(img, (hipmid[0] - int(sh[0] * L), hipmid[1] - int(sh[1] * L)),
                (hipmid[0] + int(sh[0] * L), hipmid[1] + int(sh[1] * L)), (255, 210, 40), 2)
    a_sh = np.degrees(np.arctan2(sh[1], sh[0])); a_hp = np.degrees(np.arctan2(hp[1], hp[0]))
    s, e = sorted([a_sh, a_hp])
    if e - s > 180:
        s, e = e, s + 360
    cv2.ellipse(img, hipmid, (int(L * 0.55), int(L * 0.55)), 0, s, e, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{hss_val:.0f} deg", (hipmid[0] + 14, hipmid[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "shoulder line", (rsh[0] + 10, rsh[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 210, 40), 2, cv2.LINE_AA)
    cv2.putText(img, "pelvis line", (rh[0] + 10, rh[1] + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 200, 255), 2, cv2.LINE_AA)

    banner(img, "STATION: OVERHEAD  [simulated view]",
           f"hip-shoulder separation: {hss_val:.0f} deg   (recoverable only from overhead)")
    out = os.path.join(config.OBP_VALIDATION_DIR, "hss_overhead.png")
    cv2.imwrite(out, img)
    print(f"HSS overhead: idx={index} arm={arm} frame={fr} hss={hss_val:.1f} -> {out}")


# ------------------------------------------------------------- release height
def render_release_height(name="pitching_lateral_02"):
    df = load_coords(name)
    arm = detect_arm(df)
    vpath = os.path.join(VIDEO_DIR, f"{name}.MOV")
    cap = cv2.VideoCapture(vpath); fps = cap.get(cv2.CAP_PROP_FPS) or config.FPS_DEFAULT
    cap.release()
    cand = M.compute_candidates(df, fps=fps, arm=arm, view="side")
    vals = {k: v for k, (v, _) in cand.items()}
    rel = M.release_frame(df, arm, fps, M.JOINTS, view="side")

    lany = df["left_ankle_y"].to_numpy(float); rany = df["right_ankle_y"].to_numpy(float)
    ground = float(np.nanmax(np.concatenate([lany, rany])))
    wx = float(df[f"{arm}_wrist_x"].iloc[rel]); wy = float(df[f"{arm}_wrist_y"].iloc[rel])

    img = frame_from_video(vpath, rel)
    if img is None:
        print("could not read frame"); return
    draw_skeleton(img, df, rel)
    h, w = img.shape[:2]
    cv2.line(img, (0, int(ground)), (w, int(ground)), (180, 180, 180), 2, cv2.LINE_AA)
    cv2.line(img, (int(wx), int(wy)), (int(wx), int(ground)), (60, 170, 255), 4, cv2.LINE_AA)
    cv2.circle(img, (int(wx), int(wy)), 8, (60, 170, 255), -1, cv2.LINE_AA)
    cv2.putText(img, f"release height {vals['release_height']:.2f}",
                (int(wx) + 12, int((wy + ground) / 2)), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.7, w / 1600.0) * 0.9, (60, 170, 255), 2, cv2.LINE_AA)
    banner(img, "STATION: SIDE   metric: RELEASE HEIGHT",
           f"release height: {vals['release_height']:.2f}  (ankle-ground to wrist / torso)")
    out = os.path.join(OUT_ROOT, name, f"{name}_release_height.png")
    cv2.imwrite(out, img)
    print(f"release height: {name} arm={arm} rel={rel} val={vals['release_height']:.2f} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="OBP pitch index for HSS")
    a = ap.parse_args()
    render_hss_overhead(a.index)
    render_release_height("pitching_lateral_02")


if __name__ == "__main__":
    main()
