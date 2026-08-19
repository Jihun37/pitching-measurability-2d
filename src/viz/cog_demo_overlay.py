"""
Diamond - COG metric demo overlay (real video). Shows the two newly-adopted COG
metrics + the new peak-knee-height event working on a real RTMPose clip:
  - whole-body Winter COM (metrics.body_com) drawn + trailed each frame,
  - the pkh event (peak knee height) marked, with COG Velo @PKH,
  - the frame of peak forward COM speed marked, with COG Fwd Velo,
  - a banner with the two measured values (m/s, assumed height).

Reuses render_station.draw_skeleton + the sequential-read / VideoWriter pattern
(cv2 random seek is keyframe-snapped on these clips - sequential only).

Run:  cd src\viz
      python cog_demo_overlay.py --clip angle00_00 --height-m 1.85
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import cv2

_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3", "../tests"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)

import config
import metrics as M
from render_station import draw_skeleton
from real_station_test import detect_arm_2d

MAG = (230, 0, 230); CYAN = (240, 220, 40); AMBER = (40, 150, 240); WHITE = (255, 255, 255)


def load(name):
    base = os.path.join(config.ROOT, "data", "outputs", name)
    df = pd.read_csv(config.refined_or(os.path.join(base, f"{name}_smoothed_rtmp.csv")))
    rawp = config.refined_or(os.path.join(base, f"{name}_coords_rtmp.csv"))
    raw = pd.read_csv(rawp) if os.path.exists(rawp) else None
    for d in (df, raw):
        if d is not None and "nose_x" in d.columns and "head_x" not in d.columns:
            d.rename(columns={"nose_x": "head_x", "nose_y": "head_y"}, inplace=True)
    return df, raw


def banner(img, lines, color=(30, 30, 30)):
    h, w = img.shape[:2]
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (w, 40 + 34 * len(lines)), color, -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    for i, (txt, c) in enumerate(lines):
        cv2.putText(img, txt, (18, 40 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, c, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="angle00_00")
    ap.add_argument("--height-m", type=float, default=1.85)
    a = ap.parse_args()
    name, H = a.clip, a.height_m

    df, raw = load(name)
    vpath = os.path.join(config.ROOT, "data", "videos", f"{name}.mov")
    if not os.path.exists(vpath):
        vpath = os.path.join(config.ROOT, "data", "videos", f"{name}.MOV")
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or config.FPS_DEFAULT
    w, h = int(cap.get(3)), int(cap.get(4))

    arm = detect_arm_2d(df)
    lead = "left" if arm == "right" else "right"
    rel = M.release_frame(df, arm, fps, M.JOINTS, raw_df=raw)
    fp = M.foot_plant_frame(df, lead, fps, M.JOINTS, rel)
    pkh = M.peak_knee_height_frame(df, lead, fp, M.JOINTS)

    comx, comy = M.body_com(df, M.JOINTS)
    stat = M.pixel_stature(df, M.JOINTS)
    vfwd = np.abs(np.gradient(comx)) * fps / stat            # statures/s series
    pk_frame = int(np.nanargmax(vfwd[:rel + 1]))
    cog_fwd = M.cog_fwd_velo(df, fps, rel, M.JOINTS, H)
    cog_pkh = M.cog_velo_at_pkh(df, fps, pkh, M.JOINTS, H)
    kkey = "l_kn" if lead == "left" else "r_kn"
    print(f"{name}: arm={arm} rel={rel} fp={fp} pkh={pkh} peakFwd@{pk_frame}  "
          f"cog_fwd={cog_fwd:.2f} m/s  cog_pkh={cog_pkh:.2f} m/s")

    outdir = os.path.join(config.ROOT, "data", "outputs", name)
    mp4 = os.path.join(outdir, f"{name}_cog_demo.mp4")
    # iPhone clips report a fractional fps (e.g. 113.223) whose timebase the
    # mpeg4 encoder rejects (denominator > 65535) -> round to an integer fps.
    out_fps = float(int(round(fps))) or 30.0
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("VideoWriter failed to open")
    trail = []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i < len(df):
            draw_skeleton(fr, df, i)
            cx, cy = comx[i], comy[i]
            if np.isfinite([cx, cy]).all():
                trail.append((int(cx), int(cy)))
                for j, (tx, ty) in enumerate(trail[-40:]):
                    cv2.circle(fr, (tx, ty), 2, MAG, -1)
                cv2.circle(fr, (int(cx), int(cy)), 9, MAG, -1)
                cv2.circle(fr, (int(cx), int(cy)), 9, WHITE, 2)
            # mark the lead knee whose height defines pkh
            kx = df[f"{M.JOINTS[kkey]}_x"].iloc[i]; ky = df[f"{M.JOINTS[kkey]}_y"].iloc[i]
            if np.isfinite([kx, ky]).all():
                cv2.circle(fr, (int(kx), int(ky)), 6, CYAN, 2)

        lines = [(f"{name}  (assumed height {H:.2f} m)", WHITE),
                 (f"COM (magenta) = Winter whole-body centre of mass", MAG)]
        run_pk = float(np.nanmax(vfwd[:min(i, rel) + 1])) * H if i > 0 else 0.0
        if i >= pkh:
            lines.append((f"COG Velo @PKH  = {cog_pkh:.2f} m/s   (peak knee height, f{pkh})", CYAN))
        if i >= pk_frame:
            lines.append((f"COG Fwd Velo   = {cog_fwd:.2f} m/s   (peak forward, f{pk_frame})", AMBER))
        else:
            lines.append((f"COG fwd so far = {run_pk:.2f} m/s", AMBER))
        banner(fr, lines)

        # event flashes
        if abs(i - pkh) <= 2:
            cv2.putText(fr, "PEAK KNEE HEIGHT", (int(w * 0.30), int(h * 0.15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, CYAN, 3, cv2.LINE_AA)
        if abs(i - pk_frame) <= 2:
            cv2.putText(fr, "PEAK FORWARD COM SPEED", (int(w * 0.18), int(h * 0.15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, AMBER, 3, cv2.LINE_AA)
        vw.write(fr)
        i += 1
    cap.release(); vw.release()
    print(f"saved -> {mp4}  ({i} frames)")


if __name__ == "__main__":
    main()
