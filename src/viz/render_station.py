"""
Diamond - render the STATION-classifier verdict onto a real pitch video.

Draws the pose skeleton (from the extracted coords) on the actual video frame(s)
plus a banner: predicted station + confidence + that station's metric readout.
Distinct output names (_station.png / _station.mp4) so nothing overwrites the
existing metrics.mp4 / armslot.png in the same folder.

The classifier is trained on OBP projections (cached to station_train_features
.csv on first run; instant thereafter). Same event-free ratio features as
station_classify, so the meter(OBP) vs pixel(video) scale difference cancels.

Run:  cd src\viz
      python render_station.py                       # PNG for both videos
      python render_station.py --video pitching_frontier_03 --mp4
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
from station_classify import build, features, knn, FEATS

CACHE = os.path.join(config.OBP_VALIDATION_DIR, "station_train_features.csv")
VIDEO_DIR = os.path.join(config.ROOT, "data", "videos")
OUT_ROOT = os.path.join(config.ROOT, "data", "outputs")
DEFAULT_VIDEOS = ["pitching_lateral_02", "pitching_frontier_03"]

CONNECT = [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
           ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
           ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
           ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
           ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
           ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")]
STATION_METRICS = {
    "side":  ["lead_knee_angle", "stride_pct_height", "trunk_anterior_tilt",
              "knee_ext_velo_br", "wrist_speed"],
    "front": ["arm_slot"],
    "overhead": ["hss (overhead only)"],
    "reject": [],
}
LABELFMT = {"lead_knee_angle": ("knee", "deg"), "stride_pct_height": ("stride", "xH"),
            "trunk_anterior_tilt": ("trunk", "deg"), "knee_ext_velo_br": ("knee velo", "deg/s"),
            "wrist_speed": ("wrist spd", ""), "arm_slot": ("arm slot", "deg")}


def get_classifier():
    if os.path.exists(CACHE):
        tr = pd.read_csv(CACHE)
    else:
        print("building OBP training features (first run)...")
        tr = build(250, 0.0).dropna(subset=FEATS).reset_index(drop=True)
        tr.to_csv(CACHE, index=False)
        print(f"cached -> {CACHE}")
    return tr[FEATS].to_numpy(float), tr["y"].to_numpy()


def load_coords(name, backbone="rtmp", raw=False):
    sfx = "_rtmp" if backbone == "rtmp" else ""
    p = config.refined_or(
        os.path.join(OUT_ROOT, name, f"{name}_smoothed{sfx}.csv"), raw)
    df = pd.read_csv(p)
    if "nose_x" in df.columns and "head_x" not in df.columns:
        df = df.rename(columns={"nose_x": "head_x", "nose_y": "head_y", "nose_v": "head_v"})
    return df


def detect_arm(df):
    def pk(j):
        s = np.hypot(np.diff(df[f"{j}_x"]), np.diff(df[f"{j}_y"]))
        s = s[np.isfinite(s)]
        return np.percentile(s, 95) if len(s) else 0.0
    return "right" if pk("right_wrist") >= pk("left_wrist") else "left"


def draw_skeleton(img, df, f):
    for a, b in CONNECT:
        try:
            xa, ya = df[f"{a}_x"].iloc[f], df[f"{a}_y"].iloc[f]
            xb, yb = df[f"{b}_x"].iloc[f], df[f"{b}_y"].iloc[f]
            if np.isfinite([xa, ya, xb, yb]).all():
                cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (80, 220, 80), 3)
        except Exception:
            pass
    for j in ["left_shoulder", "right_shoulder", "left_hip", "right_hip",
              "left_knee", "right_knee", "left_ankle", "right_ankle",
              "left_wrist", "right_wrist", "head"]:
        try:
            x, y = df[f"{j}_x"].iloc[f], df[f"{j}_y"].iloc[f]
            if np.isfinite([x, y]).all():
                cv2.circle(img, (int(x), int(y)), 5, (40, 220, 255), -1)
        except Exception:
            pass


from skeleton import square_crop     # shared 1:1 crop (single definition)


def draw_banner(img, station, share, vals, event=None):
    h, w = img.shape[:2]
    usable = station in ("side", "front", "overhead")
    color = (60, 170, 60) if usable else (40, 120, 230)   # BGR green / amber
    lines = STATION_METRICS.get(station, [])
    bh = int(48 + 30 * (len(lines) + 1))
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (w, bh), (30, 30, 30), -1)
    cv2.rectangle(ov, (0, 0), (int(w * 0.012), bh), color, -1)
    cv2.addWeighted(ov, 0.72, img, 0.28, 0, img)
    fs = max(0.7, w / 1600.0)
    mark = "OK" if usable else "REJECT"
    title = f"STATION: {station.upper()}  [{mark}]  conf {share:.2f}"
    if event:
        title += f"  -  {event}"
    cv2.putText(img, title,
                (int(w * 0.03), int(38 * fs) + 6), cv2.FONT_HERSHEY_SIMPLEX,
                fs * 1.05, color if usable else (60, 150, 240), 2, cv2.LINE_AA)
    y = int(38 * fs) + 40
    for k in lines:
        lbl, unit = LABELFMT.get(k, (k, ""))
        v = vals.get(k, float("nan"))
        txt = f"  {lbl}: {v:.2f} {unit}".rstrip()
        cv2.putText(img, txt, (int(w * 0.03), y), cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 0.85, (235, 235, 235), 2, cv2.LINE_AA)
        y += int(30 * fs) + 6
    if not usable:
        cv2.putText(img, "  re-shoot from a valid viewpoint", (int(w * 0.03), y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (60, 150, 240), 2, cv2.LINE_AA)


def frame_from_video(path, idx):
    """Frame-exact read via sequential grab: cv2 random seek
    (CAP_PROP_POS_FRAMES) is keyframe-snapped on iPhone HEVC clips and
    lands a few frames off the requested index."""
    cap = cv2.VideoCapture(path)
    for _ in range(int(idx)):
        if not cap.grab():
            cap.release()
            return None
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def process(name, Xtr, ytr, want_mp4, backbone="rtmp", square=False, raw=False):
    df = load_coords(name, backbone, raw)
    arm = detect_arm(df)
    f = features(df, arm)
    xv = np.array([f[k] for k in FEATS], float)
    lab, shr = knn(Xtr, ytr, xv[None, :], k=25)
    station, share = lab[0], float(shr[0])

    vpath = os.path.join(VIDEO_DIR, f"{name}.MOV")
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or config.FPS_DEFAULT
    cap.release()
    view = "frontal" if station == "front" else "side"
    sfx = "_rtmp" if backbone == "rtmp" else ""
    raw_p = config.refined_or(
        os.path.join(OUT_ROOT, name, f"{name}_coords{sfx}.csv"), raw)
    raw_df = pd.read_csv(raw_p) if os.path.exists(raw_p) else None
    cand = M.compute_candidates(df, fps=fps, arm=arm, view=view, raw_df=raw_df)
    vals = {k: v for k, (v, _) in cand.items()}
    rel = M.release_frame(df, arm, fps, M.JOINTS, view=view, raw_df=raw_df)
    outdir = os.path.join(OUT_ROOT, name)

    print(f"{name}: station={station} conf={share:.2f} arm={arm} rel_frame={rel}")

    if square:
        # paper stills: one per event, person-centered 1:1 crop
        events = {"RELEASE": rel}
        if station == "side":
            lead = "left" if arm == "right" else "right"
            events["FOOT PLANT"] = M.foot_plant_frame(df, lead, fps, M.JOINTS, rel)
        for ev, f in events.items():
            fr = frame_from_video(vpath, f)
            if fr is None:
                continue
            draw_skeleton(fr, df, f)
            fr = square_crop(fr, df, f)
            draw_banner(fr, station, share, vals, event=ev)
            tag = "" if ev == "RELEASE" else "_footplant"
            png = os.path.join(outdir, f"{name}_station{tag}_sq.png")
            cv2.imwrite(png, fr)
            print(f"  saved -> {png}  ({ev} f{f})")
    else:
        # PNG at release frame (legacy full-frame still)
        fr = frame_from_video(vpath, rel)
        if fr is not None:
            draw_skeleton(fr, df, rel)
            draw_banner(fr, station, share, vals)
            png = os.path.join(outdir, f"{name}_station.png")
            cv2.imwrite(png, fr)
            print(f"  saved -> {png}")

    if want_mp4:
        cap = cv2.VideoCapture(vpath)
        w = int(cap.get(3)); h = int(cap.get(4)); n = int(cap.get(7))
        mp4 = os.path.join(outdir, f"{name}_station.mp4")
        vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i < len(df):
                draw_skeleton(fr, df, i)
            draw_banner(fr, station, share, vals)
            vw.write(fr)
            i += 1
        cap.release(); vw.release()
        print(f"  saved -> {mp4}  ({i} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="single video name (default: both)")
    ap.add_argument("--mp4", action="store_true", help="also render full overlay video")
    ap.add_argument("--backbone", default="rtmp", choices=["rtmp", "mediapipe"])
    ap.add_argument("--square", action="store_true",
                    help="paper stills: per-event, person-centered 1:1 crop")
    ap.add_argument("--raw", action="store_true",
                    help="use RAW coords; default prefers the recovery-refined "
                         "coords when present (the best pose)")
    a = ap.parse_args()
    Xtr, ytr = get_classifier()
    for name in ([a.video] if a.video else DEFAULT_VIDEOS):
        process(name, Xtr, ytr, a.mp4, backbone=a.backbone, square=a.square,
                raw=a.raw)


if __name__ == "__main__":
    main()
