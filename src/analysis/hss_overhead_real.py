"""
Diamond - Real-video overhead HSS (peak definition).

Pipeline per video (default --backbone rtmp, --anchor signature):
  1. Pose extraction (RTMPose official / MediaPipe legacy), cached CSV.
     Detection rate + landmark visibility reported; check frames dumped so
     survival is answered by eye.
  2. Savitzky-Golay smoothing (same engine as every other video).
  3. HSS = metrics.hss_peak_overhead: chord-validity gate -> 90 ms medfilt
     -> persistence-checked sustained-swing anchor -> windowed |max|.
     RELEASE-FREE (overhead HSS peaks near foot plant, not release; the old
     wrist-speed release localization is a verified no-op and is gone).
     OBP-validated r2=0.63 (n=408, el>=85). --anchor release keeps the
     legacy wrist-speed-window path for comparison runs only.

Run:  cd src\analysis
      python hss_overhead_real.py --backbone rtmp                 # 3 clips
      python hss_overhead_real.py --backbone rtmp --video pitching_overhead_01 --square
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import cv2
from scipy.signal import medfilt

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "stage1"))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
import config
import metrics as M
from pose_extractor import extract_pose
from smoother import smooth_coordinates
# HSS definitions live in metrics.py (promoted 2026-07-04) — M.hss_*

VIDEOS = ["pitching_overhead_01", "pitching_overhead_02", "pitching_overhead_03"]
CONNECT = [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
           ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
           ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
           ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
           ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
           ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")]


def detect_arm(df):
    def pk(j):
        x = df[f"{j}_x"].to_numpy(float); y = df[f"{j}_y"].to_numpy(float)
        return np.nanmax(np.hypot(np.diff(x), np.diff(y)))
    return "right" if pk("right_wrist") >= pk("left_wrist") else "left"


def grab_frames_exact(video_path, wanted):
    """Frame-exact reads via one sequential pass. cv2 random seek
    (CAP_PROP_POS_FRAMES) is keyframe-snapped on iPhone HEVC clips and lands
    a few frames off — sequential grab() is the only frame-accurate way and
    matches what the video renderer shows."""
    wanted = sorted(set(int(f) for f in wanted))
    out = {}
    cap = cv2.VideoCapture(video_path)
    f = 0
    for target in wanted:
        while f < target:
            if not cap.grab():
                cap.release()
                return out
            f += 1
        ok, fr = cap.read()
        if not ok:
            break
        out[target] = fr
        f += 1
    cap.release()
    return out


def save_check_frames(video_path, df, frames, out_dir, name, scale=0.5):
    """Skeleton overlay on selected frames -> PNG, for eyeball survival check."""
    frames = {t: int(np.clip(f, 0, len(df) - 1)) for t, f in frames.items()}
    imgs = grab_frames_exact(video_path, frames.values())
    for tag, f in frames.items():
        fr = imgs.get(f)
        if fr is None:
            continue
        fr = fr.copy()
        for a, b in CONNECT:
            try:
                p1 = (int(df[f"{a}_x"].iloc[f]), int(df[f"{a}_y"].iloc[f]))
                p2 = (int(df[f"{b}_x"].iloc[f]), int(df[f"{b}_y"].iloc[f]))
                cv2.line(fr, p1, p2, (0, 255, 0), 4)
            except Exception:
                pass
        cv2.putText(fr, f"frame {f}  {tag}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        fr = cv2.resize(fr, None, fx=scale, fy=scale)
        out = os.path.join(out_dir, f"{name}_check_{tag}.png")
        cv2.imwrite(out, fr)
        print(f"  check frame -> {out}")


YEL, RED, WHITE, GREEN = (0, 255, 255), (0, 0, 255), (255, 255, 255), (0, 255, 0)


def _torso_pts(df, f, r, ox=0.0, oy=0.0):
    return {j: (int((df[f"{j}_x"].iloc[f] - ox) * r),
                int((df[f"{j}_y"].iloc[f] - oy) * r))
            for j in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]}


def _draw_torso(fr, pts, arc_val=None):
    """Shoulder line (yellow) + hip line (red); optionally the angle between
    them as an arc at the shoulder midpoint (same idiom as the side-view
    knee/tilt arcs): thin white reference ray parallel to the hip line,
    then an ellipse arc sweeping to the shoulder line."""
    cv2.line(fr, pts["left_shoulder"], pts["right_shoulder"], YEL, 3)
    cv2.line(fr, pts["left_hip"], pts["right_hip"], RED, 3)
    for j, p in pts.items():
        cv2.circle(fr, p, 6, YEL if "shoulder" in j else RED, -1)
    if arc_val is None:
        return
    sh = np.array(pts["right_shoulder"], float) - np.array(pts["left_shoulder"], float)
    hip = np.array(pts["right_hip"], float) - np.array(pts["left_hip"], float)
    ms = ((pts["left_shoulder"][0] + pts["right_shoulder"][0]) // 2,
          (pts["left_shoulder"][1] + pts["right_shoulder"][1]) // 2)
    a_sh = np.degrees(np.arctan2(sh[1], sh[0]))
    a_hip = np.degrees(np.arctan2(hip[1], hip[0]))
    d = (a_sh - a_hip + 180.0) % 360.0 - 180.0        # signed sweep hip->shoulder
    hn = hip / (np.linalg.norm(hip) + 1e-9)
    L = 130
    # white reference: hip-line direction re-drawn through the shoulder mid
    cv2.line(fr, (int(ms[0] - hn[0] * L), int(ms[1] - hn[1] * L)),
             (int(ms[0] + hn[0] * L), int(ms[1] + hn[1] * L)), WHITE, 2)
    cv2.ellipse(fr, ms, (80, 80), 0, a_hip, a_hip + d, YEL, 3)
    cv2.putText(fr, f"hss {abs(arc_val):.1f}", (ms[0] + 20, ms[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, YEL, 2)


def _banner(fr, pk_f, anchor_f, sep_val, fps):
    # overhead HSS is release-free: report the peak and its signature anchor,
    # never a release frame (release is not used and is unstable top-down)
    cv2.putText(fr, "HSS PEAK", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, YEL, 3)
    cv2.putText(fr, f"hss {abs(sep_val):.1f} deg", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, YEL, 2)


def draw_hss_still(video_path, df, pk_f, anchor_f, sep_val, out_png, fps,
                   square=False):
    """Presentation still (same spirit as the side/front event PNGs):
    actual frame at the HSS peak, shoulder/hip lines + angle arc + banner.
    Frame is normalized to 1920 px wide (or a person-centered 1200x1200
    square crop with square=True, for the paper's 1:1 figure set) so
    fonts/lines render the same regardless of source resolution."""
    fr = grab_frames_exact(video_path, [int(pk_f)]).get(int(pk_f))
    if fr is None:
        return
    if square:
        H, W = fr.shape[:2]
        xs, ys = [], []
        for c in df.columns:
            if c.endswith("_x"):
                j = c[:-2]
                x, y = df[f"{j}_x"].iloc[pk_f], df[f"{j}_y"].iloc[pk_f]
                if np.isfinite(x) and np.isfinite(y):
                    xs.append(float(x)); ys.append(float(y))
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        side = min(max(x1 - x0, y1 - y0) * 1.36, W, H)
        ox = float(np.clip((x0 + x1) / 2 - side / 2, 0, W - side))
        oy = float(np.clip((y0 + y1) / 2 - side / 2, 0, H - side))
        fr = fr[int(oy):int(oy + side), int(ox):int(ox + side)]
        r = 1200.0 / side
        fr = cv2.resize(fr, (1200, 1200), interpolation=cv2.INTER_AREA)
        pts = _torso_pts(df, pk_f, r, ox, oy)
    else:
        r = 1920.0 / fr.shape[1]
        fr = cv2.resize(fr, None, fx=r, fy=r)
        pts = _torso_pts(df, pk_f, r)
    _draw_torso(fr, pts, arc_val=sep_val)
    cv2.putText(fr, f"frame {int(pk_f)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    _banner(fr, pk_f, anchor_f, sep_val, fps)
    cv2.imwrite(out_png, fr)
    print(f"  hss still -> {out_png}")


def render_hss_video(video_path, df, sep_arr, pk_f, anchor_f, out_mp4, fps,
                     valid=None):
    """Overlay video like the side/front metrics.mp4: torso lines every frame,
    live sep readout, 1.5 s hold at the HSS peak and at the signature anchor
    (the sustained-swing transition, release-free). Chord-gated frames show
    an 'undefined' readout. Output normalized to 1280 px wide."""
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    r = 1280.0 / W
    size = (1280, int(H * r))
    vfps = int(round(fps))
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), vfps, size)
    hold = int(round(vfps * 1.5))
    f = 0
    while True:
        ok, fr = cap.read()
        if not ok or f >= len(df):
            break
        fr = cv2.resize(fr, size)
        is_pk, is_anchor = (f == pk_f), (f == anchor_f)
        _draw_torso(fr, _torso_pts(df, f, r), arc_val=sep_arr[f] if is_pk else None)
        cv2.putText(fr, f"frame {f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
        if valid is None or valid[f]:
            cv2.putText(fr, f"sep {sep_arr[f]:+.0f} deg", (20, size[1] - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        else:
            cv2.putText(fr, "sep -- (torso end-on)", (20, size[1] - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        if is_pk:
            _banner(fr, pk_f, anchor_f, sep_arr[f], fps)
        if is_anchor and not is_pk:
            cv2.putText(fr, "signature anchor", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN, 3)
        vw.write(fr)
        if is_pk or is_anchor:
            for _ in range(hold):
                vw.write(fr)
        f += 1
    cap.release(); vw.release()
    print(f"  hss video -> {out_mp4}")


def run_video(name, win_pre, win_post, model=None, tag="", mp4=False,
              backbone="mediapipe", anchor="signature", slomo=1.0,
              square=False):
    video = next(p for p in
                 (os.path.join(config.ROOT, "data", "videos", name + ext)
                  for ext in (".mov", ".mp4", ".MOV"))
                 if os.path.exists(p))
    out_dir = os.path.join(config.ROOT, "data", "outputs", name)
    os.makedirs(out_dir, exist_ok=True)
    sfx = f"_{tag}" if tag else ""
    coords_csv = os.path.join(out_dir, f"{name}_coords{sfx}.csv")
    smoothed_csv = os.path.join(out_dir, f"{name}_smoothed{sfx}.csv")
    model = model or config.MODEL_PATH

    print(f"\n=== {name} ===")
    # DEFAULT BEST POSE: occlusion-recovery-refined RTMPose. When the tag asks
    # for the refined coords and they are not cached, build them here (extract
    # stock RTMPose if needed, then recover). On overhead this removes the
    # confidently-wrong wrist/elbow jumps; on side/front it is a near no-op.
    if backbone == "rtmp" and tag == "rtmp_refined" and not os.path.exists(coords_csv):
        stock_csv = os.path.join(out_dir, f"{name}_coords_rtmp.csv")
        if not os.path.exists(stock_csv):
            from rtmp_extractor import extract_pose_rtmp   # stage1 engine
            extract_pose_rtmp(video, save_csv=stock_csv)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "occlusion"))
        from refine import refine_csv
        print("  building occlusion-refined coords (best pose) ...")
        refine_csv(stock_csv, save_fig=False)   # -> {name}_coords_rtmp_refined.csv

    if os.path.exists(coords_csv):
        raw = pd.read_csv(coords_csv)
        cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
        print(f"  cached coords ({len(raw)} frames)")
    elif backbone == "rtmp":
        from rtmp_extractor import extract_pose_rtmp   # stage1 engine
        raw, meta = extract_pose_rtmp(video, save_csv=coords_csv)
        fps = meta["fps"]
    else:
        raw, meta = extract_pose(video, model, save_csv=coords_csv)
        fps = meta["fps"]

    # survival report: detection rate + core-landmark visibility
    det = raw["left_shoulder_x"].notna().mean()
    vis = {j: float(np.nanmean(raw[f"{j}_v"])) for j in
           ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]}
    print(f"  detection rate: {det*100:.1f}%   "
          + "  ".join(f"{j} v={v:.2f}" for j, v in vis.items()))

    df = smooth_coordinates(raw)
    df.to_csv(smoothed_csv, index=False)

    arm = detect_arm(df)
    sep = M.hss_sep_series(df, M.JOINTS)
    k = max(3, int(0.09 * fps) // 2 * 2 + 1)

    if anchor == "signature":
        # the full adopted recipe lives in metrics.hss_peak_overhead
        # (gate -> medfilt -> persistence anchor). Release-free: overhead HSS
        # never needs release detection.
        R = M.hss_peak_overhead(df, fps, M.JOINTS, slomo=slomo)
        if R is None:
            raise RuntimeError("signature anchor not found")
        sep_f, valid = R["sep_f"], R["valid"]
        lo, hi = R["window"]
        anchor_f = R["anchor_f"]
        # take the peak from the function — it is COIL-direction now; an
        # inline argmax(|sep_f|) here silently reinstates the retired
        # sign-blind rule (it grabbed the in-window rebound on refined coords)
        pk_f = R["peak_f"]
        hss_peak = R["hss"]
    else:
        # legacy release-anchored path (kept for comparison runs only;
        # deliberately keeps the old sign-blind |max| behavior)
        rel = M.release_frame(df, arm, fps, M.JOINTS, view="side")
        valid = M.hss_chord_valid(df, M.JOINTS)
        sep_f = medfilt(np.where(valid, np.nan_to_num(sep, nan=0.0), 0.0),
                        kernel_size=k)
        lo = max(0, rel - int(win_pre * fps))
        hi = min(len(sep) - 1, rel + int(win_post * fps))
        anchor_f = rel
        seg = np.abs(sep_f[lo:hi + 1])
        pk_f = lo + int(np.nanargmax(seg))
        hss_peak = float(np.nanmax(seg))
    raw_pk = float(np.nanmax(np.abs(sep[lo:hi + 1])))
    hss_full = float(np.nanmax(np.abs(sep_f)))
    full_f = int(np.nanargmax(np.abs(sep_f)))

    print(f"  arm={arm}  anchor={anchor}(signature f{anchor_f})"
          f"  window=[f{lo},f{hi}]  medfilt k={k}"
          f"  gated={100 * (1 - valid.mean()):.0f}%")
    print(f"  HSS peak (windowed, medfilt) = {hss_peak:.1f} deg @ f{pk_f}"
          f"   (raw windowed peak was {raw_pk:.1f})")
    print(f"  HSS peak (full clip, medfilt)= {hss_full:.1f} deg @ f{full_f}"
          + ("   <- differs: window is doing real work" if full_f != pk_f else ""))

    save_check_frames(video, df, {f"anchor{sfx}": anchor_f, f"hsspeak{sfx}": pk_f,
                                  f"early{sfx}": int(len(df)*0.25)}, out_dir, name)
    still_sfx = f"{sfx}_sq" if square else sfx
    draw_hss_still(video, df, pk_f, anchor_f, float(sep_f[pk_f]),
                   os.path.join(out_dir, f"{name}_hss{still_sfx}.png"), fps,
                   square=square)
    if mp4:
        render_hss_video(video, df, sep_f, pk_f, anchor_f,
                         os.path.join(out_dir, f"{name}_hss{sfx}.mp4"), fps,
                         valid=valid)
    return {"video": name, "arm": arm, "fps": fps,
            "anchor": anchor, "anchor_f": anchor_f,
            "gated_frac": float(1 - valid.mean()),
            "hss_peak_win": hss_peak, "peak_f": pk_f, "hss_peak_win_raw": raw_pk,
            "hss_peak_full": hss_full,
            "detection_rate": det, **{f"vis_{j}": v for j, v in vis.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="single video name (no ext)")
    ap.add_argument("--win-pre", type=float, default=0.6,
                    help="window seconds before release anchor")
    ap.add_argument("--win-post", type=float, default=0.1,
                    help="window seconds after release anchor")
    ap.add_argument("--model", default=None, help="alternative .task model path")
    ap.add_argument("--tag", default="", help="suffix for outputs (model variant)")
    ap.add_argument("--mp4", action="store_true", help="also render overlay video")
    ap.add_argument("--backbone", default="rtmp",
                    choices=["mediapipe", "rtmp"],
                    help="pose backbone (default rtmp, the official backbone)")
    ap.add_argument("--raw", action="store_true",
                    help="use RAW RTMPose coords; default is the "
                         "occlusion-recovery-refined coords (the best pose)")
    ap.add_argument("--anchor", default="signature",
                    choices=["signature", "release"],
                    help="window anchor; signature = release-free torso anchor "
                         "(OBP r2 0.63), release = legacy wrist-speed argmax")
    ap.add_argument("--slomo", type=float, default=1.0,
                    help="slow-motion factor of the video (e.g. 4 for 4x)")
    ap.add_argument("--square", action="store_true",
                    help="render the HSS still as a person-centered 1:1 crop")
    a = ap.parse_args()
    if a.backbone == "rtmp" and not a.tag:
        # default = the recovery-refined best pose; --raw opts out to stock
        a.tag = "rtmp" if a.raw else "rtmp_refined"

    rows = [run_video(n, a.win_pre, a.win_post, model=a.model, tag=a.tag,
                      mp4=a.mp4, backbone=a.backbone, anchor=a.anchor,
                      slomo=a.slomo, square=a.square)
            for n in ([a.video] if a.video else VIDEOS)]
    sfx = f"_{a.tag}" if a.tag else ""
    out = os.path.join(config.OBP_VALIDATION_DIR, f"hss_overhead_real{sfx}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
