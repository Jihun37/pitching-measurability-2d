"""
Diamond - Backbone occlusion diagnostic (MediaPipe heavy vs RTMPose).

Question this probe answers: when the throwing arm "disappears" in the
overhead view, is that (a) true occlusion no 2D model can solve, or
(b) MediaPipe being out-of-distribution while a stronger backbone still
tracks the arm?  The answer decides whether the fix is a backbone swap
or an explicit unobserved-joint treatment.

Per video (pitching_overhead_01..03):
  1. RTMPose-x extraction via rtmlib BodyWithFeet (Halpe26, det per frame),
     cached to <name>_coords_rtmp.csv in the same schema as pose_extractor
     (joint_x/_y in px, joint_v = keypoint score).
  2. Load the cached MediaPipe heavy raw coords (<name>_coords_heavy.csv).
  3. Confidence comparison on the throwing-arm wrist/elbow inside the
     delivery window (release-anchored, same window as hss_overhead_real):
     per-frame series plot + summary stats.
     Note: MediaPipe "visibility" and RTMPose keypoint score are different
     quantities; compare each model against its own baseline, not the
     absolute numbers.
  4. Divergence check frames: both skeletons drawn on the frames where the
     two models disagree most on the wrist (plus release / HSS peak), so
     the occlusion-vs-weakness question is answered by eye.
  5. HSS recomputed from RTMPose coords through the identical pipeline
     (smoother -> hss_sep_series -> medfilt -> windowed peak) next to the
     MediaPipe heavy value.

Run:  cd src\tests
      python backbone_diag_test.py                  # all three videos
      python backbone_diag_test.py --video pitching_overhead_01 --mp4
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import medfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "stage1"))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import config
import metrics as M
from smoother import smooth_coordinates
from hss_elevation_test import hss_sep_series
from hss_overhead_real import detect_arm, CONNECT

VIDEOS = ["pitching_overhead_01", "pitching_overhead_02", "pitching_overhead_03"]

# extractor promoted to stage1 (2026-07-04) — single source of the definition
from rtmp_extractor import extract_pose_rtmp, HALPE26, CORE

GREEN, MAGENTA, WHITE = (0, 255, 0), (255, 0, 255), (255, 255, 255)


def extract_rtmpose(video_path, save_csv, mode="performance"):
    """Back-compat wrapper around the stage1 engine (returns df, fps)."""
    df, meta = extract_pose_rtmp(video_path, save_csv=save_csv, mode=mode)
    return df, meta["fps"]


def draw_skel(frame, df, f, color, alpha=0.45, thickness=3):
    """One model's skeleton, blended at 45% so overlapping models stay readable."""
    ov = frame.copy()
    for a, b in CONNECT:
        try:
            p1 = (int(df[f"{a}_x"].iloc[f]), int(df[f"{a}_y"].iloc[f]))
            p2 = (int(df[f"{b}_x"].iloc[f]), int(df[f"{b}_y"].iloc[f]))
            cv2.line(ov, p1, p2, color, thickness)
        except (ValueError, KeyError):
            pass
    return cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0)


def annotate(fr, tag, f, mp_v, rt_v, dist, scale=1.0):
    s = scale
    cv2.putText(fr, f"frame {f}  {tag}", (int(20*s), int(50*s)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1*s, WHITE, max(2, int(2*s)))
    cv2.putText(fr, "MediaPipe heavy", (int(20*s), int(95*s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9*s, GREEN, max(2, int(2*s)))
    cv2.putText(fr, "RTMPose-x", (int(20*s), int(135*s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9*s, MAGENTA, max(2, int(2*s)))
    cv2.putText(fr, f"wrist  mp_vis={mp_v:.2f}  rtmp_score={rt_v:.2f}  dist={dist:.0f}px",
                (int(20*s), int(175*s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9*s, WHITE, max(2, int(2*s)))


def save_diag_frames(video_path, mp_df, rt_df, frames, arm, out_dir, name, scale=0.5):
    cap = cv2.VideoCapture(video_path)
    for tag, f in frames.items():
        f = int(np.clip(f, 0, min(len(mp_df), len(rt_df)) - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            continue
        fr = draw_skel(fr, mp_df, f, GREEN)
        fr = draw_skel(fr, rt_df, f, MAGENTA)
        mp_v = float(mp_df[f"{arm}_wrist_v"].iloc[f])
        rt_v = float(rt_df[f"{arm}_wrist_v"].iloc[f])
        d = float(np.hypot(mp_df[f"{arm}_wrist_x"].iloc[f] - rt_df[f"{arm}_wrist_x"].iloc[f],
                           mp_df[f"{arm}_wrist_y"].iloc[f] - rt_df[f"{arm}_wrist_y"].iloc[f]))
        fr = cv2.resize(fr, None, fx=scale, fy=scale)
        annotate(fr, tag, f, mp_v, rt_v, d * scale, scale=1.0)
        out = os.path.join(out_dir, f"{name}_diag_{tag}.png")
        cv2.imwrite(out, fr)
        print(f"  diag frame -> {out}")
    cap.release()


def render_diag_video(video_path, mp_df, rt_df, arm, out_mp4, fps):
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    r = 1280.0 / W
    size = (1280, int(H * r))
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"),
                         int(round(fps)), size)
    n = min(len(mp_df), len(rt_df))
    f = 0
    while f < n:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, size)
        # draw on the resized frame: scale coords via temporary views
        mp_s = mp_df.iloc[[f]].copy()
        rt_s = rt_df.iloc[[f]].copy()
        for c in mp_s.columns:
            if c.endswith("_x") or c.endswith("_y"):
                mp_s[c] = mp_s[c] * r
                rt_s[c] = rt_s[c] * r
        fr = draw_skel(fr, mp_s, 0, GREEN)
        fr = draw_skel(fr, rt_s, 0, MAGENTA)
        mp_v = float(mp_df[f"{arm}_wrist_v"].iloc[f])
        rt_v = float(rt_df[f"{arm}_wrist_v"].iloc[f])
        d = float(np.hypot(mp_df[f"{arm}_wrist_x"].iloc[f] - rt_df[f"{arm}_wrist_x"].iloc[f],
                           mp_df[f"{arm}_wrist_y"].iloc[f] - rt_df[f"{arm}_wrist_y"].iloc[f])) * r
        annotate(fr, "", f, mp_v, rt_v, d, scale=0.7)
        vw.write(fr)
        f += 1
    cap.release(); vw.release()
    print(f"  diag video -> {out_mp4}")


def confidence_plot(mp_df, rt_df, arm, lo, hi, rel, out_png, name):
    n = min(len(mp_df), len(rt_df))
    x = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, joint in zip(axes, ["wrist", "elbow"]):
        ax.plot(x, mp_df[f"{arm}_{joint}_v"].to_numpy(float)[:n],
                color="green", label="MediaPipe heavy (visibility)")
        ax.plot(x, rt_df[f"{arm}_{joint}_v"].to_numpy(float)[:n],
                color="magenta", label="RTMPose-x (score)")
        ax.axvspan(lo, hi, alpha=0.15, color="gray", label="delivery window")
        ax.axvline(rel, color="black", ls="--", lw=1, label="release anchor")
        ax.set_ylabel(f"{arm} {joint}")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=8)
    axes[1].set_xlabel("frame")
    fig.suptitle(f"{name}: throwing-arm confidence, MediaPipe vs RTMPose")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  confidence plot -> {out_png}")


def pick_divergence_frames(mp_df, rt_df, arm, lo, hi, k=3, min_gap=5):
    """Frames inside the window where the two models disagree most on the wrist."""
    n = min(len(mp_df), len(rt_df))
    d = np.hypot(mp_df[f"{arm}_wrist_x"].to_numpy(float)[:n] - rt_df[f"{arm}_wrist_x"].to_numpy(float)[:n],
                 mp_df[f"{arm}_wrist_y"].to_numpy(float)[:n] - rt_df[f"{arm}_wrist_y"].to_numpy(float)[:n])
    d = np.nan_to_num(d, nan=0.0)
    order = lo + np.argsort(d[lo:hi + 1])[::-1]
    picked = []
    for f in order:
        if all(abs(f - p) >= min_gap for p in picked):
            picked.append(int(f))
        if len(picked) == k:
            break
    return picked, d


def run_video(name, win_pre, win_post, mode, mp4=False):
    video = next(p for p in
                 (os.path.join(config.ROOT, "data", "videos", name + ext)
                  for ext in (".mov", ".mp4", ".MOV"))
                 if os.path.exists(p))
    out_dir = os.path.join(config.ROOT, "data", "outputs", name)
    os.makedirs(out_dir, exist_ok=True)
    rtmp_csv = os.path.join(out_dir, f"{name}_coords_rtmp.csv")
    heavy_csv = os.path.join(out_dir, f"{name}_coords_heavy.csv")

    print(f"\n=== {name} ===", flush=True)
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()

    if os.path.exists(rtmp_csv):
        rt_raw = pd.read_csv(rtmp_csv)
        print(f"  cached rtmpose coords ({len(rt_raw)} frames)")
    else:
        rt_raw, fps = extract_rtmpose(video, rtmp_csv, mode=mode)

    if not os.path.exists(heavy_csv):
        raise FileNotFoundError(f"MediaPipe heavy baseline missing: {heavy_csv}")
    mp_raw = pd.read_csv(heavy_csv)

    # identical downstream pipeline for both models
    rt_df = smooth_coordinates(rt_raw)
    mp_df = smooth_coordinates(mp_raw)

    arm_side = detect_arm(rt_df)
    arm = f"{arm_side}"
    rel = M.release_frame(rt_df, arm_side, fps, M.JOINTS, view="side")
    lo = max(0, rel - int(win_pre * fps))
    hi = min(min(len(rt_df), len(mp_df)) - 1, rel + int(win_post * fps))

    # --- confidence stats inside the delivery window (raw, unsmoothed _v) ---
    n = min(len(mp_raw), len(rt_raw))
    stats = {}
    for tag, df in (("mp", mp_raw), ("rtmp", rt_raw)):
        for joint in ("wrist", "elbow"):
            v = df[f"{arm}_{joint}_v"].to_numpy(float)[:n][lo:hi + 1]
            stats[f"{tag}_{joint}_mean_v"] = float(np.nanmean(v))
            stats[f"{tag}_{joint}_min_v"] = float(np.nanmin(v))
            stats[f"{tag}_{joint}_frac_low"] = float(np.mean(np.nan_to_num(v) < 0.5))
    print(f"  arm={arm}  release=f{rel}  window=[f{lo},f{hi}]")
    print(f"  wrist  mp: mean_v={stats['mp_wrist_mean_v']:.2f} "
          f"min={stats['mp_wrist_min_v']:.2f} frac<0.5={stats['mp_wrist_frac_low']:.2f}"
          f"   rtmp: mean={stats['rtmp_wrist_mean_v']:.2f} "
          f"min={stats['rtmp_wrist_min_v']:.2f} frac<0.5={stats['rtmp_wrist_frac_low']:.2f}")
    print(f"  elbow  mp: mean_v={stats['mp_elbow_mean_v']:.2f} "
          f"min={stats['mp_elbow_min_v']:.2f} frac<0.5={stats['mp_elbow_frac_low']:.2f}"
          f"   rtmp: mean={stats['rtmp_elbow_mean_v']:.2f} "
          f"min={stats['rtmp_elbow_min_v']:.2f} frac<0.5={stats['rtmp_elbow_frac_low']:.2f}")

    confidence_plot(mp_raw, rt_raw, arm, lo, hi, rel,
                    os.path.join(out_dir, f"{name}_diag_conf.png"), name)

    # --- HSS through the identical pipeline, RTMPose coords ---
    sep = hss_sep_series(rt_df, M.JOINTS)
    k = int(0.09 * fps) // 2 * 2 + 1
    sep_f = medfilt(np.nan_to_num(sep, nan=0.0), kernel_size=k)
    seg = np.abs(sep_f[lo:hi + 1])
    pk_f = lo + int(np.nanargmax(seg))
    hss_rtmp = float(np.nanmax(seg))
    print(f"  HSS peak (rtmpose, windowed, medfilt) = {hss_rtmp:.1f} deg @ f{pk_f}")

    # --- divergence check frames ---
    div, dist = pick_divergence_frames(mp_df, rt_df, arm, lo, hi)
    frames = {"release": rel, "hsspeak": pk_f}
    for i, f in enumerate(div):
        frames[f"div{i+1}"] = f
    save_diag_frames(video, mp_df, rt_df, frames, arm, out_dir, name)
    print(f"  wrist model-disagreement in window: median={np.median(dist[lo:hi+1]):.0f}px "
          f"max={np.max(dist[lo:hi+1]):.0f}px (frame width 3840)")

    if mp4:
        render_diag_video(video, mp_df, rt_df, arm,
                          os.path.join(out_dir, f"{name}_diag.mp4"), fps)

    return {"video": name, "arm": arm, "release_f": rel,
            "window_lo": lo, "window_hi": hi,
            "hss_peak_rtmp": hss_rtmp, "hss_peak_f_rtmp": pk_f,
            "wrist_dist_median_px": float(np.median(dist[lo:hi + 1])),
            "wrist_dist_max_px": float(np.max(dist[lo:hi + 1])), **stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="single video name (no ext)")
    ap.add_argument("--win-pre", type=float, default=0.6)
    ap.add_argument("--win-post", type=float, default=0.1)
    ap.add_argument("--mode", default="performance",
                    choices=["performance", "balanced", "lightweight"])
    ap.add_argument("--mp4", action="store_true", help="render overlay comparison video")
    a = ap.parse_args()

    rows = [run_video(n, a.win_pre, a.win_post, a.mode, mp4=a.mp4)
            for n in ([a.video] if a.video else VIDEOS)]
    out = os.path.join(config.OBP_VALIDATION_DIR, "backbone_diag_rtmp.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
