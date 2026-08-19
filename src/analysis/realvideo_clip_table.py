"""Build the real-video clip table the feasibility driver consumes.

POPULATION (author's decision, 2026-08-01). Four kinds of self-filmed clip:

    angle00_00 .. angle11_04        60   orbit sweep
    real_video_test_01 .. _15       15
    pitching_overhead_01 .. _03      3
    consistency_test_{side,front}    2   left-handed subject
                                  ----
                                    80

`pitching_lateral_02` and `pitching_frontier_03` are dropped; the externally sourced
and non-pitch clips were never in this population.

The 78 clips that were in the frozen pilot keep their frozen anchors verbatim: nothing
is re-detected for them, so every number already reported for those clips is unchanged.
The two consistency clips were not in the pilot and had no RTMPose extraction until
2026-08-01, so their anchors are detected here, through the same `metrics` functions and
the same viewpoint routing the pilot used. That is stated rather than hidden: for these
two the detection is this script's, not the pilot's.

⚠ The consistency subject is LEFT-handed. `true_arm` is set accordingly, which swaps the
lead and trail sides for every foot-plant, stride and knee quantity. Azimuth is
handedness-relative throughout this project, so az 0 is the OPEN side for this pitcher
too, not a field bearing.

Run:  conda activate diamond
      cd src\\analysis
      python realvideo_clip_table.py
"""
import os, sys
import numpy as np, pandas as pd, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../stage2", "../deploy"):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
import metrics as M
from angle_zone_sweep import release_view
from fp_routing import fp_view

PILOT = os.path.join(config.ROOT, "data", "outputs", "realvideo_pilot",
                     "pilot_clips_eligible.csv")
OUT = os.path.join(config.OBP_VALIDATION_DIR, "realvideo_clips.csv")

KEEP_PREFIX = ("angle", "real_video_test_", "pitching_overhead_")
DROP = {"pitching_lateral_02", "pitching_frontier_03"}

# clip -> (video file, throwing arm, azimuth, elevation)
NEW = {
    "consistency_test_side_10s":  ("consistency_test_side_10s.mp4", "left", 0, 0),
    "consistency_test_front_6s":  ("consistency_test_front_6s.mp4", "left", 90, 0),
}
MER_LAG_S = 11.0 / 360.0        # angle_map_2d.MER_LAG_S


def clip_fps(fn):
    cap = cv2.VideoCapture(os.path.join(config.ROOT, "data", "videos", fn))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps


def load_pose(clip):
    base = os.path.join(config.ROOT, "data", "outputs", clip)
    sm = config.refined_or(os.path.join(base, f"{clip}_smoothed_rtmp.csv"))
    rw = config.refined_or(os.path.join(base, f"{clip}_coords_rtmp.csv"))
    df = pd.read_csv(sm)
    raw = pd.read_csv(rw) if os.path.exists(rw) else None
    for d in (df, raw):
        if d is not None and "nose_x" in d.columns and "head_x" not in d.columns:
            d.rename(columns={"nose_x": "head_x", "nose_y": "head_y",
                              "nose_v": "head_v"}, inplace=True)
    return df, raw


def detect(clip, fn, arm, az, el):
    """Anchors for one clip, through the same functions and routing as the pilot."""
    df, raw = load_pose(clip)
    fps = clip_fps(fn)
    lead = "left" if arm == "right" else "right"
    rv = release_view(az, el)
    rel = int(M.release_frame(df, arm, fps, M.JOINTS, view=rv, raw_df=raw))
    fv = fp_view(az, el)
    fp = int(M.foot_plant_frame(df, lead, fps, M.JOINTS, rel, view=fv))
    # foot_plant_frame returns rel - 0.13*fps when it finds no candidate
    fallback = max(0, rel - int(0.13 * fps))
    detected = fp != fallback
    pkh = M.peak_knee_height_frame(df, lead, fp, M.JOINTS)
    mer = int(round(rel - MER_LAG_S * fps))
    return dict(clip=clip, fps=fps, video_path=os.path.join(
                    config.ROOT, "data", "videos", fn),
                n_frames=len(df), duration_s=len(df) / fps,
                true_arm=arm, mirrored=False, az=az, el=el, vote_share=np.nan,
                fp_detector=fv, release_view=rv, release_f=float(rel),
                release_gate="", fp_status="detected" if detected
                else "fallback_no_candidate",
                fp_detected=bool(detected), fp_f=float(fp),
                pkh_f=float(pkh) if pkh is not None and np.isfinite(pkh) else np.nan,
                mer_f=float(mer))


def main():
    old = pd.read_csv(PILOT)
    keep = old[old["clip"].astype(str).str.startswith(KEEP_PREFIX)
               & ~old["clip"].isin(DROP)].copy()
    print(f"kept from the frozen pilot: {len(keep)}")
    for pre in KEEP_PREFIX:
        print(f"  {pre:<22s} {int(keep['clip'].str.startswith(pre).sum())}")
    dropped = sorted(set(old["clip"]) - set(keep["clip"]))
    print(f"dropped: {dropped}")

    rows = []
    for clip, (fn, arm, az, el) in NEW.items():
        r = detect(clip, fn, arm, az, el)
        rows.append(r)
        print(f"detected {clip}: fps {r['fps']:.2f}  n {r['n_frames']}  "
              f"arm {arm}  az{az}/el{el}  release {r['release_f']:.0f}  "
              f"fp {r['fp_f']:.0f} ({r['fp_status']})  pkh {r['pkh_f']}  "
              f"mer {r['mer_f']:.0f}")

    new = pd.DataFrame(rows)
    out = pd.concat([keep, new], ignore_index=True, sort=False)
    assert len(out) == 80, len(out)
    out.to_csv(OUT, index=False)
    print(f"\npopulation {len(out)} clips -> {OUT}")
    print("handedness:", out.true_arm.value_counts().to_dict())
    print("elevation :", out.el.value_counts().sort_index().to_dict())
    print("fp detected:", int(out.fp_detected.sum()), "of", len(out))


if __name__ == "__main__":
    main()
