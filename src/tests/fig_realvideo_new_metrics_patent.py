"""Patent Fig. 7 EXTENSION: the three 2026-07-24 metrics drawn on real phone video,
matching the cv2 overlay style of validate_overlay.py without touching it.

  (e) front  / MER      : torso lateral tilt @MER   (angle03_04, MER = release-4f)
  (f) front  / MER      : glove-arm abduction @MER  (angle03_04, MER = release-4f)
  (g) overhead / release: torso transverse rotation (pitching_overhead_02, f389 =
                          the manual HSS anchor; the shoulder line is pose-
                          estimated -- unlike the pelvis it is not self-occluded)

Frames are grabbed SEQUENTIALLY (cv2 CAP_PROP_POS_FRAMES random-seek is keyframe-
snapped on iPhone HEVC and lands off, per CLAUDE.md). Release frames are the human
GT already used by the paper stills: side f493, front f511, overhead HSS f389.

Run:
  conda activate diamond
  cd src\\tests
  python fig_realvideo_new_metrics_patent.py
"""
import os, sys
import numpy as np, pandas as pd, cv2

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(HERE, "..", "viz"))
import config
import metrics as M
from validate_overlay import CONNECT, pt
from skeleton import square_crop

J = M.JOINTS
FPS = 120.0
GREEN = (0, 255, 0); WHITE = (255, 255, 255); YEL = (0, 255, 255); RED = (0, 0, 255)
LAT = (200, 40, 190)      # torso lateral tilt (magenta/fuchsia, BGR)
ABD = (255, 120, 20)      # glove-arm abduction (blue, BGR)
KNEE = (139, 139, 0); WRST = (147, 20, 255); BLU = (255, 80, 0)   # side-release colors
PLUM = (119, 39, 219)     # release extension (magenta ground dimension, BGR)
ORNG = (12, 88, 234)      # COG velo @PKH (orange, BGR)
VIO = (237, 58, 124)      # COG fwd velo / pelvis rot velo (violet, BGR)
STRA = (211, 85, 186)     # stride angle (purple, BGR)
GRAY = (150, 150, 150)    # COM trajectory


def grab_seq(video, target):
    """Sequential frame grab (no CAP_PROP_POS_FRAMES seek)."""
    cap = cv2.VideoCapture(video)
    f, out = 0, None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if f == target:
            out = fr.copy(); break
        f += 1
    cap.release()
    return out


def load_coords(clip):
    p = os.path.join(config.ROOT, "data", "outputs", clip, f"{clip}_smoothed_rtmp.csv")
    df = pd.read_csv(p)
    if "nose_x" in df.columns and "head_x" not in df.columns:
        df = df.rename(columns={"nose_x": "head_x", "nose_y": "head_y"})
    return df


def detect_arm(df):
    def pk(j):
        x = df[f"{j}_x"].to_numpy(float); y = df[f"{j}_y"].to_numpy(float)
        return np.nanmax(np.hypot(np.diff(x), np.diff(y)))
    return "right" if pk("right_wrist") >= pk("left_wrist") else "left"


def draw_skeleton(frame, df, f):
    for j1, j2 in CONNECT:
        try:
            cv2.line(frame, pt(df, j1, f), pt(df, j2, f), GREEN, 2)
        except Exception:
            pass


def arc_minor(frame, c, p1, p2, r, color, lw=3):
    """Minor cv2 arc from ray c->p1 toward ray c->p2."""
    a1 = np.degrees(np.arctan2(p1[1] - c[1], p1[0] - c[0]))
    a2 = np.degrees(np.arctan2(p2[1] - c[1], p2[0] - c[0]))
    d = ((a2 - a1 + 180.0) % 360.0) - 180.0
    cv2.ellipse(frame, c, (r, r), 0, a1, a1 + d, color, lw)


def banner(sq, f, event, label, col):
    cv2.putText(sq, f"frame {f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    cv2.putText(sq, event, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
    cv2.putText(sq, label, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)


def panel_torso_lat(clip, rel_gt):
    df = load_coords(clip)
    video = os.path.join(config.ROOT, "data", "videos", f"{clip}.mov")
    mer = int(round(rel_gt - (11.0 / 360.0) * FPS))
    val = float(M.trunk_lean_2d(df, mer, J))
    frame = grab_seq(video, mer)
    draw_skeleton(frame, df, mer)
    def mid(a, b): return ((pt(df, a, mer)[0] + pt(df, b, mer)[0]) // 2,
                           (pt(df, a, mer)[1] + pt(df, b, mer)[1]) // 2)
    mh = mid("left_hip", "right_hip"); ms = mid("left_shoulder", "right_shoulder")
    seg = int(np.hypot(ms[0] - mh[0], ms[1] - mh[1])) or 80
    vtop = (mh[0], mh[1] - seg)
    cv2.line(frame, mh, vtop, WHITE, 2)
    cv2.line(frame, mh, ms, LAT, 3)
    arc_minor(frame, mh, vtop, ms, 42, LAT)
    cv2.putText(frame, f"lat tilt {val:.0f}", (mh[0] + 14, mh[1] - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, LAT, 2)
    sq = square_crop(frame, df, mer, pad=0.22)
    banner(sq, mer, "MER (release-4f)", f"torso lat tilt {val:.0f} deg", LAT)
    out = os.path.join(config.ROOT, "data", "outputs", clip, f"{clip}_rtmp_torsolat_sq.png")
    cv2.imwrite(out, sq); print(f"  lat tilt {val:.1f} @f{mer} -> {out}")
    return out


def panel_glove_abd(clip, rel_gt):
    df = load_coords(clip); arm = detect_arm(df)
    s = "left" if arm == "right" else "right"
    video = os.path.join(config.ROOT, "data", "videos", f"{clip}.mov")
    mer = int(round(rel_gt - (11.0 / 360.0) * FPS))
    val = float(M.shoulder_abduction_2d(df, "glove", mer, J))
    frame = grab_seq(video, mer)
    draw_skeleton(frame, df, mer)
    E, S, Hh = pt(df, f"{s}_elbow", mer), pt(df, f"{s}_shoulder", mer), pt(df, f"{s}_hip", mer)
    cv2.line(frame, S, E, ABD, 3); cv2.line(frame, S, Hh, ABD, 3)
    cv2.circle(frame, S, 7, RED, -1)
    arc_minor(frame, S, E, Hh, 46, ABD)
    cv2.putText(frame, f"glove abd {val:.0f}", (S[0] + 14, S[1] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, ABD, 2)
    sq = square_crop(frame, df, mer, pad=0.22)
    banner(sq, mer, "MER (release-4f)", f"glove-arm abd {val:.0f} deg", ABD)
    out = os.path.join(config.ROOT, "data", "outputs", clip, f"{clip}_rtmp_gloveabd_sq.png")
    cv2.imwrite(out, sq); print(f"  glove abd {val:.1f} @f{mer} -> {out}")
    return out


def panel_mer_combined(clip, rel_gt):
    """(f) front / MER, torso lateral tilt AND glove-arm abduction on ONE still.
    Both are the same event (MER = release-4f) at the same frame, so they share a
    single crop -- the split into two panels was redundant. Geometry/colors are
    unchanged from panel_torso_lat (magenta) and panel_glove_abd (blue)."""
    df = load_coords(clip); arm = detect_arm(df)
    s = "left" if arm == "right" else "right"   # glove-side shoulder
    video = os.path.join(config.ROOT, "data", "videos", f"{clip}.mov")
    mer = int(round(rel_gt - (11.0 / 360.0) * FPS))
    lat = float(M.trunk_lean_2d(df, mer, J))
    abd = float(M.shoulder_abduction_2d(df, "glove", mer, J))
    frame = grab_seq(video, mer)
    draw_skeleton(frame, df, mer)
    # torso lateral tilt (magenta): mid-hip vertical vs mid-hip -> mid-shoulder
    def mid(a, b): return ((pt(df, a, mer)[0] + pt(df, b, mer)[0]) // 2,
                           (pt(df, a, mer)[1] + pt(df, b, mer)[1]) // 2)
    mh = mid("left_hip", "right_hip"); ms = mid("left_shoulder", "right_shoulder")
    seg = int(np.hypot(ms[0] - mh[0], ms[1] - mh[1])) or 80
    vtop = (mh[0], mh[1] - seg)
    cv2.line(frame, mh, vtop, WHITE, 2)
    cv2.line(frame, mh, ms, LAT, 3)
    arc_minor(frame, mh, vtop, ms, 42, LAT)
    cv2.putText(frame, f"lat tilt {lat:.0f}", (mh[0] + 14, mh[1] - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, LAT, 2)
    # glove-arm abduction (blue): glove shoulder->elbow vs glove shoulder->hip
    E, S, Hh = pt(df, f"{s}_elbow", mer), pt(df, f"{s}_shoulder", mer), pt(df, f"{s}_hip", mer)
    cv2.line(frame, S, E, ABD, 3); cv2.line(frame, S, Hh, ABD, 3)
    cv2.circle(frame, S, 7, RED, -1)
    arc_minor(frame, S, E, Hh, 46, ABD)
    cv2.putText(frame, f"glove abd {abd:.0f}", (S[0] + 14, S[1] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, ABD, 2)
    sq = square_crop(frame, df, mer, pad=0.22)
    cv2.putText(sq, f"frame {mer}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    cv2.putText(sq, "MER (release-4f)", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2)
    cv2.putText(sq, f"torso lat tilt {lat:.0f} deg", (20, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, LAT, 2)
    cv2.putText(sq, f"glove-arm abd {abd:.0f} deg", (20, 162),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, ABD, 2)
    out = os.path.join(config.ROOT, "data", "outputs", clip, f"{clip}_rtmp_mer_sq.png")
    cv2.imwrite(out, sq)
    print(f"  MER combined: lat tilt {lat:.1f}, glove abd {abd:.1f} @f{mer} -> {out}")
    return out


def panel_torso_rot(clip, frame_gt):
    df = load_coords(clip)
    video = os.path.join(config.ROOT, "data", "videos", f"{clip}.mov")
    raw = np.degrees(np.arctan2(
        df["right_shoulder_y"].iloc[frame_gt] - df["left_shoulder_y"].iloc[frame_gt],
        df["right_shoulder_x"].iloc[frame_gt] - df["left_shoulder_x"].iloc[frame_gt])) % 180.0
    acute = raw if raw <= 90 else 180.0 - raw
    frame = grab_seq(video, frame_gt)
    draw_skeleton(frame, df, frame_gt)
    ls, rs = pt(df, "left_shoulder", frame_gt), pt(df, "right_shoulder", frame_gt)
    smid = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
    L = int(np.hypot(rs[0] - ls[0], rs[1] - ls[1]) * 0.7 + 44)
    cv2.line(frame, (smid[0] - L, smid[1]), (smid[0] + L, smid[1]), WHITE, 2)
    cv2.line(frame, ls, rs, YEL, 3)
    sr = rs if rs[0] >= ls[0] else ls
    arc_minor(frame, smid, (smid[0] + L, smid[1]), sr, 50, RED)
    cv2.putText(frame, f"torso rot {acute:.0f}", (smid[0] + 18, smid[1] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)
    sq = square_crop(frame, df, frame_gt, pad=0.28)
    banner(sq, frame_gt, "RELEASE (ballsep)", f"torso rot {acute:.0f} deg", RED)
    out = os.path.join(config.ROOT, "data", "outputs", clip, f"{clip}_torsorot_sq.png")
    cv2.imwrite(out, sq); print(f"  torso rot {acute:.1f} @f{frame_gt} -> {out}")
    return out


def panel_side_release(clip, rel_gt):
    """(b) side release WITHOUT knee ext velo (de-adopted 2026-07-24). Redraws the
    adopted release metrics only: lead knee, trunk anterior tilt, wrist speed,
    release height -- same cv2 idioms as validate_overlay.draw_event."""
    df = load_coords(clip); arm = detect_arm(df)
    lead = "left" if arm == "right" else "right"
    video = os.path.join(config.ROOT, "data", "videos", f"{clip}.mov")
    cand = {k: v for k, (v, _) in M.compute_candidates(
        df, fps=FPS, arm=arm, view="side", rel=rel_gt).items()}
    f = int(rel_gt)
    frame = grab_seq(video, f)
    draw_skeleton(frame, df, f)
    # lead knee angle (NO knee-ext-velo arrow)
    hp, kp, ap_ = pt(df, f"{lead}_hip", f), pt(df, f"{lead}_knee", f), pt(df, f"{lead}_ankle", f)
    cv2.line(frame, hp, kp, KNEE, 3); cv2.line(frame, kp, ap_, KNEE, 3)
    cv2.circle(frame, kp, 8, KNEE, -1)
    arc_minor(frame, kp, hp, ap_, 45, KNEE)
    cv2.putText(frame, f"knee {cand['lead_knee_at_release']:.0f}", (kp[0] + 12, kp[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, KNEE, 2)
    # trunk anterior tilt
    mh = (int((df['left_hip_x'].iloc[f] + df['right_hip_x'].iloc[f]) / 2),
          int((df['left_hip_y'].iloc[f] + df['right_hip_y'].iloc[f]) / 2))
    ms = (int((df['left_shoulder_x'].iloc[f] + df['right_shoulder_x'].iloc[f]) / 2),
          int((df['left_shoulder_y'].iloc[f] + df['right_shoulder_y'].iloc[f]) / 2))
    seg = int(np.hypot(ms[0] - mh[0], ms[1] - mh[1])) or 80
    vtop = (mh[0], mh[1] - seg)
    cv2.line(frame, mh, vtop, WHITE, 2); cv2.line(frame, mh, ms, YEL, 3)
    arc_minor(frame, mh, vtop, ms, 40, YEL)
    cv2.putText(frame, f"tilt {cand['trunk_anterior_tilt']:.0f}", (mh[0] + 12, mh[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, YEL, 2)
    # wrist speed vector (1-frame diff)
    wp = pt(df, f"{arm}_wrist", f)
    dx = (df[f"{arm}_wrist_x"].iloc[f] - df[f"{arm}_wrist_x"].iloc[f - 1]) * FPS
    dy = (df[f"{arm}_wrist_y"].iloc[f] - df[f"{arm}_wrist_y"].iloc[f - 1]) * FPS
    mag = (dx * dx + dy * dy) ** 0.5 + 1e-6
    tip = (int(wp[0] + dx / mag * 90), int(wp[1] + dy / mag * 90))
    cv2.arrowedLine(frame, wp, tip, WRST, 3, tipLength=0.3); cv2.circle(frame, wp, 7, WRST, -1)
    cv2.putText(frame, f"wrist_spd {cand['wrist_speed']:.1f}", (wp[0] + 10, wp[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WRST, 2)
    # release height dimension (wrist -> ground)
    gy = int(np.nanmax(np.concatenate([df["left_ankle_y"].to_numpy(float),
                                       df["right_ankle_y"].to_numpy(float)])))
    ov = frame.copy()
    cv2.line(ov, (wp[0], wp[1]), (wp[0], gy), BLU, 3)
    cv2.line(ov, (wp[0] - 10, gy), (wp[0] + 10, gy), BLU, 3)
    cv2.line(ov, (wp[0] - 10, wp[1]), (wp[0] + 10, wp[1]), BLU, 3)
    cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, f"rel_height {cand['release_height']:.2f}", (wp[0] + 14, (wp[1] + gy) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLU, 2)
    # release extension: horizontal ground dimension, trail-foot rubber anchor -> release point
    fp = M.foot_plant_frame(df, lead, FPS, J, f)
    trail = f"{'right' if arm == 'right' else 'left'}_ankle"
    anchor_x = int(M.trail_anchor_x(df[f"{trail}_x"].to_numpy(float), fp, FPS))
    rex = float(M.release_extension(df, arm, f, fp, FPS, J))
    y_ext = gy + 42
    for xx in (anchor_x, wp[0]):
        cv2.line(frame, (xx, y_ext - 10), (xx, y_ext + 10), PLUM, 3)
    cv2.arrowedLine(frame, (anchor_x, y_ext), (wp[0], y_ext), PLUM, 3, tipLength=0.04)
    cv2.arrowedLine(frame, (wp[0], y_ext), (anchor_x, y_ext), PLUM, 3, tipLength=0.04)
    cv2.putText(frame, f"rel_ext {rex:.2f}", ((anchor_x + wp[0]) // 2 - 70, y_ext + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, PLUM, 2)
    sq = square_crop(frame, df, f, pad=0.22)
    cv2.putText(sq, f"frame {f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    cv2.putText(sq, "RELEASE", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.1, RED, 3)
    lines = [f"knee {cand['lead_knee_at_release']:.0f} deg",
             f"trunk_tilt {cand['trunk_anterior_tilt']:.0f} deg",
             f"wrist_spd {cand['wrist_speed']:.1f}",
             f"rel_height {cand['release_height']:.2f}",
             f"rel_ext {rex:.2f} stat"]
    for i, t in enumerate(lines):
        cv2.putText(sq, t, (20, 130 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
    out = os.path.join(config.ROOT, "data", "outputs", clip, f"{clip}_rtmp_patent_release_sq.png")
    cv2.imwrite(out, sq)
    print(f"  release(no knee-velo) knee={cand['lead_knee_at_release']:.0f} "
          f"tilt={cand['trunk_anterior_tilt']:.0f} wrist={cand['wrist_speed']:.1f} "
          f"relh={cand['release_height']:.2f} @f{f} -> {out}")
    return out


def panel_front_stride_angle(clip, rel_gt):
    """Stride ANGLE at the FRONTAL foot plant (trail->lead ankle line vs horizontal)."""
    df = load_coords(clip); arm = detect_arm(df)
    lead = "left" if arm == "right" else "right"
    trail = "right" if lead == "left" else "left"
    video = os.path.join(config.ROOT, "data", "videos", f"{clip}.mov")
    fpf = M.foot_plant_frame(df, lead, FPS, J, rel_gt, view="front")
    val = float(M.stride_angle_2d(df, arm, fpf, J))
    frame = grab_seq(video, fpf)
    draw_skeleton(frame, df, fpf)
    la, ta = pt(df, f"{lead}_ankle", fpf), pt(df, f"{trail}_ankle", fpf)
    L = int(max(abs(la[0] - ta[0]), 70) * 1.4)
    href = (ta[0] + (L if la[0] >= ta[0] else -L), ta[1])
    cv2.line(frame, ta, href, WHITE, 2)
    cv2.line(frame, ta, la, STRA, 3); cv2.circle(frame, ta, 7, RED, -1)
    arc_minor(frame, ta, href, la, 42, STRA)
    cv2.putText(frame, f"stride angle {val:.0f}", (ta[0] + 14, ta[1] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, STRA, 2)
    sq = square_crop(frame, df, fpf, pad=0.22)
    banner(sq, fpf, "FOOT PLANT (front)", f"stride angle {val:.0f} deg", STRA)
    out = os.path.join(config.ROOT, "data", "outputs", clip, f"{clip}_rtmp_strideangle_sq.png")
    cv2.imwrite(out, sq); print(f"  stride angle {val:.0f} @f{fpf} -> {out}")
    return out


def panel_side_cog(clip, rel_gt):
    """Whole-body COM: trajectory over [0,release] + forward-velocity arrows at
    peak knee height (COG Velo @PKH) and at the peak (COG Fwd Velo). No subject
    height on real video -> values are statures/s (OBP fig uses m/s)."""
    df = load_coords(clip); arm = detect_arm(df); lead = "left" if arm == "right" else "right"
    video = os.path.join(config.ROOT, "data", "videos", f"{clip}.mov")
    comx, comy = M.body_com(df, J); stat = M.pixel_stature(df, J)
    v = np.abs(np.gradient(comx)) * FPS / stat
    pk = int(np.nanargmax(v[:int(rel_gt) + 1]))
    fp = M.foot_plant_frame(df, lead, FPS, J, rel_gt); pkh = M.peak_knee_height_frame(df, lead, fp, J)
    cog_fwd = float(M.cog_fwd_velo(df, FPS, rel_gt, J))
    cog_pkh = float(M.cog_velo_at_pkh(df, FPS, pkh, J))
    frame = grab_seq(video, pkh)
    draw_skeleton(frame, df, pkh)
    traj = [(int(comx[i]), int(comy[i])) for i in range(0, int(rel_gt) + 1)
            if np.isfinite(comx[i]) and np.isfinite(comy[i])]
    for i in range(1, len(traj)):
        cv2.line(frame, traj[i - 1], traj[i], GRAY, 2)

    def com_arrow(fr, color, label, dy):
        c = (int(comx[fr]), int(comy[fr]))
        cv2.circle(frame, c, 8, color, -1)
        cv2.arrowedLine(frame, c, (c[0] + 74, c[1]), color, 3, tipLength=0.3)
        cv2.putText(frame, label, (c[0] + 18, c[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    com_arrow(pkh, ORNG, f"COG@PKH {cog_pkh:.2f}", -16)
    com_arrow(pk, VIO, f"COG fwd {cog_fwd:.2f} (max)", 34)
    sq = square_crop(frame, df, pkh, pad=0.26)
    cv2.putText(sq, f"frame {pkh}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    cv2.putText(sq, "PEAK KNEE HEIGHT", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, ORNG, 3)
    cv2.putText(sq, f"COG fwd {cog_fwd:.2f} / @PKH {cog_pkh:.2f} stat/s", (20, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, ORNG, 2)
    out = os.path.join(config.ROOT, "data", "outputs", clip, f"{clip}_rtmp_cog_sq.png")
    cv2.imwrite(out, sq); print(f"  COG fwd {cog_fwd:.2f} @f{pk}, @PKH {cog_pkh:.2f} @f{pkh} -> {out}")
    return out


def panel_overhead_pelvis(clip):
    """Pelvis rotational velocity from the MANUAL hip GT time series
    (hip_gt_metrics.csv rot_velo_degps); the pose pelvis is self-occluded from
    overhead, so this uses the same manual hip line as the HSS panel. Absolute
    scale is uncalibrated (the printed value is a raw, relative reading)."""
    df = load_coords(clip)
    video = os.path.join(config.ROOT, "data", "videos", f"{clip}.mov")
    d = os.path.join(config.ROOT, "data", "outputs", clip)
    met = pd.read_csv(os.path.join(d, f"{clip}_hip_gt_metrics.csv"))
    fin = pd.read_csv(os.path.join(d, f"{clip}_hip_gt_final.csv"))
    i = met.rot_velo_degps.abs().idxmax()
    rf = int(met.frame.iloc[i]); rv = float(met.rot_velo_degps.iloc[i])
    frame = grab_seq(video, rf)
    for j1, j2 in CONNECT:                  # faint skeleton (pose is jumbled mid-rotation)
        try:
            cv2.line(frame, pt(df, j1, rf), pt(df, j2, rf), (185, 185, 185), 1)
        except Exception:
            pass
    row = fin[fin.frame == rf]
    if len(row):
        lh = (int(row.lhip_x.iloc[0]), int(row.lhip_y.iloc[0]))
        rh = (int(row.rhip_x.iloc[0]), int(row.rhip_y.iloc[0]))
        cv2.line(frame, lh, rh, VIO, 6)
        cv2.circle(frame, lh, 8, VIO, -1); cv2.circle(frame, rh, 8, VIO, -1)
        pc = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
        cv2.putText(frame, "pelvis line", (pc[0] + 26, pc[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, VIO, 2)
        # rotation-velocity curved arrow, offset into the empty field so it never
        # covers the pelvis line or the tangled pose
        s = 1.0 if rv >= 0 else -1.0
        cc = (pc[0] - 175, pc[1]); ar = 74
        angs = np.radians(np.linspace(-115, 115, 30) * s)
        arc = [(int(cc[0] + ar * np.cos(a)), int(cc[1] + ar * np.sin(a))) for a in angs]
        for i in range(1, len(arc)):
            cv2.line(frame, arc[i - 1], arc[i], VIO, 4)
        cv2.arrowedLine(frame, arc[-2], arc[-1], VIO, 4, tipLength=1.6)
        cv2.putText(frame, f"pelvis rot {abs(rv):.0f}/s", (cc[0] - 78, cc[1] + ar + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, VIO, 2)
    sq = square_crop(frame, df, rf, pad=0.32)
    banner(sq, rf, "PEAK ROT VELO", f"pelvis rot {abs(rv):.0f} deg/s", VIO)
    out = os.path.join(config.ROOT, "data", "outputs", clip, f"{clip}_pelvisrot_sq.png")
    cv2.imwrite(out, sq); print(f"  pelvis rot velo {abs(rv):.0f}/s @f{rf} (manual GT) -> {out}")
    return out


def overhead_release_frame():
    """Human-labeled ball-release frame from label_overhead_release.py."""
    p = os.path.join(config.ROOT, "data", "outputs", "overhead_release_gt.csv")
    row = pd.read_csv(p).iloc[0]
    return int(float(row["true_release"]))


def main():
    orel = overhead_release_frame()
    print(f"overhead release GT = f{orel}")
    print("(b) side release + release ext [angle00_00]")
    panel_side_release("angle00_00", 493)
    print("(c-cog) side whole-body COM [angle00_00]")
    panel_side_cog("angle00_00", 493)
    print("(stride angle) front foot plant [angle03_04]")
    panel_front_stride_angle("angle03_04", 511)
    print("(f) front MER: torso lateral tilt + glove-arm abduction [angle03_04]")
    panel_mer_combined("angle03_04", 511)
    print("(pelvis) overhead pelvis rot velo  [pitching_overhead_02, manual GT]")
    panel_overhead_pelvis("pitching_overhead_02")
    print("(g) overhead torso rotation @BR    [pitching_overhead_02, GT release]")
    panel_torso_rot("pitching_overhead_02", orel)


if __name__ == "__main__":
    main()
