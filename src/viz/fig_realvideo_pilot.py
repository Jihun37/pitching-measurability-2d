"""Act 3 §6.6 — the real-video pilot, as FEASIBILITY and FAILURE EXAMPLES.

⚠ FRAMING. Real phone clips have NO simultaneous 3D ground truth. This figure must
never be described as an accuracy validation. What it shows is (a) that the closed
loop runs end to end on real video and places its four events on plausible frames,
and (b) the failure modes we actually observed. Metric values are printed on the
normal case as pipeline OUTPUT, not as verified measurements.

Four cases, one per row, four events per row (PKH / FP / MER proxy / Release):
  1 normal run              real_video_test_14   fp DETECTED
  2 side FP fallback        real_video_test_04   side detector found no candidate
  3 stride angle wrap       real_video_test_11   raw atan2 wraps at a rear azimuth
  4 overhead occlusion      pitching_overhead_01 lowest pose confidence in the set

Every panel is a REAL frame grabbed from the source clip. Frame reads go through the
sequential grab helper: cv2 random seek is keyframe-snapped on iPhone HEVC and lands
frames off (CLAUDE.md). Nothing is synthesised and no threshold is changed here.

Reads the ELIGIBLE pilot tables `pilot_{clips,metrics}_eligible.csv` (n=86).
The paper population excludes non-pitch diagnostic clips (set_01/02) and externally
sourced footage without confirmed reuse permission (video_test_*), per
`analysis/pilot_eligibility.py`. Only self-filmed / rights-confirmed clips appear here.

Run:  conda activate diamond
      cd src\\viz
      python fig_realvideo_pilot.py
"""
import os, sys, textwrap
import numpy as np, pandas as pd, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../stage2", "../analysis"):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
import metrics as M
from hss_overhead_real import grab_frames_exact

J = M.JOINTS
FONT = cv2.FONT_HERSHEY_SIMPLEX
PW, PH = 430, 560
PILOT = os.path.join(config.ROOT, "data", "outputs", "realvideo_pilot")
OUT = os.path.join(config.ROOT, "data", "outputs", "viz",
                   "fig_realvideo_pilot.png")

COL = {"PKH": (240, 190, 60), "FP": (60, 190, 250),
       "MER": (220, 110, 230), "RELEASE": (70, 70, 240)}
OK, BAD = (90, 210, 90), (60, 60, 235)
SKEL = [("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle")]
DOTS = [a for a, _ in SKEL] + ["left_ankle", "right_ankle", "left_wrist",
                               "right_wrist"]

# (clip, kind, case title, status line, cause / note lines)
CASES = [
    ("real_video_test_14", "normal", "1  NORMAL RUN",
     "fp detector side -> DETECTED",
     ["all four events land on plausible frames",
      "joint lines drawn are the ones each estimator reads"]),
    ("real_video_test_04", "fail", "2  SIDE FOOT-PLANT FALLBACK",
     "fp detector side -> FALLBACK",
     ["no frame satisfied fwd>0.70*max AND y_norm>0.97 AND vy<0.15*peak",
      "returned frame is release - 0.13 s, a constant, not a measurement",
      "fp-dependent outputs are therefore NOT counted as measured"]),
    ("real_video_test_11", "fail", "3  STRIDE ANGLE RAW WRAP (rear arc)",
     "fp DETECTED, but the output is not reportable",
     ["stride angle is a raw atan2 of the ankle line at foot plant",
      "at rear azimuths the trail ankle crosses to the other side, so the",
      "value wraps to -179.3 deg; it is a CALIBRATE metric printed",
      "uncalibrated, so this is NOT counted as a measurement"]),
    ("pitching_overhead_01", "fail", "4  OVERHEAD POSE / OCCLUSION",
     "fp detector side -> FALLBACK",
     ["lowest keypoint confidence in the set, 36 % of frames occluded",
      "automatic pose does not work overhead -> the hand-labelled pelvis GT",
      "is the source for overhead results, not this track"]),
]
EVENTS = ["PKH", "FP", "MER", "RELEASE"]


def P(df, name, f):
    try:
        x, y = df[f"{name}_x"].iloc[int(f)], df[f"{name}_y"].iloc[int(f)]
        return (int(x), int(y)) if np.isfinite([x, y]).all() else None
    except Exception:
        return None


def blend(img, fn, alpha=0.55):
    ov = img.copy(); fn(ov); cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def skeleton(img, df, f):
    blend(img, lambda o: [cv2.line(o, P(df, a, f), P(df, b, f), (95, 195, 95), 2,
                                   cv2.LINE_AA)
                          for a, b in SKEL if P(df, a, f) and P(df, b, f)])
    for j in set(DOTS):
        p = P(df, j, f)
        if p:
            cv2.circle(img, p, 4, (45, 215, 250), -1, cv2.LINE_AA)


def hi(img, p1, p2, col, w=4):
    if p1 and p2:
        blend(img, lambda o: cv2.line(o, p1, p2, col, w, cv2.LINE_AA))


def mid(df, a, b, f):
    pa, pb = P(df, a, f), P(df, b, f)
    return ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2) if pa and pb else None


def box(df, frames, shape, pad=0.42):
    Hh, Ww = shape[:2]
    xs, ys = [], []
    for f in frames:
        for j in set(DOTS) | {"nose"}:
            p = P(df, j, f)
            if p:
                xs.append(p[0]); ys.append(p[1])
    if not xs:
        return 0, 0, Ww, Hh
    asp = PW / PH
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    bw = (max(xs) - min(xs)) * (1 + 2 * pad)
    bh = max((max(ys) - min(ys)) * (1 + 2 * pad), 140.0)
    bw = max(bw, bh * asp); bh = max(bh, bw / asp)
    if bw > Ww: bw, bh = Ww, Ww / asp
    if bh > Hh: bh, bw = Hh, Hh * asp
    return (int(np.clip(cx - bw / 2, 0, Ww - bw)),
            int(np.clip(cy - bh / 2, 0, Hh - bh)), int(bw), int(bh))


def panel(img, bx, tag, f, lines, col):
    x, y, w, h = bx
    p = img[y:y + h, x:x + w]
    p = cv2.resize(p if p.size else img, (PW, PH), interpolation=cv2.INTER_AREA)
    bh = 24 + 17 * len(lines) + 8
    ov = p.copy(); cv2.rectangle(ov, (0, 0), (PW, bh), (26, 26, 26), -1)
    cv2.rectangle(ov, (0, 0), (5, bh), col, -1)
    cv2.addWeighted(ov, 0.80, p, 0.20, 0, p)
    cv2.putText(p, f"{tag}   f{int(f)}", (11, 19), FONT, 0.58, col, 2, cv2.LINE_AA)
    yy = 19
    for s in lines:
        yy += 17
        cv2.putText(p, s[:60], (11, yy), FONT, 0.40, (232, 232, 232), 1, cv2.LINE_AA)
    cv2.rectangle(p, (0, 0), (PW - 1, PH - 1), col, 2)
    return p


def val(mt, clip, metric):
    r = mt[(mt["clip"] == clip) & (mt.metric == metric)]
    if r.empty:
        return None, "absent"
    v = float(r.iloc[0].value)
    return (v if np.isfinite(v) else None), str(r.iloc[0].status)


def main():
    cl = pd.read_csv(os.path.join(PILOT, "pilot_clips_eligible.csv"))
    mt = pd.read_csv(os.path.join(PILOT, "pilot_metrics_eligible.csv"))
    rows_img = []

    for clip, kind, title, status, notes in CASES:
        r = cl[cl["clip"] == clip].iloc[0]
        df = pd.read_csv(config.refined_or(os.path.join(
            config.OUTPUT_DIR.replace(config.VIDEO_NAME, clip),
            f"{clip}_smoothed_rtmp.csv")))
        if "nose_x" in df.columns and "head_x" not in df.columns:
            df = df.rename(columns={"nose_x": "head_x", "nose_y": "head_y"})
        want = {"PKH": r.pkh_f, "FP": r.fp_f, "MER": r.mer_f,
                "RELEASE": r.release_f}
        want = {k: int(v) for k, v in want.items() if np.isfinite(v)}
        g = grab_frames_exact(r.video_path, list(want.values()))
        if not g:
            print("SKIP", clip); continue
        bx = box(df, list(g.keys()), next(iter(g.values())).shape)
        lead = "left" if r.true_arm == "right" else "right"
        thr = "right" if r.true_arm == "right" else "left"

        panels = []
        for tag in EVENTS:
            f = want.get(tag)
            if f is None or f not in g:
                continue
            im = g[f].copy(); skeleton(im, df, f)
            lines = []
            if tag == "PKH":
                p = P(df, f"{lead}_knee", f)
                if p: hi(im, (0, p[1]), (im.shape[1], p[1]), COL[tag], 2)
                lines = ["lead knee at max height, argmin image-y over [0, fp]"]
            elif tag == "FP":
                hi(im, P(df, f"{lead}_hip", f), P(df, f"{lead}_knee", f), COL[tag])
                hi(im, P(df, f"{lead}_knee", f), P(df, f"{lead}_ankle", f), COL[tag])
                hi(im, P(df, f"{lead}_ankle", f), P(df, f"{thr}_ankle", f),
                   (250, 220, 90), 3)
                lines = [f"detector {r.fp_detector}  ->  "
                         f"{'DETECTED' if r.fp_detected else 'FALLBACK'}"]
                if not r.fp_detected:
                    lines.append(f"blind release-0.13s frame ({r.fp_status})")
            elif tag == "MER":
                hi(im, P(df, "left_shoulder", f), P(df, "right_shoulder", f),
                   COL[tag], 4)
                hi(im, mid(df, "left_hip", "right_hip", f),
                   mid(df, "left_shoulder", "right_shoulder", f), COL[tag], 3)
                lines = [f"proxy = release - {int(r.release_f - r.mer_f)} f "
                         f"(11 f at 360 Hz)"]
            else:
                hi(im, P(df, f"{thr}_shoulder", f), P(df, f"{thr}_wrist", f),
                   COL[tag])
                hi(im, mid(df, "left_hip", "right_hip", f),
                   mid(df, "left_shoulder", "right_shoulder", f), (250, 160, 70), 3)
                lines = [f"strategy: {r.release_view}"]

            if kind == "normal":
                for lbl, key in (("Lead Knee", "Lead Knee Angle [O]"),
                                 ("Arm Slot", "Arm Slot [O]"),
                                 ("Rel Height", "Release Height [O]"),
                                 ("Stride", "Stride (anchor) [O]")):
                    v, st = val(mt, clip, key)
                    if tag == "RELEASE" and v is not None and st == "measured":
                        lines.append(f"{lbl} {v:.1f}")
            panels.append(panel(im, bx, tag, f, lines,
                                COL[tag] if kind == "normal" or tag == "PKH"
                                else COL[tag]))
        if panels:
            rows_img.append((title, status, notes, r, panels))

    # ---- compose -------------------------------------------------------
    LW = 362
    W = LW + len(EVENTS) * (PW + 8) + 8
    rowh = PH + 10
    Hh = 96 + len(rows_img) * rowh
    out = np.full((Hh, W, 3), 22, np.uint8)
    cv2.putText(out, "Real-video pilot - feasibility and failure examples",
                (16, 34), FONT, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(out, "80 eligible pitching clips (self-filmed / rights-confirmed only), closed loop end to end. NO "
                "simultaneous 3D ground truth exists for these clips, so this is "
                "NOT an accuracy validation:", (16, 60), FONT, 0.50,
                (185, 185, 185), 1, cv2.LINE_AA)
    cv2.putText(out, "printed values are pipeline OUTPUT. Every panel is a real "
                "frame from the source video.", (16, 80), FONT, 0.50,
                (185, 185, 185), 1, cv2.LINE_AA)

    for i, (title, status, notes, r, panels) in enumerate(rows_img):
        y0 = 96 + i * rowh
        col = OK if r.fp_detected else BAD
        cv2.rectangle(out, (0, y0), (5, y0 + PH), col, -1)
        for k, tl in enumerate(textwrap.wrap(title, 30)):
            cv2.putText(out, tl, (16, y0 + 26 + 22 * k), FONT, 0.58,
                        (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(out, r["clip"], (16, y0 + 72), FONT, 0.46, (170, 170, 170),
                    1, cv2.LINE_AA)
        cv2.putText(out, f"az={r.az} el={r.el}  {r.fps:.0f} fps", (16, y0 + 92),
                    FONT, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        yy = y0 + 118
        for sl in textwrap.wrap(status, 44):
            cv2.putText(out, sl, (16, yy), FONT, 0.44, col, 1, cv2.LINE_AA)
            yy += 17
        yy += 8
        for s in notes:
            for chunk in textwrap.wrap(s, 47):
                cv2.putText(out, chunk, (16, yy), FONT, 0.395,
                            (185, 185, 185), 1, cv2.LINE_AA)
                yy += 15
            yy += 4
        cv2.putText(out, f"valid metrics {int(r.n_metrics_valid)}", (16, yy + 8),
                    FONT, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        x = LW
        for p in panels:
            out[y0:y0 + PH, x:x + PW] = p
            x += PW + 8

    cv2.imwrite(OUT, out)
    print(f"cases {len(rows_img)}  ->  {OUT}")


if __name__ == "__main__":
    main()
