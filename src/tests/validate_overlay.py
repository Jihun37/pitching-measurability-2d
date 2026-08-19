"""
Visual check on the measurement.

Draws the skeleton onto the original clip from an already extracted coordinate
CSV, marks the foot plant and release frames metrics.py actually found, and
draws what is measured at those instants: the knee angle, the trunk tilt and the
stride. It answers whether the measurement landed where it should, by eye.

The pose is not re-run; the coordinate CSV is read.

Usage:
    python validate_overlay.py --coords ../../data/outputs/pitching_test/pitching_test_smoothed.csv \
                               --video  ../../data/videos/pitching_test.MOV --fps 120
"""
import os, sys, argparse, re, json
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "viz"))
import metrics as M
from skeleton import square_crop

# Skeleton segments, in our joint names
CONNECT = [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
]


def pt(df, name, f):
    return (int(df[f"{name}_x"].iloc[f]), int(df[f"{name}_y"].iloc[f]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--fps", type=float, default=config.FPS_DEFAULT)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--view", default="side", choices=["side", "frontal"],
                    help="side = the three side overlays, frontal = the arm slot overlay")
    ap.add_argument("--rel", type=int, default=None,
                    help="set the release frame by hand, for a dry-form clip where detection has no ball to find")
    ap.add_argument("--metrics", default=None,
                    help="viewpoint-aware subset mode (deployment overlay): "
                         "comma list from knee,trunk,stride,wrist,relheight,"
                         "armslot,kneevelo — draws ONLY those geometries and "
                         "holds only at the events they need (stride -> foot "
                         "plant, rest -> release). 'none' = event hold only. "
                         "Omit for the legacy full side/frontal overlays.")
    ap.add_argument("--raw", action="store_true",
                    help="use the given RAW coords; default swaps to the "
                         "recovery-refined sibling CSV when it exists")
    ap.add_argument("--stills-only", action="store_true",
                    help="event PNG only, skip mp4 encoding")
    ap.add_argument("--square", action="store_true",
                    help="save event stills as person-centered 1:1 crops "
                         "(paper figures); video stays full-frame")
    ap.add_argument("--pad", type=float, default=0.32,
                    help="square-crop padding around the person bbox "
                         "(fraction per side; larger = more breathing room)")
    ap.add_argument("--info", default=None,
                    help="JSON file with a deployment report panel drawn on "
                         "every VIDEO frame (viewpoint estimate + zone-map "
                         "r2 per metric). Format: {'rows': [{'cols': "
                         "[str, ...], 'color': name}, ...]}; colors: white/"
                         "green/yellow/red/gray. Stills are unaffected.")
    ap.add_argument("--out-width", type=int, default=None,
                    help="downscale the OUTPUT VIDEO to this width (4K "
                         "sources -> manageable files); stills stay full-res")
    ap.add_argument("--out-tag", default=None,
                    help="suffix inserted into every output name (mp4 and "
                         "stills), e.g. 'kin' / 'ballsep' when rendering "
                         "the same clip with two release detectors")
    a = ap.parse_args()
    want = None
    if a.metrics is not None:
        want = set() if a.metrics.strip().lower() in ("", "none") else \
               {m.strip().lower() for m in a.metrics.split(",") if m.strip()}

    a.coords = config.refined_or(a.coords, a.raw)   # prefer the best pose
    df = pd.read_csv(a.coords)
    if "nose_x" in df.columns and "head_x" not in df.columns:
        df = df.rename(columns={"nose_x": "head_x", "nose_y": "head_y"})
    J = M.JOINTS

    def detect_arm():
        def pk(j):
            x = df[f"{j}_x"].to_numpy(float); y = df[f"{j}_y"].to_numpy(float)
            return np.nanmax(np.hypot(np.diff(x), np.diff(y)))
        return "right" if pk("right_wrist") >= pk("left_wrist") else "left"

    arm = a.arm or detect_arm()
    lead = "left" if arm == "right" else "right"
    # raw coords (if cached next to the smoothed CSV) refine the release
    # frame: smoothing can shift an asymmetric speed peak by one frame
    raw_path = a.coords.replace("_smoothed", "_coords")
    raw_df = None
    if raw_path != a.coords and os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        if "nose_x" in raw_df.columns and "head_x" not in raw_df.columns:
            raw_df = raw_df.rename(columns={"nose_x": "head_x", "nose_y": "head_y"})
    rel = M.release_frame(df, arm, a.fps, J, view=a.view, raw_df=raw_df)
    # Where the release frame is supplied from outside, for instance the
    # corrected release from the deployment path, it replaces the detected one
    # and every release-anchored quantity is computed at that frame
    # (compute_candidates passes rel straight through).
    cand = {k: v for k, (v, _) in M.compute_candidates(
        df, fps=a.fps, arm=arm, view=a.view, raw_df=raw_df, rel=rel).items()}
    print(f"arm={arm}  foot_plant=frame{fp}  release=frame{rel}")
    print(f"  knee@fp={cand['lead_knee_at_fp']:.0f}  knee@br={cand['lead_knee_at_release']:.0f}"
          f"  knee_ext={cand['lead_knee_extension']:+.0f}")
    print(f"  trunk_tilt={cand['lateral_trunk_tilt']:.0f}  stride%h={cand['stride_pct_height']:.2f}")

    cap = cv2.VideoCapture(a.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ts = max(1.0, W / 1920.0)              # banner text scale for >1080p sources
    vfps = cap.get(cv2.CAP_PROP_FPS) or a.fps
    vfps = round(vfps)                       # mpeg4 refuses a fractional rate such as 119.939
    if vfps <= 0 or vfps > 1000: vfps = int(round(a.fps))
    # output base: works for both mediapipe (_smoothed.csv) and rtmp
    # (_smoothed_rtmp.csv) coords; rtmp outputs keep the _rtmp marker
    base = re.sub(r"_(smoothed|coords)(_rtmp)?\.csv$", "", a.coords)
    osfx = "_rtmp" if a.coords.endswith("_rtmp.csv") else ""
    if a.out_tag:
        osfx += "_" + a.out_tag
    OW, OH = W, H                          # video-writer size (stills stay W x H)
    if a.out_width and a.out_width < W:
        OW = a.out_width
        OH = int(round(H * OW / W / 2)) * 2
    out_path = base + osfx + "_metrics.mp4"
    vw = None
    if not a.stills_only:
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), vfps, (OW, OH))
        if not vw.isOpened():                 # codec fallback
            for fourcc in ("avc1", "MJPG", "XVID"):
                vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), vfps, (OW, OH))
                if vw.isOpened(): break
    hold_frames = int(round(vfps * 1.5))   # hold 1.5 s on each event

    # deployment report panel (--info): drawn on every OUTPUT frame at the
    # output scale (post-resize) so the text size is viewing-resolution true
    panel = None
    if a.info:
        with open(a.info, encoding="utf-8") as fh:
            panel = json.load(fh)["rows"]
    PANEL_COL = {"white": (235, 235, 235), "green": (80, 220, 80),
                 "yellow": (0, 200, 255), "red": (70, 70, 235),
                 "gray": (160, 160, 160)}

    def draw_panel(img):
        s = img.shape[1] / 1920.0             # panel geometry scales with width
        fs, lh, pad = 0.52 * s, int(26 * s), int(12 * s)
        colx = [int(c * s) for c in (0, 330, 520)]
        boxw = int(800 * s)
        x0 = img.shape[1] - boxw - int(14 * s)
        y0 = int(14 * s)
        ov = img.copy()
        cv2.rectangle(ov, (x0, y0), (x0 + boxw, y0 + len(panel) * lh + 2 * pad),
                      (25, 25, 25), -1)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
        for i, row in enumerate(panel):
            col = PANEL_COL.get(row.get("color", "white"), PANEL_COL["white"])
            y = y0 + pad + int(18 * s) + i * lh
            for cx, txt in zip(colx, row["cols"]):
                cv2.putText(img, txt, (x0 + pad + cx, y),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)

    def vwrite(fr):
        if vw is None:
            return
        out = cv2.resize(fr, (OW, OH)) if (OW, OH) != (W, H) else fr.copy()
        if panel:
            draw_panel(out)
        vw.write(out)

    GREEN, RED, YEL, CYAN, WHITE = (0,255,0),(0,0,255),(0,255,255),(255,255,0),(255,255,255)
    BLU = (255, 80, 0)                    # release height dimension line
    KNEE = (139, 139, 0)                  # dark teal   - lead knee angle
    KVEL = (0, 88, 204)                   # dark orange - knee ext velocity
    WRST = (147, 20, 255)                 # deep pink - wrist speed (pops on dark trees)
    lk = ("left" if lead == "left" else "right")

    def _wrist_speed_vec(f):
        """Instantaneous wrist velocity (dx, dy) per second at frame f, from a
        one-frame difference."""
        j = f"{arm}_wrist"
        f0 = max(0, f-1)
        dx = (df[f"{j}_x"].iloc[f] - df[f"{j}_x"].iloc[f0]) * a.fps
        dy = (df[f"{j}_y"].iloc[f] - df[f"{j}_y"].iloc[f0]) * a.fps
        return dx, dy

    def draw_event(frame, f, is_fp, is_rel, want=None):
        # Metric-to-event split (matches visualize_3d_2d draw_side_footplant/release):
        #   foot plant -> stride only
        #   release    -> knee angle, trunk tilt, knee ext velo, wrist speed, release height
        # want: viewpoint-aware subset (--metrics); None = legacy full set.
        def on(key):
            return want is None or key in want
        # (1) release only: lead knee angle + arc + knee extension velocity arrow
        if is_rel and on("knee"):
            try:
                hp, kp, ap_ = pt(df, f"{lk}_hip", f), pt(df, f"{lk}_knee", f), pt(df, f"{lk}_ankle", f)
                cv2.line(frame, hp, kp, KNEE, 3); cv2.line(frame, kp, ap_, KNEE, 3)
                cv2.circle(frame, kp, 8, KNEE, -1)
                a1 = np.degrees(np.arctan2(hp[1]-kp[1], hp[0]-kp[0]))
                a2 = np.degrees(np.arctan2(ap_[1]-kp[1], ap_[0]-kp[0]))
                cv2.ellipse(frame, kp, (45,45), 0, a1, a2, KNEE, 3)
                cv2.putText(frame, f"knee {cand['lead_knee_at_release']:.0f}", (kp[0]+12, kp[1]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, KNEE, 2)
                if on("kneevelo"):
                    kvel = cand.get('knee_ext_velo_br', 0.0)
                    # arrow along the rotation tangent (perpendicular to knee->ankle)
                    vx, vy = ap_[0]-kp[0], ap_[1]-kp[1]
                    n = (vx*vx+vy*vy) ** 0.5 + 1e-6
                    s = 1.0 if kvel >= 0 else -1.0
                    px, py = -vy/n, vx/n
                    tip = (int(kp[0]+s*px*55), int(kp[1]+s*py*55))
                    cv2.arrowedLine(frame, kp, tip, KVEL, 3, tipLength=0.35)
                    cv2.putText(frame, f"knee_velo {kvel:.0f}/s", (kp[0]+12, kp[1]+22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, KVEL, 2)
            except Exception: pass
        # (2) release only: trunk anterior tilt = trunk line vs vertical + arc
        if is_rel and on("trunk"):
            try:
                mh = (int((df['left_hip_x'].iloc[f]+df['right_hip_x'].iloc[f])/2),
                      int((df['left_hip_y'].iloc[f]+df['right_hip_y'].iloc[f])/2))
                ms = (int((df['left_shoulder_x'].iloc[f]+df['right_shoulder_x'].iloc[f])/2),
                      int((df['left_shoulder_y'].iloc[f]+df['right_shoulder_y'].iloc[f])/2))
                seg = int(np.hypot(ms[0]-mh[0], ms[1]-mh[1])) or 80
                vtop = (mh[0], mh[1]-seg)                     # vertical reference (up)
                cv2.line(frame, mh, vtop, WHITE, 2)
                cv2.line(frame, mh, ms, YEL, 3)               # actual trunk line
                av = np.degrees(np.arctan2(vtop[1]-mh[1], vtop[0]-mh[0]))
                asg = np.degrees(np.arctan2(ms[1]-mh[1], ms[0]-mh[0]))
                cv2.ellipse(frame, mh, (40,40), 0, av, asg, YEL, 3)
                # subset mode measures the zone metric (anterior tilt);
                # legacy full overlay keeps the original lateral label
                tval = cand['trunk_anterior_tilt'] if want is not None \
                       else cand['lateral_trunk_tilt']
                cv2.putText(frame, f"tilt {tval:.0f}", (mh[0]+12, mh[1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, YEL, 2)
            except Exception: pass
        # (3) foot plant only: stride line (front ankle @fp <-> trail-foot anchor)
        if is_fp and on("stride"):
            try:
                lead_j = f"{lead}_ankle"
                trail_j = f"{'right' if lead=='left' else 'left'}_ankle"
                lp = pt(df, lead_j, f)
                tp = pt(df, trail_j, f)
                # trail-foot anchor x (rubber) - same function as metrics
                trail_x_arr = df[f"{trail_j}_x"].to_numpy(float)
                anchor_x = int(M.trail_anchor_x(trail_x_arr, fp, a.fps))
                ap2 = (anchor_x, tp[1])                      # anchor point (y = trail ankle)
                # vertical dashed line at anchor (rubber mark)
                for yy in range(max(0, tp[1]-60), tp[1]+30, 12):
                    cv2.line(frame, (anchor_x, yy), (anchor_x, yy+6), WHITE, 2)
                cv2.line(frame, lp, ap2, RED, 3)          # measured span: front ankle <-> anchor
                cv2.line(frame, tp, ap2, RED, 1)          # trail ankle -> anchor (drag amount)
                cv2.circle(frame, ap2, 6, WHITE, -1)
                mid = ((lp[0]+ap2[0])//2, (lp[1]+ap2[1])//2 + 28)
                cv2.putText(frame, f"stride {cand['stride_pct_height']:.2f}xH", mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)
            except Exception: pass
        # (4) release only: wrist velocity vector arrow (arm speed)
        if is_rel and on("wrist"):
            try:
                wp = pt(df, f"{arm}_wrist", f)
                dx, dy = _wrist_speed_vec(f)
                mag = (dx*dx+dy*dy) ** 0.5 + 1e-6
                L = 90.0                                   # max arrow length (px)
                tip = (int(wp[0]+dx/mag*L), int(wp[1]+dy/mag*L))
                cv2.arrowedLine(frame, wp, tip, WRST, 3, tipLength=0.3)
                cv2.circle(frame, wp, 7, WRST, -1)
                cv2.putText(frame, f"wrist_spd {cand['wrist_speed']:.1f}",
                            (wp[0]+10, wp[1]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, WRST, 2)
            except Exception: pass
        # (5) release only: release height = wrist -> ground (lowest ankle) dimension line
        if is_rel and on("relheight"):
            try:
                wp = pt(df, f"{arm}_wrist", f)
                ank_y = np.concatenate([df["left_ankle_y"].to_numpy(float),
                                        df["right_ankle_y"].to_numpy(float)])
                gy = int(np.nanmax(ank_y))
                # semi-transparent dimension line so it doesn't fight the skeleton
                ov = frame.copy()
                cv2.line(ov, (wp[0], wp[1]), (wp[0], gy), BLU, 3)
                cv2.line(ov, (wp[0]-10, gy), (wp[0]+10, gy), BLU, 3)          # ground tick
                cv2.line(ov, (wp[0]-10, wp[1]), (wp[0]+10, wp[1]), BLU, 3)    # wrist tick
                cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
                cv2.putText(frame, f"rel_height {cand['release_height']:.2f}",
                            (wp[0]+14, (wp[1]+gy)//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLU, 2)
            except Exception: pass

    def draw_armslot_event(frame, f):
        # frontal: shoulder->hand against the vertical, with the arc and the angle
        try:
            S = pt(df, f"{arm}_shoulder", f); Wr = pt(df, f"{arm}_wrist", f)
            vtop = (S[0], S[1] - (abs(S[1]-Wr[1]) + 40))
            cv2.line(frame, S, vtop, WHITE, 2)                  # vertical reference
            cv2.line(frame, S, Wr, YEL, 3)                      # shoulder->hand
            cv2.circle(frame, S, 7, RED, -1)
            a1 = np.degrees(np.arctan2(vtop[1]-S[1], vtop[0]-S[0]))
            a2 = np.degrees(np.arctan2(Wr[1]-S[1], Wr[0]-S[0]))
            cv2.ellipse(frame, S, (46,46), 0, a1, a2, YEL, 3)
            cv2.putText(frame, f"arm slot {cand['arm_slot']:.0f}", (S[0]+12, S[1]-14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, YEL, 2)
        except Exception: pass

    event_stills = {}
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok or f >= len(df):
            break
        for j1, j2 in CONNECT:                       # the skeleton
            try: cv2.line(frame, pt(df, j1, f), pt(df, j2, f), GREEN, 2)
            except Exception: pass
        if not a.square:
            # In square mode the full-frame text is skipped: where the crop
            # includes the top left, it would print twice over the crop's own.
            cv2.putText(frame, f"frame {f}", (20, int(40*ts)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0*ts, WHITE, int(2*ts))

        if want is not None:
            # viewpoint-aware subset mode: hold only at the events the
            # in-zone metrics need (stride -> foot plant; rest -> release;
            # empty set -> release hold with the event banner only)
            is_fp = (f == fp) and "stride" in want
            is_rel = (f == rel)
            if is_fp or is_rel:
                draw_event(frame, f, is_fp, is_rel, want=want)
                if is_rel and "armslot" in want:
                    draw_armslot_event(frame, f)
                label, col = (("RELEASE", RED) if is_rel
                              else ("FOOT PLANT", GREEN))
                if is_rel:
                    txt = []
                    if "knee" in want:
                        txt.append(f"knee {cand['lead_knee_angle']:.0f} deg")
                    if "trunk" in want:
                        txt.append(f"trunk_tilt {cand['trunk_anterior_tilt']:.0f} deg")
                    if "armslot" in want:
                        txt.append(f"arm slot {cand['arm_slot']:.0f} deg")
                    if "wrist" in want:
                        txt.append(f"wrist_spd {cand['wrist_speed']:.1f}")
                    if "relheight" in want:
                        txt.append(f"rel_height {cand['release_height']:.2f}")
                else:
                    txt = [f"stride {cand['stride_pct_height']:.2f} x height"]
                cv2.putText(frame, label, (20, int(85*ts)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2*ts, col, int(3*ts))
                for i, t in enumerate(txt):
                    cv2.putText(frame, t, (20, int((130+i*36)*ts)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9*ts, col, int(2*ts))
                still = base + osfx + f"_{'release' if is_rel else 'footplant'}.png"
                cv2.imwrite(still, frame)
                event_stills[label] = still
                for _ in range(hold_frames):
                    vwrite(frame)
            vwrite(frame)
            if a.stills_only and f >= (max(fp, rel) if "stride" in want else rel):
                break
            f += 1; continue

        if a.view == "frontal":
            # frontal: arm slot at release only
            if f == rel:
                draw_armslot_event(frame, f)
                if a.square:
                    sq = square_crop(frame, df, f, pad=a.pad)
                    cv2.putText(sq, f"frame {f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
                    cv2.putText(sq, "RELEASE", (20,85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, RED, 3)
                    cv2.putText(sq, f"arm slot {cand['arm_slot']:.0f} deg", (20,130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)
                cv2.putText(frame, "RELEASE", (20,85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, RED, 3)
                cv2.putText(frame, f"arm slot {cand['arm_slot']:.0f} deg", (20,130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)
                still = base + osfx + ("_armslot_sq.png" if a.square else "_armslot.png")
                cv2.imwrite(still, sq if a.square else frame)
                event_stills["ARM SLOT"] = still
                for _ in range(hold_frames):
                    vwrite(frame)
            vwrite(frame)
            if a.stills_only and f >= rel:
                break
            f += 1; continue

        # side view, the default: all three
        is_fp, is_rel = (f == fp), (f == rel)
        if is_fp or is_rel:
            draw_event(frame, f, is_fp, is_rel)
        label, col = ("", WHITE)
        if is_fp: label, col = "FOOT PLANT", GREEN
        if is_rel: label, col = "RELEASE", RED
        if label:
            if is_fp:
                txt = [f"stride {cand['stride_pct_height']:.2f} x height"]
            else:
                txt = [f"knee {cand['lead_knee_at_release']:.0f}deg",
                       f"trunk_tilt {cand['lateral_trunk_tilt']:.0f}deg",
                       f"knee_velo {cand['knee_ext_velo_br']:.0f} deg/s",
                       f"wrist_spd {cand['wrist_speed']:.1f}",
                       f"rel_height {cand['release_height']:.2f}"]
            if a.square:
                # crop with geometry only, then banner at output scale
                sq = square_crop(frame, df, f, pad=a.pad)
                cv2.putText(sq, f"frame {f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
                cv2.putText(sq, label, (20,85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
                for i, t in enumerate(txt):
                    cv2.putText(sq, t, (20,130+i*36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            cv2.putText(frame, label, (20,85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
            for i, t in enumerate(txt):
                cv2.putText(frame, t, (20,130+i*36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            still = base + osfx + \
                    f"_{'footplant' if is_fp else 'release'}{'_sq' if a.square else ''}.png"
            cv2.imwrite(still, sq if a.square else frame)
            event_stills[label] = still
            for _ in range(hold_frames):
                vwrite(frame)
        vwrite(frame)
        if a.stills_only and f >= max(fp, rel):
            break                                # both stills saved, nothing left to do
        f += 1
    cap.release()
    if vw is not None:
        vw.release()
        print(f"saved video -> {out_path}")
    for k, v in event_stills.items():
        print(f"saved still -> {v}  ({k})")


if __name__ == "__main__":
    main()