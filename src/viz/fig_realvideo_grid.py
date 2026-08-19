"""Real-video execution grid, drawn in the patent overlay style.

WHAT IS DRAWN WHERE IS DECIDED BY THE MAP, NOT BY THE FIGURE. Each row of the grid is
one capture viewpoint, and a panel draws only quantities the graded map holds a cell for
AT THAT (az, el). Hip-shoulder separation therefore appears on the overhead row and
nowhere else, pelvic rotational velocity likewise, and the patent composite's overhead
torso-rotation panel is dropped because that row is not graded at az 0 / el 85.
`main()` asserts every drawn row against gate_map.csv before rendering.

VALUES ARE THE CANONICAL ONES. Every printed number is checked against
realvideo_feasibility_cells.csv and the script exits if one disagrees. The patent
composite's numbers do not match that table (knee 137 against 135.2 on angle00_00)
because it hardcoded release frame 493 and 120 fps; the pilot's frozen anchor is 492 at
113.22 fps, and rendering at the pilot anchor reproduces the canonical value exactly.

VIEWPOINTS ARE THE CAPTURE STATIONS. `pilot_clips_eligible.csv`'s az/el are the
viewpoint CLASSIFIER's estimate (note `vote_share` beside them) -- the five angle03_*
clips are one az-90 station but are labelled 135/75/90/60/60 there. Sec. III-C makes no
quantitative self-identification claim, so the convention angleNN -> az = 30 N at ground
level is used, and the overhead station is el 85.

THE OVERHEAD HIP LINE IS THE HAND-LABELLED GROUND TRUTH. Overhead occlusion destroys the
automatic hip keypoints, so those two panels draw the manual track and say so; the
automatic pipeline's own output at that anchor is reported in the text as flagged.

Design (colours, arc radii, banner, square crop) is taken unchanged from
tests/fig_realvideo_new_metrics_patent.py so this figure matches the overlay stills.

Run:  conda activate diamond
      cd src\\viz
      python fig_realvideo_grid.py
"""
import os, sys
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../stage2", "../tests", "../analysis"):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
import metrics as M
from validate_overlay import CONNECT, pt
from skeleton import square_crop
from fig_realvideo_new_metrics_patent import grab_seq, load_coords, detect_arm, arc_minor
from fig_graded_map import INK, MUTE, BODY_W

V = config.OBP_VALIDATION_DIR
PILOT = os.path.join(config.ROOT, "data", "outputs", "realvideo_pilot")
OUT = os.path.join(config.ROOT, "data", "outputs", "viz")
J = M.JOINTS

GREEN = (0, 255, 0); WHITE = (255, 255, 255); RED = (0, 0, 255)
KNEE = (139, 139, 0); YEL = (0, 255, 255); BLU = (255, 80, 0)
PLUM = (119, 39, 219); ARM = (0, 215, 255); STRA = (211, 85, 186)
LAT = (200, 40, 190); ABD = (255, 120, 20); ORNG = (12, 88, 234)
VIO = (237, 58, 124); HSSC = (60, 200, 255)

# (clip, viewpoint label, az, el). One station per row of the grid.
STATIONS = [("angle00_00", "side", 0, 0),
            ("angle03_04", "front", 90, 0),
            ("pitching_overhead_01", "overhead", 0, 85)]


# cv2 text is drawn on a 1200 px crop that lands at ~2.2 in on the page, so it is
# downscaled about 0.55x by the tiling. Everything drawn is scaled up by SC to
# compensate; without it the patent's 0.9 font scale is unreadable at body width.
SC = 1.9


def canon(cells, clip):
    """metric_id -> (value, status), straight from the canonical feasibility table.

    Values are NOT recomputed here. Calling metrics.compute_candidates on this path
    returns NaN for release extension and both COM velocities -- the driver reaches
    them through the pilot's explicit foot-plant and peak-knee-height frames -- and a
    figure that printed those NaNs would disagree with the table it cites."""
    d = cells[cells["clip"] == clip].set_index("metric_id")
    return {m: (float(d.loc[m, "value"]), str(d.loc[m, "status"])) for m in d.index}


def vs(vals, mid, dec=0, unit=" deg"):
    """Formatted value plus a marker when the driver flagged it."""
    v, st = vals[mid]
    s = "--" if not np.isfinite(v) else f"{v:.{dec}f}{unit}"
    return s if st == "ok" else f"{s} [{st}]"


def bgr2hex(c):
    return "#%02x%02x%02x" % (c[2], c[1], c[0])


def skel(frame, df, f, col=GREEN, lw=2):
    for a, b in CONNECT:
        try:
            cv2.line(frame, pt(df, a, f), pt(df, b, f), col, int(lw * SC))
        except Exception:
            pass


def mid(df, a, b, f):
    return (int((df[f"{a}_x"].iloc[f] + df[f"{b}_x"].iloc[f]) / 2),
            int((df[f"{a}_y"].iloc[f] + df[f"{b}_y"].iloc[f]) / 2))


def txt(frame, s, org, col, sc=0.9, lw=2, crop=None):
    """Outlined label, kept inside the crop that will actually be shown.

    `crop` is the (x, y, side) square_crop will take. Without the clamp a label
    anchored near the right of the subject is drawn past the crop edge and the tile
    shows it cut in half, which is what the first draft did to `hip-shoulder sep`."""
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, sc * SC, int(lw * SC))
    x, y = org
    if crop is not None:
        cx, cy, side = crop
        x = int(min(max(x, cx + 12), cx + side - tw - 12))
        y = int(min(max(y, cy + th + 12), cy + side - 12))
    cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc * SC, (0, 0, 0),
                int(lw * SC) + 4, cv2.LINE_AA)
    cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc * SC, col,
                int(lw * SC), cv2.LINE_AA)


def crop_rect(df, f, pad):
    """The square square_crop will take, in source pixels."""
    from skeleton import _JOINTS
    xs, ys = [], []
    for j in _JOINTS:
        try:
            a, b = df[f"{j}_x"].iloc[f], df[f"{j}_y"].iloc[f]
            if np.isfinite([a, b]).all():
                xs.append(float(a)); ys.append(float(b))
        except Exception:
            pass
    if not xs:
        return None
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    return (int((x0 + x1) / 2 - side / 2), int((y0 + y1) / 2 - side / 2), int(side))


def banner(sq, f, event, lines):
    """Top-left block. Drawn on a translucent plate so it never fights the scene."""
    n = len(lines) + 2
    ov = sq.copy()
    cv2.rectangle(ov, (0, 0), (sq.shape[1], int((44 + n * 56) * SC / 1.9)),
                  (18, 12, 6), -1)
    cv2.addWeighted(ov, 0.55, sq, 0.45, 0, sq)
    s = SC / 1.9
    cv2.putText(sq, f"frame {f}", (int(24 * s), int(52 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5 * s, WHITE, int(3 * s), cv2.LINE_AA)
    cv2.putText(sq, event, (int(24 * s), int(110 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5 * s, WHITE, int(3 * s), cv2.LINE_AA)
    for i, (t, col) in enumerate(lines):
        cv2.putText(sq, t, (int(24 * s), int((166 + i * 56) * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3 * s, col, int(3 * s), cv2.LINE_AA)


def vert_ref(frame, mh, ms, col):
    seg = int(np.hypot(ms[0] - mh[0], ms[1] - mh[1])) or 80
    top = (mh[0], mh[1] - seg)
    cv2.line(frame, mh, top, WHITE, int(2 * SC))
    cv2.line(frame, mh, ms, col, int(3 * SC))
    arc_minor(frame, mh, top, ms, int(42 * SC), col, int(3 * SC))
    return top


def horiz_ref(frame, a, b, col, r=44):
    L = int(max(abs(b[0] - a[0]), 70) * 1.35)
    href = (a[0] + (L if b[0] >= a[0] else -L), a[1])
    cv2.line(frame, a, href, WHITE, int(2 * SC))
    cv2.line(frame, a, b, col, int(3 * SC))
    cv2.circle(frame, a, int(7 * SC), RED, -1)
    arc_minor(frame, a, href, b, int(r * SC), col, int(3 * SC))
    return href


# ---------------------------------------------------------------- panels

def panel_side_release(df, arm, r, vals):
    lead = "left" if arm == "right" else "right"
    f = int(r.release_f)
    frame = grab_seq(r.video_path, f); skel(frame, df, f)
    CR = crop_rect(df, f, 0.24)
    hp, kp, ap_ = pt(df, f"{lead}_hip", f), pt(df, f"{lead}_knee", f), pt(df, f"{lead}_ankle", f)
    cv2.line(frame, hp, kp, KNEE, int(3 * SC)); cv2.line(frame, kp, ap_, KNEE, int(3 * SC))
    cv2.circle(frame, kp, int(8 * SC), KNEE, -1)
    arc_minor(frame, kp, hp, ap_, int(45 * SC), KNEE, int(3 * SC))
    txt(frame, f"knee {vals['Lead Knee Angle [O]'][0]:.0f}", (kp[0] + 12, kp[1] - 12), KNEE, crop=CR)
    mh, ms = mid(df, "left_hip", "right_hip", f), mid(df, "left_shoulder", "right_shoulder", f)
    vert_ref(frame, mh, ms, YEL)
    txt(frame, f"tilt {vals['Trunk Tilt (ant) [O]'][0]:.0f}", (mh[0] + 12, mh[1] - 20), YEL, crop=CR)
    sp, wp = pt(df, f"{arm}_shoulder", f), pt(df, f"{arm}_wrist", f)
    horiz_ref(frame, sp, wp, ARM, r=50)
    txt(frame, f"arm slot {vals['Arm Slot [O]'][0]:.0f}", (sp[0] + 14, sp[1] - 18), ARM, crop=CR)
    gy = int(np.nanmax(np.concatenate([df["left_ankle_y"].to_numpy(float),
                                       df["right_ankle_y"].to_numpy(float)])))
    ov = frame.copy()
    cv2.line(ov, (wp[0], wp[1]), (wp[0], gy), BLU, int(3 * SC))
    for yy in (gy, wp[1]):
        cv2.line(ov, (wp[0] - int(10 * SC), yy), (wp[0] + int(10 * SC), yy), BLU, int(3 * SC))
    cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
    txt(frame, f"rel_height {vals['Release Height [O]'][0]:.2f}",
        (wp[0] + 14, (wp[1] + gy) // 2), BLU, 0.8)
    trail = f"{'right' if arm == 'right' else 'left'}_ankle"
    ax_ = int(M.trail_anchor_x(df[f"{trail}_x"].to_numpy(float), int(r.fp_f), r.fps))
    y_ext = gy + int(42 * SC)
    for xx in (ax_, wp[0]):
        cv2.line(frame, (xx, y_ext - int(10 * SC)), (xx, y_ext + int(10 * SC)), PLUM, int(3 * SC))
    cv2.arrowedLine(frame, (ax_, y_ext), (wp[0], y_ext), PLUM, int(3 * SC), tipLength=0.04)
    cv2.arrowedLine(frame, (wp[0], y_ext), (ax_, y_ext), PLUM, int(3 * SC), tipLength=0.04)
    txt(frame, f"rel_ext {vals['Release Ext [O]'][0]:.2f}",
        ((ax_ + wp[0]) // 2 - int(70 * SC), y_ext + int(32 * SC)), PLUM, 0.8)
    sq = square_crop(frame, df, f, pad=0.24)
    banner(sq, f, "RELEASE", [
        ("lead knee " + vs(vals, "Lead Knee Angle [O]"), KNEE),
        ("trunk tilt " + vs(vals, "Trunk Tilt (ant) [O]"), YEL),
        ("arm slot " + vs(vals, "Arm Slot [O]"), ARM),
        ("release height " + vs(vals, "Release Height [O]", 2, " xH"), BLU),
        ("release ext " + vs(vals, "Release Ext [O]", 2, " xH"), PLUM)])
    return sq, "Side | release", ["Lead Knee Angle [O]", "Trunk Tilt (ant) [O]",
                                  "Arm Slot [O]", "Release Height [O]", "Release Ext [O]"]


def panel_side_com(df, arm, r, vals):
    comx, comy = M.body_com(df, J)
    f = int(r.pkh_f)
    frame = grab_seq(r.video_path, f); skel(frame, df, f)
    CR = crop_rect(df, f, 0.28)
    traj = [(int(comx[i]), int(comy[i])) for i in range(0, int(r.release_f) + 1)
            if np.isfinite(comx[i]) and np.isfinite(comy[i])]
    for i in range(1, len(traj)):
        cv2.line(frame, traj[i - 1], traj[i], (150, 150, 150), int(2 * SC))
    v = np.abs(np.gradient(comx)) * r.fps / M.pixel_stature(df, J)
    pk = int(np.nanargmax(v[:int(r.release_f) + 1]))
    for fr, col, lab, dy in (
            (f, ORNG, f"COG@PKH {vals['COG Velo @PKH [O]'][0]:.2f}", -16),
            (pk, VIO, f"COG fwd {vals['COG Fwd Velo [O]'][0]:.2f} (max)", 34)):
        c = (int(comx[fr]), int(comy[fr]))
        cv2.circle(frame, c, int(8 * SC), col, -1)
        cv2.arrowedLine(frame, c, (c[0] + int(74 * SC), c[1]), col, int(3 * SC), tipLength=0.3)
        txt(frame, lab, (c[0] + 18, c[1] + dy), col, 0.8, crop=CR)
    sq = square_crop(frame, df, f, pad=0.28)
    banner(sq, f, "PEAK KNEE HEIGHT", [
        ("COG velo @PKH " + vs(vals, "COG Velo @PKH [O]", 2, " H/s"), ORNG),
        ("COG fwd velo " + vs(vals, "COG Fwd Velo [O]", 2, " H/s"), VIO)])
    return sq, "Side | whole-body COM", ["COG Velo @PKH [O]", "COG Fwd Velo [O]"]


def panel_front_release(df, arm, r, vals):
    f = int(r.release_f)
    frame = grab_seq(r.video_path, f); skel(frame, df, f)
    CR = crop_rect(df, f, 0.24)
    sp, wp = pt(df, f"{arm}_shoulder", f), pt(df, f"{arm}_wrist", f)
    horiz_ref(frame, sp, wp, ARM, r=52)
    txt(frame, f"arm slot {vals['Arm Slot [O]'][0]:.0f}", (sp[0] + 14, sp[1] - 18), ARM, crop=CR)
    sq = square_crop(frame, df, f, pad=0.24)
    banner(sq, f, "RELEASE", [("arm slot " + vs(vals, "Arm Slot [O]"), ARM)])
    return sq, "Front | release", ["Arm Slot [O]"]


def panel_front_fp(df, arm, r, vals):
    lead = "left" if arm == "right" else "right"
    trail = "right" if lead == "left" else "left"
    f = int(r.fp_f)
    frame = grab_seq(r.video_path, f); skel(frame, df, f)
    CR = crop_rect(df, f, 0.24)
    la, ta = pt(df, f"{lead}_ankle", f), pt(df, f"{trail}_ankle", f)
    horiz_ref(frame, ta, la, STRA)
    txt(frame, f"stride angle {vals['Stride Angle [O]'][0]:.0f}", (ta[0] + 14, ta[1] - 16), STRA, crop=CR)
    mh, ms = mid(df, "left_hip", "right_hip", f), mid(df, "left_shoulder", "right_shoulder", f)
    vert_ref(frame, mh, ms, YEL)
    txt(frame, f"torso tilt {vals['torso_anterior_tilt_fp'][0]:.0f}",
        (mh[0] + 12, mh[1] - 20), YEL)
    sq = square_crop(frame, df, f, pad=0.24)
    banner(sq, f, "FOOT PLANT", [
        ("stride angle " + vs(vals, "Stride Angle [O]"), STRA),
        ("torso ant tilt " + vs(vals, "torso_anterior_tilt_fp"), YEL)])
    return sq, "Front | foot plant", ["Stride Angle [O]", "torso_anterior_tilt_fp"]


def panel_front_mer(df, arm, r, vals):
    s = "left" if arm == "right" else "right"
    f = int(r.mer_f)
    frame = grab_seq(r.video_path, f); skel(frame, df, f)
    CR = crop_rect(df, f, 0.24)
    mh, ms = mid(df, "left_hip", "right_hip", f), mid(df, "left_shoulder", "right_shoulder", f)
    vert_ref(frame, mh, ms, LAT)
    txt(frame, f"lat tilt {vals['torso_lateral_tilt_mer'][0]:.0f}",
        (mh[0] + 14, mh[1] - 22), LAT)
    E, S, Hh = (pt(df, f"{s}_elbow", f), pt(df, f"{s}_shoulder", f), pt(df, f"{s}_hip", f))
    cv2.line(frame, S, E, ABD, int(3 * SC)); cv2.line(frame, S, Hh, ABD, int(3 * SC))
    cv2.circle(frame, S, int(7 * SC), RED, -1)
    arc_minor(frame, S, E, Hh, int(46 * SC), ABD, int(3 * SC))
    txt(frame, f"glove abd {vals['glove_shoulder_abduction_mer'][0]:.0f}",
        (S[0] + 14, S[1] - 16), ABD)
    sq = square_crop(frame, df, f, pad=0.24)
    banner(sq, f, "MER PROXY (release - 11 f)", [
        ("torso lat tilt " + vs(vals, "torso_lateral_tilt_mer"), LAT),
        ("glove-arm abd " + vs(vals, "glove_shoulder_abduction_mer"), ABD)])
    return sq, "Front | MER proxy", ["torso_lateral_tilt_mer",
                                     "glove_shoulder_abduction_mer"]


def panel_oh_hss(df, r, gt, met, vals):
    """Hip-shoulder separation at its own signature anchor. The hip line is the manual
    track; the shoulder line is the automatic pose, which overhead occlusion leaves
    usable while it destroys the hips."""
    f = int(r.hss_anchor_f) if np.isfinite(r.hss_anchor_f) else int(r.release_f)
    row = gt.loc[f]
    frame = grab_seq(r.video_path, f); skel(frame, df, f, col=(185, 185, 185), lw=1)
    CR = crop_rect(df, f, 0.34)
    lh = (int(row.lhip_x), int(row.lhip_y)); rh = (int(row.rhip_x), int(row.rhip_y))
    ls, rs = pt(df, "left_shoulder", f), pt(df, "right_shoulder", f)
    cv2.line(frame, lh, rh, VIO, int(6 * SC))
    cv2.circle(frame, lh, int(8 * SC), VIO, -1); cv2.circle(frame, rh, int(8 * SC), VIO, -1)
    cv2.line(frame, ls, rs, HSSC, int(4 * SC))
    pc = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
    sc_ = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
    dh = np.array(rh) - np.array(lh); ds = np.array(rs) - np.array(ls)
    R = int(70 * SC)
    p1 = (int(pc[0] + dh[0] / np.hypot(*dh) * R), int(pc[1] + dh[1] / np.hypot(*dh) * R))
    p2 = (int(pc[0] + ds[0] / np.hypot(*ds) * R), int(pc[1] + ds[1] / np.hypot(*ds) * R))
    cv2.line(frame, pc, p2, HSSC, int(2 * SC))
    arc_minor(frame, pc, p1, p2, R, HSSC, int(4 * SC))
    txt(frame, "pelvis line (manual GT)", (pc[0] + 26, pc[1] + int(40 * SC)), VIO, 0.75, crop=CR)
    txt(frame, f"hip-shoulder sep {vals['Hip-Shoulder Sep [O]'][0]:.0f}",
        (sc_[0] + 24, sc_[1] - 20), HSSC, 0.78, crop=CR)
    sq = square_crop(frame, df, f, pad=0.34)
    banner(sq, f, "HSS ANCHOR", [
        ("hip-shoulder sep " + vs(vals, "Hip-Shoulder Sep [O]"), HSSC),
        ("pelvis line = hand-labelled GT", VIO)])
    return sq, "Overhead | HSS anchor", ["Hip-Shoulder Sep [O]"]


def panel_oh_pelvis(df, r, gt, met, vals):
    """Pelvic rotational velocity from the manual hip track. The automatic pipeline's
    own output for this row at this clip is an unwrapped angle, reported in the text;
    the value drawn here is the hand-labelled one and is labelled as such."""
    i = met.rot_velo_degps.abs().idxmax()
    f = int(met.frame.iloc[i]); rv = float(met.rot_velo_degps.iloc[i])
    frame = grab_seq(r.video_path, f); skel(frame, df, f, col=(185, 185, 185), lw=1)
    CR = crop_rect(df, f, 0.34)
    row = gt.loc[f]
    lh = (int(row.lhip_x), int(row.lhip_y)); rh = (int(row.rhip_x), int(row.rhip_y))
    cv2.line(frame, lh, rh, VIO, int(6 * SC))
    cv2.circle(frame, lh, int(8 * SC), VIO, -1); cv2.circle(frame, rh, int(8 * SC), VIO, -1)
    pc = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
    txt(frame, "pelvis line (manual GT)", (pc[0] + 26, pc[1] - 8), VIO, 0.75, crop=CR)
    s = 1.0 if rv >= 0 else -1.0
    cc = (pc[0] - int(175 * SC), pc[1]); ar = int(74 * SC)
    arc = [(int(cc[0] + ar * np.cos(a)), int(cc[1] + ar * np.sin(a)))
           for a in np.radians(np.linspace(-115, 115, 30) * s)]
    for k in range(1, len(arc)):
        cv2.line(frame, arc[k - 1], arc[k], VIO, int(4 * SC))
    cv2.arrowedLine(frame, arc[-2], arc[-1], VIO, int(4 * SC), tipLength=1.6)
    txt(frame, f"pelvis rot {abs(rv):.0f}/s", (cc[0] - int(78 * SC), cc[1] + ar + int(46 * SC)), VIO, crop=CR)
    sq = square_crop(frame, df, f, pad=0.34)
    banner(sq, f, "PEAK ROT VELO", [
        (f"pelvis rot velo {abs(rv):.0f} deg/s", VIO),
        ("from the hand-labelled pelvis", VIO),
        ("automatic output: " + vs(vals, "Pelvis Rot Velo [O]", 0, " deg/s"), WHITE)])
    return sq, "Overhead | peak rot velo", ["Pelvis Rot Velo [O]"]


# ---------------------------------------------------------------- driver

LAYOUT = {"angle00_00": [panel_side_release, panel_side_com],
          "angle03_04": [panel_front_release, panel_front_fp, panel_front_mer],
          "pitching_overhead_01": [panel_oh_hss, panel_oh_pelvis]}


def main():
    os.makedirs(OUT, exist_ok=True)
    pilot = pd.read_csv(os.path.join(PILOT, "pilot_clips_eligible.csv")).set_index("clip")
    cells = pd.read_csv(os.path.join(V, "realvideo_feasibility_cells.csv"))
    gm = pd.read_csv(os.path.join(V, "gate_map.csv"))
    graded = {(m, a, e) for m, a, e in
              gm.loc[gm.grade.isin(["strong", "moderate"]), ["metric", "az", "el"]].to_numpy()}

    rows = []
    for clip, tier, az, el in STATIONS:
        r = pilot.loc[clip]
        df = load_coords(clip)
        arm = detect_arm(df)
        vals = canon(cells, clip)
        gt = met = None
        d = os.path.join(config.ROOT, "data", "outputs", clip)
        if os.path.exists(os.path.join(d, f"{clip}_hip_gt_final.csv")):
            gt = pd.read_csv(os.path.join(d, f"{clip}_hip_gt_final.csv")).set_index("frame")
            met = pd.read_csv(os.path.join(d, f"{clip}_hip_gt_metrics.csv"))
        panels = []
        for fn in LAYOUT[clip]:
            out = fn(df, r, gt, met, vals) if fn in (panel_oh_hss, panel_oh_pelvis) \
                else fn(df, arm, r, vals)
            sq, title, mids = out
            for m in mids:                        # the map decides what may be drawn
                if (m, az, el) not in graded:
                    raise SystemExit(f"{clip}: {m} is not graded at az {az} / el {el}")
            panels.append((sq, title))
        n_ok = int((cells[cells["clip"] == clip].status == "ok").sum())
        n_gr = sum(1 for (m, a, e) in graded if a == az and e == el)
        rows.append((tier, clip, az, el, n_ok, n_gr, panels))
        print(f"{clip:<22} az {az:>3} el {el:>2}  {len(panels)} panels  "
              f"{n_gr} rows graded here  {n_ok} of 35 return a value")

    ncol = max(len(r[-1]) for r in rows)
    nrow = len(rows)
    lab_w = 0.118
    pw = (1.0 - lab_w) / ncol
    # a title strip above every row, so a panel title can never land on the row above
    title_in = 0.15
    row_in = BODY_W * pw + title_in
    fig_h = row_in * nrow
    fig = plt.figure(figsize=(BODY_W, fig_h))
    ph = (BODY_W * pw) / fig_h                    # panel height, figure fraction
    for i, (tier, clip, az, el, n_ok, n_gr, panels) in enumerate(rows):
        ytop = 1.0 - (i + 1) * (row_in / fig_h) + (title_in / fig_h) * 0.06
        ax = fig.add_axes([0.004, ytop, lab_w - 0.014, ph]); ax.axis("off")
        ax.text(0, 0.93, tier, fontsize=6.6, weight="bold", color=INK,
                va="top", ha="left", transform=ax.transAxes)
        ax.text(0, 0.83, f"{clip}\naz {az}°, el {el}°\n\n{n_gr} rows graded\nhere; "
                f"{n_ok} of 35\nreturn a value", fontsize=4.8, color=MUTE,
                va="top", ha="left", transform=ax.transAxes, linespacing=1.6)
        for j, (sq, title) in enumerate(panels):
            axp = fig.add_axes([lab_w + j * pw, ytop, pw * 0.985, ph])
            axp.set_xticks([]); axp.set_yticks([])
            axp.imshow(sq[:, :, ::-1], aspect="auto")
            axp.set_title(title, fontsize=5.8, color=INK, weight="bold", pad=2.2)
            for sp in axp.spines.values():
                sp.set_color("0.80"); sp.set_linewidth(0.7)

    out = os.path.join(OUT, "fig_realvideo_grid.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"figure {BODY_W:.2f} x {fig_h:.2f} in")
    print("saved ->", out)


if __name__ == "__main__":
    main()
