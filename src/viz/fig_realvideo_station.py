"""Real-video execution, one figure per capture station.

WHAT IS DRAWN IS DECIDED BY THE MAP. For a station at (az, el) the script takes EVERY
row the graded map holds a cell for at that exact cell, groups them by the event they
are read at, and draws one panel per event containing all of them. Nothing is selected
by hand: the deployment pipeline's old 15-metric list is not consulted anywhere.

  side  az 0  / el 0   14 rows -> pkh 1 | fp 1 | mer 1 | release(+fp) 10 | own 1
  front az 90 / el 0    7 rows -> fp 2 | mer 2 | release 3
  overhead az 0 / el 85 8 rows -> pkh 1 | fp 1 | release(+fp) 4 | own 2

ONE FIGURE PER STATION, because they do not fit in one. Fifteen panels at body width
give 1.3 in each, and the side release panel alone carries ten labels.

VALUES ARE THE CANONICAL ONES, read from realvideo_feasibility_cells.csv at draw time,
with two stated exceptions on the overhead station: hip-shoulder separation and pelvic
rotational velocity are taken from the HAND-LABELLED pelvis track
(*_hip_gt_metrics.csv `hss_deg` / `rot_velo_degps`), because overhead occlusion destroys
the automatic hip keypoints. The automatic pipeline's own output for those rows is
printed beside the manual one so the two can be compared rather than confused.

NO BANNER PLATE. Labels are outlined text drawn next to their own geometry; the station
and the event are given by the figure's row/column headers, not painted over the scene.

VIEWPOINTS ARE THE CAPTURE STATIONS, not `pilot_clips_eligible.csv`'s az/el, which are
the viewpoint classifier's estimate (note `vote_share` beside them) and disagree with the
capture design on four of the five angle03_* clips.

Run:  conda activate diamond
      cd src\\viz
      python fig_realvideo_station.py
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
from fig_realvideo_new_metrics_patent import grab_seq, load_coords, detect_arm, arc_minor
from fig_graded_map import INK, MUTE, BODY_W

V = config.OBP_VALIDATION_DIR
PILOT = os.path.join(config.ROOT, "data", "outputs", "realvideo_pilot")
OUT = os.path.join(config.ROOT, "data", "outputs", "viz")
J = M.JOINTS

GREEN = (0, 255, 0); WHITE = (255, 255, 255); RED = (0, 0, 255)
KNEE = (139, 139, 0); YEL = (0, 255, 255); BLU = (255, 80, 0); PLUM = (119, 39, 219)
ARM = (0, 215, 255); FARM = (40, 160, 255); STRA = (211, 85, 186); LAT = (200, 40, 190)
ABD = (255, 120, 20); ORNG = (12, 88, 234); VIO = (237, 58, 124); HSSC = (60, 200, 255)
WRS = (147, 20, 255); GREY = (150, 150, 150)

NCOL = 3          # tiles per row. Widening this narrows every tile, and because the
                  # label is drawn as a fixed fraction of the tile (see emit), the
                  # printed label shrinks with it.
# Compact mode, for a caller that packs these tiles too small to carry a full label.
# The numeric value comes off the image and the execution flag moves to the panel
# title, so the remaining text is the quantity name alone.
LABEL_VALUES = True
# Independent of the above. Every label in a panel carries the same execution
# status, so the flag can sit in the panel title instead of on each label, which
# also puts it where the caption says it is. Asserted, not assumed.
FLAG_IN_TITLE = False
LABEL_SCALE = 1.0  # raise it back when a caller packs the same tiles into narrower
                   # columns, so the label keeps its printed size rather than its
                   # fraction of the crop. 1.0 = the per-station figures as authored.
STATIONS = [("angle00_00", "side", 0, 0, "side"),
            ("angle03_04", "front", 90, 0, "front"),
            ("pitching_overhead_01", "overhead", 0, 85, "overhead")]

# event key -> (frame column, column title). `own` is a row that reads no external
# anchor and locates its own instant.
EVENTS = [("pkh", "pkh_f", "peak knee height"), ("fp", "fp_f", "foot plant"),
          ("mer", "mer_f", "MER proxy"), ("release", "release_f", "release"),
          ("own", None, "own anchor")]
ANCHOR_TO_EVENT = {"pkh": "pkh", "fp": "fp", "mer": "mer",
                   "release": "release", "release+fp": "release", "none": "own"}


def unit_of(mid):
    from realvideo_feasibility import ADOPTED_UNIT, OBS_RANGE
    U = {"deg_signed": " deg", "deg_joint": " deg", "deg_per_s": " deg/s",
         "ratio": " xH", "stature_rate": " H/s", "stature_rate_signed": " H/s",
         "s": " s", "m_per_s": " m/s"}
    return U.get(ADOPTED_UNIT.get(mid, ""), " deg")


def short(mid):
    n = mid.replace(" [O]", "")
    return {"arm_slot": "arm slot",
            "Lead Knee Angle": "lead knee", "Release Height": "release height",
            "Release Ext": "release ext", "Stride (anchor)": "stride",
            "Trunk Tilt (ant)": "trunk tilt", "COG Fwd Velo": "COG fwd velo",
            "COG Velo @PKH": "COG velo @PKH", "Wrist Speed": "wrist speed",
            "Stride Angle": "stride angle", "Hip-Shoulder Sep": "hip-shoulder sep",
            "Pelvis Rot Velo": "pelvis rot velo",
            "lead_knee_extension_angular_velo_max": "knee ext velo (max)",
            "lead_knee_extension_from_fp_to_br": "knee ext fp-BR",
            "torso_lateral_tilt_fp": "torso lat tilt",
            "torso_lateral_tilt_mer": "torso lat tilt",
            "torso_anterior_tilt_fp": "torso ant tilt",
            "torso_anterior_tilt_mer": "torso ant tilt",
            "glove_shoulder_abduction_mer": "glove-arm abd"}.get(n, n)


class Panel:
    """One event at one station: the frame, the geometry, and the label stack."""

    def __init__(self, df, frame, f, arm, lead, sc):
        self.df, self.img, self.f, self.arm, self.lead, self.sc = df, frame, f, arm, lead, sc
        self.labels = []                       # (anchor_xy, text, colour, below)
        self.flags = set()                     # execution statuses, compact mode only

    def P(self, name):
        return pt(self.df, name, self.f)

    def mid(self, a, b):
        pa, pb = self.P(a), self.P(b)
        return ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)

    def line(self, a, b, col, w=3):
        cv2.line(self.img, a, b, col, int(w * self.sc), cv2.LINE_AA)

    def arc(self, v, p1, p2, r, col):
        arc_minor(self.img, v, p1, p2, int(r * self.sc), col, int(3 * self.sc))

    def dot(self, p, col, r=7):
        cv2.circle(self.img, p, int(r * self.sc), col, -1)

    def label(self, xy, text, col, below=False):
        """`below` puts the text under the anchor instead of over it, for anchors
        whose own limb is above them."""
        self.labels.append((xy, text, col, below))

    # -- geometry primitives ------------------------------------------------
    def vs_vertical(self, col):
        mh, ms = self.mid("left_hip", "right_hip"), self.mid("left_shoulder", "right_shoulder")
        seg = int(np.hypot(ms[0] - mh[0], ms[1] - mh[1])) or 80
        self.line(mh, (mh[0], mh[1] - seg), WHITE, 2)
        self.line(mh, ms, col)
        self.arc(mh, (mh[0], mh[1] - seg), ms, 42, col)
        return mh

    def vs_horizontal(self, a, b, col, r=46):
        L = int(max(abs(b[0] - a[0]), 70) * 1.35)
        h = (a[0] + (L if b[0] >= a[0] else -L), a[1])
        self.line(a, h, WHITE, 2); self.line(a, b, col); self.dot(a, RED)
        self.arc(a, h, b, r, col)
        return a

    def joint_arc(self, v, a, b, col, r=45):
        self.line(v, a, col); self.line(v, b, col); self.dot(v, col, 8)
        self.arc(v, a, b, r, col)
        return v

    def ground_y(self):
        return int(np.nanmax(np.concatenate([self.df["left_ankle_y"].to_numpy(float),
                                             self.df["right_ankle_y"].to_numpy(float)])))


# -- one drawer per quantity; returns the point its label hangs off -----------

def d_arm_slot(p):
    return p.vs_horizontal(p.P(f"{p.arm}_shoulder"), p.P(f"{p.arm}_wrist"), ARM, 52), ARM


def d_arm_slot_forearm(p):
    return p.vs_horizontal(p.P(f"{p.arm}_elbow"), p.P(f"{p.arm}_wrist"), FARM, 38), FARM


def d_lead_knee(p):
    return p.joint_arc(p.P(f"{p.lead}_knee"), p.P(f"{p.lead}_hip"),
                       p.P(f"{p.lead}_ankle"), KNEE), KNEE


def d_trunk_ant(p):
    return p.vs_vertical(YEL), YEL


def d_trunk_lat(p):
    return p.vs_vertical(LAT), LAT


def d_release_height(p):
    w, gy = p.P(f"{p.arm}_wrist"), p.ground_y()
    ov = p.img.copy()
    cv2.line(ov, w, (w[0], gy), BLU, int(3 * p.sc))
    for yy in (gy, w[1]):
        cv2.line(ov, (w[0] - int(10 * p.sc), yy), (w[0] + int(10 * p.sc), yy), BLU,
                 int(3 * p.sc))
    cv2.addWeighted(ov, 0.5, p.img, 0.5, 0, p.img)
    return (w[0], (w[1] + gy) // 2), BLU


def d_release_ext(p, fp):
    w, gy = p.P(f"{p.arm}_wrist"), p.ground_y()
    trail = f"{'right' if p.arm == 'right' else 'left'}_ankle"
    ax = int(M.trail_anchor_x(p.df[f"{trail}_x"].to_numpy(float), fp, 120))
    y = gy + int(40 * p.sc)
    for xx in (ax, w[0]):
        cv2.line(p.img, (xx, y - int(10 * p.sc)), (xx, y + int(10 * p.sc)), PLUM,
                 int(3 * p.sc))
    cv2.arrowedLine(p.img, (ax, y), (w[0], y), PLUM, int(3 * p.sc), tipLength=0.04)
    cv2.arrowedLine(p.img, (w[0], y), (ax, y), PLUM, int(3 * p.sc), tipLength=0.04)
    return ((ax + w[0]) // 2, y), PLUM


def d_stride(p, fp):
    """From the FIXED trail anchor to the lead ankle -- the same rubber-side origin
    release extension uses, and the origin event_inventory.py records for this row.

    Do not redraw this ankle-to-ankle. By release the trail foot has dragged well
    forward of the rubber, so a line between the two ankles at the frame being drawn
    spans far less than the value it labels: a stride is ~0.8-0.9 of stature, which
    an instantaneous ankle separation at release cannot be. The observable's name in
    realvideo_support_matrix (`ankle_sep_ratio`, joints `l_an`/`r_an`) names its two
    INPUTS -- the trail-ankle series the anchor is averaged over, and the lead ankle
    read at the anchor -- not an instantaneous separation. That misreading reverted
    this drawing once on 2026-08-07; it was wrong."""
    la = p.P(f"{p.lead}_ankle")
    trail = f"{'right' if p.arm == 'right' else 'left'}_ankle"
    ax = int(M.trail_anchor_x(p.df[f"{trail}_x"].to_numpy(float), fp, 120))
    y = la[1]
    for xx in (ax, la[0]):
        cv2.line(p.img, (xx, y - int(9 * p.sc)), (xx, y + int(9 * p.sc)), STRA,
                 int(3 * p.sc))
    cv2.arrowedLine(p.img, (ax, y), (la[0], y), STRA, int(3 * p.sc), tipLength=0.04)
    cv2.arrowedLine(p.img, (la[0], y), (ax, y), STRA, int(3 * p.sc), tipLength=0.04)
    return ((ax + la[0]) // 2, y), STRA


def d_stride_angle(p):
    la = p.P(f"{p.lead}_ankle")
    ta = p.P(f"{'right' if p.lead == 'left' else 'left'}_ankle")
    return p.vs_horizontal(ta, la, STRA, 42), STRA


def d_glove_abd(p):
    g = "left" if p.arm == "right" else "right"
    return p.joint_arc(p.P(f"{g}_shoulder"), p.P(f"{g}_elbow"), p.P(f"{g}_hip"),
                       ABD, 46), ABD


def d_wrist_speed(p):
    w = p.P(f"{p.arm}_wrist")
    p.dot(w, WRS, 8)
    cv2.arrowedLine(p.img, w, (w[0] + int(80 * p.sc), w[1] - int(30 * p.sc)), WRS,
                    int(3 * p.sc), tipLength=0.3)
    return w, WRS


def d_com(p, col, at):
    cx, cy = M.body_com(p.df, J)
    c = (int(cx[at]), int(cy[at]))
    p.dot(c, col, 8)
    cv2.arrowedLine(p.img, c, (c[0] + int(74 * p.sc), c[1]), col, int(3 * p.sc),
                    tipLength=0.3)
    return c, col


def d_hip_shoulder(p, hip):
    lh, rh = hip
    ls, rs = p.P("left_shoulder"), p.P("right_shoulder")
    p.line(lh, rh, VIO, 6); p.dot(lh, VIO, 8); p.dot(rh, VIO, 8)
    p.line(ls, rs, HSSC, 4)
    pc = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
    dh = np.array(rh) - np.array(lh); ds = np.array(rs) - np.array(ls)
    R = int(72 * p.sc)
    q1 = (int(pc[0] + dh[0] / np.hypot(*dh) * R), int(pc[1] + dh[1] / np.hypot(*dh) * R))
    q2 = (int(pc[0] + ds[0] / np.hypot(*ds) * R), int(pc[1] + ds[1] / np.hypot(*ds) * R))
    p.line(pc, q2, HSSC, 2)
    p.arc(pc, q1, q2, 72, HSSC)
    return pc, HSSC


def d_pelvis_rot(p, hip, sign):
    lh, rh = hip
    p.line(lh, rh, VIO, 6); p.dot(lh, VIO, 8); p.dot(rh, VIO, 8)
    pc = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
    cc = (pc[0] - int(170 * p.sc), pc[1]); ar = int(72 * p.sc)
    a = [(int(cc[0] + ar * np.cos(t)), int(cc[1] + ar * np.sin(t)))
         for t in np.radians(np.linspace(-115, 115, 30) * sign)]
    for k in range(1, len(a)):
        cv2.line(p.img, a[k - 1], a[k], VIO, int(4 * p.sc), cv2.LINE_AA)
    cv2.arrowedLine(p.img, a[-2], a[-1], VIO, int(4 * p.sc), tipLength=1.6)
    return cc, VIO


# metric_id -> drawer. Every graded row of every station appears here; emit() raises if
# the map hands it a row this table does not cover, so the figure can never quietly
# draw a subset of what the station supports.
DRAW = {
    "Arm Slot [O]": d_arm_slot, "arm_slot": d_arm_slot_forearm,
    "Lead Knee Angle [O]": d_lead_knee,
    "lead_knee_extension_angular_velo_max": d_lead_knee,
    "lead_knee_extension_from_fp_to_br": d_lead_knee,
    "Trunk Tilt (ant) [O]": d_trunk_ant,
    "torso_anterior_tilt_fp": d_trunk_ant, "torso_anterior_tilt_mer": d_trunk_ant,
    "torso_lateral_tilt_fp": d_trunk_lat, "torso_lateral_tilt_mer": d_trunk_lat,
    "Release Height [O]": d_release_height, "Release Ext [O]": "ext",
    "Stride (anchor) [O]": "stride", "Stride Angle [O]": d_stride_angle,
    "glove_shoulder_abduction_mer": d_glove_abd,
    "Wrist Speed [O]": d_wrist_speed,
    "COG Fwd Velo [O]": "com_fwd", "COG Velo @PKH [O]": "com_pkh",
    "Hip-Shoulder Sep [O]": "hss", "Pelvis Rot Velo [O]": "pelvis",
}


# Anchors that carry the label UNDER the point. The arm slot is read at the throwing
# shoulder at release, so the space above it is the head.
LABEL_BELOW = {"arm_slot"}


def place(panel, crop, sc):
    """Draw the label stack: outlined text on the scene, inside the crop, off each other.

    No plate is painted behind the labels. Collision is resolved by pushing a colliding
    label down one line, and every box is clamped into the crop before drawing, so a
    label can neither overprint another nor fall outside the tile that is shown."""
    cx, cy, side = crop
    sc = sc * LABEL_SCALE
    fs, th = 0.78 * sc, max(int(2 * sc), 2)
    lh = int(38 * sc)
    taken = []
    for xy, text, col, below in panel.labels:
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        x = int(min(max(xy[0] + int(16 * sc), cx + 10), cx + side - tw - 10))
        off = int((tht + 20) * sc) if below else -int(14 * sc)
        y = int(min(max(xy[1] + off, cy + tht + 10), cy + side - 10))
        for _ in range(24):
            if not any(abs(t[1] - y) < lh and x < t[0] + t[2] and x + tw > t[0]
                       for t in taken):
                break
            y += lh
        y = int(min(y, cy + side - 10))
        taken.append((x, y, tw))
        cv2.line(panel.img, xy, (x - int(6 * sc), y - int(4 * sc)), col,
                 max(int(1 * sc), 1), cv2.LINE_AA)
        cv2.putText(panel.img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 0, 0), th + 4, cv2.LINE_AA)
        cv2.putText(panel.img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, col,
                    th, cv2.LINE_AA)


def crop_of(df, frames, pad=0.30):
    xs, ys = [], []
    for f in frames:
        for a, b in CONNECT:
            for j in (a, b):
                try:
                    x, y = df[j + "_x"].iloc[f], df[j + "_y"].iloc[f]
                    if np.isfinite([x, y]).all():
                        xs.append(float(x)); ys.append(float(y))
                except Exception:
                    pass
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    return (int((x0 + x1) / 2 - side / 2), int((y0 + y1) / 2 - side / 2), int(side))


def build_panel(df, r, arm, f, mids, vals, gt, met, sc):
    lead = "left" if arm == "right" else "right"
    img = grab_seq(r.video_path, f)
    p = Panel(df, img, f, arm, lead, sc)
    for a, b in CONNECT:
        try:
            cv2.line(img, pt(df, a, f), pt(df, b, f), GREEN, int(2 * sc), cv2.LINE_AA)
        except Exception:
            pass
    hip = None
    if gt is not None and f in gt.index:
        row = gt.loc[f]
        hip = ((int(row.lhip_x), int(row.lhip_y)), (int(row.rhip_x), int(row.rhip_y)))
    for mid in mids:
        d = DRAW[mid]
        if d == "ext":
            xy, col = d_release_ext(p, int(r.fp_f))
        elif d == "stride":
            xy, col = d_stride(p, int(r.fp_f))
        elif d == "com_fwd":
            cx, _cy = M.body_com(df, J)
            v = np.abs(np.gradient(cx)) * r.fps / M.pixel_stature(df, J)
            xy, col = d_com(p, VIO, int(np.nanargmax(v[:int(r.release_f) + 1])))
        elif d == "com_pkh":
            xy, col = d_com(p, ORNG, int(r.pkh_f))
        elif d == "hss":
            xy, col = d_hip_shoulder(p, hip)
        elif d == "pelvis":
            rv = 0.0
            if met is not None and len(met.loc[met.frame == f]):
                rv = float(met.loc[met.frame == f, "rot_velo_degps"].iloc[0])
            xy, col = d_pelvis_rot(p, hip, 1.0 if rv >= 0 else -1.0)
        else:
            xy, col = d(p)
        v, st = vals[mid]
        dec = 2 if unit_of(mid) in (" xH", " H/s") else 0
        s = "--" if not np.isfinite(v) else "%.*f%s" % (dec, v, unit_of(mid))
        # the two overhead transverse rows are read off the hand-labelled pelvis; the
        # automatic value is kept beside it rather than replaced, so the gap is visible
        if met is not None and mid in ("Hip-Shoulder Sep [O]", "Pelvis Rot Velo [O]"):
            key = "hss_deg" if mid.startswith("Hip") else "rot_velo_degps"
            g = met.loc[met.frame == f, key]
            if len(g):
                auto = "--" if not np.isfinite(v) else "%.0f" % v
                s = "%.0f%s (manual GT)  auto %s [%s]" % (
                    abs(float(g.iloc[0])), unit_of(mid), auto, st)
                st = "ok"
        txt = "%s %s" % (short(mid), s) if LABEL_VALUES else short(mid)
        if st != "ok":
            if FLAG_IN_TITLE:
                p.flags.add(st)
            else:
                txt += " [%s]" % st
        p.label(xy, txt, col, below=mid in LABEL_BELOW)
    return p


def hss_gt_panel(pilot, sc_target=None):
    """The overhead hip-shoulder separation panel, drawn from the AUTHOR'S ground truth.

    The GT is not derived here and is not the automatic pipeline's output. It is the
    hip line the author placed by hand, logged by scratch/hss_manual_still.py to
    data/outputs/hss_overhead_gt.csv as clip / frame / two endpoints / separation:

        pitching_overhead_02, f389, L(2017.28,808.96)-R(1699.84,1026.56), sep = -48.6

    That frame, those endpoints and that angle are what this panel draws. The shoulder
    line comes from the refined pose, which overhead occlusion leaves usable while it
    destroys the hips; the separation is the angle between the two lines, drawn as an
    arc at their intersection exactly as the disclosure still draws it."""
    import re
    from smoother import smooth_coordinates
    g = pd.read_csv(os.path.join(config.ROOT, "data", "outputs",
                                 "hss_overhead_gt.csv"), encoding="utf-8-sig").iloc[0]
    clip, f = str(g["clip"]), int(g["hss_frame"])
    m = re.search(r"L\(([\d.]+),([\d.]+)\)-R\(([\d.]+),([\d.]+)\)\s*sep=(-?[\d.]+)",
                  str(g["user_note"]))
    lh = (float(m.group(1)), float(m.group(2)))
    rh = (float(m.group(3)), float(m.group(4)))
    sep = float(m.group(5))

    coords = os.path.join(config.ROOT, "data", "outputs", clip,
                          clip + "_coords_rtmp_refined.csv")
    df = smooth_coordinates(pd.read_csv(coords))
    r = pilot.loc[clip]
    img = grab_seq(r.video_path, f)
    ls = (int(df["left_shoulder_x"].iloc[f]), int(df["left_shoulder_y"].iloc[f]))
    rs = (int(df["right_shoulder_x"].iloc[f]), int(df["right_shoulder_y"].iloc[f]))
    lhp, rhp = (int(lh[0]), int(lh[1])), (int(rh[0]), int(rh[1]))

    xs = [ls[0], rs[0], lhp[0], rhp[0]]; ys = [ls[1], rs[1], lhp[1], rhp[1]]
    side = int(max(max(xs) - min(xs), max(ys) - min(ys)) * 2.6)
    H, W = img.shape[:2]
    side = min(side, H, W)
    x = int(np.clip((min(xs) + max(xs)) / 2 - side / 2, 0, W - side))
    y = int(np.clip((min(ys) + max(ys)) / 2 - side / 2, 0, H - side))
    sc = side / 1000.0

    for a, b in CONNECT:                       # faint pose for context only
        try:
            cv2.line(img, pt(df, a, f), pt(df, b, f), (185, 185, 185),
                     max(int(1 * sc), 1), cv2.LINE_AA)
        except Exception:
            pass
    cv2.line(img, ls, rs, YEL, int(3 * sc), cv2.LINE_AA)
    cv2.line(img, lhp, rhp, RED, int(3 * sc), cv2.LINE_AA)
    for q, c in ((ls, YEL), (rs, YEL), (lhp, RED), (rhp, RED)):
        cv2.circle(img, q, int(6 * sc), c, -1)

    sh = np.array(rs, float) - np.array(ls, float)
    hp = np.array(rhp, float) - np.array(lhp, float)
    M2 = np.array([sh, -hp]).T
    lab_at = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
    if abs(np.linalg.det(M2)) > 1e-9:
        t, _ = np.linalg.solve(M2, np.array(lhp, float) - np.array(ls, float))
        ix, iy = (np.array(ls, float) + t * sh).astype(int)
        a_sh = np.degrees(np.arctan2(sh[1], sh[0]))
        a_hp = np.degrees(np.arctan2(hp[1], hp[0]))
        d = (a_sh - a_hp + 540) % 360 - 180
        cv2.ellipse(img, (ix, iy), (int(64 * sc), int(64 * sc)), 0, a_hp, a_hp + d,
                    YEL, int(2 * sc), cv2.LINE_AA)
        lab_at = (ix, iy)

    p = Panel(df, img, f, "right", "left", sc)
    p.label(lab_at, "hip-shoulder sep %.1f deg (manual GT)" % abs(sep), YEL)
    p.label(((lhp[0] + rhp[0]) // 2, (lhp[1] + rhp[1]) // 2),
            "hip line: author-placed endpoints", RED)
    place(p, (x, y, side), sc)
    return p.img[y:y + side, x:x + side], f, clip


def station_tiles(station, cells, gm, reg, pilot):
    """Render one station's tiles: [(event title, n graded rows, square BGR image)].

    The per-station figures and the merged anchor-orientation figure (§VIII Fig. 1)
    both draw from here, so the two can never disagree about what a station reads or
    how it is drawn. The CALLER owns the panel title text and the grid: §VIII prints
    the anchor name alone, `emit` appends the graded-row count. The manual-GT tile
    carries `n = None` because no graded row is read on it.
    """
    clip, tier, az, el, tag = station
    r = pilot.loc[clip]
    df = load_coords(clip)
    arm = detect_arm(df)
    vals = {m: (float(v), str(s)) for m, v, s in
            cells.loc[cells["clip"] == clip,
                      ["metric_id", "value", "status"]].to_numpy()}
    g = gm[(gm.az == az) & (gm.el == el) & gm.grade.isin(["strong", "moderate"])]
    by_ev = {}
    for m in sorted(g.metric):
        if m not in DRAW:
            raise SystemExit("%s: no drawer for graded row %s" % (clip, m))
        by_ev.setdefault(ANCHOR_TO_EVENT[reg.loc[m, "anchor_type"]], []).append(m)
    d = os.path.join(config.ROOT, "data", "outputs", clip)
    gt = met = None
    if os.path.exists(os.path.join(d, clip + "_hip_gt_final.csv")):
        gt = pd.read_csv(os.path.join(d, clip + "_hip_gt_final.csv")).set_index("frame")
        met = pd.read_csv(os.path.join(d, clip + "_hip_gt_metrics.csv"))

    panels = []
    for key, colname, title in EVENTS:
        mids = by_ev.get(key)
        if not mids:
            continue
        if key == "own":
            # Hip-shoulder separation is drawn from the author's ground truth, on its
            # own clip and frame, so it leaves this panel rather than sharing one.
            if tag == "overhead" and "Hip-Shoulder Sep [O]" in mids:
                mids = [m for m in mids if m != "Hip-Shoulder Sep [O]"]
                if not mids:
                    continue
            f = int(r.hss_anchor_f) if np.isfinite(r.hss_anchor_f) else int(r.release_f)
        else:
            f = int(r[colname])
        panels.append((title, f, mids))

    tiles = []
    for title, f, mids in panels:
        # The drawing scale is set by the CROP, not by the panel's width in inches.
        # Tying it to inches made every label the same pixel size on crops that differ
        # by 3x in subject size, so the overhead panels came out with text wider than
        # the pitcher. Scaling with the crop keeps the printed label a fixed fraction
        # of the tile whatever the pose does.
        x, y, side = crop_of(df, [f])
        sc = side / 1000.0
        p = build_panel(df, r, arm, f, mids, vals, gt, met, sc)
        H, W = p.img.shape[:2]
        side = min(side, min(H, W))
        x = max(0, min(x, W - side)); y = max(0, min(y, H - side))
        place(p, (x, y, side), sc)
        if FLAG_IN_TITLE and p.flags:
            # one flag per panel, which is what makes moving it to the title exact
            assert len(p.flags) == 1, "%s carries %s" % (title, sorted(p.flags))
            title = "%s  [%s]" % (title, p.flags.pop())
        tiles.append((title, len(mids), p.img[y:y + side, x:x + side]))
        print("  %-18s f%-5d %d rows: %s"
              % (title, f, len(mids), ", ".join(short(m) for m in mids)))
    if tag == "overhead":
        sq, gf, gclip = hss_gt_panel(pilot)
        # Kept short on purpose: a panel is ~2.3 in wide and a centred title longer than
        # about 40 characters overruns its axes and collides with its neighbour's. The
        # fuller statement belongs in the caption, and the panel's own label already
        # carries "(manual GT)".
        tiles.append(("hip-shoulder separation  —  manual GT", None, sq))
        print("  %-18s f%-5d ground-truth hip line on %s" % ("HSS (GT)", gf, gclip))
    return tiles


def draw_tile(fig, ax_rect, sq, title):
    ax = fig.add_axes(ax_rect)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(sq[:, :, ::-1], aspect="auto")
    ax.set_title(title, fontsize=6.6, color=INK, weight="bold", pad=2.6)
    for spn in ax.spines.values():
        spn.set_color("0.80"); spn.set_linewidth(0.7)
    return ax


def grid(tiles, ncol, head=0.19):
    """A BODY_W figure of square tiles: (fig, [rect per tile]).

    Never save the result with bbox_inches="tight" -- the tight bbox would refit the
    PNG to its content and \\includegraphics[width=7.16in] would then rescale every
    font in it."""
    nrow = int(np.ceil(len(tiles) / float(ncol)))
    pw_in = BODY_W / ncol
    fig_h = (pw_in + head) * nrow
    fig = plt.figure(figsize=(BODY_W, fig_h))
    rects = []
    for i in range(len(tiles)):
        rr, cc = divmod(i, ncol)
        # a left inset so the first panel's centred title is not clipped at x = 0
        inset = 0.012
        rects.append([inset + cc * (1 - 2 * inset) / float(ncol),
                      1 - (rr + 1) * (pw_in + head) / fig_h,
                      (1 - 2 * inset) / ncol * 0.98, pw_in / fig_h])
    return fig, rects, fig_h


def emit(station, cells, gm, reg, pilot):
    tag = station[4]
    tiles = station_tiles(station, cells, gm, reg, pilot)
    fig, rects, fig_h = grid(tiles, min(NCOL, len(tiles)))
    for (title, n, sq), rect in zip(tiles, rects):
        draw_tile(fig, rect, sq, title if n is None else
                  "%s  —  %d graded row%s read here"
                  % (title, n, "" if n == 1 else "s"))
    out = os.path.join(OUT, "fig_realvideo_%s.png" % tag)
    fig.savefig(out, dpi=300); plt.close(fig)
    print("  -> %s   %.2f x %.2f in\n" % (out, BODY_W, fig_h))


def main():
    os.makedirs(OUT, exist_ok=True)
    pilot = pd.read_csv(os.path.join(PILOT, "pilot_clips_eligible.csv")).set_index("clip")
    cells = pd.read_csv(os.path.join(V, "realvideo_feasibility_cells.csv"))
    gm = pd.read_csv(os.path.join(V, "gate_map.csv"))
    reg = pd.read_csv(os.path.join(V, "paper_registry.csv")).set_index("metric_id")
    assert cells.metric_id.nunique() == 35 and len(cells) == 2800
    for st in STATIONS:
        print("%s  az %d el %d" % (st[0], st[2], st[3]))
        emit(st, cells, gm, reg, pilot)


if __name__ == "__main__":
    main()
