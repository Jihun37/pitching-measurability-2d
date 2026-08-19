"""Fig. 3 of section VIII: overhead hip-shoulder separation, two pelvis sources.

SINGLE COLUMN, PANELS STACKED. The frame is an overhead one: the pitcher is a wide flat
band across the middle and the rest is court. Cropped to the subject each panel is about
2:1, so two of them stacked fit a \\columnwidth float -- a `figure`, not a `figure*`.
Stacking also puts the two hip lines directly above each other, which is the comparison
the figure exists to make. The old full-width side-by-side spent most of its area on
empty court and made the reader's eye travel to compare two lines.

NOTHING IS PAINTED ON THE IMAGE. The old panels carried cv2 text -- a different typeface
from the manuscript, a different one in each panel (they came from two scripts, months
apart), and a "frame 377 / HSS PEAK" banner that belongs in the caption. Every word here
is matplotlib text outside the image.

BOTH NUMBERS ARE READ FROM THEIR CANONICAL FILE AT DRAW TIME, never typed in:
  (a) `hss_overhead_gt.csv`      -- the author's hand-placed endpoints and their angle
  (b) `pilot_clips_eligible.csv` -- `hss_peak_f` / `hss_raw_deg`, the frozen pilot's
      output under the adopted `metrics.hss_peak_overhead` definition
The retired still said 18.0 deg at f377; the frozen pilot says 17.85 at f376. A figure
that carries its numbers as literals cannot notice that, which is how the stale pair
survived into the manuscript.

BOTH PANELS ARE THE SAME FRAME. The retired figure was not: (a) was the manual frame
389 and (b) the automatic pipeline's own HSS peak, 377 -- two different poses, so a
reader could put the difference down to the instant rather than the pelvis. VIII-B ¶3
claims the comparison is controlled to one variable, and only one frame delivers that:
same image, same automatic shoulders, the red line is the only thing that moves.

Setting SAME_FRAME = False restores the two-peak form, an end-to-end comparison of what
each pelvis source reports for the clip: 48.6 against 17.85 deg, the automatic pelvis at
its own best frame. It is the reading most favourable to the automatic pose, it costs
0.8 in of height because the sprawled legs at f376 widen the crop, and ¶3's wording
would have to change with it.

Run:  conda activate diamond
      cd src\\viz
      python fig_overhead_hss.py
"""
import math
import os, re, sys

import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../stage2", "../tests", "../analysis"):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
import metrics as M
from smoother import smooth_coordinates
from validate_overlay import CONNECT, pt
from fig_realvideo_new_metrics_patent import grab_seq, arc_minor
from fig_graded_map import INK

OUT = os.path.join(config.ROOT, "data", "outputs", "viz")
PILOT = os.path.join(config.ROOT, "data", "outputs", "realvideo_pilot")

CLIP = "pitching_overhead_02"
SAME_FRAME = True      # False: draw the automatic panel at its own HSS peak instead
COL_W = 3.50           # IEEE \columnwidth
HEAD_IN = 0.20         # title band per panel
PAD = 0.10             # crop margin, fraction of the subject's larger side

SHOULDER = (60, 200, 255)     # BGR amber -- the SAME automatic shoulders in both panels
HIP = (0, 0, 235)             # BGR red -- the pelvis, whichever source drew it
POSE = (190, 190, 190)
ARC = (60, 200, 255)


def manual_gt():
    """The author's hand-placed pelvis: frame, both endpoints, and its angle."""
    g = pd.read_csv(os.path.join(config.ROOT, "data", "outputs", "hss_overhead_gt.csv"),
                    encoding="utf-8-sig")
    g = g[g["clip"] == CLIP].iloc[0]
    m = re.search(r"L\(([\d.]+),([\d.]+)\)-R\(([\d.]+),([\d.]+)\)\s*sep=(-?[\d.]+)",
                  str(g["user_note"]))
    return (int(g["hss_frame"]),
            (float(m.group(1)), float(m.group(2))),
            (float(m.group(3)), float(m.group(4))),
            abs(float(m.group(5))))


def line_angle(p, q):
    return math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 180.0


def same_anchor_and_length(man, auto):
    """The manual pelvis line on the automatic one's LEFT-HIP end and its length.

    Left is the pitcher's left hip, not the left of the image: from overhead with
    the pitcher folded over, the left hip is the one still visible on the image
    right and the right hip is the occluded one that projects near the head. Both
    pairs arrive as (left_hip, right_hip), so the anchor is simply the first.
    Only the direction of the manual line survives, and that is the whole of what
    its angle is read from."""
    (ax, ay), (bx, by) = auto
    (mx, my), (nx, ny) = man
    k = math.hypot(bx - ax, by - ay) / math.hypot(nx - mx, ny - my)
    out = ((ax, ay), (ax + (nx - mx) * k, ay + (ny - my) * k))
    assert abs(line_angle(*out) - line_angle((mx, my), (nx, ny))) < 1e-6, \
        "the redraw moved the angle the figure reports"
    return out


def draw(img, df, f, lh, rh, crop):
    """Faint pose for context, the automatic shoulder line, the pelvis line, the angle.

    TWO LINES AND NOTHING ELSE. The angle is struck where the shoulder line and the hip
    line actually cross, so nothing is drawn that is not one of the two measured lines.
    An earlier version added a shoulder-direction reference through the pelvis so that
    an arc could always be drawn; it read as a third line of unclear status.

    An arc is struck only where the two DRAWN segments cross, never on their
    extensions. Since 2026-08-18 the labelled pelvis is redrawn on the automatic
    pelvis's anchor and length, and that shortened line no longer reaches the
    shoulder line, so neither panel carries an arc now. The lower one never could:
    at 4.4 deg its lines meet about 1,950 px away, far outside the frame."""
    ls = (int(df["left_shoulder_x"].iloc[f]), int(df["left_shoulder_y"].iloc[f]))
    rs = (int(df["right_shoulder_x"].iloc[f]), int(df["right_shoulder_y"].iloc[f]))
    lh, rh = (int(lh[0]), int(lh[1])), (int(rh[0]), int(rh[1]))
    for a, b in CONNECT:
        try:
            cv2.line(img, pt(df, a, f), pt(df, b, f), POSE, 3, cv2.LINE_AA)
        except Exception:
            pass
    cv2.line(img, ls, rs, SHOULDER, 7, cv2.LINE_AA)
    cv2.line(img, lh, rh, HIP, 7, cv2.LINE_AA)
    for q, c in ((ls, SHOULDER), (rs, SHOULDER), (lh, HIP), (rh, HIP)):
        cv2.circle(img, q, 11, c, -1)
    sh = np.array(rs, float) - np.array(ls, float)
    hp = np.array(rh, float) - np.array(lh, float)
    A = np.array([sh, -hp]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t, u = np.linalg.solve(A, np.array(lh, float) - np.array(ls, float))
    # ON the drawn segments, not on their extensions. Checking only the crop let an
    # arc be struck where the produced lines would have met, which drew a vertex at a
    # point neither line reaches.
    if not (0.0 <= t <= 1.0 and 0.0 <= u <= 1.0):
        return None
    ix, iy = (np.array(ls, float) + t * sh).astype(int)
    xa, ya, w, h = crop
    if not (xa + 60 < ix < xa + w - 60 and ya + 60 < iy < ya + h - 60):
        return None                    # the lines meet outside the panel: no arc
    a_sh = np.degrees(np.arctan2(sh[1], sh[0]))
    a_hp = np.degrees(np.arctan2(hp[1], hp[0]))
    d = (a_sh - a_hp + 540) % 360 - 180
    # The radius is the geometry's, not a constant, and so is the SIDE. The two lines
    # cut four sectors and the angle is the same in the pair of vertical ones, so the
    # arc goes wherever there is more line to sit on. Here that is the far side: the
    # crossing falls near the lower end of the pelvis line, leaving 79 px that way
    # against 115 px back along it. A fixed 110 px radius on the near side, set when
    # the labelled line was 385 px long, ran clean off the end of the 251 px one.
    X = np.array((ix, iy), float)
    near = min(np.hypot(*(np.array(rs, float) - X)), np.hypot(*(np.array(rh, float) - X)))
    far = min(np.hypot(*(np.array(ls, float) - X)), np.hypot(*(np.array(lh, float) - X)))
    flip = 0.0 if near >= far else 180.0
    a_hp += flip
    rad = int(max(min(0.62 * max(near, far), 110), 16))
    cv2.ellipse(img, (int(ix), int(iy)), (rad, rad), 0, a_hp, a_hp + d, ARC, 5,
                cv2.LINE_AA)
    # where the value goes: outward along the bisector, clear of both lines
    bis = np.radians(a_hp + d / 2.0)
    lab = (ix + 2.35 * rad * np.cos(bis), iy + 2.35 * rad * np.sin(bis))
    return (int(ix), int(iy)), (float(lab[0]), float(lab[1]))


def main():
    os.makedirs(OUT, exist_ok=True)
    pilot = pd.read_csv(os.path.join(PILOT, "pilot_clips_eligible.csv")).set_index("clip")
    r = pilot.loc[CLIP]
    f_man, lh, rh, val_man = manual_gt()
    f_auto = f_man if SAME_FRAME else int(r.hss_peak_f)
    df = smooth_coordinates(pd.read_csv(os.path.join(
        config.ROOT, "data", "outputs", CLIP, CLIP + "_coords_rtmp_refined.csv")))

    R = M.hss_peak_overhead(df, float(r.fps), M.JOINTS)
    assert R["peak_f"] == int(r.hss_peak_f) and abs(R["hss"] - float(r.hss_raw_deg)) < 0.05, \
        "current code gives %.2f @ f%d, the frozen pilot %.2f @ f%d" % (
            R["hss"], R["peak_f"], r.hss_raw_deg, r.hss_peak_f)
    val_auto = abs(float(R["sep_f"][f_auto]))
    a_lh = (df["left_hip_x"].iloc[f_auto], df["left_hip_y"].iloc[f_auto])
    a_rh = (df["right_hip_x"].iloc[f_auto], df["right_hip_y"].iloc[f_auto])
    print("%s  fps %.2f" % (CLIP, r.fps))
    print("  (a) hand-labelled pelvis   f%-4d  %.1f deg" % (f_man, val_man))
    print("  (b) automatic pose         f%-4d  %.1f deg%s"
          % (f_auto, val_auto, "" if SAME_FRAME else "   (its own HSS peak)"))

    # The labelled pelvis is REDRAWN, not relabelled: same direction, but anchored on
    # the automatic pelvis's left-hand endpoint and cut to its length, so the panels
    # differ in direction alone. As labelled the two lines shared neither end nor
    # length and the comparison read as one of position.
    lh, rh = same_anchor_and_length((lh, rh), (a_lh, a_rh))

    # no "(a)"/"(b)": the panels are named, as they are in fig_anchor_orientation, so a
    # caption refers to them by name rather than by a letter the reader has to map
    # the title names the panel, the value sits on it beside its own angle
    panels = [(f_man, lh, rh, val_man, "hand-labelled pelvis"),
              (f_auto, a_lh, a_rh, val_auto, "automatic pose")]

    # A COMMON WINDOW SIZE, CENTRED PER PANEL. One shared window would have to span the
    # union of two frames the pitcher has moved between, and both panels would then
    # carry the other's empty court -- that cost 1.4 in of figure height. Same size means
    # the same scale, which is all the comparison needs; the subject sits in the middle
    # of each panel instead.
    boxes = []
    for f, hl, hr, _, _ in panels:
        xs, ys = [hl[0], hr[0]], [hl[1], hr[1]]
        for a, b in CONNECT:
            for j in (a, b):
                x, y = df[j + "_x"].iloc[f], df[j + "_y"].iloc[f]
                if np.isfinite([x, y]).all():
                    xs += [float(x)]; ys += [float(y)]
        boxes.append((min(xs), max(xs), min(ys), max(ys)))
    pad = PAD * max(max(b[1] - b[0], b[3] - b[2]) for b in boxes)
    win_w = int(max(b[1] - b[0] for b in boxes) + 2 * pad)
    win_h = int(max(b[3] - b[2] for b in boxes) + 2 * pad)

    imgs = []
    for (f, hl, hr, val, title), b in zip(panels, boxes):
        img = grab_seq(r.video_path, f)
        H, W = img.shape[:2]
        w, h = min(win_w, W), min(win_h, H)
        xa = int(np.clip((b[0] + b[1]) / 2 - w / 2, 0, W - w))
        ya = int(np.clip((b[2] + b[3]) / 2 - h / 2, 0, H - h))
        # the crop goes IN, because whether the two lines meet inside it decides
        # whether there is an angle to draw at all
        got = draw(img, df, f, hl, hr, (xa, ya, w, h))
        print("     arc at (%d, %d)" % got[0] if got else "     no arc: the lines "
              "meet outside the panel")
        # panel coordinates for the value. With an arc it goes beside it; without
        # one there is no vertex to sit at, so it goes under the pelvis line.
        if got:
            lx, ly = got[1][0] - xa, got[1][1] - ya
        else:
            lx = (hl[0] + hr[0]) / 2.0 - xa
            ly = (hl[1] + hr[1]) / 2.0 - ya + 0.055 * h
        imgs.append((img[ya:ya + h, xa:xa + w], title, val, lx, ly))

    ph = COL_W * imgs[0][0].shape[0] / float(imgs[0][0].shape[1])
    fig_h = 2 * (ph + HEAD_IN)
    fig = plt.figure(figsize=(COL_W, fig_h))
    for i, (sq, title, val, lx, ly) in enumerate(imgs):
        ax = fig.add_axes([0.0, 1 - (i + 1) * (ph + HEAD_IN) / fig_h, 1.0, ph / fig_h])
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(sq[:, :, ::-1], aspect="auto")
        ax.set_title(title, fontsize=8, color=INK, weight="bold", pad=2.6)
        # The value sits on the panel, beside the angle it measures. Drawn as
        # matplotlib text over the axes rather than painted into the frame, so the
        # typeface is the manuscript's and the rule that nothing is baked into the
        # image holds.
        ax.text(lx, ly, "%.1f°" % val, fontsize=9, color="#FFC83C",
                weight="bold", ha="center", va="center", zorder=5)
        for spn in ax.spines.values():
            spn.set_color("0.80"); spn.set_linewidth(0.7)
    out = os.path.join(OUT, "fig_overhead_hss.png")
    fig.savefig(out, dpi=300)          # never bbox_inches="tight"
    print("-> %s   %.2f x %.2f in" % (out, COL_W, fig_h))


if __name__ == "__main__":
    main()
