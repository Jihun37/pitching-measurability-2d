"""
Diamond — 검증 데이터 시각화 (실제 OBP 투구 1개)
visualize_3d_2d.py
"""
import os, sys, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import savgol_filter

def _setup_korean_font():
    names = ["Malgun Gothic","AppleGothic","NanumGothic","NanumBarunGothic",
             "Noto Sans CJK KR","Noto Sans KR"]
    paths = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
             "C:/Windows/Fonts/malgun.ttf"]
    for pth in paths:
        if os.path.exists(pth):
            try: fm.fontManager.addfont(pth)
            except Exception: pass
    have = {f.name for f in fm.fontManager.ttflist}
    for n in names:
        if n in have:
            plt.rcParams["font.family"] = n
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["axes.unicode_minus"] = False
    return False

KO = False
def T(ko, en): return en

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import config
import obp_project as O
import metrics as M
from hss_elevation_test import project_cam, hss_sep_series

BONES = [
    ("head", "left_shoulder"), ("head", "right_shoulder"),
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
]
INK="#0E1B33"; TEAL="#0FA3B1"; AMBER="#F2A900"; RED="#E0533D"; MUTE="#64748B"
GREEN="#16A34A"; VIOLET="#7C3AED"


def pick_frame(df, arm, fps, which):
    lead = "left" if arm == "right" else "right"
    rel = M.release_frame(df, arm, fps, M.JOINTS)
    fp = M.foot_plant_frame(df, lead, fps, M.JOINTS, rel)
    if which == "release": return rel, rel, fp
    if which == "footplant": return fp, rel, fp
    return int(which), rel, fp


def draw_3d(joints, f, arm, out_png, title):
    fig = plt.figure(figsize=(6, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    P = {n: joints[n][:, f] for n in joints}
    for a, b in BONES:
        if a in P and b in P:
            xa, ya, za = P[a]; xb, yb, zb = P[b]
            ax.plot([xa, xb], [ya, yb], [za, zb], color=INK, lw=2.2)
    xs=[P[n][0] for n in P]; ys=[P[n][1] for n in P]; zs=[P[n][2] for n in P]
    ax.scatter(xs, ys, zs, color=TEAL, s=22, depthshade=False)
    wr = f"{arm}_wrist"
    if wr in P: ax.scatter(*P[wr], color=RED, s=55, depthshade=False)
    _set_equal_3d(ax, xs, ys, zs)
    ax.set_xlabel(T("X (투구 방향)","X (pitch dir)"), fontsize=9, labelpad=2)
    ax.set_ylabel(T("Y (깊이)","Y (depth)"), fontsize=9, labelpad=2)
    ax.set_zlabel(T("Z (수직)","Z (up)"), fontsize=9, labelpad=2)
    ax.set_title(title, fontsize=13, color=INK, weight="bold", pad=6)
    ax.view_init(elev=10, azim=-60)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _set_equal_3d(ax, xs, ys, zs):
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    cx, cy, cz = xs.mean(), ys.mean(), zs.mean()
    r = max(np.ptp(xs), np.ptp(ys), np.ptp(zs)) / 2 + 0.1
    ax.set_xlim(cx-r, cx+r); ax.set_ylim(cy-r, cy+r); ax.set_zlim(cz-r, cz+r)
    try: ax.set_box_aspect((1, 1, 1))
    except Exception: pass


def draw_3d_gif(joints, f, arm, out_gif):
    fig = plt.figure(figsize=(6, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    P = {n: joints[n][:, f] for n in joints}
    xs=[P[n][0] for n in P]; ys=[P[n][1] for n in P]; zs=[P[n][2] for n in P]

    def render(az):
        ax.cla()
        for a, b in BONES:
            if a in P and b in P:
                xa, ya, za = P[a]; xb, yb, zb = P[b]
                ax.plot([xa, xb], [ya, yb], [za, zb], color=INK, lw=2.2)
        ax.scatter(xs, ys, zs, color=TEAL, s=22, depthshade=False)
        wr = f"{arm}_wrist"
        if wr in P: ax.scatter(*P[wr], color=RED, s=55, depthshade=False)
        _set_equal_3d(ax, xs, ys, zs)
        ax.set_xlabel("X", fontsize=8); ax.set_ylabel("Y", fontsize=8); ax.set_zlabel("Z", fontsize=8)
        ax.set_title(T("3D 모션캡처 (OBP)","3D mocap (OBP)"), fontsize=12, color=INK, weight="bold")
        ax.view_init(elev=10, azim=az)
        ax.grid(True, alpha=0.25)

    frames = np.arange(0, 360, 4)
    anim = FuncAnimation(fig, render, frames=frames, interval=60)
    anim.save(out_gif, writer=PillowWriter(fps=18))
    plt.close(fig)


def _ang(a, b, c):
    v1 = np.array(a) - np.array(b); v2 = np.array(c) - np.array(b)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _arc(ax, center, p1, p2, r, color, label, dx=0, dy=0, fs=15):
    a1 = np.degrees(np.arctan2(p1[1]-center[1], p1[0]-center[0]))
    a2 = np.degrees(np.arctan2(p2[1]-center[1], p2[0]-center[0]))
    if a2 < a1: a1, a2 = a2, a1
    if a2 - a1 > 180: a1, a2 = a2, a1 + 360
    th = np.radians(np.linspace(a1, a2, 40))
    ax.plot(center[0]+r*np.cos(th), center[1]+r*np.sin(th), color=color, lw=2.6)
    mid = np.radians((a1+a2)/2)
    ax.text(center[0]+(r+22)*np.cos(mid)+dx, center[1]+(r+22)*np.sin(mid)+dy,
            label, color=color, fontsize=fs, weight="bold", ha="center", va="center")


def _line_intersect(p1, p2, p3, p4):
    """Intersection of line (p1,p2) with line (p3,p4); None if near-parallel."""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(den) < 1e-9:
        return None
    a = x1*y2 - y1*x2; b = x3*y4 - y3*x4
    return ((a*(x3-x4) - (x1-x2)*b) / den, (a*(y3-y4) - (y1-y2)*b) / den)


FIG_W, FIG_H, FIG_DPI = 6.2, 6.6, 160

def _fit_canvas(ax, margin=0.03):
    target = FIG_W / FIG_H
    fig = ax.figure
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    inverted = y0 > y1
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)
    # grow the window to include text/annotation labels so nothing clips when the
    # axes fills the whole figure
    try:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        for t in ax.texts:
            if not t.get_text():
                continue
            bb = t.get_window_extent(rend)
            (px0, py0) = inv.transform((bb.x0, bb.y0))
            (px1, py1) = inv.transform((bb.x1, bb.y1))
            xmin = min(xmin, px0, px1); xmax = max(xmax, px0, px1)
            ymin = min(ymin, py0, py1); ymax = max(ymax, py0, py1)
    except Exception:
        pass
    cx = (xmin + xmax) / 2; cy = (ymin + ymax) / 2
    w = (xmax - xmin) * (1 + margin); h = (ymax - ymin) * (1 + margin)
    if w / h < target:
        w = h * target
    else:
        h = w / target
    ax.set_xlim(cx - w/2, cx + w/2)
    ax.set_ylim(cy + h/2, cy - h/2) if inverted else ax.set_ylim(cy - h/2, cy + h/2)
    # let the axes fill the figure -> minimal outer padding, size stays fixed
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)


def draw_2d(df, rel, fp, arm, cand, out_png, fps):
    lead = "left" if arm == "right" else "right"
    def pt(j, f): return (df[f"{j}_x"].iloc[f], df[f"{j}_y"].iloc[f])

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for a, b in BONES:
        ax.plot([pt(a, rel)[0], pt(b, rel)[0]], [pt(a, rel)[1], pt(b, rel)[1]],
                color=TEAL, lw=2.4, zorder=2)
    for j in M.JOINTS.values():
        ax.scatter(*pt(j, rel), color=INK, s=18, zorder=3)

    def mid(j1, j2, f): return ((pt(j1,f)[0]+pt(j2,f)[0])/2, (pt(j1,f)[1]+pt(j2,f)[1])/2)
    mh = mid("left_hip","right_hip",rel); ms = mid("left_shoulder","right_shoulder",rel)

    # (1) trunk tilt
    vtop = (mh[0], mh[1] - np.hypot(ms[0]-mh[0], ms[1]-mh[1]))
    ax.plot([mh[0], vtop[0]], [mh[1], vtop[1]], color=MUTE, lw=1.3, ls="--", zorder=1)
    ax.plot([mh[0], ms[0]], [mh[1], ms[1]], color=AMBER, lw=2.6, zorder=2)
    _arc(ax, mh, vtop, ms, 55, AMBER, f"tilt {cand['lateral_trunk_tilt']:.0f}\u00b0", dx=-45)

    # (2) lead knee
    hp, kn, an = pt(f"{lead}_hip",rel), pt(f"{lead}_knee",rel), pt(f"{lead}_ankle",rel)
    ax.plot([hp[0],kn[0]],[hp[1],kn[1]], color=INK, lw=2.6, zorder=2)
    ax.plot([kn[0],an[0]],[kn[1],an[1]], color=INK, lw=2.6, zorder=2)
    _arc(ax, kn, hp, an, 42, RED, f"knee {cand['lead_knee_at_release']:.0f}\u00b0", dx=-15, dy=40, fs=13)

    # (3) knee extension velocity
    sh_v = np.array([an[0]-kn[0], an[1]-kn[1]])
    sh_v = sh_v / (np.linalg.norm(sh_v) + 1e-9)
    sgn = 1.0 if cand['knee_ext_velo_br'] >= 0 else -1.0
    tip = (an[0] + sgn*sh_v[0]*38, an[1] + sgn*sh_v[1]*38)
    ax.annotate("", xy=tip, xytext=an,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0), zorder=4)
    ax.text(kn[0]+50, kn[1]+22,
            f"ext velo {cand['knee_ext_velo_br']:+.0f}\u00b0/s",
            color=RED, fontsize=14, weight="bold", ha="left", va="center")

    # (4) wrist speed
    wxs = df[f"{arm}_wrist_x"].to_numpy(); wys = df[f"{arm}_wrist_y"].to_numpy()
    spd = np.hypot(np.gradient(wxs), np.gradient(wys)) * fps
    f0 = max(0, min(fp, rel) - int(0.10*fps)); f1 = min(len(wxs)-1, rel + int(0.05*fps))
    pk = f0 + int(np.nanargmax(spd[f0:f1+1]))
    ax.plot(wxs[f0:f1+1], wys[f0:f1+1], color=GREEN, lw=1.8, alpha=0.8, zorder=2)
    ax.scatter(wxs[pk], wys[pk], color=GREEN, s=52, zorder=5,
               edgecolors="white", linewidths=1.0)
    _all_y = np.concatenate([df[f"{j}_y"].to_numpy() for j in M.JOINTS.values()])
    _ytop, _ybot = np.nanmin(_all_y), np.nanmax(_all_y)
    _near_top = wys[pk] < _ytop + 0.25*(_ybot - _ytop)
    _ty = wys[pk] - 32 if _near_top else wys[pk] - 32
    _all_x = np.concatenate([df[f"{j}_x"].to_numpy() for j in M.JOINTS.values()])
    _tx = min(wxs[pk] + 16, np.nanmax(_all_x) - 40)
    ax.annotate(f"wrist speed {cand['wrist_speed']:.2f} H/s",
                (wxs[pk], wys[pk]), xytext=(_tx, _ty),
                color=GREEN, fontsize=12, weight="bold", ha="left",
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=1.0))

    # (5) stride
    for a, b in BONES:
        ax.plot([pt(a, fp)[0], pt(b, fp)[0]], [pt(a, fp)[1], pt(b, fp)[1]],
                color=MUTE, lw=1.0, ls=":", alpha=0.5, zorder=1)
    la, ra = pt("left_ankle",fp), pt("right_ankle",fp)
    ax.plot([la[0],ra[0]],[la[1],ra[1]], color=VIOLET, lw=2.6, zorder=2)
    smid = ((la[0]+ra[0])/2, (la[1]+ra[1])/2)
    ax.text(smid[0], smid[1]+48, f"stride {cand['stride_pct_height']:.2f}\u00d7H",
            color=VIOLET, fontsize=15, weight="bold", ha="center")

    # (6) release height: vertical ankle-ground -> wrist at release
    BLUE = "#2563EB"
    _ank_y = np.concatenate([df["left_ankle_y"].to_numpy(), df["right_ankle_y"].to_numpy()])
    ground_y = np.nanmax(_ank_y)
    wpt = pt(f"{arm}_wrist", rel)
    x_dim = max(pt(j, rel)[0] for j in M.JOINTS.values()) + 34   # dimension line to the right of the body
    ax.plot([x_dim, x_dim], [wpt[1], ground_y], color=BLUE, lw=2.0, zorder=2)
    ax.plot([x_dim-9, x_dim+9], [wpt[1], wpt[1]], color=BLUE, lw=1.6, zorder=2)
    ax.plot([x_dim-9, x_dim+9], [ground_y, ground_y], color=BLUE, lw=1.6, zorder=2)
    ax.plot([wpt[0], x_dim], [wpt[1], wpt[1]], color=BLUE, lw=0.8, ls=":", alpha=0.6, zorder=1)
    ax.text(x_dim-13, (wpt[1]+ground_y)/2, f"rel. height {cand['release_height']:.2f}",
            color=BLUE, fontsize=11, weight="bold", ha="center", va="center", rotation=90)

    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    _fit_canvas(ax)
    ax.set_title(T("측면(0°) 2D 투영 + 측정 지표 6종","Lateral (0\u00b0) 2D projection + 6 metrics"),
                 fontsize=16, color=INK, weight="bold", pad=14)
    fig.savefig(out_png, dpi=FIG_DPI)
    plt.close(fig)


def draw_side_footplant(df, fp, arm, cand, out_png):
    """Foot-plant stage figure: the pose at front-foot contact + the one metric
    measured there (stride)."""
    def pt(j, f): return (df[f"{j}_x"].iloc[f], df[f"{j}_y"].iloc[f])
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for a, b in BONES:
        ax.plot([pt(a, fp)[0], pt(b, fp)[0]], [pt(a, fp)[1], pt(b, fp)[1]],
                color=TEAL, lw=2.4, zorder=2)
    for j in M.JOINTS.values():
        ax.scatter(*pt(j, fp), color=INK, s=18, zorder=3)
    la, ra = pt("left_ankle", fp), pt("right_ankle", fp)
    ax.plot([la[0], ra[0]], [la[1], ra[1]], color=VIOLET, lw=2.8, zorder=2)
    smid = ((la[0]+ra[0])/2, (la[1]+ra[1])/2)
    ax.text(smid[0], smid[1]+48, f"stride {cand['stride_pct_height']:.2f}×H",
            color=VIOLET, fontsize=15, weight="bold", ha="center")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    _fit_canvas(ax)
    fig.savefig(out_png, dpi=FIG_DPI); plt.close(fig)


def draw_front_stride_angle(df, fp, arm, val, out_png):
    """Front-view foot-plant figure: stride ANGLE (adopted 2026-07-23), the
    image-plane orientation of the trail->lead ankle line at foot plant. Read
    from the FRONT (az90) where its zone lives; a separate panel from the front
    arm-slot figure because it is measured at a different event (foot plant, not
    release). The printed value is the raw image-plane angle, which differs from
    the OBP convention by a near-constant offset (metrics.stride_angle_2d)."""
    lead = "left" if arm == "right" else "right"
    trail = "right" if lead == "left" else "left"
    def pt(j, f): return (df[f"{j}_x"].iloc[f], df[f"{j}_y"].iloc[f])

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for a, b in BONES:
        ax.plot([pt(a, fp)[0], pt(b, fp)[0]], [pt(a, fp)[1], pt(b, fp)[1]],
                color=TEAL, lw=2.4, zorder=2)
    for j in M.JOINTS.values():
        ax.scatter(*pt(j, fp), color=INK, s=18, zorder=3)

    la, ta = pt(f"{lead}_ankle", fp), pt(f"{trail}_ankle", fp)
    ax.plot([ta[0], la[0]], [ta[1], la[1]], color=VIOLET, lw=3.0, zorder=4)
    ax.scatter(*ta, color=RED, s=45, zorder=5)
    # horizontal reference through the trail ankle (the angle datum)
    L = max(abs(la[0]-ta[0]), 70) * 1.6
    href = (ta[0] + (L if la[0] >= ta[0] else -L), ta[1])
    ax.plot([ta[0], href[0]], [ta[1], href[1]], color=MUTE, lw=1.4, ls="--", zorder=1)
    _arc(ax, ta, href, la, 40, VIOLET, f"stride angle {val:.0f}°",
         dx=-78, dy=30, fs=13)

    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    _fit_canvas(ax)
    fig.savefig(out_png, dpi=FIG_DPI); plt.close(fig)


def draw_side_release(df, rel, arm, cand, out_png, fps):
    """Release stage figure: the pose at ball release + the five metrics measured
    there (trunk tilt, lead knee angle, knee ext. velocity, wrist speed, release
    height)."""
    lead = "left" if arm == "right" else "right"
    def pt(j, f): return (df[f"{j}_x"].iloc[f], df[f"{j}_y"].iloc[f])
    def mid(j1, j2, f): return ((pt(j1,f)[0]+pt(j2,f)[0])/2, (pt(j1,f)[1]+pt(j2,f)[1])/2)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for a, b in BONES:
        ax.plot([pt(a, rel)[0], pt(b, rel)[0]], [pt(a, rel)[1], pt(b, rel)[1]],
                color=TEAL, lw=2.4, zorder=2)
    for j in M.JOINTS.values():
        ax.scatter(*pt(j, rel), color=INK, s=18, zorder=3)

    mh = mid("left_hip", "right_hip", rel); ms = mid("left_shoulder", "right_shoulder", rel)
    vtop = (mh[0], mh[1] - np.hypot(ms[0]-mh[0], ms[1]-mh[1]))
    ax.plot([mh[0], vtop[0]], [mh[1], vtop[1]], color=MUTE, lw=1.3, ls="--", zorder=1)
    ax.plot([mh[0], ms[0]], [mh[1], ms[1]], color=AMBER, lw=2.6, zorder=2)
    _arc(ax, mh, vtop, ms, 55, AMBER, f"tilt {cand['lateral_trunk_tilt']:.0f}°", dx=-72, dy=52)

    hp, kn, an = pt(f"{lead}_hip",rel), pt(f"{lead}_knee",rel), pt(f"{lead}_ankle",rel)
    ax.plot([hp[0],kn[0]], [hp[1],kn[1]], color=INK, lw=2.6, zorder=2)
    ax.plot([kn[0],an[0]], [kn[1],an[1]], color=INK, lw=2.6, zorder=2)
    _arc(ax, kn, hp, an, 42, RED, f"knee {cand['lead_knee_at_release']:.0f}°", dx=-15, dy=40, fs=13)

    sh_v = np.array([an[0]-kn[0], an[1]-kn[1]]); sh_v = sh_v/(np.linalg.norm(sh_v)+1e-9)
    sgn = 1.0 if cand['knee_ext_velo_br'] >= 0 else -1.0
    tip = (an[0]+sgn*sh_v[0]*46, an[1]+sgn*sh_v[1]*46)
    ax.annotate("", xy=tip, xytext=an, arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0), zorder=4)
    # the arrow IS the extension-velocity vector -> label it right at the arrow
    ax.text(tip[0]+12, tip[1], f"ext velo {cand['knee_ext_velo_br']:+.0f}°/s",
            color=RED, fontsize=12, weight="bold", ha="left", va="center")

    wxs = df[f"{arm}_wrist_x"].to_numpy(); wys = df[f"{arm}_wrist_y"].to_numpy()
    spd = np.hypot(np.gradient(wxs), np.gradient(wys)) * fps
    fp = M.foot_plant_frame(df, lead, fps, M.JOINTS, rel)
    f0 = max(0, min(fp, rel) - int(0.10*fps)); f1 = min(len(wxs)-1, rel + int(0.05*fps))
    pk = f0 + int(np.nanargmax(spd[f0:f1+1]))
    ax.plot(wxs[f0:f1+1], wys[f0:f1+1], color=GREEN, lw=1.8, alpha=0.8, zorder=2)
    ax.scatter(wxs[pk], wys[pk], color=GREEN, s=52, zorder=5, edgecolors="white", linewidths=1.0)
    _all_x = np.concatenate([df[f"{j}_x"].to_numpy() for j in M.JOINTS.values()])
    _tx = min(wxs[pk]+16, np.nanmax(_all_x)-40)
    ax.annotate(f"wrist speed {cand['wrist_speed']:.2f} H/s", (wxs[pk], wys[pk]),
                xytext=(_tx, wys[pk]-32), color=GREEN, fontsize=12, weight="bold", ha="left",
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=1.0))

    BLUE = "#2563EB"
    _ank_y = np.concatenate([df["left_ankle_y"].to_numpy(), df["right_ankle_y"].to_numpy()])
    ground_y = np.nanmax(_ank_y)
    wpt = pt(f"{arm}_wrist", rel)
    x_dim = wpt[0]                               # at the wrist's actual x (true drop)
    ax.plot([x_dim, x_dim], [wpt[1], ground_y], color=BLUE, lw=2.2, alpha=0.4, zorder=1)
    ax.plot([x_dim-9, x_dim+9], [ground_y, ground_y], color=BLUE, lw=1.6, alpha=0.4, zorder=1)
    y_lbl = (wpt[1] + ground_y) / 2
    x_lbl = max(pt(j, rel)[0] for j in M.JOINTS.values()) + 18   # out in front (throwing side)
    ax.plot([x_dim, x_lbl], [y_lbl, y_lbl], color=BLUE, lw=0.8, ls=":", alpha=0.35, zorder=1)
    ax.text(x_lbl, y_lbl, f"rel. height {cand['release_height']:.2f}",
            color=BLUE, fontsize=12, weight="bold", ha="left", va="center")

    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    _fit_canvas(ax)
    fig.savefig(out_png, dpi=FIG_DPI); plt.close(fig)


def draw_arm_slot(df, rel, arm, cand, out_png, azimuth):
    def pt(j, f): return (df[f"{j}_x"].iloc[f], df[f"{j}_y"].iloc[f])
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for a, b in BONES:
        ax.plot([pt(a, rel)[0], pt(b, rel)[0]], [pt(a, rel)[1], pt(b, rel)[1]],
                color=TEAL, lw=2.2, zorder=2)
    for j in M.JOINTS.values():
        ax.scatter(*pt(j, rel), color=INK, s=16, zorder=3)
    S, W = pt(f"{arm}_shoulder", rel), pt(f"{arm}_wrist", rel)
    Vtop = (S[0], S[1] - abs(S[1]-W[1]) - 30)
    ax.plot([S[0], W[0]], [S[1], W[1]], color=AMBER, lw=3.0, zorder=4)
    ax.plot([S[0], Vtop[0]], [S[1], Vtop[1]], color=MUTE, lw=1.4, ls="--", zorder=1)
    ax.scatter(*S, color=RED, s=45, zorder=5)
    _arc(ax, S, Vtop, W, 46, AMBER, f"arm slot {cand['arm_slot']:.0f}\u00b0", dy=-26)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    _fit_canvas(ax)
    fig.savefig(out_png, dpi=FIG_DPI); plt.close(fig)


def draw_hss_overhead(joints, arm, fps, out_png):
    """Overhead (90 deg elevation) skeleton showing hip-shoulder separation: the
    angle between the shoulder line and the pelvis line in the transverse plane.
    Same visual language as draw_2d. Coords scaled to the pixel range the _arc
    offsets assume (project_cam uses ppm=1)."""
    lead = "left" if arm == "right" else "right"
    df0 = O.project_view(joints, azimuth_deg=0.0)
    rel = M.release_frame(df0, arm, fps, M.JOINTS)
    fp = M.foot_plant_frame(df0, lead, fps, M.JOINTS, rel)
    dfo = project_cam(joints, 0, 90)
    sep = hss_sep_series(dfo, M.JOINTS)
    for c in dfo.columns:                       # meters -> pixel-ish scale
        dfo[c] = dfo[c] * 300.0
    lo, hi = min(fp, rel), max(fp, rel) + 1
    fr = lo + int(np.nanargmax(np.abs(sep[lo:hi])))
    hss = float(abs(sep[fr]))

    def pt(j): return (dfo[f"{j}_x"].iloc[fr], dfo[f"{j}_y"].iloc[fr])
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for a, b in BONES:
        ax.plot([pt(a)[0], pt(b)[0]], [pt(a)[1], pt(b)[1]],
                color=TEAL, lw=2.0, alpha=0.55, zorder=2)
    for j in M.JOINTS.values():
        ax.scatter(*pt(j), color=INK, s=15, zorder=3)

    lsh, rsh = pt("left_shoulder"), pt("right_shoulder")
    lh, rh = pt("left_hip"), pt("right_hip")
    ax.plot([lsh[0], rsh[0]], [lsh[1], rsh[1]], color=AMBER, lw=3.4, zorder=4)
    ax.plot([lh[0], rh[0]], [lh[1], rh[1]], color=VIOLET, lw=3.4, zorder=4)
    ax.text(rsh[0]+10, rsh[1]-6, "shoulder line", color=AMBER, fontsize=12, weight="bold", va="center", ha="left")
    ax.text(lh[0]-10, lh[1]-6, "pelvis line", color=VIOLET, fontsize=12, weight="bold", va="center", ha="right")

    # draw the separation angle right where the shoulder and pelvis lines actually
    # cross on the skeleton (no translated / extended reference rays). Keep the arc
    # radius smaller than the crossing-to-joint distance so it stays inside the lines.
    X = _line_intersect(lsh, rsh, lh, rh) or ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
    d_sh = np.hypot(rsh[0]-X[0], rsh[1]-X[1]); d_hp = np.hypot(rh[0]-X[0], rh[1]-X[1])
    r_hss = float(np.clip(0.5*min(d_sh, d_hp), 16.0, 46.0))
    _arc(ax, X, rsh, rh, r_hss, RED, "")                 # arc only; label placed above
    ax.text(X[0]+18, X[1]-(r_hss+1), f"HSS {hss:.0f}\u00b0", color=RED, fontsize=15,
            weight="bold", ha="center", va="center")

    # pelvis rotational velocity (peak) - overhead-only transverse metric. Peak of
    # the hip-line yaw rate in the release window; drawn as a rotation arrow on the
    # pelvis line (a velocity, so annotated like the release-still 'ext velo').
    hipang = np.unwrap(np.arctan2(
        dfo["right_hip_y"].to_numpy(float) - dfo["left_hip_y"].to_numpy(float),
        dfo["right_hip_x"].to_numpy(float) - dfo["left_hip_x"].to_numpy(float)))
    w = max(5, int(round(0.05 * fps))); w += (w % 2 == 0)
    w = min(w, len(hipang) - (len(hipang) % 2 == 0))
    prvel = (np.degrees(savgol_filter(hipang, w, 3, deriv=1, delta=1.0 / fps, mode="interp"))
             if w > 3 else np.degrees(np.gradient(hipang) * fps))
    lo2 = max(0, rel - int(0.40 * fps)); hi2 = min(len(prvel) - 1, rel + int(0.05 * fps))
    prv = float(np.nanmax(np.abs(prvel[lo2:hi2 + 1]))) if hi2 >= lo2 else float("nan")

    pc = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
    plen = np.hypot(rh[0] - lh[0], rh[1] - lh[1]) + 1e-9
    dx, dy = (rh[0] - lh[0]) / plen, (rh[1] - lh[1]) / plen
    nx, ny = -dy, dx
    if ny < 0:
        nx, ny = -nx, -ny                       # perpendicular toward screen-down (+y)
    ctr = (pc[0] + nx * 0.7 * plen, pc[1] + ny * 0.7 * plen)
    arc_r = 0.5 * plen
    base = np.arctan2(pc[1] - ctr[1], pc[0] - ctr[0])
    th = base + np.radians(np.linspace(-72, 72, 40))
    axx = ctr[0] + arc_r * np.cos(th); ayy = ctr[1] + arc_r * np.sin(th)
    ax.plot(axx, ayy, color=GREEN, lw=2.4, zorder=5, solid_capstyle="round")
    ax.annotate("", xy=(axx[-1], ayy[-1]), xytext=(axx[-4], ayy[-4]),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.4, mutation_scale=15),
                zorder=5)
    if np.isfinite(prv):
        ax.text(ctr[0], ctr[1] + arc_r + 13, f"pelvis rot. {prv:.0f}\u00b0/s",
                color=GREEN, fontsize=13, weight="bold", ha="center", va="center")

    txs = [lsh[0], rsh[0], lh[0], rh[0]]; tys = [lsh[1], rsh[1], lh[1], rh[1]]
    cx, cy = np.mean(txs), np.mean(tys)
    sp = max(np.ptp(txs), np.ptp(tys)) + 1e-6
    ax.set_xlim(cx-sp*1.7, cx+sp*1.7)
    ax.set_ylim(cy+sp*1.7, cy-sp*1.7)           # inverted y
    ax.set_aspect("equal"); ax.axis("off")
    _fit_canvas(ax)
    fig.savefig(out_png, dpi=FIG_DPI); plt.close(fig)


def auto_pick_c3d():
    import config
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    r = md.iloc[0]
    path = os.path.join(config.OBP_DATA_DIR, "c3d", f"{int(r.user):06d}", r.filename_new)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3d", default=None)
    ap.add_argument("--out", default=os.path.join(config.ROOT, "data", "outputs", "viz"))
    ap.add_argument("--frame", default="release")
    ap.add_argument("--armslot-azim", type=float, default=90.0)
    ap.add_argument("--with-3d", action="store_true", help="also render 3d png + rotating gif")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c3d = a.c3d or auto_pick_c3d()

    joints, fps = O.load_c3d_joints(c3d)
    arm = O.detect_throwing_arm(joints, fps)
    df = O.project_view(joints, azimuth_deg=0.0)
    f, rel, fp = pick_frame(df, arm, fps, a.frame)
    cand = {k: v for k, (v, _) in M.compute_candidates(df, fps=fps, arm=arm).items()}
    print(f"c3d={os.path.basename(c3d)}  arm={arm} fps={fps:.0f}  release=f{rel} footplant=f{fp}")
    print(f"  [0deg]  trunk_tilt={cand['lateral_trunk_tilt']:.1f}  "
          f"lead_knee@rel={cand['lead_knee_at_release']:.1f}  stride%H={cand['stride_pct_height']:.2f}"
          f"  release_height={cand['release_height']:.2f}")

    # station figures, same visual language (fig_metrics_*.png). The side view
    # is split by event stage so each figure shows only what is measured there.
    p_fp = os.path.join(a.out, "fig_metrics_footplant.png")
    p_rel = os.path.join(a.out, "fig_metrics_release.png")
    p_front = os.path.join(a.out, "fig_metrics_front.png")
    p_stridea = os.path.join(a.out, "fig_metrics_stride_angle.png")
    p_over = os.path.join(a.out, "fig_metrics_overhead.png")
    draw_side_footplant(df, fp, arm, cand, p_fp)
    draw_side_release(df, rel, arm, cand, p_rel, fps)

    az = a.armslot_azim
    df_az = O.project_view(joints, azimuth_deg=az)
    rel_az = M.release_frame(df_az, arm, fps, M.JOINTS)
    cand_az = {k: v for k, (v, _) in M.compute_candidates(df_az, fps=fps, arm=arm).items()}
    print(f"  [{az:.0f}deg] arm_slot={cand_az['arm_slot']:.1f}")
    draw_arm_slot(df_az, rel_az, arm, cand_az, p_front, az)

    # stride ANGLE: same front view, but the foot-plant event -- detected with the
    # FRONTAL detector (the side one cannot fire here, see foot_plant_frame_front).
    lead = "left" if arm == "right" else "right"
    fp_az = M.foot_plant_frame(df_az, lead, fps, M.JOINTS, rel_az, view="front")
    sa = M.stride_angle_2d(df_az, arm, fp_az, M.JOINTS)
    print(f"  [{az:.0f}deg] stride_angle={sa:.1f}  (footplant f{fp_az})")
    draw_front_stride_angle(df_az, fp_az, arm, sa, p_stridea)

    draw_hss_overhead(joints, arm, fps, p_over)
    print("saved:", p_fp, p_rel, p_front, p_stridea, p_over, sep="\n  ")

    if a.with_3d:
        name = os.path.splitext(os.path.basename(c3d))[0]
        draw_3d(joints, f, arm, os.path.join(a.out, f"{name}_3d.png"),
                T("3D 모션캡처 (OBP)", "3D mocap (OBP)"))
        draw_3d_gif(joints, f, arm, os.path.join(a.out, f"{name}_3d_rotate.gif"))


if __name__ == "__main__":
    main()