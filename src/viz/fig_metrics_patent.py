"""Patent-only extension of the patent Fig. 6 metric-overlay panels. Adds the three metrics
that the paper Fig 3 generator (visualize_3d_2d.py) does not draw, WITHOUT
touching that script or its PNGs:

  - Release Ext (forward reach, statures) -> added to the release panel
    (fig_metrics_release_patent.png), as a horizontal ground dimension from the
    trail-foot setup position to the release point (the orthogonal partner of
    release height).
  - COG Fwd Velo (peak) + COG Velo @PKH -> a NEW panel
    (fig_metrics_cog_patent.png): the pose at peak knee height, the whole-body
    COM trajectory, and a forward-velocity arrow at each of the two read frames.

The footplant / front / overhead panels are unchanged -> reuse the paper PNGs.
Same example c3d and visual language as visualize_3d_2d.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import visualize_3d_2d as V
from visualize_3d_2d import (BONES, INK, TEAL, AMBER, RED, MUTE, GREEN, VIOLET,
                             FIG_W, FIG_H, FIG_DPI, _arc, _fit_canvas)
import config
import obp_project as O
import metrics as M

BLUE = "#2563EB"
PLUM = "#DB2777"     # release-ext dimension (magenta, deliberately NOT the
                     # violet used for STRIDE in the foot-plant panel -- the two
                     # are both horizontal ground dimensions and read as the
                     # same metric if they share a colour)
ORANGE = "#EA580C"   # COG @PKH
LAT = "#C026D3"      # torso lateral tilt @MER (fuchsia, distinct from anterior tilt)


def draw_side_release_patent(df, rel, arm, cand, out_png, fps):
    """The paper release panel + one added metric: Release Ext, drawn as a
    horizontal ground dimension (trail-foot setup x -> release point x)."""
    lead = "left" if arm == "right" else "right"
    trail = "right" if lead == "left" else "left"
    def pt(j, f): return (df[f"{j}_x"].iloc[f], df[f"{j}_y"].iloc[f])
    def mid(j1, j2, f): return ((pt(j1, f)[0]+pt(j2, f)[0])/2, (pt(j1, f)[1]+pt(j2, f)[1])/2)

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

    hp, kn, an = pt(f"{lead}_hip", rel), pt(f"{lead}_knee", rel), pt(f"{lead}_ankle", rel)
    ax.plot([hp[0], kn[0]], [hp[1], kn[1]], color=INK, lw=2.6, zorder=2)
    ax.plot([kn[0], an[0]], [kn[1], an[1]], color=INK, lw=2.6, zorder=2)
    _arc(ax, kn, hp, an, 42, RED, f"knee {cand['lead_knee_at_release']:.0f}°", dx=-15, dy=40, fs=13)

    # Knee extension velocity was DE-ADOPTED 2026-07-24 (event-precision wall);
    # its arrow/label is intentionally not drawn here.
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

    _ank_y = np.concatenate([df["left_ankle_y"].to_numpy(), df["right_ankle_y"].to_numpy()])
    ground_y = np.nanmax(_ank_y)
    wpt = pt(f"{arm}_wrist", rel)
    x_dim = wpt[0]
    ax.plot([x_dim, x_dim], [wpt[1], ground_y], color=BLUE, lw=2.2, alpha=0.4, zorder=1)
    ax.plot([x_dim-9, x_dim+9], [ground_y, ground_y], color=BLUE, lw=1.6, alpha=0.4, zorder=1)
    y_lbl = (wpt[1] + ground_y) / 2
    x_lbl = max(pt(j, rel)[0] for j in M.JOINTS.values()) + 18
    ax.plot([x_dim, x_lbl], [y_lbl, y_lbl], color=BLUE, lw=0.8, ls=":", alpha=0.35, zorder=1)
    ax.text(x_lbl, y_lbl, f"rel. height {cand['release_height']:.2f}",
            color=BLUE, fontsize=12, weight="bold", ha="left", va="center")

    # --- ADDED: Release Ext = horizontal ground dimension, setup foot -> release ---
    trail_ax = df[f"{trail}_ankle_x"].to_numpy()
    setup_x = float(M.trail_anchor_x(trail_ax, fp, fps))
    y_ext = ground_y + 34
    for xx in (setup_x, x_dim):
        ax.plot([xx, xx], [y_ext-9, y_ext+9], color=PLUM, lw=1.6, zorder=3)
    ax.annotate("", xy=(x_dim, y_ext), xytext=(setup_x, y_ext),
                arrowprops=dict(arrowstyle="<|-|>", color=PLUM, lw=2.0), zorder=3)
    ax.text((setup_x+x_dim)/2, y_ext+22, f"ext {cand['release_ext']:.2f} m",
            color=PLUM, fontsize=12, weight="bold", ha="center", va="center")

    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    _fit_canvas(ax)
    fig.savefig(out_png, dpi=FIG_DPI); plt.close(fig)


def draw_cog_panel(df, arm, fps, cand, out_png):
    """NEW patent panel: whole-body COM forward velocity. Pose at peak knee
    height, the COM trajectory over [0, release], and a forward-velocity arrow at
    the two read frames -- COG Velo @PKH (balance point) and COG Fwd Velo (peak)."""
    lead = "left" if arm == "right" else "right"
    def pt(j, f): return (df[f"{j}_x"].iloc[f], df[f"{j}_y"].iloc[f])

    rel = M.release_frame(df, arm, fps, M.JOINTS)
    fp = M.foot_plant_frame(df, lead, fps, M.JOINTS, rel)
    pkh = M.peak_knee_height_frame(df, lead, fp, M.JOINTS)
    comx, comy = M.body_com(df, M.JOINTS)
    stat = M.pixel_stature(df, M.JOINTS)
    v = np.abs(np.gradient(comx)) * fps / stat
    pk = int(np.nanargmax(v[:rel+1]))                    # COG-fwd-velo peak frame

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    # skeleton at peak knee height (the balance point)
    for a, b in BONES:
        ax.plot([pt(a, pkh)[0], pt(b, pkh)[0]], [pt(a, pkh)[1], pt(b, pkh)[1]],
                color=TEAL, lw=2.4, zorder=2)
    for j in M.JOINTS.values():
        ax.scatter(*pt(j, pkh), color=INK, s=18, zorder=3)

    # COM trajectory windup -> release
    ax.plot(comx[:rel+1], comy[:rel+1], color=MUTE, lw=1.6, ls="--", alpha=0.7, zorder=1)

    def com_arrow(f, color, label, tx, ty, ha):
        cx, cy = comx[f], comy[f]
        ax.scatter(cx, cy, color=color, s=70, zorder=5, edgecolors="white", linewidths=1.2)
        L = 46
        ax.annotate("", xy=(cx+L, cy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2), zorder=5)
        ax.text(cx+tx, cy+ty, label, color=color, fontsize=12, weight="bold",
                ha=ha, va="center")

    # the peak-velocity point sits far to the right of the frame, so its label
    # goes BELOW the arrow (ha=center) instead of trailing off the canvas.
    # @PKH label sits ABOVE its arrow: at the balance point the COM is inside the
    # torso, so a label trailing to the right runs straight through the arm bones.
    com_arrow(pkh, ORANGE, f"COG@PKH {cand['cog_velo_pkh']:.2f} m/s",
              tx=23, ty=-58, ha="center")
    com_arrow(pk, VIOLET, f"COG fwd velo {cand['cog_fwd_velo']:.2f} m/s (max)",
              tx=23, ty=44, ha="center")

    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    _fit_canvas(ax)
    fig.savefig(out_png, dpi=FIG_DPI); plt.close(fig)


def draw_torso_lat_tilt(joints, arm, fps, out_png):
    """NEW patent panel (adopted 2026-07-24): torso LATERAL (coronal) tilt @MER.
    Read from the front, slightly elevated (az90/el30), at the MER proxy frame
    (release - 11 frames). Same trunk-lean observable as the anterior tilt, but the
    front view makes it the coronal lean; drawn in a distinct colour so it does not
    read as the amber anterior-tilt metric of the release panel."""
    df0 = O.project_view(joints, azimuth_deg=0.0)
    rel = M.release_frame(df0, arm, fps, M.JOINTS)
    mer = max(0, int(round(rel - (11.0 / 360.0) * fps)))
    dfv = O.project_view(joints, azimuth_deg=90.0, elevation_deg=30.0)
    val = M.trunk_lean_2d(dfv, mer, M.JOINTS)
    def pt(j): return (dfv[f"{j}_x"].iloc[mer], dfv[f"{j}_y"].iloc[mer])
    def mid(a, b): return ((pt(a)[0] + pt(b)[0]) / 2, (pt(a)[1] + pt(b)[1]) / 2)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for a, b in BONES:
        ax.plot([pt(a)[0], pt(b)[0]], [pt(a)[1], pt(b)[1]], color=TEAL, lw=2.4, zorder=2)
    for j in M.JOINTS.values():
        ax.scatter(*pt(j), color=INK, s=18, zorder=3)
    mh = mid("left_hip", "right_hip"); ms = mid("left_shoulder", "right_shoulder")
    vtop = (mh[0], mh[1] - np.hypot(ms[0] - mh[0], ms[1] - mh[1]))
    ax.plot([mh[0], vtop[0]], [mh[1], vtop[1]], color=MUTE, lw=1.3, ls="--", zorder=1)
    ax.plot([mh[0], ms[0]], [mh[1], ms[1]], color=LAT, lw=2.8, zorder=2)
    _arc(ax, mh, vtop, ms, 55, LAT, f"lat tilt {val:.0f}°", dx=44, dy=44)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    _fit_canvas(ax)
    fig.savefig(out_png, dpi=FIG_DPI); plt.close(fig)


def draw_glove_sh_abd(joints, arm, fps, out_png):
    """NEW patent panel (adopted 2026-07-24): glove-arm shoulder abduction @MER =
    the elbow-shoulder-hip angle on the glove (non-throwing) side, from the front
    (az75/el15) at the MER proxy frame (release - 11 frames)."""
    df0 = O.project_view(joints, azimuth_deg=0.0)
    rel = M.release_frame(df0, arm, fps, M.JOINTS)
    mer = max(0, int(round(rel - (11.0 / 360.0) * fps)))
    dfv = O.project_view(joints, azimuth_deg=75.0, elevation_deg=15.0)
    val = M.shoulder_abduction_2d(dfv, "glove", mer, M.JOINTS)
    s = "left" if arm == "right" else "right"          # glove = non-throwing side
    def pt(j): return (dfv[f"{j}_x"].iloc[mer], dfv[f"{j}_y"].iloc[mer])

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for a, b in BONES:
        ax.plot([pt(a)[0], pt(b)[0]], [pt(a)[1], pt(b)[1]], color=TEAL, lw=2.4, zorder=2)
    for j in M.JOINTS.values():
        ax.scatter(*pt(j), color=INK, s=18, zorder=3)
    E, S, H = pt(f"{s}_elbow"), pt(f"{s}_shoulder"), pt(f"{s}_hip")
    ax.plot([S[0], E[0]], [S[1], E[1]], color=BLUE, lw=2.8, zorder=4)
    ax.plot([S[0], H[0]], [S[1], H[1]], color=BLUE, lw=2.8, zorder=4)
    ax.scatter(*S, color=RED, s=45, zorder=5)
    _arc(ax, S, E, H, 40, BLUE, f"glove abd {val:.0f}°", dx=-46, dy=-30, fs=13)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    _fit_canvas(ax)
    fig.savefig(out_png, dpi=FIG_DPI); plt.close(fig)


def draw_torso_rot(joints, arm, fps, out_png):
    """NEW patent panel (adopted 2026-07-24): torso transverse rotation @release,
    recovered from OVERHEAD (az90/el85). The image-plane orientation of the
    shoulder line vs a horizontal datum; the printed angle is the raw image-plane
    value (a near-constant convention offset from the OBP column, like stride
    angle -- calibrated per pitcher in deployment)."""
    df0 = O.project_view(joints, azimuth_deg=0.0)
    rel = M.release_frame(df0, arm, fps, M.JOINTS)
    dfv = O.project_view(joints, azimuth_deg=90.0, elevation_deg=85.0)
    def pt(j): return (dfv[f"{j}_x"].iloc[rel], dfv[f"{j}_y"].iloc[rel])

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for a, b in BONES:
        ax.plot([pt(a)[0], pt(b)[0]], [pt(a)[1], pt(b)[1]], color=TEAL, lw=2.0,
                alpha=0.6, zorder=2)
    for j in M.JOINTS.values():
        ax.scatter(*pt(j), color=INK, s=15, zorder=3)
    ls, rs = pt("left_shoulder"), pt("right_shoulder")
    smid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    L = np.hypot(rs[0] - ls[0], rs[1] - ls[1]) * 0.7 + 34
    ax.plot([smid[0] - L, smid[0] + L], [smid[1], smid[1]], color=MUTE, lw=1.4,
            ls="--", zorder=1)
    ax.plot([ls[0], rs[0]], [ls[1], rs[1]], color=AMBER, lw=3.4, zorder=4)
    # acute image-plane angle between the shoulder line and the horizontal datum
    # (raw value; a near-constant convention offset to the OBP column, calibrated
    # per pitcher in deployment, like stride angle). Minor arc to the +x shoulder ray.
    raw = np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0])) % 180.0
    acute = raw if raw <= 90 else 180.0 - raw
    sr = rs if rs[0] >= ls[0] else ls
    a_s = np.arctan2(sr[1] - smid[1], sr[0] - smid[0])
    d = (a_s + np.pi) % (2 * np.pi) - np.pi            # signed minor diff from +x
    r = 50
    th = np.linspace(0.0, d, 40)
    ax.plot(smid[0] + r * np.cos(th), smid[1] + r * np.sin(th), color=RED, lw=2.6, zorder=5)
    mt = d / 2.0
    ax.text(smid[0] + (r + 34) * np.cos(mt), smid[1] + (r + 34) * np.sin(mt),
            f"torso rot {acute:.0f}°", color=RED, fontsize=13, weight="bold",
            ha="center", va="center")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    _fit_canvas(ax)
    fig.savefig(out_png, dpi=FIG_DPI); plt.close(fig)


def main():
    out = os.path.join(config.ROOT, "data", "outputs", "viz")
    c3d = V.auto_pick_c3d()
    # release_ext / cog_* are absolute (stature-scaled) metrics -> they return
    # NaN unless the subject height is supplied, so read it from the metadata
    # row that auto_pick_c3d selected.
    import pandas as pd
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    row = md[md["filename_new"] == os.path.basename(c3d)]
    height_m = float(row["session_height_m"].iloc[0]) if len(row) else None
    joints, fps = O.load_c3d_joints(c3d)
    arm = O.detect_throwing_arm(joints, fps)
    df = O.project_view(joints, azimuth_deg=0.0)
    rel = M.release_frame(df, arm, fps, M.JOINTS)
    cand = {k: v for k, (v, _) in
            M.compute_candidates(df, fps=fps, arm=arm, height_m=height_m).items()}
    print(f"c3d={os.path.basename(c3d)} rel=f{rel} "
          f"release_ext={cand['release_ext']:.2f} cog_fwd={cand['cog_fwd_velo']:.2f} "
          f"cog_pkh={cand['cog_velo_pkh']:.2f}")

    p_rel = os.path.join(out, "fig_metrics_release_patent.png")
    p_cog = os.path.join(out, "fig_metrics_cog_patent.png")
    p_lat = os.path.join(out, "fig_metrics_torso_lat_tilt.png")
    p_abd = os.path.join(out, "fig_metrics_glove_sh_abd.png")
    p_rot = os.path.join(out, "fig_metrics_torso_rot.png")
    draw_side_release_patent(df, rel, arm, cand, p_rel, fps)
    draw_cog_panel(df, arm, fps, cand, p_cog)
    draw_torso_lat_tilt(joints, arm, fps, p_lat)
    draw_glove_sh_abd(joints, arm, fps, p_abd)
    draw_torso_rot(joints, arm, fps, p_rot)
    print("saved:", p_rel, p_cog, p_lat, p_abd, p_rot, sep="\n  ")


if __name__ == "__main__":
    main()
