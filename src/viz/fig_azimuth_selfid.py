"""
Figure (도7): camera-VIEWPOINT self-ID features (구성 B).

A single 2D pose carries the geometry needed to recover its own camera viewpoint
on BOTH axes:
  - AZIMUTH: projected shoulder-line width, hip-line width and foot separation
    shrink monotonically from side (0 deg) to front (90 deg).
  - ELEVATION: vertical body extent (torso/shoulder ratio, head-ankle height,
    bounding-box aspect) shrinks monotonically as the camera rises from ground
    (0 deg) to overhead (85 deg) - foreshortening of the standing body.

Left: two feature panels (mean +/- 1 SD over OBP). Right: the same release pose
projected to the two extremes of each axis. Annotated with the measured self-ID
accuracy (azimuth_classify.py: 3-class 95.5%, azimuth MAE 3.8 deg at el=0;
viewpoint_zone_classify.py: elevation median error 0 deg, mean 5.3 deg, n=411).
NOTE these are HARDCODED display strings -- re-render alone will NOT refresh
them. Re-run azimuth_classify.py / viewpoint_zone_classify.py and copy the
printed values here whenever the bank or the adopted set changes.

Run:
  conda activate diamond
  cd src\\viz
  python fig_azimuth_selfid.py
"""
import os, sys, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(__file__)
for p in ("..", "../stage2", "../stage3", "../analysis"):
    sys.path.insert(0, os.path.join(HERE, p))
import config
import obp_project as O
import metrics as M
from hss_elevation_test import project_cam

INK = "#0E1B33"; TEAL = "#0FA3B1"; AMBER = "#F2A900"; VIOLET = "#7C3AED"; GRAY = "#64748B"
BONES = [("head", "left_shoulder"), ("head", "right_shoulder"),
         ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
         ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
         ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")]

AZ_FEATS = [("foot_sep",  "foot separation", TEAL),
            ("sh_w_med",  "shoulder-line width", AMBER),
            ("hip_w_med", "hip-line width", VIOLET)]
EL_FEATS = [("torso", "torso height / shoulder", TEAL),
            ("headv", "head-ankle height", AMBER),
            ("bbox",  "bounding-box aspect", VIOLET)]

# --deployed: azimuth curves from the DEPLOYED kNN ratio features (the frozen
# OBP bank used by measure_auto; station_classify.FEATS). shhip (hip/shoulder
# ratio) was dropped 2026-07-04 for backbone-convention sensitivity, and raw
# shoulder/hip line widths are only NORMALIZERS there — so the legacy panel
# above must not be quoted as the deployed feature set.
AZ_FEATS_DEP = [("footsep", "foot sep / shoulder width", TEAL),
                ("stride",  "stride range / shoulder width", AMBER),
                ("wrist",   "wrist travel aspect", VIOLET)]


def draw_skeleton(ax, joints, fr, az, el, title, mode):
    df = project_cam(joints, az, el)

    def pt(j):
        return (df[f"{j}_x"].iloc[fr], df[f"{j}_y"].iloc[fr])
    for a, b in BONES:
        ax.plot([pt(a)[0], pt(b)[0]], [pt(a)[1], pt(b)[1]],
                color=TEAL, lw=1.7, alpha=0.55, zorder=2)
    if mode == "az":                       # azimuth cue: shoulder & hip widths
        ls, rs = pt("left_shoulder"), pt("right_shoulder")
        lh, rh = pt("left_hip"), pt("right_hip")
        ax.plot([ls[0], rs[0]], [ls[1], rs[1]], color=AMBER, lw=3.2, zorder=4)
        ax.plot([lh[0], rh[0]], [lh[1], rh[1]], color=VIOLET, lw=3.2, zorder=4)
    elif mode == "az_dep":                 # deployed azimuth cue: foot sep
        la, ra = pt("left_ankle"), pt("right_ankle")
        ls, rs = pt("left_shoulder"), pt("right_shoulder")
        ax.plot([la[0], ra[0]], [la[1], ra[1]], color=TEAL, lw=3.4, zorder=4)
        ax.plot([ls[0], rs[0]], [ls[1], rs[1]], color=GRAY, lw=2.6, zorder=4,
                ls=(0, (4, 2)))            # shoulder width = the normaliser
    else:                                  # elevation cue: vertical torso extent
        sm = tuple((np.array(pt("left_shoulder")) + pt("right_shoulder")) / 2)
        hm = tuple((np.array(pt("left_hip")) + pt("right_hip")) / 2)
        ax.plot([sm[0], hm[0]], [sm[1], hm[1]], color=AMBER, lw=3.4, zorder=4)
    for j in M.JOINTS.values():
        ax.scatter(*pt(j), color=INK, s=7, zorder=3)
    ax.set_title(title, fontsize=10, color=INK, fontweight="medium", pad=3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.invert_yaxis()          # project_cam y is image-down -> flip for upright


def feature_panel(ax, xs, g, feats, xlabel, note, ymax=1.2,
                  note_pos=(0.98, 0.94), note_va="top", xticks=None,
                  logy=False):
    for key, label, col in feats:
        m = g[key].mean(); s = g[key].std()
        base = m.loc[xs[0]]
        ax.plot(xs, m.loc[xs] / base, color=col, lw=2.4, marker="o", ms=5,
                label=label, zorder=3)
        ax.fill_between(xs, (m.loc[xs] - s.loc[xs]) / base,
                        (m.loc[xs] + s.loc[xs]) / base, color=col,
                        alpha=0.13, zorder=1)
    ax.set_xticks(xticks if xticks is not None else xs)
    ax.set_xlabel(xlabel, fontsize=10.5)
    ax.set_ylabel("feature magnitude\n(rel. to first view)", fontsize=10)
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(0.35, ymax)
        ax.set_yticks([0.5, 1, 2, 4])
        ax.set_yticklabels(["0.5", "1", "2", "4"])
    else:
        ax.set_ylim(0, ymax)
    ax.grid(True, color="#E3E1D8", lw=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    leg = ax.legend(loc="lower left", frameon=True, fontsize=9.5,
                    facecolor="white", edgecolor="none", framealpha=0.88)
    leg.set_zorder(7)
    ax.text(*note_pos, note, transform=ax.transAxes, ha="right", va=note_va,
            fontsize=9.8, color=INK, zorder=8,
            bbox=dict(boxstyle="round,pad=0.45", fc="#F2F5FB", ec="#B7C6DF",
                      lw=1, alpha=1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployed", action="store_true",
                    help="azimuth curves from the deployed kNN ratio "
                         "features (frozen OBP bank) instead of the legacy "
                         "raw-width study")
    a = ap.parse_args()

    bank = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR,
                                    "viewpoint_zone_features.csv"))
    if a.deployed:
        # FULL 360-deg orbit: the deployed estimator works on 15-deg bins all
        # around, and the ~180-deg periodicity visible here is exactly the
        # pose symmetry the sign bits must break (lower panel of 도3).
        az_csv = bank[(bank["aug"] == 0) & (bank["el"] == 0)]
        az_feats = AZ_FEATS_DEP
        az_note = ("deployed classifier features (scale-free ratios,\n"
                   "OBP bank, el 0°) — near-180° periodicity\n"
                   "= the symmetry the sign bits break")
        az_mode = "az_dep"
        az_ymax = 5.2
        out_name = "fig_azimuth_selfid_deployed.png"
    else:
        az_csv = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR,
                                          "azimuth_classify_features.csv"))
        az_feats = AZ_FEATS
        az_note = "azimuth recoverable\n3-class 95.5%  ·  MAE 3.8° (el 0°)"
        az_mode = "az"
        az_ymax = 1.2
        out_name = "fig_azimuth_selfid.png"
    el_csv = bank
    # fix azimuth (side view) so the elevation SD band reflects pitcher-to-
    # pitcher variation only, not azimuth spread - comparable to the az panel.
    el_csv = el_csv[(el_csv["aug"] == 0) & (el_csv["az"] == 0)]
    az_x = sorted(az_csv["az"].unique())
    el_x = sorted(el_csv["el"].unique())

    fig = plt.figure(figsize=(13, 6.6))
    gs = GridSpec(2, 3, width_ratios=[2.5, 0.72, 0.72],
                  height_ratios=[1, 1], wspace=0.05, hspace=0.42)
    axAz = fig.add_subplot(gs[0, 0])
    axEl = fig.add_subplot(gs[1, 0])
    ax_side = fig.add_subplot(gs[0, 1]); ax_front = fig.add_subplot(gs[0, 2])
    ax_gnd = fig.add_subplot(gs[1, 1]);  ax_over = fig.add_subplot(gs[1, 2])

    az_np = (0.98, 0.06) if a.deployed else (0.98, 0.94)
    az_xt = list(range(0, 360, 45)) if a.deployed else None
    az_xl = ("camera azimuth (deg)      0° = side · 90° = front · "
             "180° = far side · 270° = rear" if a.deployed else
             "camera azimuth (deg)      0° = side · 90° = front")
    feature_panel(axAz, az_x, az_csv.groupby("az"), az_feats,
                  az_xl, az_note, ymax=az_ymax, note_pos=az_np,
                  note_va="bottom" if a.deployed else "top", xticks=az_xt,
                  logy=a.deployed)
    if a.deployed:
        for gx in (90, 180, 270):
            axAz.axvline(gx, color="#B7C6DF", lw=0.9, ls=(0, (3, 3)), zorder=0)
    feature_panel(axEl, el_x, el_csv.groupby("el"), EL_FEATS,
                  "camera elevation (deg)      0° = ground · 85° = overhead",
                  "elevation recoverable\nmedian err 0°  ·  mean 5.3° (n=411)")

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    r = md.iloc[0]
    path = os.path.join(config.OBP_DATA_DIR, "c3d", f"{int(r.user):06d}", r.filename_new)
    joints, fps = O.load_c3d_joints(path)
    arm = O.detect_throwing_arm(joints, fps)
    fr = M.release_frame(O.project_view(joints, azimuth_deg=0.0), arm, fps, M.JOINTS)

    draw_skeleton(ax_side,  joints, fr, 0,  0,  "side (0°): wide",     az_mode)
    draw_skeleton(ax_front, joints, fr, 90, 0,  "front (90°): narrow", az_mode)
    draw_skeleton(ax_gnd,   joints, fr, 0,  0,  "ground (0°): tall",   "el")
    draw_skeleton(ax_over,  joints, fr, 0,  85, "overhead (85°): flat", "el")

    fig.suptitle("A single 2D pose self-identifies its camera viewpoint "
                 "(azimuth + elevation)", fontsize=12.5, fontweight="medium",
                 color=INK, y=0.99)

    out = os.path.join(config.OBP_VALIDATION_DIR, out_name)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
