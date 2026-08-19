"""Skeleton gallery of REJECTED columns: what each failed metric would have been
measured from, drawn at its own best viewpoint, with the reason it did not make
the 16-metric map.

Same visual language as fig_elbow_mer_views (bones, arc, straight-arm reference)
-- drawing helpers are imported from it, not re-written. Every number on the
figure is read from column_coverage_audit.csv at render time; nothing about the
verdicts is hardcoded here, so a re-screen that changes a column changes the
figure.

The five panels are one per failure TYPE, so the gallery reads as a taxonomy
rather than a list:

  Elbow Flex @FP      F-below-floor  the SAME geometry as the adopted metric,
                                     read at foot plant: one lucky cell out of
                                     168, which is a spike, not a zone.
  Elbow Flex (peak)   D-redundant    measurable (4 cells) but the same lay-back
                                     signal already adopted at MER.
  Shoulder IR Velo    E-not-observable  rotation about the limb's own axis: the
                                     shoulder-elbow-wrist chain is invariant to
                                     it at every viewpoint.
  Elbow Varus Moment  A-kinetics     a torque. No kinematic observable at all --
                                     the pose is identical whatever the load.
  Torso Rot @FP       B-viewpoint    transverse rotation, degenerate from the
                                     ground; alive only overhead, where
                                     torso_rotation_br was adopted instead.

Events are GT (obp_gt_events), never detected -- a rejection must not be a
detector artefact.

Run:  conda activate diamond
      cd src\\viz
      python fig_failed_metrics_gallery.py
      python fig_failed_metrics_gallery.py --pitch 1031_2
"""
import os, sys, argparse, textwrap
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../stage2", "../stage3", "../analysis"):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)

import config
import obp_project as O
import metrics as M
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from obp_gt_events import load_gt_events
from visualize_3d_2d import INK, TEAL, AMBER, MUTE, VIOLET
from fig_elbow_mer_views import (draw_skeleton, draw_elbow, _frame_axes, _pts,
                                 az_words, CRIMSON, PPM, pick_pitch)

AUDIT = os.path.join(config.OBP_VALIDATION_DIR, "column_coverage_audit.csv")
GREY = "#94A3B8"

# (audit column, panel title, event, geometry). ANCHOR_VIEW is used only for the
# kinetics row, which has no sweep view because it was never screenable.
ANCHOR_VIEW = (330, 60)
PANELS = [
    ("elbow_flexion_fp",                      "Elbow Flex @ foot plant", "fp",   "elbow"),
    ("max_elbow_flexion",                     "Elbow Flex (peak)",       "peak", "elbow"),
    ("max_shoulder_internal_rotational_velo", "Shoulder IR Velocity",    "mer",  "spin"),
    ("elbow_varus_moment",                    "Elbow Varus Moment",      "mer",  "moment"),
    ("torso_rotation_fp",                     "Torso Rotation @ foot plant", "fp", "transverse"),
]


def flex_curve(df, arm):
    """Elbow flexion for every frame (same definition as metrics.elbow_flexion_2d,
    vectorised) -- only used to LOCATE the peak-flexion frame."""
    J = M.JOINTS
    def xy(k):
        return (df[f"{J[k]}_x"].to_numpy(float), df[f"{J[k]}_y"].to_numpy(float))
    a = "r" if arm == "right" else "l"
    sx, sy = xy(f"{a}_sh"); ex, ey = xy(f"{a}_el"); wx, wy = xy(f"{a}_wr")
    return 180.0 - M._angle(sx, sy, ex, ey, wx, wy)


def spin_arrow(ax, p0, p1, label):
    """A dashed rotation symbol around the segment p0->p1, i.e. about the limb's
    OWN long axis. Deliberately drawn in grey: it marks a quantity the image does
    not contain, so it must not read like a measured overlay."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    mid = (p0 + p1) / 2
    L = float(np.hypot(*(p1 - p0)))
    ang = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
    ax.add_patch(Arc(mid, width=0.34 * L, height=0.95 * L, angle=ang,
                     theta1=0, theta2=310, color=GREY, lw=2.0, ls=(0, (4, 3)),
                     zorder=6))
    tip = mid + 0.42 * L * np.array([-np.sin(np.radians(ang)), np.cos(np.radians(ang))])
    ax.annotate("", xy=tip, xytext=mid + 0.30 * L * np.array(
        [np.cos(np.radians(ang)), np.sin(np.radians(ang))]),
        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.8), zorder=6)
    ax.text(mid[0], mid[1] - 0.62 * L, label, color=GREY, fontsize=11,
            weight="bold", ha="center", va="center", zorder=6)


def moment_arrow(ax, joint, scale, label):
    """A curved torque symbol at a joint. Grey for the same reason as spin_arrow:
    a moment has no image counterpart at all."""
    j = np.asarray(joint, float)
    a = j + np.array([-0.55, -0.42]) * scale
    b = j + np.array([0.55, -0.42]) * scale
    ax.add_patch(FancyArrowPatch(a, b, connectionstyle="arc3,rad=-0.62",
                                 arrowstyle="-|>", mutation_scale=17,
                                 color=GREY, lw=2.2, ls=(0, (4, 3)), zorder=6))
    ax.text(j[0], j[1] - 1.05 * scale, label, color=GREY, fontsize=11,
            weight="bold", ha="center", va="center", zorder=6)


def draw_transverse(ax, P, arm):
    """Shoulder line and hip line -- the two segments any transverse-rotation
    metric has to read. Off the overhead view their projected length, not their
    orientation, carries the rotation, and that is what goes degenerate."""
    for (a, b), col, lab, dy in (
            (("left_shoulder", "right_shoulder"), AMBER, "shoulder line", -26),
            (("left_hip", "right_hip"), VIOLET, "hip line", 30)):
        p, q = np.array(P[a]), np.array(P[b])
        ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=4.0, zorder=5,
                solid_capstyle="round")
        mid = (p + q) / 2
        ax.text(mid[0], mid[1] + dy, lab, color=col, fontsize=11,
                weight="bold", ha="center", va="center", zorder=6)


def event_frame(kind, gt_ev, df, arm, view):
    if kind in ("fp", "mer", "rel"):
        return int(gt_ev[kind])
    if kind == "peak":
        fx = flex_curve(df, arm)
        lo, hi = int(gt_ev["fp"]), int(gt_ev["rel"])
        return lo + int(np.nanargmax(fx[lo:hi + 1]))
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitch", default=None)
    ap.add_argument("--out", default=os.path.join(config.ROOT, "data", "outputs", "viz"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    audit = pd.read_csv(AUDIT).set_index("column")
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    gt = load_gt_events(["fp", "mer", "rel"])

    sp, path = pick_pitch(md, gt, poi, a.pitch)
    joints, fps = load_feet(path)
    arm = O.detect_throwing_arm(joints, fps)
    g = gt[sp]
    print(f"pitch {sp}  fps={fps:.0f}  fp=f{g['fp']}  MER=f{g['mer']}  rel=f{g['rel']}\n")

    # The kinetics rows carry no sweep, so their audit "reason" is just the
    # inference number. Summarise the axis from the CSV itself rather than
    # restating a hardcoded sentence about it.
    # Count only the rows carrying their OWN inference score: the GRF direction
    # rows quote a proxy column's number and are not separate targets.
    kin = audit[(audit.reason_code == "A-kinetics") & audit.infer_r2.notna()]
    kin = kin[~kin.reason.str.contains("not separately inferred")]
    n_pass = int((kin.infer_r2 >= 0.50).sum())
    kin_note = (f"the whole kinetics axis: {n_pass} of {len(kin)} inferred "
                f"targets reach R² 0.50 (best {kin.infer_r2.max():.3f})")

    n = len(PANELS)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 6.9 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, (col, title, ev, geom) in zip(axes, PANELS):
        row = audit.loc[col]
        r2 = float(row.sweep_best_r2) if pd.notna(row.sweep_best_r2) else np.nan
        if isinstance(row.sweep_view, str) and "/" in row.sweep_view:
            az, el = (int(v) for v in row.sweep_view.split("/"))
            view_note = f"best cell az {az}° / el {el}°"
        else:
            az, el = ANCHOR_VIEW
            view_note = f"no screenable view · drawn at az {az}° / el {el}°"
        df, P = _pts(joints, az, el, 0)
        f = event_frame(ev, g, df, arm, (az, el))
        _, P = _pts(joints, az, el, f)

        draw_skeleton(ax, P, arm)
        sh, elb = np.array(P[f"{arm}_shoulder"]), np.array(P[f"{arm}_elbow"])
        scale = float(np.hypot(*(elb - sh)))
        if geom == "elbow":
            draw_elbow(ax, P, arm, M.elbow_flexion_2d(df, arm, f, M.JOINTS))
        elif geom == "spin":
            draw_elbow(ax, P, arm, 0.0, label=False)
            spin_arrow(ax, sh, elb, "spin about the arm's own axis")
        elif geom == "moment":
            draw_elbow(ax, P, arm, 0.0, label=False)
            moment_arrow(ax, elb, 0.55 * scale, "torque (N·m)")
        elif geom == "transverse":
            draw_transverse(ax, P, arm)
        _frame_axes(ax, P, pad=0.30)

        code = str(row.reason_code)
        r2s = f"best r² {r2:.2f}" if np.isfinite(r2) else "never screenable"
        cells = (f"{int(row.sweep_cells)} of 168 cells ≥ 0.60"
                 if pd.notna(row.sweep_cells) else "no 2D observable")
        inf = (f"   ·   inference R² {float(row.infer_r2):.2f}"
               if pd.notna(row.infer_r2) else "")
        ax.set_title(f"{title}\n{code}", fontsize=14, weight="bold",
                     color=INK, pad=16)
        ax.text(0.5, 1.005, f"{view_note}  ·  {ev.upper()} event (GT)",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10.5, color=MUTE)
        reason = str(row.reason)
        if reason.startswith("inference R2="):      # duplicates the line above
            reason = kin_note
        body = f"{r2s}   ·   {cells}{inf}\n" + "\n".join(
            textwrap.wrap(reason, 58))
        ax.text(0.5, -0.02, body, transform=ax.transAxes, ha="center", va="top",
                fontsize=9.8, color=INK)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Rejected columns — what each would have been measured from, "
                 "and why it failed", fontsize=17, weight="bold", color=INK, y=0.985)
    fig.text(0.5, 0.955, f"pitch {sp} · GT events · every number read from "
                         f"column_coverage_audit.csv · grey = a quantity the "
                         f"image does not contain",
             ha="center", va="top", fontsize=11.5, color=MUTE)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.085,
                        wspace=0.05, hspace=0.50)
    out = os.path.join(a.out, "fig_failed_metrics_gallery.png")
    fig.savefig(out, dpi=165)
    plt.close(fig)
    print("saved ->", out)

    print(f"\n{'column':40}{'code':18}{'best r2':>9}{'view':>10}{'cells':>7}")
    print("-" * 84)
    for col, title, ev, geom in PANELS:
        r = audit.loc[col]
        r2 = f"{float(r.sweep_best_r2):.3f}" if pd.notna(r.sweep_best_r2) else "-"
        cl = f"{int(r.sweep_cells)}" if pd.notna(r.sweep_cells) else "-"
        print(f"{col:40}{str(r.reason_code):18}{r2:>9}"
              f"{str(r.sweep_view):>10}{cl:>7}")


if __name__ == "__main__":
    main()
