"""Skeleton panels for the 16th metric, Elbow Flex @MER -- the one metric that is
on the paper measurability map but NOT in the deployment map (16 = 15 + 1).

What it draws, per camera cell: the projected 2D skeleton at the GT MER frame,
the throwing-arm chain shoulder-elbow-wrist, and the flexion angle itself as the
opening between the straight-arm reference (the shoulder->elbow direction carried
past the elbow) and the forearm. That opening IS metrics.elbow_flexion_2d
(180 - the included shoulder-elbow-wrist angle), so the picture and the number
come from the same definition -- nothing is re-implemented here.

Two figures:
  fig_elbow_mer_views.png    the optimal viewpoints (r2 >= 0.80 cells of
                             angle_zone_sweep_gt_clean.csv), one panel each.
  fig_elbow_mer_frames.png   why it is GT-only: the same view read at MER +- a
                             few frames, plus the flexion-vs-frame curve. The
                             metric moves ~4.3 deg per 360 Hz frame, so one
                             phone frame at 120 fps is ~13 deg -- the event
                             precision wall, not a projection failure.
  --with-fail                adds fig_elbow_mer_fail.png (ground-level side and
                             front views, for contrast; off by default).

MER is a rotation instant 2D cannot detect, so the frame comes from the OBP
landmarks (obp_gt_events), never from a detector -- same convention as the
GT-clean map this figure quotes.

Run:  conda activate diamond
      cd src\\viz
      python fig_elbow_mer_views.py
      python fig_elbow_mer_views.py --views 330/60,315/60 --pitch 1031_2
"""
import os, sys, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../stage2", "../stage3", "../analysis"):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)

import config
import obp_project as O
import metrics as M
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from obp_gt_events import load_gt_events
from gt_landmark_outlier_effect import outlier_pitches
from visualize_3d_2d import BONES, INK, TEAL, AMBER, MUTE, _arc

CRIMSON = "#D92B4B"      # the measured chain + its arc
PPM = 300.0              # m -> pixel-like units, so the _arc label offsets
                         # (tuned in pixel space for the paper figures) behave

# 4 of the 6 r2 >= 0.80 cells, chosen to span the zone rather than to crowd its
# centre (the top four by r2 are all el=60 and look nearly identical).
DEFAULT_VIEWS = [(300, 60), (330, 60), (345, 45), (0, 45)]
FAIL_VIEWS = [(0, 0), (90, 0)]
MAP_CSV = os.path.join(config.OBP_VALIDATION_DIR, "angle_zone_sweep_gt_clean.csv")
METRIC = "Elbow Flex @MER [O]"
TRUTH_COL = "elbow_flexion_mer"


def az_words(az):
    """Plain-language camera placement. Official convention (2026-07-15) is
    handedness-relative: az0 = the OPEN side (3B for a RHP), az increases toward
    home, az90 = front/catcher, az180 = closed side, az270 = rear/2B."""
    names = {0: "open side", 90: "front", 180: "closed side", 270: "rear"}
    az = az % 360
    card = min(names, key=lambda c: min(abs(az - c), 360 - abs(az - c)))
    d = (az - card + 180) % 360 - 180
    if abs(d) < 1:
        return names[card]
    return f"{names[card]} {abs(d):.0f}° {'front' if d > 0 else 'rear'}"


def load_r2():
    """{(az, el): r2} for this metric from the GT-clean paper map."""
    d = pd.read_csv(MAP_CSV)
    d = d[d.metric == METRIC]
    return {(int(r.az), int(r.el)): float(r.r2) for r in d.itertuples(index=False)}


def pick_pitch(md, gt, poi, want=None):
    """A clean pitch with a GT MER frame. Deterministic: the first metadata row
    that survives the gt_clean filter, so the figure is reproducible."""
    bad = outlier_pitches()
    for r in md.itertuples(index=False):
        sp = r.session_pitch
        if want and sp != want:
            continue
        if sp in bad or sp not in gt or "mer" not in gt[sp]:
            continue
        if sp not in poi.index or not np.isfinite(poi.loc[sp, TRUTH_COL]):
            continue
        path = os.path.join(config.OBP_DATA_DIR, "c3d",
                            f"{int(r.user):06d}", r.filename_new)
        if os.path.exists(path):
            return sp, path
    raise SystemExit(f"no usable pitch found (want={want})")


def _pts(joints, az, el, frame):
    """Projected joint dict at one frame, in pixel-like units, y already down."""
    df = project_cam(joints, az, el)
    return df, {n: (df[f"{n}_x"].iloc[frame] * PPM, df[f"{n}_y"].iloc[frame] * PPM)
                for n in joints}


def draw_skeleton(ax, P, arm, alpha=1.0, lw=2.4, dots=True):
    for a, b in BONES:
        if a in P and b in P:
            ax.plot([P[a][0], P[b][0]], [P[a][1], P[b][1]],
                    color=TEAL, lw=lw, alpha=alpha, zorder=2, solid_capstyle="round")
    if dots:
        for j in M.JOINTS.values():
            if j in P:
                ax.scatter(*P[j], color=INK, s=14, alpha=alpha, zorder=3)


def draw_elbow(ax, P, arm, value, label=True, color=CRIMSON, alpha=1.0, lw=3.6):
    """The measured chain + the flexion opening. The dashed ray is the straight
    arm (0 deg flexion); the arc from it to the forearm is the returned angle."""
    sh = np.array(P[f"{arm}_shoulder"]); el = np.array(P[f"{arm}_elbow"])
    wr = np.array(P[f"{arm}_wrist"])
    ax.plot([sh[0], el[0]], [sh[1], el[1]], color=color, lw=lw, alpha=alpha,
            zorder=4, solid_capstyle="round")
    ax.plot([el[0], wr[0]], [el[1], wr[1]], color=color, lw=lw, alpha=alpha,
            zorder=4, solid_capstyle="round")
    for p in (sh, el, wr):
        ax.scatter(*p, color=color, s=30, alpha=alpha, zorder=5,
                   edgecolors="white", linewidths=0.8)
    if not label:
        return
    fore = float(np.hypot(*(wr - el)))
    u = (el - sh) / (np.linalg.norm(el - sh) + 1e-9)
    ext = el + u * fore                                   # straight-arm reference
    ax.plot([el[0], ext[0]], [el[1], ext[1]], color=MUTE, lw=1.4, ls="--",
            alpha=0.9, zorder=3)
    _arc(ax, tuple(el), tuple(ext), tuple(wr), 0.42 * fore, color,
         f"flex {value:.0f}°", fs=13)


def _frame_axes(ax, P, pad=0.14):
    xs = [p[0] for p in P.values()]; ys = [p[1] for p in P.values()]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    r = max(max(xs) - min(xs), max(ys) - min(ys)) / 2
    r *= (1 + pad)
    ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy + r, cy - r)   # y down
    ax.set_aspect("equal"); ax.axis("off")


def views_figure(joints, arm, mer, truth, views, r2map, out_png, title, sub):
    n = len(views)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.8))
    axes = np.atleast_1d(axes)
    for ax, (az, el) in zip(axes, views):
        df, P = _pts(joints, az, el, mer)
        val = M.elbow_flexion_2d(df, arm, mer, M.JOINTS)
        draw_skeleton(ax, P, arm)
        draw_elbow(ax, P, arm, val)
        _frame_axes(ax, P, pad=0.22)      # room for the arc label above the arm
        r2v = r2map.get((az, el), np.nan)
        r2s = f"r² = {r2v:.2f}" if np.isfinite(r2v) else "r² = n/a"
        ok = np.isfinite(r2v) and r2v >= 0.80
        ax.set_title(f"az {az}°  ·  el {el}°", fontsize=15, weight="bold",
                     color=INK, pad=22)
        ax.text(0.5, 1.012, f"{az_words(az)}  ·  camera {el}° high",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=11.5, color=MUTE)
        ax.text(0.5, -0.02, f"{r2s}   │   2D read {val:.1f}°"
                            f"   (3D {truth:.1f}°)",
                transform=ax.transAxes, ha="center", va="top", fontsize=12.5,
                weight="bold", color=(INK if ok else "#B45309"))
    fig.suptitle(title, fontsize=16.5, weight="bold", color=INK, y=0.982)
    fig.text(0.5, 0.928, sub, ha="center", va="top", fontsize=11.5, color=MUTE)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.07, wspace=0.04)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print("saved ->", out_png)


def truth_sd(md, poi, gt):
    """SD of the 3D truth over the gt_clean population the map is scored on.
    Computed, never hardcoded: the frame-slope only means something against it."""
    bad = outlier_pitches()
    sps = [r.session_pitch for r in md.itertuples(index=False)
           if r.session_pitch in gt and "mer" in gt[r.session_pitch]
           and r.session_pitch not in bad and r.session_pitch in poi.index]
    v = poi.loc[sps, TRUTH_COL].to_numpy(float)
    return float(np.nanstd(v[np.isfinite(v)], ddof=1)), len(sps)


def frames_figure(joints, arm, mer, truth, view, fps, out_png, tsd, n_sd):
    """The event wall: same camera, MER read a few frames early/late."""
    az, el = view
    df = project_cam(joints, az, el)
    offs = np.arange(-6, 7)
    vals = np.array([M.elbow_flexion_2d(df, arm, mer + int(o), M.JOINTS) for o in offs])
    slope = float(np.mean(np.abs(np.diff(vals))))
    phone = slope * (fps / 120.0)            # 1 frame at 120 fps, in c3d frames

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axes[0]
    _, P0 = _pts(joints, az, el, mer)
    draw_skeleton(ax, P0, arm, alpha=0.30, lw=2.0, dots=False)
    ghosts = [-3, 0, 3]
    cols = {-3: "#7C3AED", 0: CRIMSON, 3: "#0EA5E9"}
    allP = dict(P0)
    for k, o in enumerate(ghosts):
        _, P = _pts(joints, az, el, mer + o)
        v = M.elbow_flexion_2d(df, arm, mer + o, M.JOINTS)
        draw_elbow(ax, P, arm, v, label=False, color=cols[o],
                   alpha=1.0 if o == 0 else 0.75, lw=3.6 if o == 0 else 2.6)
        lab = "MER (GT event)" if o == 0 else f"MER {o:+d} frames"
        ax.text(0.02, 0.985 - 0.062 * k, f"{lab}:  {v:.0f}°",
                transform=ax.transAxes, color=cols[o], fontsize=13,
                weight="bold", va="top", ha="left", zorder=6)
        allP.update({f"{k}#{o}": v2 for k, v2 in P.items()})
    _frame_axes(ax, allP, pad=0.16)
    ax.set_title(f"az {az}° / el {el}° — one camera, event read ±3 frames off\n"
                 f"(±3 frames @ {fps:.0f} Hz = ±1 frame on a 120 fps phone)",
                 fontsize=12.5, weight="bold", color=INK, pad=10)

    ax = axes[1]
    ax.axhline(truth, color=MUTE, ls="--", lw=1.4)
    ax.text(offs[0], truth, f" 3D truth {truth:.1f}°", color=MUTE,
            fontsize=11, va="bottom")
    ax.axvspan(-3, 3, color=AMBER, alpha=0.16, lw=0)
    ax.plot(offs, vals, color=CRIMSON, lw=2.6, marker="o", ms=5)
    ax.scatter([0], [vals[list(offs).index(0)]], color=CRIMSON, s=110,
               zorder=5, edgecolors="white", linewidths=1.4)
    ax.set_xlabel(f"frame offset from GT MER  (1 frame = {1000.0/fps:.1f} ms)",
                  fontsize=11.5)
    ax.set_ylabel("elbow flexion read from 2D (°)", fontsize=11.5)
    ax.set_title(f"{slope:.1f}° per c3d frame  →  {phone:.0f}° per "
                 f"phone frame at 120 fps\n(truth SD across pitches "
                 f"{tsd:.1f}°, n={n_sd})",
                 fontsize=12.5, weight="bold", color=INK, pad=10)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # The precise claim (docs/legacy_pre_dedup/GT_EVENT_MAP_HANDOFF.md 6g): a CONSTANT offset is
    # absorbed by calibration -- reading at MER +- 2 f still scores r2 0.86. What
    # kills the metric is the per-pitch JITTER of any anchor we can actually
    # detect. Stated here so the steep curve is not misread as "any offset dies".
    fig.text(0.5, 0.012,
             "a constant offset is absorbed by calibration (MER ± 2 f still "
             "r² ≈ 0.86); it is the per-pitch jitter of the anchor that destroys "
             "it  —  docs/legacy_pre_dedup/GT_EVENT_MAP_HANDOFF.md §6g",
             ha="center", va="bottom", fontsize=10.5, color=MUTE)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print("saved ->", out_png)
    print(f"  slope {slope:.2f} deg / frame @ {fps:.0f} Hz "
          f"({phone:.1f} deg per 120 fps frame)")


def parse_views(s):
    out = []
    for tok in s.split(","):
        az, el = tok.strip().split("/")
        out.append((int(az), int(el)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitch", default=None, help="session_pitch, e.g. 1031_2")
    ap.add_argument("--views", default=None, help="az/el list, e.g. 330/60,0/45")
    ap.add_argument("--with-fail", action="store_true",
                    help="also render the ground-level views that cannot see it")
    ap.add_argument("--title", default=None,
                    help="headline for the views figure (default assumes the "
                         "views are the measurable ones)")
    ap.add_argument("--tag", default="",
                    help="filename suffix, so a custom --views run does not "
                         "overwrite the default figures")
    ap.add_argument("--out", default=os.path.join(config.ROOT, "data", "outputs", "viz"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = f"_{a.tag}" if a.tag else ""

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    gt = load_gt_events(["mer", "rel"])
    r2map = load_r2()

    top = sorted(r2map.items(), key=lambda kv: -kv[1])[:10]
    print(f"[{METRIC}]  GT-clean map, best cells")
    for (az, el), v in top:
        print(f"   az{az:>3d} / el{el:>2d}   r2 = {v:.3f}   {az_words(az)}")
    print(f"   cells r2>=0.80: {sum(v >= 0.80 for v in r2map.values())}"
          f"   r2>=0.60: {sum(v >= 0.60 for v in r2map.values())}  of {len(r2map)}\n")

    sp, path = pick_pitch(md, gt, poi, a.pitch)
    joints, fps = load_feet(path)
    arm = O.detect_throwing_arm(joints, fps)          # always "right" (reflected)
    mer, rel = gt[sp]["mer"], gt[sp].get("rel")
    truth = float(poi.loc[sp, TRUTH_COL])
    throws = str(poi.loc[sp, "p_throws"]) if "p_throws" in poi.columns else "?"
    print(f"pitch {sp}  ({throws}HP, reflected to RHP)  fps={fps:.0f}  "
          f"MER=f{mer}  release=f{rel}  (rel - MER = {rel - mer} f)")
    print(f"3D truth (poi.{TRUTH_COL}) = {truth:.2f} deg")

    views = parse_views(a.views) if a.views else DEFAULT_VIEWS
    for az, el in views:
        df = project_cam(joints, az, el)
        print(f"   az{az:>3d}/el{el:>2d}  2D = "
              f"{M.elbow_flexion_2d(df, arm, mer, M.JOINTS):6.2f} deg")

    views_figure(joints, arm, mer, truth, views, r2map,
                 os.path.join(a.out, f"fig_elbow_mer_views{tag}.png"),
                 a.title or "Elbow Flex @MER — the viewpoints that can measure it",
                 f"pitch {sp} · GT MER frame f{mer} (OBP landmark, "
                 f"release − {rel - mer} frames) · 3D truth {truth:.1f}°"
                 f" · arc = flexion from the straight-arm reference")

    best = max(views, key=lambda v: r2map.get(v, -1))
    tsd, n_sd = truth_sd(md, poi, gt)
    print(f"truth SD (gt_clean population) = {tsd:.2f} deg, n={n_sd}")
    frames_figure(joints, arm, mer, truth, best, fps,
                  os.path.join(a.out, f"fig_elbow_mer_frames{tag}.png"), tsd, n_sd)

    if a.with_fail:
        views_figure(joints, arm, mer, truth, FAIL_VIEWS, r2map,
                     os.path.join(a.out, f"fig_elbow_mer_fail{tag}.png"),
                     "Elbow Flex @MER — ground-level views (for contrast)",
                     f"pitch {sp} · same GT MER frame f{mer} · 3D truth "
                     f"{truth:.1f}° · these cells are below the 0.60 floor")


if __name__ == "__main__":
    main()
