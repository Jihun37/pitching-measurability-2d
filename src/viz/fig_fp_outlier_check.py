"""Visual check of the 8 k-outlier pitches that the GT-clean filter removes.

The filter that takes the scored population from 401 to 394 is statistical: a
+-3 MAD rule on k = (fp_100 - pkh)/(BR - pkh). This script asks the question a
reader actually asks, which is whether the flagged pitches are visibly wrong at
the landmark rather than merely unusual in a distribution.

Per pitch it draws
  (a) the lead-ankle height trace with pkh / fp_10 / fp_100 / BR marked, so the
      real touchdown (where the ankle stops descending and settles) can be read
      straight off the curve, and
  (b) side-view stick figures at pkh, at the fp_100 landmark and at release.

If fp_100 is sound the middle skeleton stands on a planted lead foot and the
landmark sits at the ankle-height plateau. On the flagged pitches it does not.

Run:  conda activate diamond; cd src\\viz; python fig_fp_outlier_check.py
Outputs: data/outputs/obp_validation/fp_outlier_check_*.png  (+ a printed table)
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(HERE, "..", "stage3"))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import config
import obp_project as O
from obp_gt_events import load_gt_events
from gt_landmark_outlier_effect import outlier_pitches
from master_angle_table import load_feet

INK = "#0E1B33"; TEAL = "#0FA3B1"; RED = "#E0533D"
AMBER = "#F2A900"; GREEN = "#16A34A"; GREY = "0.55"

BONES = [("head", "left_shoulder"), ("head", "right_shoulder"),
         ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
         ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
         ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")]

# k range of the 391 pitches the rule keeps (docstring of
# gt_landmark_outlier_effect); printed on each panel for reference.
K_OK = (0.763, 0.884)


def clip_path(md_row):
    return os.path.join(config.OBP_DATA_DIR, "c3d",
                        f"{int(md_row.user):06d}", md_row.filename_new)


def draw_pose(ax, df, f, title, colour):
    """Side view (az 0) stick figure at frame f. Image y already grows down."""
    for a, b in BONES:
        ax.plot([df[f"{a}_x"][f], df[f"{b}_x"][f]],
                [df[f"{a}_y"][f], df[f"{b}_y"][f]], color=colour, lw=1.8,
                solid_capstyle="round")
    xs = [df[f"{n}_x"][f] for n in ("head", "left_ankle", "right_ankle",
                                    "left_wrist", "right_wrist")]
    ys = [df[f"{n}_y"][f] for n in ("head", "left_ankle", "right_ankle",
                                    "left_wrist", "right_wrist")]
    ax.scatter(xs, ys, s=9, color=colour, zorder=3)
    ax.set_title(title, fontsize=8, color=colour, pad=3)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")


def panel_row(axes, sp, joints, fps, df, ev, k, note):
    ax = axes[0]
    lead = "left"                      # every pitch is RHP after reflection
    z = joints[f"{lead}_ankle"][2]
    n = len(z)
    ax.plot(np.arange(n), z, color=INK, lw=1.3)
    for name, col, ls in (("pkh", GREEN, "-"), ("fp10", AMBER, ":"),
                          ("fp", RED, "-"), ("rel", TEAL, "-")):
        f = ev.get(name)
        if f is None:
            continue
        ax.axvline(f, color=col, lw=1.5, ls=ls)
        ax.text(f, ax.get_ylim()[1], f" {name}", fontsize=6.5, color=col,
                rotation=90, va="top", ha="left")
    lo = max(0, (ev.get("pkh") or 0) - 40)
    hi = min(n - 1, (ev.get("rel") or n - 1) + 40)
    ax.set_xlim(lo, hi)
    ax.set_ylabel("lead ankle\nheight (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(f"{sp}   k = {k:.3f}   {note}", fontsize=8,
                 color=RED if note else INK, loc="left")

    draw_pose(axes[1], df, ev["pkh"], f"peak knee height  f{ev['pkh']}", GREEN)
    draw_pose(axes[2], df, ev["fp"], f"fp_100 landmark  f{ev['fp']}", RED)
    draw_pose(axes[3], df, ev["rel"], f"release  f{ev['rel']}", TEAL)


def main():
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    md = md.set_index("session_pitch")
    gt = load_gt_events()
    bad = sorted(outlier_pitches())
    an = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR,
                                  "fp_front_error_anatomy.csv")).set_index("sp")

    # two healthy controls, k nearest the median, for side-by-side reading
    ok = an.loc[[s for s in an.index if s not in bad], "k_true"]
    ctrl = list((ok - ok.median()).abs().sort_values().index[:2])

    # The money column is `above`: how high the lead ankle still is at the claimed
    # 100 %-load landmark, relative to the height it settles at once the foot is
    # genuinely down. A sound landmark reads ~0.
    print(f"{'pitch':<12} {'k':>7}  {'pkh':>5} {'fp10':>5} {'fp100':>5} "
          f"{'BR':>5}  {'above (m)':>9}  {'fp>BR':>5}  note")
    groups = [("outliers_a", bad[:4]), ("outliers_b", bad[4:]),
              ("controls", ctrl)]
    for tag, sps in groups:
        sps = [s for s in sps if s in gt and s in md.index]
        if not sps:
            continue
        fig, AX = plt.subplots(len(sps), 4, figsize=(13, 2.6 * len(sps)),
                               gridspec_kw={"width_ratios": [2.1, 1, 1, 1]})
        AX = np.atleast_2d(AX)
        for i, sp in enumerate(sps):
            joints, fps = load_feet(clip_path(md.loc[sp]))
            df = O.project_view(joints, azimuth_deg=0.0)
            ev = gt[sp]
            k = float(an.loc[sp, "k_true"])
            note = "" if sp in ctrl else "FLAGGED"
            panel_row(AX[i], sp, joints, fps, df, ev, k, note)
            za = joints["left_ankle"][2]
            above = float(za[ev["fp"]] - np.nanmin(za))
            late = "yes" if ev["fp"] > ev["rel"] else "no"
            print(f"{sp:<12} {k:>7.3f}  {ev.get('pkh',-1):>5} "
                  f"{ev.get('fp10',-1):>5} {ev.get('fp',-1):>5} "
                  f"{ev.get('rel',-1):>5}  {above:>9.3f}  {late:>5}  {note}")
        fig.suptitle("GT foot-plant landmark (fp_100) at the flagged pitches"
                     if tag.startswith("outliers") else
                     "Controls, k near the population median",
                     fontsize=10, color=INK, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = os.path.join(config.OBP_VALIDATION_DIR,
                           f"fp_outlier_check_{tag}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("saved ->", out)


if __name__ == "__main__":
    main()
