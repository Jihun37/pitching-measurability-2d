"""Real-video feasibility: one clip per row, the four anchors across the columns.

⚠ FRAMING, unchanged from the figure this replaces. The clips carry no simultaneous
3D ground truth, so this is NOT an accuracy validation. What it shows is that the closed
loop runs end to end on ordinary phone video, where each anchor lands, and the failure
modes actually observed. Every panel is a real frame from the source clip.

⚠ WHY THIS FILE EXISTS RATHER THAN `fig_realvideo_pilot.py`. That figure counts
"valid metrics" out of the pipeline's hardcoded 15-metric deployment list, which is the
PRE-DEDUP `angle_map_2d.adopted_rows()` and is not the paper's row set. The counts here
come from `realvideo_feasibility_cells.csv`, which runs all 35 canonical retained rows
over the 80 eligible clips (`analysis/realvideo_feasibility.py`). It is also drawn in
matplotlib at the body width rather than composited in OpenCV, so its type matches the
rest of the paper's figures.

Two case sets, so the choice can be made by looking rather than by argument:

  observed  the four cases the pilot reported, each showing one failure MODE.
            Note that the stride-angle case scores higher overall than the normal
            run, which is the point: a clip can look healthy in aggregate and still
            emit one plausible wrong number.
  extremes  chosen by canonical-35 outcome instead: best clip, foot-plant fallback,
            total failure, overhead.

Run:  conda activate diamond
      cd src\\viz
      python fig_realvideo_cases.py --set observed
      python fig_realvideo_cases.py --set extremes
"""
import os, sys, argparse, textwrap
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../stage2", "../stage3", "../analysis"):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
import metrics as M
from hss_overhead_real import grab_frames_exact
from realvideo_feasibility import load_pose
from fig_graded_map import INK, MUTE, BODY_W

PILOT = os.path.join(config.ROOT, "data", "outputs", "realvideo_pilot")
V = config.OBP_VALIDATION_DIR
OUT = os.path.join(config.ROOT, "data", "outputs", "viz")
J = M.JOINTS

EVENTS = [("pkh_f", "peak knee height"), ("fp_f", "foot plant"),
          ("mer_f", "MER proxy"), ("release_f", "release")]
EV_C = {"pkh_f": "#0FA3B1", "fp_f": "#F2A900",
        "mer_f": "#7C3AED", "release_f": "#D92B4B"}

# Which segments each anchor's canonical rows actually read. Taken from the `joints`
# column of realvideo_support_matrix.csv, which is itself derived from the estimator
# definitions: at foot plant, MER and release the shoulder and hip lines dominate
# because the transverse-plane rows are read there; the throwing chain is added where
# the arm quantities are read; peak knee height is the whole-body COM, so the whole
# skeleton is its read. Placeholders are resolved per pitcher from `true_arm`.
HILITE = {
    "pkh_f":     [("LEAD_hip", "LEAD_knee"), ("LEAD_knee", "LEAD_ankle")],
    "fp_f":      [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
                  ("left_ankle", "right_ankle")],
    "mer_f":     [("left_shoulder", "right_shoulder"),
                  ("THROW_shoulder", "THROW_elbow"),
                  ("THROW_elbow", "THROW_wrist")],
    "release_f": [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
                  ("THROW_shoulder", "THROW_wrist")],
}

SKEL = [("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle")]

# (clip, title, note)  -- the note states the mode, never a measurement
SETS = {
    "observed": [
        ("real_video_test_14", "Normal run",
         "all four anchors land on plausible frames"),
        ("real_video_test_04", "Foot-plant fallback",
         "no frame met the side detector's condition, so the anchor is a constant offset"),
        ("real_video_test_11", "Unusable output, healthy clip",
         "stride angle is a raw atan2 that wraps at a rear azimuth"),
        ("pitching_overhead_01", "Overhead occlusion",
         "lowest keypoint confidence in the set"),
    ],
    "extremes": [
        ("real_video_test_10", "Best clip",
         "the highest count of usable outputs in the set"),
        ("real_video_test_04", "Foot-plant fallback",
         "no frame met the side detector's condition, so the anchor is a constant offset"),
        ("angle02_00", "Lowest usable count",
         "foot plant falls back, as it does on 37 of the 80 clips"),
        ("pitching_overhead_01", "Overhead occlusion",
         "lowest keypoint confidence in the set"),
    ],
}
PANEL_AR = 0.78          # panel width / height


def crop_box(df, frames, shape):
    """One box shared by a row's four panels, so the subject does not jump."""
    xs, ys = [], []
    for f in frames:
        for name in {a for pair in SKEL for a in pair}:
            cx, cy = df[f"{name}_x"].to_numpy(float), df[f"{name}_y"].to_numpy(float)
            if 0 <= f < len(cx) and np.isfinite(cx[f]) and np.isfinite(cy[f]):
                xs.append(cx[f]); ys.append(cy[f])
    if not xs:
        h, w = shape[:2]
        return 0, 0, w, h
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = max((y1 - y0) * 1.45, (x1 - x0) * 1.45 / PANEL_AR, 40)
    w = h * PANEL_AR
    H, W = shape[:2]
    x, y = cx - w / 2, cy - h / 2
    x = float(np.clip(x, 0, max(W - w, 0))); y = float(np.clip(y, 0, max(H - h, 0)))
    return int(x), int(y), int(min(w, W)), int(min(h, H))


def resolve(name, arm):
    lead = "left" if arm == "right" else "right"
    # only the side placeholder is substituted; the joint names are already full,
    # because expanding abbreviations here would also rewrite "left_shoulder"
    return (name.replace("THROW", arm).replace("TRAIL", arm)
                .replace("GLOVE", lead).replace("LEAD", lead))


def seg(ax, df, f, a, b, x0, y0, col, lw, z, alpha=1.0):
    pa = (df[f"{a}_x"].iloc[f] - x0, df[f"{a}_y"].iloc[f] - y0)
    pb = (df[f"{b}_x"].iloc[f] - x0, df[f"{b}_y"].iloc[f] - y0)
    if all(np.isfinite(v) for v in pa + pb):
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=col, lw=lw, alpha=alpha,
                solid_capstyle="round", zorder=z)


def draw_pose(ax, df, f, x0, y0, key, arm, col):
    """The whole pose faintly, then the segments the rows at this anchor read."""
    for a, b in SKEL:
        seg(ax, df, f, a, b, x0, y0, "white", 1.6, 2, alpha=0.55)
        seg(ax, df, f, a, b, x0, y0, "#2E3B4E", 0.7, 3, alpha=0.75)
    for a, b in HILITE[key]:
        a, b = resolve(a, arm), resolve(b, arm)
        seg(ax, df, f, a, b, x0, y0, "white", 3.0, 4, alpha=0.8)
        seg(ax, df, f, a, b, x0, y0, col, 1.7, 5)


def status_line(cells, clip):
    # cells.clip would resolve to DataFrame.clip, the method
    d = cells[cells["clip"] == clip]
    ok = int((d.status == "ok").sum())
    parts = [f"{ok} of {len(d)} rows return a usable value"]
    for s, lab in (("fp_fallback", "fallback anchor"), ("mer_proxy", "MER proxy"),
                   ("angle_unwrap", "angle past a full turn"),
                   ("wrap_risk", "near the wrap boundary")):
        k = int((d.status == s).sum())
        if k:
            parts.append(f"{k} {lab}")
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="which", choices=list(SETS), default="observed")
    a = ap.parse_args()
    cases = SETS[a.which]

    clips = pd.read_csv(os.path.join(PILOT, "pilot_clips_eligible.csv")).set_index("clip")
    cells = pd.read_csv(os.path.join(V, "realvideo_feasibility_cells.csv"))
    assert cells.metric_id.nunique() == 35, cells.metric_id.nunique()
    assert cells["clip"].nunique() == 80, cells["clip"].nunique()

    nrow, ncol = len(cases), len(EVENTS)
    lab_w = 0.235
    panel_w = (1.0 - lab_w) / ncol
    fig_h = BODY_W * (1 - lab_w) / ncol / PANEL_AR * nrow + 0.30
    fig = plt.figure(figsize=(BODY_W, fig_h))

    for i, (clip, title, note) in enumerate(cases):
        r = clips.loc[clip]
        df = load_pose(clip)
        frames = [int(r[k]) for k, _ in EVENTS if np.isfinite(r[k])]
        grabbed = grab_frames_exact(r.video_path, frames) if frames else {}
        # one clip of the 80 resolves no anchor at all: its release gate rejected the
        # candidate, and release is the first anchor, so nothing downstream is defined
        shape = next(iter(grabbed.values())).shape if grabbed else (1080, 1920, 3)
        x0, y0, w, h = crop_box(df, frames, shape)

        ytop = 1.0 - (i + 1) / nrow
        ax = fig.add_axes([0.004, ytop, lab_w - 0.012, 1.0 / nrow])
        ax.axis("off")
        ax.text(0, 0.94, title, fontsize=6.4, weight="bold", color=INK,
                va="top", ha="left", transform=ax.transAxes)
        ax.text(0, 0.80, f"{clip}   az {int(r.az)}°  el {int(r.el)}°",
                fontsize=4.8, color=MUTE, va="top", ha="left",
                transform=ax.transAxes)
        yy = 0.70
        lines = textwrap.wrap(note, width=42) + [""] + status_line(cells, clip)
        for line in lines:
            ax.text(0, yy, line, fontsize=4.6, color=MUTE, va="top", ha="left",
                    transform=ax.transAxes)
            yy -= 0.072

        for j, (key, ev) in enumerate(EVENTS):
            axp = fig.add_axes([lab_w + j * panel_w, ytop, panel_w * 0.985,
                                1.0 / nrow])
            axp.set_xticks([]); axp.set_yticks([])
            for sp in axp.spines.values():
                sp.set_color(EV_C[key]); sp.set_linewidth(0.9)
            f = int(r[key]) if np.isfinite(r[key]) else None
            if f is None or f not in grabbed:
                axp.text(0.5, 0.5, "no frame", fontsize=5, color=MUTE,
                         ha="center", va="center", transform=axp.transAxes)
                continue
            img = grabbed[f][y0:y0 + h, x0:x0 + w, ::-1]
            axp.imshow(img, aspect="auto")
            draw_pose(axp, df, f, x0, y0, key, str(r.true_arm), EV_C[key])
            axp.set_xlim(0, w); axp.set_ylim(h, 0)
            axp.text(0.03, 0.035, f"{ev}   f{f}", fontsize=4.7, color="white",
                     weight="bold", va="top", ha="left", transform=axp.transAxes,
                     bbox=dict(facecolor=EV_C[key], edgecolor="none",
                               boxstyle="square,pad=0.22"))

    out = os.path.join(OUT, f"fig_realvideo_cases_{a.which}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"set={a.which}  {nrow} clips x {ncol} anchors")
    for clip, _t, _n in cases:
        print(f"  {clip:<22} " + " | ".join(status_line(cells, clip)))
    print("saved ->", out)


if __name__ == "__main__":
    main()
