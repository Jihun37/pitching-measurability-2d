"""Figure (patent Fig. 3): view-dependent release detection.
side  = wrist-speed argmax (ball travels in-plane).
front = arm-extension argmax within a 200 ms window ending at the speed peak
        (the speed peak itself lands in follow-through)."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(HERE, "..", "stage3"))
import config
import obp_project as O
import metrics as M

INK = "#0E1B33"; GREEN = "#16A34A"; AMBER = "#F2A900"; RED = "#E0533D"


def series(df, arm):
    wk = "r_wr" if arm == "right" else "l_wr"
    sk = "r_sh" if arm == "right" else "l_sh"
    wx, wy = M._xy(df, wk, M.JOINTS); sx, sy = M._xy(df, sk, M.JOINTS)
    return wx, wy, np.hypot(wx - sx, wy - sy)


def main():
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    r = md.iloc[0]
    path = os.path.join(config.OBP_DATA_DIR, "c3d", f"{int(r.user):06d}", r.filename_new)
    joints, fps = O.load_c3d_joints(path)
    arm = O.detect_throwing_arm(joints, fps)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=False)
    win = int(0.20 * fps)

    for ax, (az, title) in zip(axes, [(0, "(a) Side (0°): wrist-speed peak"),
                                      (90, "(b) Front (90°): windowed arm extension")]):
        df = O.project_view(joints, azimuth_deg=az)
        wx, wy, ext = series(df, arm)
        spd = M._speed(wx, wy, fps)
        t = np.arange(len(spd)) / fps * 1000.0
        pk = int(np.nanargmax(spd))
        ax.plot(t, spd / np.nanmax(spd), color=GREEN, lw=1.8, label="wrist speed (norm.)")
        ax.plot(t, ext / np.nanmax(ext), color=AMBER, lw=1.6, label="arm extension (norm.)")

        if az == 0:
            rel = pk
            ax.axvline(t[rel], color=RED, lw=1.8)
            ax.text(t[rel] - 3, 1.03, "release = speed peak", color=RED, fontsize=9, ha="right")
        else:
            lo = max(0, pk - win)
            rel = lo + int(np.nanargmax(ext[lo:pk + 1]))
            ax.axvspan(t[lo], t[pk], color="0.9", zorder=0)
            ax.text((t[lo]+t[pk])/2, 0.06, "200 ms window", color="0.4", fontsize=8, ha="center")
            ax.axvline(t[pk], color=GREEN, ls="--", lw=1.4)
            ax.text(t[pk] + 3, 1.03, "speed peak\n(follow-through)", color=GREEN, fontsize=8, ha="left")
            ax.axvline(t[rel], color=RED, lw=1.8)
            ax.text(t[rel] - 3, 1.03, "release = ext. max in window", color=RED, fontsize=9, ha="right")

        ax.set_title(title, fontsize=12, color=INK, weight="bold")
        ax.set_xlabel("time (ms)"); ax.set_ylim(0, 1.12)
        ax.set_xlim(t[max(0, pk-int(0.5*fps))], t[min(len(t)-1, pk+int(0.25*fps))])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if az == 0:
            ax.set_ylabel("normalized")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.04))
    out = os.path.join(config.OBP_VALIDATION_DIR, "fig_release_detection.png")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out, dpi=200, bbox_inches="tight"); print("saved ->", out)


if __name__ == "__main__":
    main()
