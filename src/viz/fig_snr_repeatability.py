"""Figure (구성 H): value measurability vs repeatability.
Per metric, r2 of the VALUE (2D vs 3D) beside r2 of the between-pitch SD
(repeatability). Some metrics recover both (lead knee angle, high SNR); others
recover the value but not its pitch-to-pitch SD (e.g. stride, low SNR).
SD-r2 values come from the validated GRADE table in
analysis/pitch_segment_consistency.py; value-r2 from the adopted validation."""
import os, sys, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import config
from pitch_segment_consistency import GRADE

INK = "#0E1B33"; TEAL = "#0FA3B1"; VIOLET = "#7C3AED"

VALUE_R2 = {  # adopted value-r2 @ best side view (0°)
    "lead_knee_angle": 0.94, "stride_pct_height": 0.82, "trunk_anterior_tilt": 0.70,
    "knee_ext_velo_br": 0.65, "wrist_speed": 0.60, "release_height": 0.99,
}
LABEL = {"lead_knee_angle": "lead knee angle", "stride_pct_height": "stride",
         "trunk_anterior_tilt": "trunk tilt", "knee_ext_velo_br": "knee ext. velo",
         "wrist_speed": "wrist speed", "release_height": "release height"}
SNR = {"lead_knee_angle": 5.15, "stride_pct_height": 0.23}   # headline SNR points


def sd_r2(key):
    m = re.search(r"SD r2=([0-9.]+)", GRADE[key])
    return float(m.group(1)) if m else np.nan


def main():
    keys = sorted(VALUE_R2, key=lambda k: VALUE_R2[k])   # ascending -> best on top
    y = np.arange(len(keys)); h = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.barh(y + h/2, [VALUE_R2[k] for k in keys], h, color=TEAL, label="value  $r^2$ (2D vs 3D)")
    ax.barh(y - h/2, [sd_r2(k) for k in keys], h, color=VIOLET, label="repeatability  $r^2$ (between-pitch SD)")

    ax.axvline(0.6, ls="--", color="0.45", lw=1.2)
    ax.text(0.6, -1.05, "$r^2\\geq0.6$", fontsize=9, color="0.35", ha="center", va="top")
    for i, k in enumerate(keys):
        ax.text(VALUE_R2[k]+0.01, i+h/2, f"{VALUE_R2[k]:.2f}", va="center", fontsize=8, color=INK)
        ax.text(sd_r2(k)+0.01, i-h/2, f"{sd_r2(k):.2f}", va="center", fontsize=8, color=INK)

    # reserved SNR column on the right (only the two headline points)
    ax.axvline(1.12, color="0.85", lw=0.8)
    ax.text(1.20, len(keys)-0.35, "SNR", fontsize=10, color=INK, weight="bold", ha="center")
    for i, k in enumerate(keys):
        if k in SNR:
            col = TEAL if SNR[k] >= 1 else VIOLET
            ax.text(1.20, i, f"{SNR[k]:.2f}", va="center", ha="center", fontsize=11,
                    color=col, weight="bold")
            tag = "both recover" if SNR[k] >= 1 else "repeatability lost"
            ax.text(1.20, i-0.32, tag, va="center", ha="center", fontsize=7.5, color=col)

    ax.set_yticks(y); ax.set_yticklabels([LABEL[k] for k in keys])
    ax.set_xlim(0, 1.3); ax.set_ylim(-1.2, len(keys)-0.2); ax.set_xlabel("$r^2$")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Measuring the value $\\neq$ measuring its repeatability (SNR)",
                 fontsize=12, color=INK, weight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    out = os.path.join(config.OBP_VALIDATION_DIR, "fig_snr_repeatability.png")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); print("saved ->", out)


if __name__ == "__main__":
    main()
