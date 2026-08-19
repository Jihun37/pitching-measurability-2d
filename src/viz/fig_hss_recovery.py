"""
SUPPLEMENT ONLY (renamed figS4, 2026-07-28).

The standalone HSS recovery curve is NOT a body figure. Hip-Shoulder Sep is
presented as a MODERATE-ONLY row of the main graded map (best CCC 0.780,
40 moderate cells) -- see docs/PAPER_MASTER.md 1.1 and FINAL_GATE_MAP_394.md.
This elevation curve may be cited as supplementary support only.
"""
"""Figure (Act 1, section 4.4): the overhead counterexample.

Two transverse-plane metrics that a ground camera cannot see, hip-shoulder
separation and pelvis rotational velocity, both recover as the camera is raised
overhead. Measurability is therefore a property of (metric x viewpoint), not of
the metric alone. Each curve is the mean r2 over azimuth at each elevation, with
a band spanning the azimuth min-max, so the recovery is shown to be
azimuth-independent as well as monotonic.

Data:
  HSS   hss_elevation_features.csv (hss_max) vs poi max_rotation_hip_shoulder_separation.
  Pelvis sequencing_overhead_deploy.csv (pel_mag, 360 fps) vs max_pelvis_rotational_velo.
Both are clean-projection ceilings.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

INK = "#0E1B33"; HSS_C = "#0C447C"; PEL_C = "#0FA3B1"
V = config.OBP_VALIDATION_DIR


def r2(a, b):
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < 3:
        return np.nan
    r = d["a"].corr(d["b"])
    return r * r if pd.notna(r) else np.nan


def curve(df, el_col, az_col, est_col, truth_col):
    """mean r2 over azimuth per elevation, plus min/max band."""
    els, mean, lo, hi = [], [], [], []
    for el in sorted(df[el_col].unique()):
        sub = df[df[el_col] == el]
        vals = [r2(sub[sub[az_col] == az][est_col], sub[sub[az_col] == az][truth_col])
                for az in sorted(sub[az_col].unique())]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        els.append(el); mean.append(np.mean(vals)); lo.append(min(vals)); hi.append(max(vals))
    return np.array(els), np.array(mean), np.array(lo), np.array(hi)


def main():
    # HSS
    h = pd.read_csv(os.path.join(V, "hss_elevation_features.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    h = h.merge(poi[["session_pitch", "max_rotation_hip_shoulder_separation"]], on="session_pitch")
    eh, mh, lh, hh = curve(h, "el", "az", "hss_max", "max_rotation_hip_shoulder_separation")

    # Pelvis rotational velocity (360 fps line)
    p = pd.read_csv(os.path.join(V, "sequencing_overhead_deploy.csv"))
    # drop the no-projection control rows (elev=-1, az=-1) so only real views remain
    p = p[(p.fps == 360) & (p.elev >= 0) & (p.az >= 0)]
    ep, mp, lp, hp = curve(p, "elev", "az", "pel_mag", "max_pelvis_rotational_velo")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.axhspan(0, 0.6, color="0.93", zorder=0)
    ax.axhline(0.6, ls="--", color="0.45", lw=1.2, zorder=1)
    ax.text(2, 0.615, "usability threshold  $r^2\\geq0.6$", fontsize=9, color="0.35", va="bottom")

    ax.fill_between(eh, lh, hh, color=HSS_C, alpha=0.13, zorder=2)
    ax.plot(eh, mh, marker="o", ms=5, lw=2.3, color=HSS_C, zorder=3,
            label="hip-shoulder separation")
    ax.fill_between(ep, lp, hp, color=PEL_C, alpha=0.15, zorder=2)
    ax.plot(ep, mp, marker="s", ms=5, lw=2.3, color=PEL_C, zorder=3,
            label="pelvis rotational velocity")

    ax.annotate("dead on the ground\n(depth-axis rotation lost)", xy=(0, 0.09),
                xytext=(12, 0.24), fontsize=9, color="0.30",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    ax.annotate("both recover overhead", xy=(85, mp[-1]), xytext=(40, 0.90), fontsize=9.5,
                color=INK, weight="bold",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))

    ax.set_xlabel("camera elevation (deg)   0 = ground, 90 = overhead", fontsize=10)
    ax.set_ylabel("$r^2$   (2D vs 3D)", fontsize=10)
    ax.set_title("Measurability is (metric $\\times$ viewpoint), not the metric alone",
                 fontsize=12, color=INK, weight="bold")
    ax.set_xlim(-3, 93); ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.grid(True, alpha=0.22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out = os.path.join(V, "figS4_hss_recovery.png")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); print("saved ->", out)


if __name__ == "__main__":
    main()
