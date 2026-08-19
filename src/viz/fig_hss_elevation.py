"""Body figure: elevation is an axis of measurability, not a nuisance parameter.

Hip-shoulder separation is the extreme case in the whole evaluation set. It is a
transverse-plane quantity, so a ground camera sees it almost not at all; raising the
camera recovers it monotonically -- and even at the top it stops at the moderate grade.
One row therefore carries two claims at once: measurability is a property of
(quantity x viewpoint), and the map's moderate contour is a real ceiling for some
quantities rather than a formality.

Source: hss_elevation.csv (analysis/viewpoint_anchor_check.py), the canonical elevation
table -- best LOCO CCC over azimuth at each swept elevation, with the azimuth attaining
it and the number of graded cells at that elevation.

NOTE this is NOT figS4_hss_recovery.png (viz/fig_hss_recovery.py). That figure predates
the freeze: it recomputes a mean-r2-over-azimuth curve from hss_elevation_features.csv
and sequencing_overhead_deploy.csv, adds a second quantity from a deployment dump, and
labels r2 = 0.60 a "usability threshold" -- which contradicts the frozen vocabulary,
where r2 0.60 is a screening pre-filter and the grade is a CCC contour. This script
reads the canonical CSV and plots the graded quantity instead. figS4 is left untouched.

Run:  conda activate diamond
      cd src\\viz
      python fig_hss_elevation.py
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", ".."):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
from fig_graded_map import INK, MUTE, BODY_W

CSV = os.path.join(config.OBP_VALIDATION_DIR, "hss_elevation.csv")
REG = os.path.join(config.OBP_VALIDATION_DIR, "paper_registry.csv")
OUT = os.path.join(config.ROOT, "data", "outputs", "viz")

LINE = "#0C447C"; BAR = "#85B7EB"
STRONG_C, MOD_C = "#D92B4B", "#F2A900"
STRONG, MODERATE = 0.80, 0.75
# half the figure width: this is a single-column float
COL_W = BODY_W / 2


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(CSV).sort_values("el").reset_index(drop=True)
    reg = pd.read_csv(REG)
    row = reg[reg.metric_id == "Hip-Shoulder Sep [O]"].iloc[0]

    peak = d.loc[d.best_ccc.idxmax()]
    graded_tot = int(d.graded_cells.sum())
    first_el = int(d[d.graded_cells > 0].el.min())
    print("elevation ladder (el, best_ccc, best_az, best_r2, graded, strong):")
    for r in d.itertuples(index=False):
        print(f"  el {r.el:>2}  ccc {r.best_ccc:.3f}  az {r.best_az:>3}  "
              f"r2 {r.best_r2:.3f}  graded {r.graded_cells:>2}  strong {r.strong_cells}")
    print(f"peak CCC {peak.best_ccc:.4f} at el {int(peak.el)} (az {int(peak.best_az)});  "
          f"graded cells {graded_tot}, all at el >= {first_el};  "
          f"strong cells {int(d.strong_cells.sum())}")
    print(f"registry: grade={row.metric_grade}  graded_cells={row.graded_cells}  "
          f"best {row.best_ccc} @ {row.best_az}/{row.best_el}")
    assert graded_tot == int(row.graded_cells), "elevation table disagrees with registry"
    assert int(d.strong_cells.sum()) == 0, "HSS is moderate-only in the registry"

    fig, ax = plt.subplots(figsize=(COL_W, 2.55))

    # graded cells per elevation, behind the curve, on their own right axis
    ax2 = ax.twinx()
    ax2.bar(d.el, d.graded_cells, width=7.5, color=BAR, alpha=0.55,
            edgecolor="white", lw=0.4, zorder=1)
    ax2.set_ylim(0, max(d.graded_cells.max() * 3.1, 1))
    ax2.set_ylabel("graded cells", fontsize=6.2, color="#5B7FA6")
    ax2.tick_params(axis="y", labelsize=5.6, colors="#5B7FA6", length=2, pad=1)
    for sp in ("top", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color("#CBD5E1")

    # the two grade contours
    for lev, c, lab in ((STRONG, STRONG_C, f"strong  {STRONG:.2f}"),
                        (MODERATE, MOD_C, f"moderate  {MODERATE:.2f}")):
        ax.axhline(lev, color=c, ls="--", lw=0.9, zorder=2)
        ax.text(-2, lev + 0.012, lab, fontsize=5.4, color=c, ha="left",
                va="bottom", weight="bold")

    ax.plot(d.el, d.best_ccc, marker="o", ms=3.4, lw=1.8, color=LINE, zorder=4)
    # Where the curve has already flattened, a label just under the marker sits on
    # the moderate contour, so those drop below it instead.
    FLAT_FROM = 60
    for r in d.itertuples(index=False):
        flat = r.el >= FLAT_FROM
        ha, dx, dy = ("center", 0, -9.5) if flat else ("left", 3.5, -3.0)
        ax.annotate(f"az{int(r.best_az)}", (r.el, r.best_ccc),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=4.6, color=MUTE, ha=ha, va="top")
    ax.annotate(f"{peak.best_ccc:.3f}", (peak.el, peak.best_ccc),
                textcoords="offset points", xytext=(0, 5.5), fontsize=5.8,
                color=LINE, ha="center", weight="bold")

    ax.set_xlim(-4, 92)
    ax.set_ylim(0.20, 0.90)
    ax.set_xticks(sorted(d.el.tolist()))   # no 90 tick: el 85 sits next to it and the
                                           # axis label already says 90 = overhead
    ax.tick_params(axis="both", labelsize=5.8, length=2, pad=1.5)
    ax.set_xlabel("camera elevation (deg)    0 = ground, 90 = overhead", fontsize=6.5)
    ax.set_ylabel("best agreement over azimuth\n(LOCO CCC)", fontsize=6.5)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.grid(axis="y", alpha=0.14, lw=0.5)
    ax.set_axisbelow(False)

    fig.tight_layout(pad=0.30)
    out = os.path.join(OUT, "fig_hss_elevation.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("saved ->", out)


if __name__ == "__main__":
    main()
