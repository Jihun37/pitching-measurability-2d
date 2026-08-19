"""The graded measurability map (2026-07-27 final design).

Two figures, exactly per docs/MAP_FIGURE_SPEC.md:

  fig_graded_map.png    small multiples, one panel per map row: the CONTINUOUS
                        LOCO-CCC field over azimuth x elevation, with two contours
                        (0.75 moderate, 0.80 strong). The field plus two lines is
                        the point -- the reader can draw their own threshold.
                        Cells with r2 >= 0.60 but CCC < 0.75 (association without
                        agreement) are hatched, but ONLY in the panels where they
                        cluster: there are just 23 in the whole map, so hatching
                        everywhere reads as speckle.
  fig_graded_strip.png  all map rows at once, azimuth on x at each metric's best
                        elevation, filled strong / moderate / off-map. This is the
                        figure that answers "where do I stand to measure X".

Colour scale is shared with the existing map figures (angle_map_2d_fig.CMAP) so the
paper's figures read as one family.

Run:  conda activate diamond
      cd src\\viz
      python fig_graded_map.py
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../analysis"):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
from angle_map_2d_fig import CMAP

# Repointed 2026-07-29 (GT dedup): the map is regenerated in place now, and the
# PRE_DEDUP_394_*.csv copies are the pre-dedup history. Reading those would draw
# the paper's headline figure on the retired 52-row evaluation set.
GATE = os.path.join(config.OBP_VALIDATION_DIR, "gate_map.csv")
SUMM = os.path.join(config.OBP_VALIDATION_DIR, "layer_summary.csv")
OUT = os.path.join(config.ROOT, "data", "outputs", "viz")
STRONG, MODERATE = 0.80, 0.75
INK, MUTE = "#0E1B33", "#64748B"
C_STRONG, C_MOD, C_OFF = "#0C447C", "#85B7EB", "#EDF2F7"

# IEEE two-column float width. Body figures are AUTHORED at this width so their
# font sizes are the sizes that print. Authoring wider and letting
# \includegraphics shrink the PNG scales every label down with it.
BODY_W = 7.16
VMIN, VMAX = 0.30, 1.00
# Contour colours, named once: the contours, the colourbar rules and the colourbar
# tick labels all have to agree, and they did not when each carried its own literal.
C_MOD_LINE, C_STRONG_LINE = "#E8A33D", "#D92B4B"
# Where the tone jumps, and how much of the ramp each band gets. The unused gaps,
# 0.40 to 0.55 and 0.70 to 0.80, ARE the jumps.
BANDS = ((0.00, 0.40), (0.55, 0.70), (0.80, 1.00))


# Rows that are the SAME estimator reached through a second pipeline (a cross-check,
# values agree to dump rounding), plus the one row that is a second, weaker METHOD for
# a quantity already listed. MAP_FIGURE_SPEC: "disclose the pairs - discovered they
# look like padding, disclosed they are validation." Marked exactly as in
# FINAL_GATE_MAP_394.md, which uses the same dagger for both kinds.
# NOTE `arm_slot` (forearm) is NOT here: it is a DIFFERENT definition from the adopted
# shoulder-wrist Arm Slot, not a duplicate path.
# EMPTIED 2026-07-29 by the GT dedup. These held the rows that shared a
# biomechanical target with another row, and the dagger marked that pairing. The
# duplicate partners were removed from the evaluation set, so no row in the map
# has a partner any more and nothing should be daggered. Do NOT repopulate these
# from memory: if a pairing is ever reintroduced, read it from
# layer_summary.pairs_with instead of hardcoding names here.
DUP_ROWS = set()
ALT_ROWS = set()


# EMPTIED 2026-08-18. This separated the two arm-slot rows by definition, and the
# shoulder-wrist one left with the five direct-3D rows, so `arm_slot' is now the only
# arm slot in the set and the "(forearm)" qualifier contrasted with nothing.
DISAMBIG = {}


# metric_id -> the OBP column the row is scored against. Read once, lazily, so a
# figure that never labels a registry row does not pay for the file.
_COLUMN_OF = None


def column_of(name):
    """The OBP column a registry row is scored against, or the name unchanged."""
    global _COLUMN_OF
    if _COLUMN_OF is None:
        reg = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR,
                                       "paper_registry.csv"))
        # every row is scored against a column, and no two share one; if that ever
        # stops holding, a silent many-to-one rename would merge rows in the axis
        assert (reg.truth_source == "obp_column").all(), "a row is not column-scored"
        assert reg.truth_quantity.is_unique, "two rows share a truth column"
        _COLUMN_OF = dict(zip(reg.metric_id, reg.truth_quantity))
    return _COLUMN_OF.get(name, name)


# Captions used to hard-code "23 cells", which a re-render silently kept while the
# data moved to 20. Every caption now interpolates this, set once from the CSV in
# main(). Never type the number into a string again.
HATCH_N = None


def stepped(cmap, cuts, spans, n=512):
    """`cmap` remapped so its tone jumps at each grade level.

    Continuous inside each band, discontinuous between them. The field is still a
    field, which is the point of drawing one, but `graded' is now legible in the
    colour instead of resting entirely on two thin contours."""
    edges = [VMIN] + list(cuts) + [VMAX]
    lut = []
    for i in range(n):
        v = VMIN + (VMAX - VMIN) * i / float(n - 1)
        k = max(j for j in range(len(edges) - 1) if v >= edges[j])
        lo, hi = edges[k], edges[k + 1]
        a, b = spans[k]
        f = 0.0 if hi <= lo else min(max((v - lo) / (hi - lo), 0.0), 1.0)
        lut.append(cmap(a + (b - a) * f))
    return ListedColormap(lut)


GRADED_CMAP = stepped(CMAP, (MODERATE, STRONG), BANDS)


def short(name):
    if name in DISAMBIG:
        return DISAMBIG[name]
    # the label is the column, not the pipeline's own name for the row
    name = column_of(name)
    # The abbreviations are what let the titles fit a 1.1 in panel: three
    # "glove shoulder ..." rows sit side by side in one grid row and overprinted each
    # other at full length. Order matters -- "glove shoulder" before "shoulder".
    s = (name.replace(" [O]", "").replace("_", " ")
             .replace("horizontal abduction", "hz abd")
             .replace("extension angular velo", "ext velo")
             .replace("extension from fp to br", "ext fp-br")
             .replace("rotational", "rot")
             .replace("rotation", "rot")
             .replace("glove shoulder", "glove sh")
             .replace("shoulder", "sh")
             .replace("abduction", "abd")
             .replace("anterior", "ant")
             .replace("lateral", "lat")
             .replace("separation", "sep")[:34])
    if name in DUP_ROWS:
        s = "‡ " + s
    elif name in ALT_ROWS:
        s = "‡ " + s + " (alt method)"
    return s


def grid(sub, col):
    """(el ascending upward) x (az) matrix for one metric."""
    az = sorted(sub.az.unique()); el = sorted(sub.el.unique())
    M = np.full((len(el), len(az)), np.nan)
    for r in sub.itertuples(index=False):
        M[el.index(r.el), az.index(r.az)] = getattr(r, col)
    return M, az, el


# --- panel grid geometry (2026-08-03) ------------------------------------------------
# The grid was 7.16 x 8.92 in, near enough a whole page, and 37 % of its height was the
# gap between rows. The gap was that wide because EVERY panel carried its own y ticks
# (7 labels) and x ticks (4 labels) -- 385 tick labels for 11 distinct values. The axes
# are shared now, so only the left column and the bottom row are labelled and the gap
# has to clear a title and nothing else.
#
# PANEL_AR is the second saving. Each field is 24 azimuths x 7 elevations, natural
# aspect 0.29; the old panels were stretched to 0.65, which inflated the height and made
# every field look taller than it is. 0.50 is still stretched -- enough to keep seven
# elevation rows resolvable -- but far closer to the data.
PANEL_AR = 0.50               # panel height / panel width
L, R, TOP = 0.052, 0.918, 0.958
WSP, HSP = 0.14, 0.32         # HSP has to clear one panel title, nothing else
BOT_IN = 0.26                 # x tick labels + the one-line azimuth axis name
FS_TITLE, FS_TICK, FS_AXLAB = 5.4, 5.0, 6.4


def panels(g, s, out_png, width_in=BODY_W):
    """Body figure, authored at its placement width (see strip()). The legend
    that used to sit in a suptitle block moved to the caption and to Sec. IV-A."""
    rows = s[s.map_cells > 0].sort_values(
        ["strong_cells", "map_cells"], ascending=False)
    n = len(rows)
    ncol, nrow = 5, int(np.ceil(n / 5))

    # height is DERIVED from the panel aspect, so no row gap is larger than the title
    # it exists to carry
    panel_w = width_in * (R - L) / (ncol + (ncol - 1) * WSP)
    panel_h = PANEL_AR * panel_w
    grid_h = nrow * panel_h + (nrow - 1) * HSP * panel_h
    fig_h = (grid_h + BOT_IN) / TOP

    fig, axes = plt.subplots(nrow, ncol, figsize=(width_in, fig_h),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, r in zip(axes, rows.itertuples(index=False)):
        sub = g[(g.metric == r.metric) & (g.source == r.source)]
        M, az, el = grid(sub, "ccc")
        H, _, _ = grid(sub, "hatch")
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=GRADED_CMAP,
                       vmin=VMIN, vmax=VMAX, interpolation="nearest",
                       extent=[-0.5, len(az) - 0.5, -0.5, len(el) - 0.5])
        # Two contours on the continuous field -- both are required (the map is graded,
        # not gated), but they are NOT co-equal marks. Drawn at the same weight they run
        # a pixel apart wherever the field is steep and read as one fuzzy double line.
        # Strong is the primary contour; moderate is a lighter, thinner outer edge.
        Mf = np.where(np.isfinite(M), M, VMIN)
        for lev, col, lw in ((MODERATE, C_MOD_LINE, 0.8),
                             (STRONG, C_STRONG_LINE, 1.05)):
            if np.nanmax(Mf) >= lev >= np.nanmin(Mf):
                ax.contour(Mf, levels=[lev], colors=[col], linewidths=lw)
        # association without agreement -- only where it clusters
        if np.nansum(H) >= 2:
            ys, xs = np.where(H == 1)
            ax.scatter(xs, ys, marker="x", s=6, c=C_STRONG_LINE, linewidths=0.7)
        # F4 (2026-08-08): graded cells that are NOT stable under a pitcher-level
        # cluster bootstrap. Marked, not hidden: 31 of the 1,151 graded cells fall
        # below P(graded) = 0.50 over B = 200 resamples, spread thinly over 17 rows,
        # so most panels carry none and none carries many. A small hollow ring reads
        # as "this cell is a coin flip" without competing with the red x, which means
        # something else entirely (association without agreement).
        if "p_graded" in sub.columns:
            P, _, _ = grid(sub, "p_graded")
            unstable = np.isfinite(M) & (M >= MODERATE) & np.isfinite(P) & (P < 0.50)
            if unstable.any():
                ys, xs = np.where(unstable)
                # WHITE, not ink: a ringed cell is graded by definition, so it always
                # sits in the dark half of the colormap and an ink ring disappears
                # into it (verified on Stride Angle and rot hip sh sep fp).
                ax.scatter(xs, ys, marker="o", s=10, facecolors="none",
                           edgecolors="white", linewidths=0.7, zorder=5)
        ax.set_xticks(range(0, len(az), 6))
        ax.set_xticklabels([str(az[i]) for i in range(0, len(az), 6)])
        ax.set_yticks(range(len(el)))
        ax.set_yticklabels([str(e) for e in el])
        ax.tick_params(labelsize=FS_TICK, length=1.8, pad=1.2, color="#CBD5E1")
        # 5.4 pt in a 1.15 in panel holds ~26 characters. The [S]/[M] tag was
        # dropped: the contours already say which grade the panel reaches, and
        # keeping it overflowed the title into the neighbouring column.
        ax.set_title(f"{short(r.metric)[:24]} {r.best_ccc:.2f}",
                     fontsize=FS_TITLE, color=INK, weight="bold", pad=1.6)
        for sp in ax.spines.values():
            sp.set_color("#CBD5E1")
    for ax in axes[n:]:
        ax.axis("off")

    fig.subplots_adjust(left=L, right=R, top=TOP, bottom=BOT_IN / fig_h,
                        wspace=WSP, hspace=HSP)

    # Shared axis NAMES only. The azimuth convention ("az0 = open side (3B for a RHP),
    # az90 = front") was printed here and does NOT belong in the figure -- it is
    # Methods/caption text, and the paper's captions carry it. Axis furniture stays,
    # explanation goes in the caption. Do not reintroduce the clause.
    fig.text((L + R) / 2.0, 0.010, "camera azimuth (deg)",
             ha="center", va="bottom", fontsize=FS_AXLAB, color=INK)
    fig.text(0.008, (TOP + BOT_IN / fig_h) / 2.0, "camera elevation (deg)",
             ha="left", va="center", rotation=90, fontsize=FS_AXLAB, color=INK)

    cax = fig.add_axes([R + 0.018, BOT_IN / fig_h, 0.011, TOP - BOT_IN / fig_h])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("agreement (LOCO CCC)", fontsize=FS_AXLAB, color=INK, labelpad=2)
    # The two levels the paper grades on, named on the bar in the colours of the
    # contours that mark them. Without this nothing in the figure said which line was
    # which, and how narrow the band between them is was invisible.
    cb.set_ticks([0.3, 0.4, 0.5, 0.6, MODERATE, STRONG, 0.9, 1.0])
    cb.ax.tick_params(labelsize=FS_TICK, length=2, pad=1.5)
    for lev, col in ((MODERATE, C_MOD_LINE), (STRONG, C_STRONG_LINE)):
        cb.ax.axhline(lev, color=col, lw=1.1)
    for lab, v in zip(cb.ax.get_yticklabels(), cb.get_ticks()):
        if abs(v - MODERATE) < 1e-9:
            lab.set_color(C_MOD_LINE); lab.set_weight("bold")
        elif abs(v - STRONG) < 1e-9:
            lab.set_color(C_STRONG_LINE); lab.set_weight("bold")
    cb.outline.set_edgecolor("#CBD5E1")

    # A title is centred, so it may spill into the column gap -- but not past its half,
    # or it collides with the neighbour. Three "glove shoulder ..." rows in one grid row
    # overprinted each other before the abbreviations in short(); this catches the next
    # long name instead of leaving it for a reader to notice.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for ax, r in zip(axes[:n], rows.itertuples(index=False)):
        ab = ax.get_window_extent(rend)
        tb = ax.title.get_window_extent(rend)
        room = WSP * ab.width / 2.0
        assert (ab.x0 - tb.x0) <= room and (tb.x1 - ab.x1) <= room, \
            f"title '{ax.get_title()}' overruns the column gap; abbreviate in short()"

    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"saved -> {out_png}   {width_in:.2f} x {fig_h:.2f} in")


def strip(g, s, out_png, width_in=BODY_W):
    """Body figure. Authored at the width it is placed at (\\includegraphics
    width=7.16in), so every font size here is the size it prints at. Drawing it
    wider and letting LaTeX shrink it is what made the old 8 pt labels render at
    4.6 pt. The in-figure title and the association-without-agreement note were
    removed 2026-07-29: the caption carries the first and Sec. IV-A the second."""
    rows = s[s.map_cells > 0].sort_values(
        ["strong_cells", "map_cells"], ascending=True)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(width_in, 0.132 * n + 0.95))
    az = sorted(g.az.unique())
    for i, r in enumerate(rows.itertuples(index=False)):
        sub = g[(g.metric == r.metric) & (g.source == r.source)]
        el = int(r.best_ccc_view.split("/")[1])
        row = sub[sub.el == el].set_index("az")
        for j, a in enumerate(az):
            if a not in row.index:
                continue
            gr = row.loc[a, "grade"]
            c = C_STRONG if gr == "strong" else (C_MOD if gr == "moderate" else C_OFF)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.42), 1, 0.84,
                                       facecolor=c, edgecolor="white", lw=0.4))
    ax.set_xlim(-0.5, len(az) - 0.5); ax.set_ylim(-0.7, n - 0.3)
    ax.set_xticks(range(len(az)))
    ax.set_xticklabels([str(a) for a in az], fontsize=5.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{short(r.metric)}  (el {r.best_ccc_view.split('/')[1]}°)"
                        for r in rows.itertuples(index=False)], fontsize=6.0)
    # axis NAME only -- the az0/az90 convention that used to be appended here is
    # caption text, same as in panels()
    ax.set_xlabel("camera azimuth (deg)", fontsize=7.0)
    ax.legend(handles=[Patch(facecolor=C_STRONG, label="strong (CCC ≥ 0.80)"),
                       Patch(facecolor=C_MOD, label="moderate (0.75–0.80)"),
                       Patch(facecolor=C_OFF, label="below 0.75")],
              loc="lower center", bbox_to_anchor=(0.5, -0.10 - 0.9 / n),
              ncol=3, frameon=False, fontsize=6.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print("saved ->", out_png)


def field_panels(g, rows, out_png, col, cmap, vmin, vmax, levels, cbar_label,
                 title, sub, ncol=5, invert_good=False):
    """Generic small-multiples renderer: one continuous field per metric row."""
    n = len(rows)
    nrow = int(np.ceil(n / ncol))
    # reserve a FIXED header height in inches: a fraction-based top margin that
    # suits an 8-row grid buries the panel titles in a 3-row one
    head_in = 1.05
    fig_h = 1.62 * nrow + head_in
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.05 * ncol, fig_h))
    axes = np.atleast_1d(axes).ravel()
    im = None
    for ax, r in zip(axes, rows.itertuples(index=False)):
        sub_g = g[(g.metric == r.metric) & (g.source == r.source)]
        M, az, el = grid(sub_g, col)
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest",
                       extent=[-0.5, len(az) - 0.5, -0.5, len(el) - 0.5])
        Mf = np.where(np.isfinite(M), M, vmin if not invert_good else vmax)
        for lev, lc, lw in levels:
            if np.nanmax(Mf) >= lev >= np.nanmin(Mf):
                ax.contour(Mf, levels=[lev], colors=[lc], linewidths=lw)
        ax.set_xticks(range(0, len(az), 6))
        ax.set_xticklabels([str(az[i]) for i in range(0, len(az), 6)], fontsize=6.5)
        ax.set_yticks(range(len(el)))
        ax.set_yticklabels([str(e) for e in el], fontsize=6)
        val = getattr(r, "best_ccc", np.nan)
        ax.set_title(f"{short(r.metric)}  {val:.2f}" if np.isfinite(val)
                     else short(r.metric),
                     fontsize=7.6, color=INK, weight="bold", pad=3)
        for sp in ax.spines.values():
            sp.set_color("#CBD5E1")
    for ax in axes[n:]:
        ax.axis("off")
    top = 1.0 - head_in / fig_h
    fig.suptitle(title, fontsize=14, weight="bold", color=INK,
                 y=1.0 - 0.22 / fig_h)
    fig.text(0.5, 1.0 - 0.52 / fig_h, sub, ha="center", va="top",
             fontsize=8.6, color=MUTE, wrap=True)
    fig.subplots_adjust(left=0.035, right=0.93, top=top, bottom=0.10 / fig_h * 3,
                        wspace=0.30, hspace=0.62)
    cax = fig.add_axes([0.945, 0.25, 0.011, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label, fontsize=9, color=INK)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print("saved ->", out_png)


def truth_sd_by_metric():
    """SD of each truth over the pitch population, from the pair dumps. Needed to turn MAE
    into NMAE, the only accuracy number that is comparable across metrics: NMAE 1
    means the typical error is the whole spread of the population."""
    out = {}
    for f in ("angle_zone_pairs_gt.csv.gz", "rejected_gt_pairs.csv.gz"):
        p = os.path.join(config.OBP_VALIDATION_DIR, f)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, usecols=["metric", "session_pitch", "truth"])
        for m, sub in d.groupby("metric"):
            v = sub.drop_duplicates("session_pitch").truth
            out[m] = float(np.nanstd(v, ddof=1))
    return out


def main():
    global HATCH_N
    os.makedirs(OUT, exist_ok=True)
    g = pd.read_csv(GATE)
    s = pd.read_csv(SUMM)
    HATCH_N = int(g.hatch.sum())
    # bootstrap stability, if it has been built. Optional on purpose: the map figure
    # must still render from gate_map.csv alone.
    bpath = os.path.join(config.OBP_VALIDATION_DIR, "review_bootstrap_cells.csv")
    if os.path.exists(bpath):
        b = pd.read_csv(bpath)[["metric", "az", "el", "p_graded", "p_strong"]]
        g = g.merge(b, on=["metric", "az", "el"], how="left")
        gr = g[g.grade.isin(["strong", "moderate"])]
        n_un = int((gr.p_graded < 0.50).sum())
        print(f"bootstrap: P(graded) median {gr.p_graded.median():.3f} over graded "
              f"cells;  {int((gr.p_graded >= 0.95).sum())} of {len(gr)} at >= 0.95;  "
              f"{n_un} below 0.50 over {gr[gr.p_graded < 0.50].metric.nunique()} rows "
              f"-- these are ringed in the panels")
    else:
        print("bootstrap stability not found -- panels render without the rings")
    m = s[s.map_cells > 0]
    print(f"map rows {len(m)}  = strong-capable "
          f"{int((m.metric_grade == 'strong-capable').sum())} + moderate-only "
          f"{int((m.metric_grade == 'moderate-only').sum())};  "
          f"cells strong {int((g.grade == 'strong').sum())} / moderate "
          f"{int((g.grade == 'moderate').sum())};  hatch {int(g.hatch.sum())}")
    panels(g, s, os.path.join(OUT, "fig_graded_map.png"))
    strip(g, s, os.path.join(OUT, "fig_graded_strip.png"))

    # ---- S1: the association layer, ALL 42 rows -------------------------
    allrows = s.sort_values(["map_cells", "r2_cells"], ascending=False)
    field_panels(
        g, allrows, os.path.join(OUT, "figS1_association_map.png"), "r2",
        CMAP, 0.0, 1.0, [(0.60, "#F2A900", 1.5), (0.80, "#D92B4B", 1.6)],
        "association (r²)",
        "S1 — Association layer: r² over azimuth × elevation, all 42 rows",
        "contours: amber r² = 0.60, red = 0.80.  This layer is NOT a gate — "
        "CCC ≤ |r| always, so the agreement map already carries an association "
        "floor inside it.  n = 394")

    # ---- S2: the rows that never reach the map --------------------------
    off = s[s.map_cells == 0].sort_values("best_ccc", ascending=False)
    field_panels(
        g, off, os.path.join(OUT, "figS2_off_map_rows.png"), "ccc",
        CMAP, VMIN, VMAX, [(MODERATE, "#F2A900", 1.5)],
        "agreement (LOCO CCC)",
        f"S2 — The {len(off)} rows that reach no cell of the map",
        "Nothing is silently dropped: every rejected quantity is shown with its "
        "own field. The amber 0.75 contour appears only where a row grazes the "
        "moderate line.  Title value = best CCC anywhere.")

    # ---- S3: accuracy in units of the population spread ------------------
    sd = truth_sd_by_metric()
    g2 = g.copy()
    g2["nmae"] = g2.mae / g2.metric.map(sd)
    field_panels(
        g2, s[s.map_cells > 0].sort_values(["strong_cells", "map_cells"],
                                           ascending=False),
        os.path.join(OUT, "figS3_nmae_map.png"), "nmae",
        CMAP + "_r" if isinstance(CMAP, str) else CMAP.reversed(),
        0.0, 1.2, [(0.5, "#D92B4B", 1.6)], "NMAE  (MAE ÷ population SD)",
        "S3 — Accuracy in units of the population spread",
        "Calibrated MAE divided by the SD of the truth over the pitch "
        "population. "
        "Red contour = 0.5, i.e. the typical error is half that entire spread; "
        "at NMAE ≥ 1 the reading cannot separate two pitchers. "
        "The full per-cell table (MAE, pitcher-bias SD, calibration model) is "
        "gate_map.csv.")


if __name__ == "__main__":
    main()
