"""Sec. IV-D — how many frames of anchor error each map cell tolerates.

Reads ONLY `event_tolerance_map.csv`. Every number in the figure is computed here
from that file; none is hardcoded. Rewritten 2026-07-29.

WHAT CHANGED. The old version read `event_tolerance.csv`, which covered the 23
SCREENED rows only, and quoted a 746-cell denominator carried over from the
pre-dedup 52-row evaluation set. Those numbers (746 / 537 / 63) are obsolete and
must not reappear. The merged table adds the 10 adopted rows that read an external
anchor, so the denominator is now the graded cells of all 33 applicable rows.

DENOMINATOR. `applicable=True` only. Two map rows read no external event anchor --
wrist speed takes a whole-clip maximum, hip-shoulder separation locates its own
signature anchor -- so a uniform shift is undefined for them and their graded cells
are excluded from every percentage. That exclusion is a scope statement, not a
finding that those rows are robust.

SCOPE. This is a TIME-SENSITIVITY layer, not a deployability criterion. A uniform
shift displaces every pitch by the same k, which a calibration absorbs; a detector
errs independently per pitch, which it cannot. The deployability verdict is the
empirical detected-event map of Sec. VI, never this figure.

Run:  conda activate diamond
      cd src\\viz
      python fig_event_tolerance.py
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", ".."):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
from fig_graded_map import INK, MUTE, C_STRONG, C_MOD, BODY_W

SRC = os.path.join(config.OBP_VALIDATION_DIR, "event_tolerance_map.csv")
OUT = os.path.join(config.ROOT, "data", "outputs", "viz",
                   "fig_event_tolerance.png")


def main():
    d = pd.read_csv(SRC)
    na = d[~d.applicable]
    m = d[d.applicable & (d.tol_map >= 0)].copy()
    N = len(m)
    counts = [int((m.tol_map == k).sum()) for k in (0, 1, 2, 3)]
    surv = [float((m.tol_map >= k).mean()) for k in (0, 1, 2, 3)]
    print(f"applicable rows      : {m.metric.nunique()}")
    print(f"excluded as N/A      : {na.metric.nunique()} rows, {len(na)} cells "
          f"({sorted(na.metric.unique())})")
    print(f"denominator          : {N} graded cells with a shiftable anchor")
    print(f"tol_map counts       : {counts}  (sum {sum(counts)})")
    for k in range(4):
        print(f"  keeps grade to +-{k}: {int((m.tol_map>=k).sum())}/{N} "
              f"= {surv[k]:.3f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(BODY_W, 2.65),
                                 gridspec_kw={"width_ratios": [1.0, 1.1]})

    cols = ["#D92B4B", "#F2A900", C_MOD, C_STRONG]
    a1.bar(range(4), counts, color=cols, width=0.70)
    for k, c in enumerate(counts):
        a1.text(k, c + N * 0.015, f"{c}\n{c/N*100:.1f}%", ha="center",
                fontsize=6.2, color=INK)
    a1.set_xticks(range(4))
    a1.set_xticklabels(["0\n(lost at ±1)", "1", "2", "≥3"], fontsize=6.4)
    a1.set_xlabel("frames of anchor error the cell still tolerates", fontsize=7)
    a1.set_ylabel(f"graded cells (of {N})", fontsize=7)
    a1.set_ylim(0, max(counts) * 1.28)

    a2.plot(range(4), surv, "-o", color=C_STRONG, lw=1.7, ms=4.5)
    for k in range(4):
        a2.annotate(f"{surv[k]:.3f}", (k, surv[k]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=6.4, color=INK,
                    weight="bold" if k == 3 else "normal")
    a2.axvline(3, color=MUTE, lw=0.8, ls=":")
    a2.text(2.92, 0.10, "±3 frames at 360 Hz\n= one frame at 120 fps",
            fontsize=6.0, color=MUTE, ha="right")
    a2.set_xticks(range(4))
    a2.set_xticklabels([f"±{k}" for k in range(4)], fontsize=6.4)
    a2.set_xlabel("anchor error tolerated (frames at 360 Hz)", fontsize=7)
    a2.set_ylabel("share still holding its grade", fontsize=7)
    a2.set_ylim(0, 1.10)

    for ax in (a1, a2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=6.4, length=2.5, pad=1.5)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.965, bottom=0.235,
                        wspace=0.30)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print("saved ->", OUT)

    # the caption is generated, never typed, so a re-render cannot go stale
    cap = (f"Grade retained under a uniform shift of the ground-truth anchor, "
           f"over the {N:,} graded cells of the {m.metric.nunique()} rows that "
           f"read one.")
    print("\nCAPTION (generated):\n  " + cap)


if __name__ == "__main__":
    main()
