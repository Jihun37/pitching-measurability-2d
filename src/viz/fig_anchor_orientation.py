"""Fig. 1 of section VIII: what each temporal anchor reads on real video.

ONE GRID, NOT TWO STACKED FIGURES. `paper/VIII_skeleton.md` left that open; the merged
form is the answer. The side station reads four anchors and the front three, so the
seven tiles sit in a 4 x 2 grid with one cell empty, which costs two rows instead of
the three a 3-wide layout would spend.

THE TILES ARE NOT REDRAWN HERE. They come from `fig_realvideo_station.station_tiles`,
which is what the per-station figures draw, so this figure and those cannot disagree
about what a station reads or how it is drawn. Two things are this figure's own:

  * the panel title is the ANCHOR NAME, with no graded-row count -- section VIII states
    no count of cells, rows or grades. The station is named beside it because a merged
    grid no longer gives the station by which block a tile sits in;
  * `LABEL_SCALE` is raised, because at four columns a tile is 1.79 in instead of 2.39
    and the on-image labels are drawn as a fixed fraction of the crop.

The panel flags (`fp_fallback`, `implausible`, `mer_proxy`) are EXECUTION STATUS, not
accuracy -- the caption has to say so.

Run:  conda activate diamond
      cd src\\viz
      python fig_anchor_orientation.py
"""
import os, sys

import matplotlib; matplotlib.use("Agg")
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../stage2", "../tests", "../analysis"):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
import fig_realvideo_station as F

NCOL = 4
STATIONS = [("angle00_00", "side", 0, 0, "side"),
            ("angle03_04", "front", 90, 0, "front")]


def main():
    os.makedirs(F.OUT, exist_ok=True)
    # The flag moves to the panel title, the value stays on the image. Dropping the
    # value bought label width, and that width was going to be spent on a larger
    # font; the larger font was reverted, so the value has no reason to be gone.
    # With the flag in the title the label is shorter than it ever was, so the
    # value fits at this scale with room to spare.
    F.LABEL_VALUES = True
    F.FLAG_IN_TITLE = True
    F.LABEL_SCALE = 4.0 / 3.0
    pilot = pd.read_csv(os.path.join(F.PILOT, "pilot_clips_eligible.csv")).set_index("clip")
    cells = pd.read_csv(os.path.join(F.V, "realvideo_feasibility_cells.csv"))
    gm = pd.read_csv(os.path.join(F.V, "gate_map.csv"))
    reg = pd.read_csv(os.path.join(F.V, "paper_registry.csv")).set_index("metric_id")
    # the pilot predates the 2026-08-12 row-set reduction and still carries the five
    # direct-3D rows; keep only what the registry still evaluates
    cells = cells[cells.metric_id.isin(reg.index)]
    assert cells.metric_id.nunique() == 30 and len(cells) == 2400

    tiles = []
    for st in STATIONS:
        print("%s  az %d el %d" % (st[0], st[2], st[3]))
        for title, n, sq in F.station_tiles(st, cells, gm, reg, pilot):
            tiles.append(("%s  ·  %s" % (st[1], title), sq))
    # seven tiles in a 4 x 2 grid, so the last cell is empty. That gap is the finding
    # of Section VIII-A drawn: the side station lost its `own anchor` tile when wrist
    # speed left the evaluated set, and hip-shoulder separation, the one row that still
    # locates its own anchor, holds no graded cell at this station.
    assert len(tiles) == 7, "%d tiles, expected 7" % len(tiles)

    fig, rects, fig_h = F.grid(tiles, NCOL)
    # centre the short last row: left-aligned, the leftover cell puts a quarter of
    # the figure's width of white space under the fourth column
    rem = len(tiles) % NCOL
    if rem:
        shift = (rects[1][0] - rects[0][0]) * (NCOL - rem) / 2.0
        for r in rects[len(tiles) - rem:]:
            r[0] += shift
    for (title, sq), rect in zip(tiles, rects):
        F.draw_tile(fig, rect, sq, title)
    out = os.path.join(F.OUT, "fig_anchor_orientation.png")
    fig.savefig(out, dpi=300)          # never bbox_inches="tight" -- see F.grid
    print("\n-> %s   %.2f x %.2f in" % (out, F.BODY_W, fig_h))


if __name__ == "__main__":
    main()
