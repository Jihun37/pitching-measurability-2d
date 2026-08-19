"""A5 -- per-pitch timing JITTER, as opposed to a systematic anchor offset (audit ST3).

WHY THIS IS A DIFFERENT EXPERIMENT FROM SEC VI-C. That sweep displaces the anchor by the
SAME k frames on every pitch and rescores under the full protocol -- which REFITS the
leave-one-pitcher-out correction at the shifted anchor. A constant displacement is a
systematic error, and the offset and linear models absorb much of it. A real detector does
not fail that way: its error differs pitch to pitch, is not absorbable by any global
correction, and degrades the correlation directly. So the 83.4 % reported in Sec VI-C is
`systematic anchor-offset tolerance after recalibration`, and it is structurally
optimistic about a detector. The recipe in Sec IX-A -- "detect it to within about one
frame at 120 fps" -- does not follow from it. This script supplies what does.

DESIGN, and the constraint that shaped it. The offset dumps hold estimates at
k = -3 .. +3 c3d frames only (+-8.33 ms at the 360 Hz reference). Drawing a Gaussian and
clipping to that support would bias every point above about 3 ms, so the jitter is drawn
from a DISCRETE distribution defined ON the support, which cannot clip:

    bell   discretised normal on {-3..3}, renormalised; sigma solved to hit the target SD
    limit  the bell family saturates at the uniform, SD = 2 frames
    worst  two-point at +-3, SD = 3 frames exactly = one 120 fps frame

Ground truth is read at offset 0 throughout -- the DETECTOR is wrong, the truth is not.
Verified: `truth` is constant across offsets in both dumps.

Reachable range is therefore SD in [0, 8.33] ms, which is exactly the range the Sec IX-A
recipe speaks about. Larger jitter would need a wider offset sweep; that is a separate cost
and is NOT approximated here.

Scoring is `gate_map.score_cell`, i.e. the map's own nested protocol, so a jittered cell is
directly comparable with a map cell.

Output: review_jitter_cells.csv, review_jitter_summary.csv
Run:  conda activate diamond; cd src\\analysis; python review_jitter.py --R 20
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../stage2"):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
import gate_map as GM
from angle_map_2d import CIRCULAR, unwrap_circular

V = config.OBP_VALIDATION_DIR
OFFSETS = np.arange(-3, 4)
FPS_C3D = 360.0
MS = 1000.0 / FPS_C3D                      # 2.778 ms per c3d frame
DUMPS = ("adopted_tolerance_pairs.csv.gz", "rejected_gt_pairs_offsets.csv.gz")


def dist_for_sd(target_frames):
    """Symmetric discrete distribution on {-3..3}, mean 0, with the requested SD.
    Bell (discretised normal) while that is attainable; the two-point at +-3 for SD 3."""
    if target_frames >= 2.999:
        p = np.zeros(len(OFFSETS)); p[0] = p[-1] = 0.5
        return p
    lo, hi = 1e-3, 60.0
    for _ in range(200):                    # bisect on sigma
        s = 0.5 * (lo + hi)
        p = np.exp(-0.5 * (OFFSETS / s) ** 2)
        p /= p.sum()
        sd = np.sqrt((p * OFFSETS ** 2).sum())
        if sd < target_frames:
            lo = s
        else:
            hi = s
    return p


def load_cells(keep_rows, pop):
    """(metric, az, el) -> (est[n,7], truth[n], pitcher codes). Chunked: the screened
    offsets dump is 133 MB gzipped and does not fit comfortably in one frame."""
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    user_of = dict(zip(md.session_pitch, md.user))
    store = {}
    for name in DUMPS:
        path = os.path.join(V, name)
        if not os.path.exists(path):
            print(f"  !! missing {name}"); continue
        n = 0
        for ch in pd.read_csv(path, chunksize=2_000_000):
            ch = ch[ch.metric.isin(keep_rows) & ch.session_pitch.isin(pop)]
            n += len(ch)
            for key, sub in ch.groupby(["metric", "az", "el"], sort=False):
                store.setdefault(key, []).append(sub)
        print(f"  {name}: {n:,} rows kept")
    cells = []
    for (m, az, el), parts in store.items():
        d = pd.concat(parts, ignore_index=True)
        w = d.pivot_table(index="session_pitch", columns="offset", values="est")
        if list(w.columns) != list(OFFSETS):
            w = w.reindex(columns=OFFSETS)
        t = d[d.offset == 0].set_index("session_pitch").truth.reindex(w.index)
        u = pd.Series(w.index.map(user_of), index=w.index)
        ok = t.notna().to_numpy() & u.notna().to_numpy() & np.isfinite(w.to_numpy()).all(1)
        if ok.sum() < 30:
            continue
        codes = pd.Categorical(u[ok]).codes
        cells.append((m, int(az), int(el), w.to_numpy(float)[ok],
                      t.to_numpy(float)[ok], np.asarray(codes)))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=20, help="replicates per jitter level")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    reg = pd.read_csv(os.path.join(V, "paper_registry.csv"))
    rows = set(reg.metric_id)
    gm = pd.read_csv(os.path.join(V, "gate_map.csv"))
    gm = gm[gm.metric.isin(rows)]
    pop = set(pd.read_csv(os.path.join(V, "population_frozen.csv")).session_pitch) \
        if os.path.exists(os.path.join(V, "population_frozen.csv")) else None
    if pop is None:
        from mer_proxy_map import map_population
        pop = set(map_population())
    print(f"population {len(pop)}  canonical rows {len(rows)}")

    t0 = time.time()
    cells = load_cells(rows, pop)
    print(f"{len(cells)} cells with a full offset sweep, in {time.time() - t0:.0f}s")

    base = {(r.metric, r.az, r.el): r.grade for r in gm.itertuples()}
    rng = np.random.default_rng(a.seed)
    targets = [0.5, 1.0, 1.5, 2.0, 3.0]           # c3d frames
    summ, percell = [], []

    for tf in targets:
        p = dist_for_sd(tf)
        sd_f = float(np.sqrt((p * OFFSETS ** 2).sum()))
        hits = np.zeros(len(cells)); strong = np.zeros(len(cells)); tot = 0
        for _ in range(a.R):
            for ci, (m, az, el, est, t, g) in enumerate(cells):
                k = rng.choice(OFFSETS, size=len(t), p=p) + 3      # index into columns
                e = est[np.arange(len(t)), k]
                if m.strip() in CIRCULAR:
                    e = unwrap_circular(e)
                s = GM.score_cell(e, t, g, 0.75, 0.80)
                if s is None:
                    continue
                hits[ci] += s["grade"] in ("strong", "moderate")
                strong[ci] += s["grade"] == "strong"
            tot += 1
        pg, ps = hits / tot, strong / tot
        keys = [(m, az, el) for m, az, el, *_ in cells]
        was_g = np.array([base.get(k, "limited") in ("strong", "moderate") for k in keys])
        was_s = np.array([base.get(k, "limited") == "strong" for k in keys])
        summ.append(dict(sd_frames=round(sd_f, 3), sd_ms=round(sd_f * MS, 2),
                         swept_cells=len(cells),
                         baseline_graded=int(was_g.sum()),
                         exp_graded=round(float(pg.sum()), 1),
                         retention_pct=round(100.0 * pg.sum() / max(was_g.sum(), 1), 1),
                         baseline_strong=int(was_s.sum()),
                         exp_strong=round(float(ps.sum()), 1),
                         strong_retention_pct=round(100.0 * ps.sum() / max(was_s.sum(), 1), 1),
                         mean_p_graded_on_baseline=round(float(pg[was_g].mean()), 4)))
        print(f"  SD {sd_f:4.2f} fr ({sd_f * MS:5.2f} ms):  expected graded "
              f"{pg.sum():7.1f} of {int(was_g.sum())}  = {100 * pg.sum() / was_g.sum():5.1f}%"
              f"   strong {100 * ps.sum() / max(was_s.sum(), 1):5.1f}%", flush=True)
        for ci, (m, az, el, *_ ) in enumerate(cells):
            percell.append(dict(metric=m, az=az, el=el, sd_ms=round(sd_f * MS, 2),
                                p_graded=round(float(pg[ci]), 3),
                                p_strong=round(float(ps[ci]), 3),
                                baseline=base.get((m, az, el), "limited")))

    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(V, "review_jitter_summary.csv"), index=False)
    pd.DataFrame(percell).to_csv(os.path.join(V, "review_jitter_cells.csv"), index=False)
    print("\n" + S.to_string(index=False))
    print("\nwrote review_jitter_summary.csv / review_jitter_cells.csv")
    print("NOTE: SD > 8.33 ms is not reachable from the +-3-frame dumps and is not "
          "extrapolated here.")


if __name__ == "__main__":
    main()
