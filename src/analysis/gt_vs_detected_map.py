"""Compare the GT-event angle map against the detected-event (official) one.

The map is meant to answer a geometric question -- is this metric recoverable from
this viewpoint -- so answering it with our own event detectors conflates a detector
failure with a projection impossibility. angle_zone_sweep.py --gt-events rebuilds
the same grid on the OBP landmark events; this prints the difference per metric.

A metric that GAINS under GT events was event-limited. A metric that LOSES had its
detected-event score propped up by the detector picking a per-pitch frame from the
same 2D pose the metric is read from -- OR by a scoring/definition defect of ours,
which is what the first pass found and what docs/legacy_pre_dedup/GT_EVENT_MAP_HANDOFF.md documents:

  - stride was read at a single foot-plant frame while the lead ankle was still
    travelling                       -> metrics.stride_settled_2d (release-anchored)
  - stride angle was scored with Pearson r2 across the atan2 branch cut
                                     -> angle_map_2d.unwrap_circular
  - knee extension velocity really is event-limited (the one genuine loss)

The ADOPTED-VIEW block is the one the paper cites; the best-cell block carries the
usual same-sample selection optimism and is a diagnostic only.

Run:  conda activate diamond; cd src\\analysis; python gt_vs_detected_map.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
import numpy as np, pandas as pd
import config
from absacc_table import ADOPTED_VIEW

V = config.OBP_VALIDATION_DIR
det = pd.read_csv(os.path.join(V, "angle_zone_sweep.csv"))
gt = pd.read_csv(os.path.join(V, "angle_zone_sweep_gt.csv"))
m = det.merge(gt, on=["metric", "az", "el"], suffixes=("_det", "_gt"))
print(f"cells compared: {len(m)}\n")

print("=" * 88)
print("[ADOPTED VIEW]  the pre-specified paper anchor for each metric")
print("=" * 88)
print(f"{'metric':>24} {'az/el':>7} {'detected':>9} {'GT events':>10} {'delta':>8}")
rows = []
for metric, (az, el) in ADOPTED_VIEW.items():
    c = m[(m.metric == metric) & (m.az == az) & (m.el == el)]
    if c.empty:
        print(f"{metric:>24} {f'{az}/{el}':>7}   (cell missing)")
        continue
    d, g = float(c.r2_det.iloc[0]), float(c.r2_gt.iloc[0])
    rows.append((metric, az, el, d, g))
for metric, az, el, d, g in sorted(rows, key=lambda x: x[4] - x[3]):
    print(f"{metric:>24} {f'{az}/{el}':>7} {d:>9.3f} {g:>10.3f} {g-d:>+8.3f}")
print(f"\n{'':>24} {'mean |delta|':>7} {np.mean([abs(g-d) for _,_,_,d,g in rows]):>26.3f}")
print(f"{'':>24} {'cells within 0.02':>7} "
      f"{sum(abs(g-d) <= 0.02 for _,_,_,d,g in rows):>19} / {len(rows)}")

print("\n" + "=" * 88)
print("[WHOLE MAP]  best observed cell (selection-optimistic) and zone counts")
print("=" * 88)
print(f"{'metric':>24} {'best det':>9} {'best GT':>9} {'delta':>8}   "
      f"{'cells>=0.6 det':>14} {'GT':>5}")
allr = []
for name, g in m.groupby("metric"):
    bd, bg = g.r2_det.max(), g.r2_gt.max()
    nd, ng = (g.r2_det >= 0.6).sum(), (g.r2_gt >= 0.6).sum()
    allr.append((name, bd, bg, bg - bd, nd, ng))
for name, bd, bg, d, nd, ng in sorted(allr, key=lambda x: x[3]):
    print(f"{name:>24} {bd:>9.3f} {bg:>9.3f} {d:>+8.3f}   {nd:>14} {ng:>5}")

print("\nGAINED under GT events (event-limited):")
hit = False
for name, bd, bg, d, nd, ng in sorted(allr, key=lambda x: -x[3]):
    if d > 0.02:
        print(f"  {name:>24} {bd:.3f} -> {bg:.3f}  ({d:+.3f})"); hit = True
if not hit:
    print("  (none)")
print("\nLOST under GT events:")
hit = False
for name, bd, bg, d, nd, ng in sorted(allr, key=lambda x: x[3]):
    if d < -0.02:
        print(f"  {name:>24} {bd:.3f} -> {bg:.3f}  ({d:+.3f})"); hit = True
if not hit:
    print("  (none)")

# el=0 mirror invariant: az and az+180 are the same plane u-flipped, so their r2
# must be equal. A violation is a scoring artifact, never geometry -- this is the
# check that caught the stride-angle branch cut.
print("\n" + "=" * 88)
print("[MIRROR INVARIANT]  el=0, |r2(az) - r2(az+180)| must be 0")
print("=" * 88)
for tag, col in (("detected", "r2_det"), ("GT events", "r2_gt")):
    w = m[m.el == 0].pivot(index="metric", columns="az", values=col)
    worst = []
    for metric in w.index:
        ds = [abs(w.loc[metric, a] - w.loc[metric, a + 180]) for a in range(0, 180, 15)
              if np.isfinite(w.loc[metric, a]) and np.isfinite(w.loc[metric, a + 180])]
        if ds:
            worst.append((metric, max(ds)))
    bad = [(k, v) for k, v in worst if v > 0.01]
    mx = max((v for _, v in worst), default=float("nan"))
    print(f"  {tag:>10}: max |delta| = {mx:.4f} over {len(worst)} metrics"
          + ("  OK" if not bad else "   VIOLATIONS: "
             + ", ".join(f"{k} ({v:.3f})" for k, v in bad)))
