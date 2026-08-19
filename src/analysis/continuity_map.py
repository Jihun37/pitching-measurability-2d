"""CONTIGUOUS AZIMUTHAL ARCS: how wide a band of viewpoints each row actually holds.

THE QUESTION. A count of graded cells says how much of the grid a row holds but not
whether that coverage is USABLE. Twelve cells scattered one bin apart and twelve
cells in one unbroken band are the same number and completely different instructions
to someone aiming a phone. This measures the second thing.

TERMINOLOGY, FIXED. The unit is a **contiguous azimuthal arc**: the longest run of
consecutive graded azimuth bins at one elevation, wrapping at 360 degrees. The word
`zone` is NOT used -- it was the name of a retired r-squared-threshold construct
(`angle_zone_table.csv`) and reusing it would silently import that definition.

    bin           one 15-degree azimuth step; 24 bins make the full circle
    arc           a maximal run of consecutive graded bins at a fixed elevation
    row max arc   the widest arc that row holds at ANY of its 7 elevations
    isolated cell a graded cell with no graded neighbour in azimuth +-15 deg or at
                  the adjacent elevation -- already computed as gate_map's `spike`

WHY "AT ANY ELEVATION" AND NOT SUMMED. Arcs at different elevations are different
camera placements, not a single wider arc, so they are never added. The row-level
number is a maximum over elevations, and the elevation it occurs at is reported with
it so the figure and the text cannot drift apart.

GRADED means strong or moderate. The arc is a contiguity statement about the
published map, so it uses the map's own membership and applies no second threshold.

Input:  gate_map.csv, paper_registry.csv
Output: continuity_rows.csv (per row), continuity_arcs.csv (every arc)
Run:  conda activate diamond; cd src\\analysis; python continuity_map.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config

V = config.OBP_VALIDATION_DIR
AZ_STEP, N_BIN = 15, 24

# A row is a RATE row when the quantity it reports is a time derivative. Explicit,
# because "velocity" appears in some column names and not others (`Wrist Speed`,
# `cog_velo_pkh`), and the angle-vs-velocity comparison is a reported result.
RATE_TOKENS = ("velo", "speed")


def kind_of(metric_id):
    low = metric_id.lower()
    return "velocity" if any(t in low for t in RATE_TOKENS) else "angle/posture"


def arcs_at(bins):
    """Maximal runs of consecutive occupied azimuth bins, wrapping at 24.

    Returns [(start_bin, length)]. A fully occupied circle is one arc of 24, not 24
    arcs of 1 -- the wrap join is what makes that come out right, and it is the case
    that a non-wrapping implementation gets wrong.
    """
    occ = sorted(bins)
    if not occ:
        return []
    if len(occ) == N_BIN:
        return [(occ[0], N_BIN)]
    s = set(occ)
    out = []
    for b in occ:
        if (b - 1) % N_BIN in s:            # not a run start
            continue
        n, c = 1, (b + 1) % N_BIN
        while c in s and n < N_BIN:
            n += 1; c = (c + 1) % N_BIN
        out.append((b, n))
    return out


def main():
    g = pd.read_csv(os.path.join(V, "gate_map.csv"))
    reg = pd.read_csv(os.path.join(V, "paper_registry.csv"))
    retained = reg[reg.retained].metric_id.tolist()
    # 35 until the row set was reduced to the 42 OBP kinematic columns on
    # 2026-08-12; the kept rows re-scored bit-identically
    assert len(retained) == 30, len(retained)

    graded = g[g.grade.isin(["strong", "moderate"])]
    # 1,151 = 819 strong + 332 moderate. Unchanged by the 2026-08-08 switch to nested
    # correction-model selection, which reproduces the map cell for cell; see
    # gate_map.nested_predictions.
    assert len(graded) == 1151, len(graded)
    isolated = int(graded.spike.sum())
    print(f"graded cells {len(graded)}   isolated cells {isolated}")
    print(graded[graded.spike][["metric", "az", "el", "grade"]]
          .to_string(index=False))
    assert isolated == 2, isolated

    rows, arc_recs = [], []
    for m in retained:
        sub = graded[graded.metric == m]
        best, best_el = 0, None
        for el, gel in sub.groupby("el"):
            bins = {int(a) // AZ_STEP for a in gel.az}
            for start, n in arcs_at(bins):
                arc_recs.append(dict(metric_id=m, el=int(el),
                                     az_start=start * AZ_STEP,
                                     az_end=((start + n - 1) % N_BIN) * AZ_STEP,
                                     bins=n, degrees=n * AZ_STEP))
                if n > best:
                    best, best_el = n, int(el)
        rows.append(dict(metric_id=m, kind=kind_of(m), max_arc_bins=best,
                         max_arc_deg=best * AZ_STEP, max_arc_el=best_el,
                         graded_cells=len(sub)))
    R = pd.DataFrame(rows).sort_values("max_arc_bins", ascending=False)
    A = pd.DataFrame(arc_recs)

    ge3 = int((R.max_arc_bins >= 3).sum())
    print(f"\nrows with a contiguous arc of at least 3 bins (45 deg): "
          f"{ge3} of {len(R)}")
    print("rows below that:")
    print(R[R.max_arc_bins < 3][["metric_id", "kind", "max_arc_bins",
                                 "graded_cells"]].to_string(index=False))

    print("\nmedian row max arc, by kind:")
    for k, sub in R.groupby("kind"):
        print(f"  {k:<14} n={len(sub):>2}  median {sub.max_arc_bins.median():.1f} "
              f"bins ({sub.max_arc_bins.median() * AZ_STEP:.0f} deg)"
              f"   range {sub.max_arc_bins.min()}-{sub.max_arc_bins.max()}")
    print(f"  {'ALL':<14} n={len(R):>2}  median {R.max_arc_bins.median():.1f} bins")

    R.to_csv(os.path.join(V, "continuity_rows.csv"), index=False)
    A.to_csv(os.path.join(V, "continuity_arcs.csv"), index=False)
    print(f"\nsaved -> {os.path.join(V, 'continuity_rows.csv')}")
    print(f"saved -> {os.path.join(V, 'continuity_arcs.csv')}")

    print("\nwidest arc per row:")
    print(R[["metric_id", "kind", "max_arc_bins", "max_arc_deg", "max_arc_el",
             "graded_cells"]].to_string(index=False))


if __name__ == "__main__":
    main()
