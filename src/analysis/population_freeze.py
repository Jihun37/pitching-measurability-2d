"""FREEZE THE LAYER-1 POPULATION and prove every output uses the same pitches.

WHY. Counts in this project have been quoted against 411, 408, 403, 401, 395 and 394
in different documents, and a cell count is only comparable across rows if every row
was scored on the identical pitch set. This script derives the population once,
asserts the exclusion arithmetic, writes the frozen id list with a checksum, and
then checks the actual Layer-1 outputs against it. Nothing here re-analyses anything;
it only proves that what is already on disk agrees.

THE ARITHMETIC THIS ENFORCES

    411 pitches in metadata
    -10 no usable ground-truth event set
    - 7 implausible ground-truth foot plant (robust k rule, gt_landmark_outlier_effect)
    = 394 pitches, 98 pitchers

⚠ WHAT "USABLE EVENT SET" MEANS, and why the split is 10/7 and not 9/8. The gate is
pkh, fp and rel all present (time > 0) **AND rel > fp + 1**. The ordering condition
is not decoration: exactly one pitch carries all three landmarks but places release
within one frame of foot plant, which leaves the window every fp-anchored estimator
reads empty. That pitch is ALSO one of the 8 the outlier rule flags. Drop the
ordering condition and it stays in the event-usable set and is excluded later, giving
9 + 8; keep it and the pitch is excluded earlier, giving 10 + 7. Both reach 394, so
the total cannot reveal the error -- only the decomposition can, and it is the
decomposition that goes into Methods. The subtraction is asserted to be disjoint.

Requiring the remaining landmarks (fp10, mer, mir) changes nothing: every pitch that
has pkh/fp/rel has those too. Requiring only rel gives 403 and only fp+rel gives 395,
which is where those two retired counts came from.

WHY `n` STILL VARIES PER CELL. `gate_map.csv`'s per-cell `n` ranges below 394 because
a cell drops pairs whose estimate or truth is non-finite AT THAT VIEWPOINT (a
landmark can project behind the camera, an estimator can defer). That is a property
of the cell, not a different population, and it is NOT to be "fixed" by padding.
What must hold is that no cell exceeds the frozen population and every cell's pitches
are a SUBSET of it, which is asserted below.

Output: population_frozen.csv (the id list) + a sha256 over the sorted ids, printed
        and written to population_frozen_checksum.txt for the freeze report.
Run:  conda activate diamond; cd src\\analysis; python population_freeze.py
"""
import os, sys, hashlib
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import pandas as pd
import config
from obp_gt_events import load_gt_events
from gt_landmark_outlier_effect import outlier_pitches
from mer_proxy_map import map_population

V = config.OBP_VALIDATION_DIR


def checksum(ids):
    h = hashlib.sha256()
    for i in sorted(map(str, ids)):
        h.update(i.encode()); h.update(b"\n")
    return h.hexdigest()


def main():
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    total = set(md.session_pitch)
    print(f"metadata pitches                {len(total)}")

    gt = load_gt_events()
    # A pitch is usable only with the full event set the map reads AND a non-empty
    # foot-plant-to-release window. See the module docstring: the ordering condition
    # is what makes the split 10/7 rather than 9/8.
    NEED = ("pkh", "fp", "rel")
    have = {sp for sp, d in gt.items()
            if all(e in d for e in NEED) and d["rel"] > d["fp"] + 1}
    with_events = total & have
    no_events = total - with_events
    print(f"  -- no usable GT event set     {len(no_events)}")

    bad = set(outlier_pitches())
    bad_in = with_events & bad
    bad_out = bad - with_events
    print(f"  -- implausible GT foot plant  {len(bad_in)}"
          f"   (rule flags {len(bad)}; {len(bad_out)} already excluded above)")

    frozen = with_events - bad_in
    users = md.set_index("session_pitch").user
    n_pitchers = users.loc[sorted(frozen)].nunique()
    print(f"= frozen population             {len(frozen)} pitches, "
          f"{n_pitchers} pitchers")

    assert len(total) == 411, len(total)
    assert len(no_events) == 10, len(no_events)
    assert len(bad_in) == 7, (len(bad_in), len(bad))
    assert len(frozen) == 394, len(frozen)
    assert n_pitchers == 98, n_pitchers
    assert len(total) - len(no_events) - len(bad_in) == len(frozen)

    # the project's existing helper must return the identical set
    mp = set(map_population())
    assert mp == frozen, (f"map_population() differs: "
                          f"+{len(mp - frozen)} / -{len(frozen - mp)}")
    print("  map_population() agrees exactly")

    cs = checksum(frozen)
    print(f"\nsha256(sorted ids)  {cs}")

    # ---- every Layer-1 output must live inside this set --------------------
    print("\nLAYER-1 INPUTS -- the raw dumps are a SUPERSET by design")
    # The dumps are produced before the clean filter exists; gate_map.py and
    # accuracy_map.py both apply `isin(keep)` at scoring time. So the requirement on
    # a dump is COVERAGE (it must contain every frozen pitch), not equality. Asserting
    # equality here would flag the normal pipeline as broken.
    ok = True
    for f in ("angle_zone_pairs_gt.csv.gz", "rejected_gt_pairs.csv.gz"):
        p = os.path.join(V, f)
        if not os.path.exists(p):
            print(f"  {f:<34} MISSING"); ok = False; continue
        s = set(pd.read_csv(p, usecols=["session_pitch"]).session_pitch.unique())
        missing = frozen - s
        print(f"  {f:<34} {len(s):>4} ids, covers frozen: "
              f"{'yes' if not missing else f'NO, {len(missing)} absent'}"
              f"   (+{len(s - frozen)} filtered at scoring)")
        ok &= not missing

    if not ok:
        sys.exit("a dump does not cover the frozen population")

    print("\nLAYER-1 SCORED OUTPUTS -- must never exceed the frozen population")
    # A cell's n falls below 394 when the estimate or the truth is non-finite AT
    # THAT VIEWPOINT. That is a property of the cell, not a second population.
    for f in ("gate_map.csv", "accuracy_map_gt_clean.csv"):
        p = os.path.join(V, f)
        if not os.path.exists(p):
            sys.exit(f"missing scored output {f}")
        d = pd.read_csv(p, usecols=["n", "n_pitcher"])
        print(f"  {f:<34} n {int(d.n.min())}-{int(d.n.max())}, "
              f"pitchers {int(d.n_pitcher.min())}-{int(d.n_pitcher.max())}"
              f"   ({int((d.n == len(frozen)).mean() * 100)} % of cells at full n)")
        assert d.n.max() <= len(frozen), (f, int(d.n.max()))
        assert d.n_pitcher.max() <= n_pitchers, (f, int(d.n_pitcher.max()))
    print("  no scored output exceeds the frozen population")

    out = pd.DataFrame({"session_pitch": sorted(frozen)})
    out["user"] = users.loc[out.session_pitch].to_numpy()
    p = os.path.join(V, "population_frozen.csv")
    out.to_csv(p, index=False)
    with open(os.path.join(V, "population_frozen_checksum.txt"), "w") as fh:
        fh.write(f"{cs}  {len(frozen)} pitches  {n_pitchers} pitchers\n")
    print(f"\nsaved -> {p}")
    print(f"saved -> {os.path.join(V, 'population_frozen_checksum.txt')}")


if __name__ == "__main__":
    main()
