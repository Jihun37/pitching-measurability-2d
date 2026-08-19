"""Remove the 5 duplicate-quantity rows from the GT measurement map only.

USER DECISION 2026-07-29. Five rows measured a quantity that another row already
measured, so the same quantity was counted twice and the evaluation set read 52
rows for 47 quantities. They are dropped outright, so from here on a row count and
a quantity count are the same number.

SCOPE. Plain `python dedup_rows.py` touches the GT map and nothing else.

`python dedup_rows.py --deploy` does the deferred half: the DETECTED-event dumps
that `deploy_map.py` consumes (2026-07-30, user decision). Until that ran, the
deployed map sat on the old 40 rows / 1,687 GT cells / 991 deployable while the
GT map sat on 35 / 1,500, and joining the two or dividing one by the other gave a
meaningless retention ratio. After it, both layers score the same 47 rows.
The two halves are separate flags rather than one run because the GT half was
frozen first and re-running it is a no-op that still rewrites 40 M rows.

WHICH SIDE OF EACH PAIR IS DROPPED. Four pairs are `same-estimator`: both code
paths produce the same number to dump rounding, so dropping either leaves the map
bit-identical, and the COLUMN-NAMED row is kept so every kinematic column is
evaluated under its own name.

    DROP  Torso Rot @BR [O]        keep  torso_rotation_br             49/39/0.8920
    DROP  Torso Lat Tilt @MER [O]  keep  torso_lateral_tilt_mer        47/37/0.9127
    DROP  Elbow Flex @MER [O]      keep  elbow_flexion_mer             24/20/0.9383
    DROP  Glove Sh Abd @MER [O]    keep  glove_shoulder_abduction_mer  19/10/0.8999

The fifth pair is `alt-estimator`: the same truth through a genuinely different 2D
observable, and the two are NOT equivalent. The adopted row is much the better of
them, so here the COLUMN-NAMED row is dropped instead. Dropping the adopted row
would have cost 52 -> 15 strong cells.

    DROP  max_pelvis_rotational_velo  (48/15/0.8066)
    keep  Pelvis Rot Velo [O]         (64/52/0.9061)

Run:  conda activate diamond; cd src\\analysis; python dedup_rows.py
Then: python gate_map.py --out-suffix _dedup ... ; python layer_report.py
"""
import os, sys, shutil, gzip, argparse
_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
import pandas as pd
import config

DROP_ADOPTED = ["Torso Rot @BR [O]", "Torso Lat Tilt @MER [O]",
                "Elbow Flex @MER [O]", "Glove Sh Abd @MER [O]"]
DROP_SCREENED = ["max_pelvis_rotational_velo"]

BACKUP = "backup_pre_dedup_20260729"

# GT dumps. The detected/redetect dumps are NOT here: they are reconciled by
# `--deploy` below, which was deliberately deferred on 2026-07-29 so the GT map
# could be frozen without waiting on the deployment layer.
# rejected_gt_pairs_offsets.csv.gz ADDED 2026-07-29: it is a GT dump and was
# missing here, not deliberately excluded, so every fresh
# `rejected_gt_full_sweep.py --event-offsets=...` run silently reintroduced
# max_pelvis_rotational_velo into the tolerance layer. Found when
# event_tolerance's offset-0 baseline disagreed with gate_map on 33 cells.
JOBS = [("angle_zone_pairs_gt.csv.gz", DROP_ADOPTED),
        ("rejected_gt_pairs.csv.gz", DROP_SCREENED),
        ("rejected_gt_pairs_offsets.csv.gz", DROP_SCREENED)]

# The DETECTED-event dumps, run with `--deploy` (2026-07-30, user decision to
# reconcile the deployment layer onto the same 47 rows as the GT map). The four
# files are the ones `deploy_map.DUMPS` consumes -- two fp policies x
# (adopted, screened). The drop lists are identical to the GT ones on purpose:
# the whole point of the reconciliation is that both layers score the same rows,
# so `Pelvis Rot Velo [O]` is kept over `max_pelvis_rotational_velo` here too,
# even though that choice was originally argued from GT cell counts.
# `_oldwin` and `_fp10` dumps are diagnostics for retired definitions and are
# left alone; do not quote a count from them either way.
JOBS_DEPLOY = [("angle_zone_pairs_redetect.csv.gz", DROP_ADOPTED),
               ("angle_zone_pairs_redetect_fpfrontal.csv.gz", DROP_ADOPTED),
               ("rejected_gt_pairs_detected.csv.gz", DROP_SCREENED),
               ("rejected_gt_pairs_detected_fpfrontal.csv.gz", DROP_SCREENED)]

# The offsets dump is ~17 M rows, so it is filtered in chunks rather than loaded
# whole. Everything else is small enough either way; one code path serves both.
CHUNK = 2_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", action="store_true",
                    help="filter the DETECTED-event dumps instead of the GT ones")
    a = ap.parse_args()
    jobs = JOBS_DEPLOY if a.deploy else JOBS

    V = config.OBP_VALIDATION_DIR
    bdir = os.path.join(V, BACKUP)
    os.makedirs(bdir, exist_ok=True)
    for src_name, drop in jobs:
        src = os.path.join(V, src_name)
        if not os.path.exists(src):
            print(f"\n{src_name}: absent, skipped")
            continue

        # pass 1: what is in there, without holding the whole file
        have, total, per = set(), 0, {m: 0 for m in drop}
        for ch in pd.read_csv(src, usecols=["metric"], chunksize=CHUNK):
            s = ch.metric.astype(str).str.strip()
            have |= set(s.unique())
            total += len(ch)
            for m in drop:
                per[m] += int((s == m).sum())
        present = [m for m in drop if m in have]
        print(f"\n{src_name}: {total:,} rows, {len(have)} metrics")
        if not present:
            print("   already clean, left alone")
            continue

        bak = os.path.join(bdir, src_name)
        if not os.path.exists(bak):
            shutil.copy2(src, bak)
            print(f"   backed up -> {BACKUP}/{src_name}")
        for m in present:
            print(f"   DROP {m:<32} {per[m]:>9,} rows")

        # pass 2: rewrite through a temp file so a crash cannot truncate the dump
        tmp = src + ".tmp"
        kept, mets, first = 0, set(), True
        with gzip.open(tmp, "wt", newline="") as out:
            for ch in pd.read_csv(src, chunksize=CHUNK):
                c = ch[~ch.metric.astype(str).str.strip().isin(present)]
                kept += len(c)
                mets |= set(c.metric.astype(str).str.strip().unique())
                c.to_csv(out, index=False, header=first)
                first = False
        os.replace(tmp, src)
        print(f"   kept: {kept:,} rows, {len(mets)} metrics")
        print("   rewrote in place")


if __name__ == "__main__":
    main()
