"""Move stale outputs out of the active generation path. MOVES, never deletes.

`.git` in this repository is empty, so there is no version-control safety net and a
delete is unrecoverable. Everything here is moved into
`data/outputs/obp_validation/archive_stale_pre_freeze/` and can be moved back.

TWO DEPLOYMENT GENERATIONS, KEPT APART. Both are stale and neither may be cited:

    deployment_pre_dedup_991/           991 cells (698 strong / 293 moderate) on the
                                        old 40-row map, superseded by the dedup
    deployment_rescore_old_dumps_863/   863 cells (617 strong / 246 moderate),
                                        produced 2026-07-30 by re-scoring the SAME
                                        old detected-event dumps after filtering the
                                        deduped rows out of them. A consistency
                                        check, NOT a fresh deployment evaluation

They go in separate subdirectories on purpose. Left in one folder it becomes
impossible to tell later which file belongs to which generation, which is how the
991/863 confusion would come back.

THE DETECTED-EVENT DUMPS GO TOO. They were rewritten in place by
`dedup_rows.py --deploy` and are stale INPUTS of the 863 generation. Their filenames
are enumerated from `deploy_map.DUMPS` and `fp_routing_cv.PAIRS` rather than typed
here, so a dump cannot be missed by a typo. No detected-event dump may remain in the
active path before the fresh post-freeze sweep.

`backup_pre_dedup_20260729/` is NOT touched -- it is the only surviving copy of the
pre-dedup dumps.

Run:  conda activate diamond; cd src\\analysis
      python archive_stale.py            # dry run, prints what would move
      python archive_stale.py --apply
"""
import os, sys, glob, shutil, argparse
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import config

V = config.OBP_VALIDATION_DIR
ROOT = os.path.join(V, "archive_stale_pre_freeze")
KEEP_DIRS = {"backup_pre_dedup_20260729", "backup_pre_quietguard_20260727",
             "archive_stale_20260728", "archive_stale_pre_freeze"}

# ---- generation 1: the pre-dedup 991-cell deployment layer -----------------
PRE_DEDUP_991 = [
    "backup_pre_dedup_20260729/deploy_map_cells.csv",
    "backup_pre_dedup_20260729/deploy_map_summary.csv",
    "backup_pre_dedup_20260729/anchor_retention_summary.csv",
]

# ---- generation 2: this session's 863-cell rescore -------------------------
RESCORE_863 = [
    "deploy_map_cells.csv", "deploy_map_summary.csv",
    "anchor_retention_summary.csv",
    "fp_routing_rule.csv", "fp_routing_cv_cells.csv",
    "gate_map_deploy_*.csv",
    "gate_map_detected*.csv",
    "deploy_tolerance_vs_error.csv",
    "fig_deploy_loss.png", "fig_deploy_strip.png", "fig_anchor_retention.png",
]

# ---- everything else retired at the freeze ---------------------------------
OTHER = [
    "column_coverage_audit.csv",          # A-G failure-reason scheme, retired
    "method_robustness_probe.csv",        # companion to the same scheme
    "absacc_table.csv", "absacc_table_gt.csv", "absacc_table_gt_clean.csv",
    "absacc_table_cand.csv",              # superseded by accuracy_map_gt_clean.csv
    "loco_calibration.csv", "loco_calibration_gt_clean.csv",
    "loco_calibration_cand.csv",          # subsumed by gate_map's per-cell LOCO
    "within_pitcher_agreement.csv", "within_pitcher_agreement_gt_clean.csv",
    "within_pitcher_agreement_cand.csv",  # superseded by accuracy_bestcell
    "angle_zone_table*.csv",              # r-squared-threshold zone tables, retired
    "angle_zone_sweep_offset_*.csv", "angle_zone_sweep_oracle.csv",
    "angle_zone_sweep_redetect*.csv",
    "fig_loco_calibration*.png", "fig_within_pitcher.png",
]


def detected_dumps():
    """Filenames enumerated FROM THE DEPLOYMENT CODE, not typed here."""
    from deploy_map import DUMPS
    from fp_routing_cv import PAIRS
    out = set()
    for ad, sc in list(DUMPS.values()) + list(PAIRS.values()):
        out.add(ad); out.add(sc)
    # the retired-definition siblings live beside them and are equally stale
    for pat in ("rejected_gt_pairs_detected*.csv.gz", "angle_zone_pairs_redetect*.csv.gz",
                "rejected_gt_full_grid_detected*.csv", "rejected_gt_full_sweep_detected*.csv"):
        for p in glob.glob(os.path.join(V, pat)):
            out.add(os.path.basename(p))
    return sorted(out)


def expand(pats):
    out = []
    for pat in pats:
        hits = sorted(glob.glob(os.path.join(V, pat)))
        if not hits:
            continue
        for h in hits:
            rel = os.path.relpath(h, V)
            if rel.split(os.sep)[0] in KEEP_DIRS and not pat.startswith("backup_"):
                continue
            out.append(rel)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually move the files")
    a = ap.parse_args()

    groups = {
        "deployment_pre_dedup_991": expand(PRE_DEDUP_991),
        "deployment_rescore_old_dumps_863": (expand(RESCORE_863)
                                             + expand(detected_dumps())),
        "": expand(OTHER),
    }

    total = 0
    for sub, files in groups.items():
        label = sub or "(top level)"
        print(f"\n{label}   {len(files)} files")
        for f in files:
            src = os.path.join(V, f)
            dst = os.path.join(ROOT, sub, os.path.basename(f))
            size = os.path.getsize(src) / 1e6
            print(f"   {os.path.basename(f):<50} {size:8.1f} MB")
            total += 1
            if a.apply:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(dst):
                    print(f"      ! destination exists, skipped")
                    continue
                shutil.move(src, dst)

    print(f"\n{total} files {'MOVED' if a.apply else 'would move'}")
    if not a.apply:
        print("dry run -- re-run with --apply")
    else:
        print(f"-> {ROOT}")
        print("backup_pre_dedup_20260729/ left in place (only pre-dedup dump copy)")


if __name__ == "__main__":
    main()
