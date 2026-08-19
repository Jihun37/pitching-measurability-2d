"""THE CANONICAL 42-ROW REGISTRY. One file decides every table, figure and
denominator in the manuscript.

WHY. Numbers in this project have repeatedly disagreed between documents because
each analysis carried its own idea of what the row set was: 12 adopted metrics here,
16 there, 52 evaluation rows, 40 map rows, 47 after the dedup. A count quoted from
the wrong generation is indistinguishable from a correct one once it is in prose.
This script derives the row set ONCE, from the code and the canonical CSVs, asserts
that the accounting closes, and writes `paper_registry.csv`. Everything downstream
reads that file; nothing downstream re-derives a row set.

THE ACCOUNTING THIS FILE ENFORCES

    81 OBP poi columns  =  5 metadata + 34 kinetic + 42 kinematic
    42 evaluated rows   =  the 42 kinematic columns, each evaluated exactly once
    42 evaluated        =  30 retained + 12 non-retained

⚠ TWO DIFFERENT 35s. They are numerically equal and semantically unrelated, and
conflating them has already produced wrong prose. The registry names them apart:

    n_same_name_pathways   35  rows that carry an OBP column's own name
    n_retained_rows        30  evaluation rows holding at least one graded cell

The first is a naming fact about where a row's truth comes from. The second is a
RESULT. Until 2026-08-12 both were 35 and conflating them produced wrong prose; they
now differ, but the distinction is the point and not the coincidence.

⚠ `graded_cells`, NOT `map_cells`. It is the number of strong-or-moderate cells a
row retains in the published map. It is NOT the row's 168 evaluated viewpoints: the
30 retained rows span 30 x 168 = 5,040 evaluated cells, of which 1,151 are graded.
`layer_summary.csv` still calls the column `map_cells`, which reads either way; the
registry renames it on the way in and the registry name is the one the paper uses.

⚠ THE 34 KINETIC TARGETS ARE NOT IN THIS FILE. This registry is the
direct-measurement row set. Kinetic inference uses its own explicit canonical list
of 34 targets and shares only the frozen 394-pitch population. Do not add them here.

WHY 42 AND NOT 35 KINEMATIC PATHWAYS. Seven kinematic columns are reached by an
adopted estimator under a paper name rather than under the column's own name
(stride_length, torso_anterior_tilt_br, max_rotation_hip_shoulder_separation,
max_pelvis_rotational_velo, max_cog_velo_x, cog_velo_pkh, stride_angle). 35 + 7 = 42,
so every kinematic column is evaluated exactly once and no column is evaluated twice.

WHY NO DIRECT-3D ROWS, since 2026-08-12. Lead knee angle, wrist speed, arm slot,
release height and release extension were evaluated against a quantity computed
directly on the c3d markers, because no column held them under the adopted definition.
They are removed, so every row now takes its truth from a published column and the
evaluated set is fixed by the dataset rather than partly by this work. Extending past
the dataset's schema is Future Work. The `arm_slot` column stays, evaluated under its
own forearm-based definition, as one of the 42.

A literature census run on 26 full texts before the removal found no quantity that
recurs in the pitching-biomechanics literature and that OBP does not publish, which is
what makes the 42 defensible as the whole evaluated set.

Inputs:  poi_metrics.csv (column census), angle_map_2d.adopted_rows(),
         rejected_gt_full_sweep.CANDS, layer_summary.csv, event_tolerance_map.csv
Output:  paper_registry.csv
Run:  conda activate diamond; cd src\\analysis; python paper_registry.py
"""
import os, sys, argparse
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config
from angle_map_2d import CIRCULAR, adopted_rows
from rejected_gt_full_sweep import CANDS

V = config.OBP_VALIDATION_DIR

# ---------------------------------------------------------------- column census
# Explicit, not pattern-matched: a regex over column names would silently
# reclassify a column if OBP ever renames one, and the 5/34/42 split is an
# assertion the paper leans on.
METADATA = ["session_pitch", "session", "p_throws", "pitch_type", "pitch_speed_mph"]
KINETIC = [
    "elbow_varus_moment", "shoulder_internal_rotation_moment",
    "shoulder_transfer_fp_br", "shoulder_generation_fp_br", "shoulder_absorption_fp_br",
    "elbow_transfer_fp_br", "elbow_generation_fp_br", "elbow_absorption_fp_br",
    "lead_hip_transfer_fp_br", "lead_hip_generation_fp_br", "lead_hip_absorption_fp_br",
    "lead_knee_transfer_fp_br", "lead_knee_generation_fp_br",
    "lead_knee_absorption_fp_br",
    "rear_hip_transfer_pkh_fp", "rear_hip_generation_pkh_fp",
    "rear_hip_absorption_pkh_fp",
    "rear_knee_transfer_pkh_fp", "rear_knee_generation_pkh_fp",
    "rear_knee_absorption_pkh_fp",
    "pelvis_lumbar_transfer_fp_br", "thorax_distal_transfer_fp_br",
    "rear_grf_x_max", "rear_grf_y_max", "rear_grf_z_max", "rear_grf_mag_max",
    "rear_grf_angle_at_max",
    "lead_grf_x_max", "lead_grf_y_max", "lead_grf_z_max", "lead_grf_mag_max",
    "lead_grf_angle_at_max",
    "peak_rfd_rear", "peak_rfd_lead",
]

# ------------------------------------------------------- adopted-row descriptors
# image-plane observable and read rule for the 12 adopted estimators, keyed by the
# estimator function name so a renamed paper label cannot silently unmap one.
ADOPTED_DESC = {
    "est_knee_abs":     ("hip-knee-ankle angle", "at release"),
    "est_stride_anchor": ("ankle-ankle separation / stature", "at settled stride"),
    "est_trunk":        ("trunk line vs image vertical", "at release"),
    "est_wrist_abs":    ("wrist point speed", "whole-clip maximum"),
    "est_armslot":      ("shoulder-wrist line orientation", "at release"),
    "est_relh":         ("wrist height / stature", "at release"),
    "est_hss":          ("hip-line minus shoulder-line angle",
                         "signature-anchored peak"),
    "est_pelvis_rot":   ("hip-line angular velocity", "window maximum near release"),
    "est_cog_velo":     ("whole-body COM forward velocity", "window maximum to release"),
    "est_cog_pkh":      ("whole-body COM forward velocity", "at peak knee height"),
    "est_release_ext":  ("wrist-to-trail-anchor forward distance", "at release"),
    "est_stride_angle": ("ankle-line orientation", "at foot plant"),
}

FAMILY_RULES = [
    ("glove_shoulder", "glove shoulder"), ("shoulder", "shoulder"),
    ("elbow", "elbow"), ("lead_knee", "lead knee"), ("knee", "lead knee"),
    ("pelvis", "pelvis"), ("torso", "torso"), ("trunk", "torso"),
    ("hip_shoulder", "hip-shoulder separation"),
    ("hip-shoulder", "hip-shoulder separation"),
    ("stride", "stride"), ("release", "release"), ("cog", "centre of mass"),
    ("arm slot", "arm slot"), ("arm_slot", "arm slot"), ("timing", "sequencing"),
    ("wrist", "wrist"),
]


def family(name):
    low = name.lower()
    for key, fam in FAMILY_RULES:
        if key in low:
            return fam
    return "other"


def read_rule_screened(obs, ev):
    """How the screened row's value is read off the image-plane series."""
    if obs.endswith("_max") or obs.endswith("_min"):
        return "window maximum" if obs.endswith("_max") else "window minimum"
    if obs == "knee_ext_fp_to_br":
        return "difference between two events"
    if obs == "torso_pelvis_timing":
        return "interval between two window maxima"
    return f"at {ev}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_registry.csv")
    a = ap.parse_args()

    # ---- 1. column census -------------------------------------------------
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"),
                      nrows=1)
    cols = list(poi.columns)
    kinematic = [c for c in cols if c not in METADATA and c not in KINETIC]
    print(f"OBP poi columns: {len(cols)}")
    print(f"  metadata  {len(METADATA)}\n  kinetic   {len(KINETIC)}"
          f"\n  kinematic {len(kinematic)}")
    assert len(cols) == 81, len(cols)
    assert len(METADATA) == 5 and len(KINETIC) == 34 and len(kinematic) == 42, (
        len(METADATA), len(KINETIC), len(kinematic))
    missing = [c for c in METADATA + KINETIC if c not in cols]
    assert not missing, f"declared column absent from poi_metrics: {missing}"

    # ---- 2. the 47 evaluation rows ---------------------------------------
    # THE ROW SET COMES FROM WHAT WAS ACTUALLY SCORED, not from the code's candidate
    # lists. `CANDS` is the full candidate dict and still contains rows the
    # 2026-07-29 dedup removed (max_pelvis_rotational_velo), so enumerating it
    # double-counts a kinematic column against the adopted row that kept it.
    # `layer_summary.csv` is the authority for membership; the code supplies only
    # the DESCRIPTORS. The assert below then proves the two agree.
    ls_raw = pd.read_csv(os.path.join(V, "layer_summary.csv"))
    scored = list(ls_raw.metric)
    adopted_desc = {lab: (estfn, truth) for lab, estfn, truth in adopted_rows()}

    rows, unknown = [], []
    for m in scored:
        if m in adopted_desc:
            estfn, truth = adopted_desc[m]
            direct = not isinstance(truth, str)
            obs, rule = ADOPTED_DESC[estfn.__name__]
            rows.append(dict(
                metric_id=m, paper_name=m.replace(" [O]", ""),
                row_class="adopted", quantity_family=family(m),
                image_plane_observable=obs,
                truth_source="3d_direct" if direct else "obp_column",
                truth_quantity=(truth[1].__name__ if direct else truth),
                read_rule=rule))
        elif m in CANDS:
            obs, ev = CANDS[m]
            rows.append(dict(
                metric_id=m, paper_name=m, row_class="screened",
                quantity_family=family(m), image_plane_observable=obs,
                truth_source="obp_column", truth_quantity=m,
                read_rule=read_rule_screened(obs, ev)))
        else:
            unknown.append(m)
    assert not unknown, f"scored rows with no descriptor in the code: {unknown}"
    reg = pd.DataFrame(rows)

    # ---- 3. accounting ----------------------------------------------------
    col_paths = reg[reg.truth_source == "obp_column"].truth_quantity.tolist()
    direct_rows = int((reg.truth_source == "3d_direct").sum())
    dupes = [c for c in set(col_paths) if col_paths.count(c) > 1]
    assert not dupes, f"a kinematic column is evaluated twice: {dupes}"
    uncovered = sorted(set(kinematic) - set(col_paths))
    print(f"\nevaluation rows: {len(reg)}"
          f"  = {len(col_paths)} kinematic-column pathways"
          + (f" + {direct_rows} direct-3D-truth rows" if direct_rows else ""))
    if uncovered:
        print(f"  !! kinematic columns with no pathway: {uncovered}")
    assert not uncovered, uncovered
    assert len(col_paths) == 42 and direct_rows == 0, (len(col_paths), direct_rows)
    assert len(reg) == 42, len(reg)

    # ---- 4. results, joined from the canonical CSVs -----------------------
    ls = pd.read_csv(os.path.join(V, "layer_summary.csv"))
    ls["best_az"] = ls.best_ccc_view.str.split("/").str[0].astype(int)
    ls["best_el"] = ls.best_ccc_view.str.split("/").str[1].astype(int)
    ls["retained"] = ls.map_cells > 0
    # RENAMED from layer_summary's `map_cells`. "map cells" reads either way: the
    # 30 retained rows span 30 x 168 = 5,040 EVALUATED cells, of which 1,151 are
    # graded. `graded_cells` is the strong-or-moderate count and nothing else.
    ls = ls.rename(columns={"map_cells": "graded_cells"})
    reg = reg.merge(
        ls[["metric", "retained", "metric_grade", "graded_cells", "strong_cells",
            "moderate_cells", "best_ccc", "best_r2", "best_az", "best_el"]],
        left_on="metric_id", right_on="metric", how="left").drop(columns="metric")
    assert reg.retained.notna().all(), \
        sorted(reg.metric_id[reg.retained.isna()])
    reg["retained"] = reg.retained.astype(bool)

    # anchor fields: the tolerance layer already classifies every RETAINED row.
    et = os.path.join(V, "event_tolerance_map.csv")
    if os.path.exists(et):
        t = pd.read_csv(et).drop_duplicates("metric")
        reg = reg.merge(
            t[["metric", "anchor_type", "shifted_boundaries", "applicable",
               "not_applicable_reason"]].rename(
                columns={"applicable": "event_tolerance_applicable"}),
            left_on="metric_id", right_on="metric", how="left").drop(columns="metric")
    # non-retained rows hold no graded cell, so a tolerance question does not
    # arise for them; that is a scope statement, not a robustness finding.
    reg.loc[~reg.retained, "event_tolerance_applicable"] = False
    reg.loc[~reg.retained, "not_applicable_reason"] = \
        reg.loc[~reg.retained, "not_applicable_reason"].fillna("row holds no graded cell")
    # fall back to the screened event key where the tolerance layer has no row
    fallback = reg.anchor_type.isna() & reg.metric_id.isin(CANDS)
    reg.loc[fallback, "anchor_type"] = [CANDS[m][1] for m in reg.metric_id[fallback]]
    # ⚠ THE FALLBACK UNDER-REPORTS fp DEPENDENCE. A window observable carries the
    # event key of its FAR end ('rel') but reads the whole [fp, rel] window, so it
    # moves when the foot plant moves too. The tolerance layer already encodes that
    # for the rows it covers; the 12 non-retained rows have no tolerance entry, so
    # without this correction six of them would be filed as release-anchored and any
    # fp-dependence denominator built from the registry would be six rows short.
    # `fp_dependent_rows()` derives the true set from the code.
    # ONE VOCABULARY, applied BEFORE the fp upgrade below. The tolerance layer
    # writes "release" and CANDS writes "rel" for the same anchor; leaving both
    # would make any group-by split one class into two and quietly halve a
    # denominator -- and would also make the upgrade's `== "release"` test miss
    # every fallback row, which is exactly what it did on the first attempt.
    reg["anchor_type"] = reg.anchor_type.replace({"rel": "release"})
    from fp_target_check import fp_dependent_rows
    reads_fp = fp_dependent_rows()
    # DOCUMENTED EXCEPTION, applied first. `fp_dependent_rows()` lists
    # COG Velo @PKH because its estimator falls back to ctx['fp'] when pkh is
    # absent. Under the GT-event convention pkh is never absent, so that branch
    # never runs and its measured fp-sensitivity is exactly 0.0000. It keeps `pkh`.
    reads_fp = reads_fp - {"COG Velo @PKH [O]"}
    upgrade = reg.metric_id.isin(reads_fp) & (reg.anchor_type == "release")
    reg.loc[upgrade, "anchor_type"] = "release+fp"
    print(f"  fp dependence upgraded release -> release+fp on {int(upgrade.sum())} "
          f"window-observable rows the event key alone could not express")
    assert (reg.loc[reg.metric_id == "COG Velo @PKH [O]", "anchor_type"] == "pkh").all()
    assert set(reg.anchor_type.dropna()) <= {"none", "release", "fp", "mer", "pkh",
                                             "release+fp"}, \
        sorted(set(reg.anchor_type.dropna()))
    # the registry's fp-dependent set must now equal the code's, exception aside
    reg_fp = set(reg.metric_id[reg.anchor_type.isin(["fp", "release+fp"])])
    assert reg_fp == (reads_fp & set(reg.metric_id)), (
        sorted(reg_fp - reads_fp), sorted((reads_fp & set(reg.metric_id)) - reg_fp))
    # shifted_boundaries is only defined where the tolerance layer scored the row;
    # elsewhere say so rather than leaving an empty cell that reads as "no anchor".
    reg["shifted_boundaries"] = reg.shifted_boundaries.fillna("not scored")
    reg["circular"] = reg.metric_id.isin(CIRCULAR)

    # ---- 5. the two different 35s ----------------------------------------
    n_same_name = int((reg.row_class == "screened").sum())
    n_retained = int(reg.retained.sum())
    print(f"\nn_same_name_pathways  {n_same_name}   (screened rows named after "
          f"their own OBP column)")
    print(f"n_retained_rows       {n_retained}   (rows holding >=1 graded cell)"
          f"   non-retained {len(reg) - n_retained}")
    # the two counts were both 35 until the row set was reduced on 2026-08-12 and
    # the coincidence that made them confusable is gone
    assert n_same_name == 35 and n_retained == 30, (n_same_name, n_retained)
    graded = int(reg.graded_cells.fillna(0).sum())
    strong = int(reg.strong_cells.fillna(0).sum())
    moderate = int(reg.moderate_cells.fillna(0).sum())
    evaluated = n_retained * 168
    print(f"graded_cells {graded} = {strong} strong + {moderate} moderate"
          f"   (out of {evaluated} evaluated cells over the retained rows)")
    # UNCHANGED by the 2026-08-08 switch to nested correction-model selection. The retired
    # rule chose the winning model by an argmax over out-of-fold CCC that included the
    # held-out pitcher; the adopted rule takes the modal choice over per-fold selections
    # each made blind to its own pitcher (gate_map.nested_predictions). The two agree on
    # 7,882 of 7,896 cells, and where they differ the CCC moves by at most 5.8e-04 -- not
    # enough to move any cell's grade or verdict. Removing the leak cost nothing.
    # 1,500 = 1,100 + 400 over 35 rows until the row set was reduced on 2026-08-12.
    # The 42 kept rows re-scored bit-identically, so the fall is the five removed rows
    # and nothing else.
    assert graded == 1151 and strong == 819 and moderate == 332, \
        (graded, strong, moderate)
    assert graded == strong + moderate

    COLS = ["metric_id", "paper_name", "row_class", "quantity_family",
            "image_plane_observable", "truth_source", "truth_quantity",
            "read_rule", "anchor_type", "shifted_boundaries", "circular",
            "event_tolerance_applicable", "not_applicable_reason", "retained",
            "metric_grade", "graded_cells", "strong_cells", "moderate_cells",
            "best_ccc", "best_r2", "best_az", "best_el"]
    reg = reg[COLS].sort_values(
        ["retained", "best_ccc"], ascending=[False, False]).reset_index(drop=True)
    p = os.path.join(V, a.out)
    reg.to_csv(p, index=False, float_format="%.6g")
    print(f"\nsaved -> {p}")

    print("\n" + "=" * 96)
    print("NON-RETAINED ROWS (reported with best cell only; no cause is inferred)")
    print("=" * 96)
    nr = reg[~reg.retained][["metric_id", "quantity_family", "best_ccc",
                             "best_r2", "best_az", "best_el"]]
    print(nr.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
