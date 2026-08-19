"""FINAL back-calculation run: can the kinetics be inferred from what 2D can measure?

Everything is kept -- passes and failures alike -- because the failures are the
paper's evidence, not debris. Previous runs printed only the headline and left the
GRF direction components with an inherited number ("not separately inferred"); here
EVERY kinetic column in the OBP poi table is its own target with its own score.

Four input sets per target, so a change in the number can be attributed:

  anthro     height / mass / age only. The floor: what body size alone predicts.
  gate       the truth columns of every metric with >=1 GATE cell (LOCO CCC >= 0.80
             somewhere), i.e. what a calibrated 2D system can actually deliver
             under the 2026-07-27 two-layer map.
  screen403  the input rule the 2026-07-24 run used (r2 >= 0.60 on the unfiltered
             403-pitch screen, union the adopted set). Kept so the new numbers are
             comparable with the recorded ones.
  allkin     EVERY kinematic poi column, measurable or not. Not a deployable
             system -- it is the ceiling. If kinetics does not fall out of the
             complete 3D kinematic description, no 2D pipeline can reach it, and
             the closure argument stops depending on which columns we can measure.

INPUTS ARE THE PERFECT 3D poi COLUMNS, never our 2D estimates: reading them
imperfectly can only do worse, so a low ceiling here is decisive. Kinetics are not
algebraic functions of kinematics, so there is no definitional leakage of the kind
that inflated rotation_hip_shoulder_separation_fp. GroupKFold by pitcher blocks
subject leakage. Adoption floor R2 = 0.50 (inference_retry_enriched.FLOOR).

Output: inference_final.csv (every target x every input set) + the printed table.

Run:  conda activate diamond; cd src\\research; python inference_final.py
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
sys.path.insert(0, HERE)
import config  # noqa: E402
from inference_retry_enriched import ANTHRO, cv_r2, FLOOR  # noqa: E402
from angle_map_2d import adopted_rows, gt_only_rows  # noqa: E402
# gt_clean, like every other paper layer: the excluded pitches have a broken
# foot-plant landmark, and the kinetic truths are computed at those landmarks.
from gt_landmark_outlier_effect import outlier_pitches  # noqa: E402

AUDIT = os.path.join(config.OBP_VALIDATION_DIR, "column_coverage_audit.csv")
LAYER = os.path.join(config.OBP_VALIDATION_DIR, "layer_summary.csv")
OLD_SCREEN = os.path.join(config.OBP_VALIDATION_DIR, "rejected_gt_full_sweep.csv")
OUT = os.path.join(config.OBP_VALIDATION_DIR, "inference_final.csv")


def main():
    audit = pd.read_csv(AUDIT).set_index("column")
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    df = poi.merge(md[["session_pitch", "user"] + ANTHRO], on="session_pitch")
    n_before = len(df)
    df = df[~df.session_pitch.isin(outlier_pitches())]
    print(f"gt_clean filter: {n_before} -> {len(df)} pitches")

    kinetic = [c for c in poi.columns
               if c in audit.index and audit.loc[c, "reason_code"] == "A-kinetics"]
    meta = [c for c in poi.columns
            if c in audit.index and "metadata" in str(audit.loc[c, "reason"])]
    kinematic = [c for c in poi.columns
                 if c != "session_pitch" and c not in kinetic and c not in meta]

    # gate set: poi truth columns of every metric with at least one gate cell
    lay = pd.read_csv(LAYER)
    ok = lay[lay.gate_cells > 0]
    label_truth = {l.strip(): t for l, _, t in adopted_rows() + gt_only_rows()
                   if isinstance(t, str)}
    gate_cols = {c for c in ok.metric if c in poi.columns}
    gate_cols |= {label_truth[m.strip()] for m in ok.metric
                  if m.strip() in label_truth}
    gate = sorted(c for c in gate_cols if c in df.columns)

    # the 2026-07-24 rule, reproduced
    old = pd.read_csv(OLD_SCREEN)
    revived = list(old.loc[old.best_r2 >= 0.60, "column"])
    adopted_input = ["stride_length", "stride_angle", "arm_slot",
                     "torso_anterior_tilt_br", "lead_knee_extension_angular_velo_br",
                     "max_cog_velo_x", "cog_velo_pkh",
                     "max_rotation_hip_shoulder_separation",
                     "max_pelvis_rotational_velo"]
    screen403 = [c for c in dict.fromkeys(adopted_input + revived) if c in df.columns]

    SETS = {"anthro": [], "gate": gate, "screen403": screen403,
            "allkin": sorted(kinematic)}

    print(f"n = {len(df)} pitches, {df.user.nunique()} pitchers")
    print(f"kinetic targets : {len(kinetic)}")
    for k, v in SETS.items():
        print(f"  input set {k:<10} {len(v):>3} mechanics columns "
              f"(+{len(ANTHRO)} anthro)")
    print(f"\ninference floor R2 = {FLOOR}; inputs are PERFECT 3D columns (a ceiling)\n")

    rows = []
    for t in kinetic:
        if t not in df.columns:
            continue
        line = {"target": t}
        for name, feats in SETS.items():
            r, n = cv_r2(df, ANTHRO + feats, t)
            line[name] = np.nan if r is None else r
            line["n"] = n
        rows.append(line)
        print(f"  scored {t}")

    res = pd.DataFrame(rows)
    for k in SETS:
        if k != "anthro":
            res[f"gain_{k}"] = res[k] - res["anthro"].clip(lower=0)
    res["best_set"] = res[list(SETS)].idxmax(axis=1)
    res["best_r2"] = res[list(SETS)].max(axis=1)
    res["clears_floor"] = res.best_r2 >= FLOOR
    res.to_csv(OUT, index=False, float_format="%.6g")

    W = 104
    print("\n" + "=" * W)
    print("BACK-CALCULATION, EVERY KINETIC TARGET, EVERY INPUT SET  (GroupKFold by "
          "pitcher, GBM)")
    print("=" * W)
    print(f"{'kinetic target':<42}{'anthro':>9}{'screen403':>11}{'gate':>9}"
          f"{'allkin':>9}{'best':>9}{'n':>7}")
    print("-" * W)
    for _, r in res.sort_values("best_r2", ascending=False).iterrows():
        flag = "  <== CLEARS" if r.clears_floor else ""
        print(f"{r.target:<42}{r.anthro:>9.3f}{r.screen403:>11.3f}{r.gate:>9.3f}"
              f"{r.allkin:>9.3f}{r.best_r2:>9.3f}{int(r.n):>7}{flag}")
    print("-" * W)
    n_pass = int(res.clears_floor.sum())
    print(f"targets clearing R2 {FLOOR}: {n_pass} of {len(res)}")
    for k in ("screen403", "gate", "allkin"):
        print(f"  best {k:<10} R2 = {res[k].max():.3f} "
              f"({res.loc[res[k].idxmax(), 'target']});  "
              f"median gain over anthro = {res[f'gain_{k}'].median():+.3f}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
