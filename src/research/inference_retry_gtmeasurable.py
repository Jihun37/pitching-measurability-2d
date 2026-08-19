"""Inference retry with the FULL 2D-measurable input set (GT-event sweep, 2026-07-24).

The 2026-07-20 retry (inference_retry_enriched) added only two transverse columns
(pelvis rot velo, HSS) plus the COG pair to the sagittal pool, because at the time
those were the only transverse quantities shown 2D-recoverable. The GT-event sweep
(analysis/rejected_gt_full_sweep) has since shown that many more kinematic columns
are recoverable once they are read at the correct GT event and from the right
elevation -- torso rotation, pelvis rotation, several shoulder-abduction columns,
the MER-anchored trunk tilts, etc.

So the perfect-input ceiling question is worth settling once more, now with an input
set that is close to "everything a calibrated 2D system could measure". If even that
cannot predict the injury-relevant kinetics out of sample, the inference door is
closed at the strongest available input set, not merely at the 2020-era one.

DESIGN (unchanged from the prior runs, so results are comparable):
  - INPUTS are the PERFECT OBP 3D columns for every measurable quantity = a ceiling.
    Reading them imperfectly in 2D can only do worse, so a low ceiling is decisive.
  - INPUTS are kinematic only; TARGETS are kinetic only (forces, moments, GRF, joint
    energy). Kinetics are not algebraic functions of the kinematic inputs, so there
    is no definitional leakage of the kind that inflated rotation_hip_shoulder_
    separation_fp (0.81, later shown to be max_HSS re-expressed).
  - GroupKFold by pitcher blocks subject leakage; anthro-only baseline is reported
    so what matters is the INCREMENTAL R2 of mechanics over body size.

The measurable-input list is read at runtime from rejected_gt_full_sweep.csv (>= the
usable floor) UNION the adopted set, so it stays in sync with the sweep.

Run:  conda activate diamond; cd src\\research; python inference_retry_gtmeasurable.py
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import config  # noqa: E402
from inference_retry_enriched import ANTHRO, cv_r2, FLOOR  # noqa: E402

SWEEP = os.path.join(config.OBP_VALIDATION_DIR, "rejected_gt_full_sweep.csv")

# adopted 12 as perfect-input columns (3D-direct ones have no poi column, dropped)
ADOPTED_INPUT = [
    "stride_length", "stride_angle", "arm_slot", "torso_anterior_tilt_br",
    "lead_knee_extension_angular_velo_br", "max_cog_velo_x", "cog_velo_pkh",
    "max_rotation_hip_shoulder_separation", "max_pelvis_rotational_velo",
]

# pure kinetics -- forces, moments, ground reaction, joint energy flow. None is an
# algebraic function of the kinematic inputs, so no definitional leakage.
TARGETS_KIN = [
    "elbow_varus_moment", "shoulder_internal_rotation_moment",
    "rear_grf_mag_max", "lead_grf_mag_max", "peak_rfd_rear", "peak_rfd_lead",
    "shoulder_transfer_fp_br", "shoulder_generation_fp_br", "shoulder_absorption_fp_br",
    "elbow_transfer_fp_br", "elbow_generation_fp_br", "elbow_absorption_fp_br",
    "lead_hip_transfer_fp_br", "lead_hip_generation_fp_br", "lead_hip_absorption_fp_br",
    "lead_knee_transfer_fp_br",
]


def main():
    if not os.path.exists(SWEEP):
        sys.exit("run analysis/rejected_gt_full_sweep.py first")
    sweep = pd.read_csv(SWEEP)
    revived = list(sweep.loc[sweep.best_r2 >= 0.60, "column"])

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    df = poi.merge(md[["session_pitch", "user"] + ANTHRO], on="session_pitch")

    inputs = [c for c in dict.fromkeys(ADOPTED_INPUT + revived) if c in df.columns]
    print(f"n = {len(df)} pitches, {df.user.nunique()} pitchers")
    print(f"measurable-input ceiling set: {len(inputs)} columns "
          f"(adopted {len(ADOPTED_INPUT)} + sweep-revived "
          f"{len([c for c in revived if c not in ADOPTED_INPUT])})")
    print("  " + ", ".join(inputs))
    print(f"\nadoption floor R2 = {FLOOR}; inputs are PERFECT 3D columns (a ceiling).")
    print(f"  {'kinetic target':<40}{'anthro':>9}{'+mech':>9}{'gain':>9}{'n':>7}")
    print("  " + "-" * 74)

    rows = []
    for t in TARGETS_KIN:
        if t not in df.columns:
            print(f"  {t:<40}  (not in poi)")
            continue
        r_base, _ = cv_r2(df, ANTHRO, t)
        r_full, n = cv_r2(df, ANTHRO + inputs, t)
        if r_full is None:
            print(f"  {t:<40}  (too few rows)")
            continue
        gain = r_full - max(r_base, 0.0)
        flag = "  <-- CLEARS FLOOR" if r_full >= FLOOR else ""
        print(f"  {t:<40}{r_base:>9.3f}{r_full:>9.3f}{gain:>+9.3f}{n:>7d}{flag}")
        rows.append(dict(target=t, anthro=r_base, full=r_full, gain=gain, n=n))

    res = pd.DataFrame(rows)
    passed = res[res.full >= FLOOR]
    print(f"\n{len(passed)} / {len(res)} kinetic targets clear R2 >= {FLOOR} "
          f"with the full measurable-input ceiling.")
    if len(passed):
        print("  passing:", ", ".join(f"{r.target} ({r.full:.2f})"
                                       for r in passed.itertuples(index=False)))
        print("  -> any pass MUST be leakage-audited before it is believed "
              "(cf. rotation_hip_shoulder_separation_fp 0.81 = max_HSS re-expressed).")
    else:
        print("  the injury/kinetics axis stays closed even at the strongest input set.")
    out = os.path.join(config.OBP_VALIDATION_DIR, "inference_retry_gtmeasurable.csv")
    res.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
