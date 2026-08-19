"""Close the coverage gap: run EVERY remaining poi column through inference.

`scratch/column_coverage_audit.py` found 27 of the 81 poi columns had never been
attempted at all -- neither measured from a projection nor used as an inference
target. They are not a random 27: 23 are kinetics siblings (GRF components, joint
energy generation/absorption) whose family representatives were tested and failed,
3 are glove-side arm angles whose throwing-side analogue was tested and rejected,
and 1 (elbow_flexion_mer) is a genuine gap from the survey's own "remaining to
test" trio.

Family reasoning is defensible, but for the manuscript it is much stronger to
attach a NUMBER to every column than an assertion about its family. This run does
that, using the same design as inference_retry_enriched (perfect-input ceiling,
GroupKFold by pitcher, anthro baseline reported so the incremental contribution of
mechanics is visible).

After this, every one of the 81 columns is either measured, directly tested, or
inference-tested.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import config  # noqa: E402
from inference_retry_enriched import (ANTHRO, INPUT_NEW, INPUT_OLD,  # noqa: E402
                                      FLOOR, cv_r2)

AUDIT = os.path.join(config.ROOT, "data", "outputs", "obp_validation",
                     "column_coverage_audit.csv")


def main():
    audit = pd.read_csv(AUDIT)
    todo = list(audit[audit.status.str.contains("NEVER")].column)
    print(f"columns never attempted: {len(todo)}\n")

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    df = poi.merge(md[["session_pitch", "user"] + ANTHRO], on="session_pitch")

    print(f"  {'column':<42}{'anthro':>9}{'+mech':>9}{'gain':>9}{'n':>7}")
    print("  " + "-" * 76)
    rows = []
    for c in todo:
        if c not in df.columns:
            print(f"  {c:<42}  (not in poi)")
            continue
        r_base, _ = cv_r2(df, ANTHRO, c)
        r_full, n = cv_r2(df, ANTHRO + INPUT_OLD + INPUT_NEW, c)
        if r_full is None:
            print(f"  {c:<42}  (insufficient n)")
            continue
        flag = "  <-- CLEARS FLOOR" if r_full >= FLOOR else ""
        print(f"  {c:<42}{r_base:>9.3f}{r_full:>9.3f}"
              f"{r_full - r_base:>+9.3f}{n:>7d}{flag}")
        rows.append(dict(target=c, anthro=r_base, full=r_full,
                         gain=r_full - r_base, n=n))

    out = pd.DataFrame(rows)
    n_ok = int((out.full >= FLOOR).sum())
    print(f"\n  clearing the {FLOOR} floor: {n_ok} / {len(out)}")
    if n_ok:
        print(out[out.full >= FLOOR].to_string(index=False))
    print(f"  best overall: {out.full.max():.3f} "
          f"({out.loc[out.full.idxmax(), 'target']})")

    dst = os.path.join(config.ROOT, "data", "outputs", "obp_validation",
                       "inference_close_remaining.csv")
    out.to_csv(dst, index=False)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
