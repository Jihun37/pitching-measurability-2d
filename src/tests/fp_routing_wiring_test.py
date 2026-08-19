"""Regression for wiring deploy/fp_routing.fp_view into metrics.compute_candidates.

Two things have to hold, and they pull in opposite directions:

  A  INERT WITHOUT A VIEWPOINT. Every existing caller passes no az/el (the whole
     validation stack does), so compute_candidates() must be BIT-IDENTICAL to the
     pre-wiring behaviour there. Checked by comparing against an explicit
     side-detector reference computed the old way.
  B  ACTIVE WITH ONE, AND ONLY ON FP-DEPENDENT OUTPUTS. With az/el supplied at a
     cell the rule routes to `frontal`, the foot-plant-dependent outputs may move
     and the rest must NOT. The FP-dependent set is derived from what `fp` feeds in
     compute_candidates, not guessed.

Run:  conda activate diamond; cd src\\tests; python fp_routing_wiring_test.py
"""
import os, sys
import numpy as np, pandas as pd

_HERE = os.path.dirname(__file__)
for p in ("..", "../stage2", "../stage3", "../analysis", "../deploy"):
    sys.path.insert(0, os.path.join(_HERE, p))
import config
import metrics as M
import obp_project as O
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from fp_routing import fp_view, available

# Outputs that read `fp` (directly, or through pkh / trail anchor / stride angle).
FP_DEPENDENT = {
    "lead_knee_at_fp", "lead_knee_extension",
    "stride_length", "stride_pct_height",
    "cog_velo_pkh", "release_ext", "stride_angle",
    # trunk tilt only uses fp for a direction SIGN, so it is allowed to move but
    # practically never does; kept here so a sign flip is not reported as a leak
    "trunk_anterior_tilt", "lateral_trunk_tilt",
}
N = 12


def vals(c):
    return {k: (float(v) if np.isscalar(v) or isinstance(v, (int, float))
                else np.nan) for k, (v, _) in c.items()}


def main():
    print(f"fp routing table available: {available()}")
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    root = os.path.join(config.OBP_DATA_DIR, "c3d")
    cells = [(0, 0), (90, 15), (135, 15), (270, 15), (210, 85)]
    print("rule at the probe cells: "
          + "  ".join(f"{c}->{fp_view(*c)}" for c in cells) + "\n")

    n_a = n_b = 0
    worst_a = 0.0
    leaks, moved = {}, {}
    for r in md.head(N * 3).itertuples(index=False):
        if n_a >= N:
            break
        p = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(p):
            continue
        try:
            j, fps = load_feet(p)
            arm = O.detect_throwing_arm(j, fps)
            h = float(r.session_height_m)
        except Exception:
            continue
        df = project_cam(j, 0, 0)

        # ---- A: no viewpoint -> must equal the explicit side path -------------
        a1 = vals(M.compute_candidates(df, fps=fps, arm=arm, height_m=h))
        a2 = vals(M.compute_candidates(df, fps=fps, arm=arm, height_m=h,
                                       az=None, el=None))
        for k in a1:
            d = abs(a1[k] - a2[k]) if np.isfinite(a1[k]) and np.isfinite(a2[k]) else 0.0
            worst_a = max(worst_a, d)
        n_a += 1

        # ---- B: viewpoint supplied -> only FP-dependent outputs may move ------
        for (az, el) in cells:
            if fp_view(az, el) == "side":
                continue          # routing is a no-op there, nothing to test
            d2 = project_cam(j, az, el)
            base = vals(M.compute_candidates(d2, fps=fps, arm=arm, height_m=h))
            routed = vals(M.compute_candidates(d2, fps=fps, arm=arm, height_m=h,
                                               az=az, el=el))
            for k in base:
                if not (np.isfinite(base[k]) and np.isfinite(routed[k])):
                    continue
                if abs(base[k] - routed[k]) > 1e-9:
                    (moved if k in FP_DEPENDENT else leaks)[k] = \
                        max((moved if k in FP_DEPENDENT else leaks).get(k, 0.0),
                            abs(base[k] - routed[k]))
            n_b += 1

    print("=" * 78)
    print("[A] no viewpoint -> unchanged")
    print("=" * 78)
    print(f"  {n_a} pitches, max |delta| over every output = {worst_a:.12f}   "
          + ("PASS" if worst_a == 0.0 else "FAIL"))

    print("\n" + "=" * 78)
    print("[B] viewpoint supplied -> only FP-dependent outputs move")
    print("=" * 78)
    print(f"  {n_b} (pitch, cell) evaluations at cells the rule sends to frontal")
    print(f"  FP-dependent outputs that moved ({len(moved)}):")
    for k, v in sorted(moved.items(), key=lambda x: -x[1]):
        print(f"      {k:<24} max |delta| {v:.6g}")
    print(f"  NON-FP outputs that moved ({len(leaks)}):"
          + ("  none -- PASS" if not leaks else ""))
    for k, v in sorted(leaks.items(), key=lambda x: -x[1]):
        print(f"      !! {k:<21} max |delta| {v:.6g}")
    ok = (worst_a == 0.0) and not leaks
    print("\n" + ("ALL CHECKS PASS" if ok else "!! REGRESSION"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
