"""
Diamond - setup-anchor rule change: regression against the frozen map.

metrics.QUIET_GUARD went 1 -> 5 on 2026-07-27 (analysis/motion_onset_candidates.py,
user-adopted). trail_anchor_x feeds `Stride (anchor)` and `Release Ext`, and it also
feeds the Release Ext TRUTH (angle_map_2d.t3_release_ext uses the same rule on the
c3d, deliberately), so a definition change moves the frozen table. This file
measures by how much, with the ADOPTED estimators -- not a re-implementation.

Two checks:
  1. guard=1 must REPRODUCE gate_map.csv for those two rows. If it does not, the
     harness is wrong and nothing below can be trusted.
  2. guard=5 minus guard=1, per cell: CCC delta and any grade change.

Population = the frozen map's 394 pitches (mer_proxy_map.map_population), events =
OBP GT landmarks, scoring = gate_map.score_cell.

Outputs: setup_anchor_regression.csv
Run:  cd src\\analysis
      python setup_anchor_regression.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)

import config
import metrics as M
import obp_project as O
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from obp_gt_events import load_gt_events
from gate_map import score_cell
from mer_proxy_map import map_population
from angle_map_2d import t3_release_ext
from motion_onset_candidates import STRIDE_CELLS, RELEXT_CELLS

CELLS = sorted(set(STRIDE_CELLS) | set(RELEXT_CELLS))
GUARDS = [1, 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--floor", type=float, default=0.75)
    ap.add_argument("--strong", type=float, default=0.80)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    root = os.path.join(config.OBP_DATA_DIR, "c3d")
    gt = load_gt_events()
    pop = map_population()

    stride = {(g, c): [] for g in GUARDS for c in STRIDE_CELLS}
    relext = {(g, c): [] for g in GUARDS for c in RELEXT_CELLS}
    t_str, t_rel, users = [], {g: [] for g in GUARDS}, []
    done = fail = 0
    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        sp = r.session_pitch
        if sp not in pop or sp not in poi.index:
            continue
        g = gt.get(sp)
        if not g or not {"fp", "rel"} <= set(g):
            fail += 1; continue
        path = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = load_feet(path)
            arm = O.detect_throwing_arm(joints, fps)
        except Exception:
            fail += 1; continue
        lead = "left" if arm == "right" else "right"
        fp, rel = int(g["fp"]), int(g["rel"])
        h = float(r.session_height_m)
        ctx = {"arm": arm, "fp": fp, "rel": rel, "fps": fps}

        t_str.append(float(poi.loc[sp, "stride_length"]))
        users.append(r.user)
        for gu in GUARDS:
            M.QUIET_GUARD = gu            # resolved at call time by _quiet_end
            t_rel[gu].append(t3_release_ext(joints, ctx))
        for c in CELLS:
            try:
                df = project_cam(joints, c[0], c[1])
            except Exception:
                df = None
            for gu in GUARDS:
                M.QUIET_GUARD = gu
                sv = rv = np.nan
                if df is not None:
                    try:
                        if (gu, c) in stride:
                            sv = M.stride_settled_2d(df, lead, rel, fps, M.JOINTS)
                        if (gu, c) in relext:
                            rv = M.release_extension(df, arm, rel, fp, fps,
                                                     M.JOINTS, h)
                    except Exception:
                        pass
                if (gu, c) in stride:
                    stride[(gu, c)].append(sv)
                if (gu, c) in relext:
                    relext[(gu, c)].append(rv)
        done += 1
        if done % 50 == 0:
            print(f"  ...{done} processed")
    M.QUIET_GUARD = 5                     # leave the adopted value in place
    print(f"processed {done} / failed {fail}\n")

    codes = pd.Series(users).astype("category").cat.codes.to_numpy()
    ts = np.asarray(t_str, float)
    rows = []
    for gu in GUARDS:
        tr = np.asarray(t_rel[gu], float)
        for c in CELLS:
            for name, store, truth in (("Stride (anchor) [O]", stride, ts),
                                       ("Release Ext [O]", relext, tr)):
                if (gu, c) not in store:
                    continue
                e = np.asarray(store[(gu, c)], float)
                s = score_cell(e, truth, codes, a.floor, a.strong)
                if s is None:
                    continue
                rows.append(dict(metric=name, guard=gu, az=c[0], el=c[1],
                                 nan_share=float(np.mean(~np.isfinite(e))), **s))
    out = pd.DataFrame(rows)
    p = os.path.join(config.OBP_VALIDATION_DIR, "setup_anchor_regression.csv")
    out.to_csv(p, index=False, float_format="%.6g")

    frozen = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR, "gate_map.csv"))
    pd.set_option("display.width", 200)
    print("=" * 92)
    print(f"SETUP-ANCHOR REGRESSION -- n={len(ts)} pitches, {len(set(users))} pitchers")
    print("=" * 92)

    print("\n[1] harness check: guard=1 vs the frozen gate_map.csv")
    for name in ("Stride (anchor) [O]", "Release Ext [O]"):
        mine = out[(out.metric == name) & (out.guard == 1)].set_index(["az", "el"])
        fr = frozen[frozen.metric == name].set_index(["az", "el"])
        j = mine.join(fr[["ccc", "grade"]], rsuffix="_frozen", how="inner")
        d = (j.ccc - j.ccc_frozen).abs()
        print(f"  {name:<22} {len(j)} cells   max |dCCC| {d.max():.6f}   "
              f"grade mismatches {(j.grade != j.grade_frozen).sum()}")

    print("\n[2] guard=5 minus guard=1")
    for name in ("Stride (anchor) [O]", "Release Ext [O]"):
        s = out[out.metric == name]
        a1 = s[s.guard == 1].set_index(["az", "el"])
        a5 = s[s.guard == 5].set_index(["az", "el"])
        j = a1.join(a5, rsuffix="_g5", how="inner")
        d = j.ccc_g5 - j.ccc
        moved = j[(j.grade != j.grade_g5)]
        print(f"\n  {name}   ({len(j)} cells)")
        print(f"    dCCC  median {d.median():+.5f}   min {d.min():+.5f}   "
              f"max {d.max():+.5f}")
        print(f"    strong cells {int((j.grade=='strong').sum())} -> "
              f"{int((j.grade_g5=='strong').sum())}   "
              f"gate cells {int(j.gate_pass.sum())} -> {int(j.gate_pass_g5.sum())}")
        print(f"    deferred (NaN) share {j.nan_share.median():.4f} -> "
              f"{j.nan_share_g5.median():.4f}")
        if len(moved):
            print("    GRADE CHANGES:")
            for r2 in moved.itertuples():
                print(f"      az{r2.Index[0]:>3} el{r2.Index[1]:>2}  "
                      f"{r2.grade} -> {r2.grade_g5}  "
                      f"CCC {r2.ccc:.4f} -> {r2.ccc_g5:.4f}")
        else:
            print("    no grade changes")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
