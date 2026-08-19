"""
Diamond - MER layer: ORACLE vs DEPLOYABLE, per cell (2026-07-27).

Why this exists. The graded map scores every MER-anchored row at the GT MER frame:
angle_zone_sweep --gt-events puts the landmark into ctx["mer"], and both the
screened columns (rejected_gt_full_sweep CANDS event key "mer") and the adopted
rows (angle_map_2d.mer_frame, which prefers ctx["mer"] and only falls back to
rel - k) read it. The frozen table proves it -- `Torso Lat Tilt @MER` and
`torso_lateral_tilt_mer` are bit-identical, and the second is unambiguously GT.

So the map's whole MER layer is ORACLE. That is legitimate for a measurability
claim (2D geometry at a known instant) and inadmissible for a deployment claim
(real video has no MER). The proxy had only ever been scored at three hand-picked
anchor viewpoints (mer_proxy_score.py), never per cell.

This file scores every MER quantity at every cell under an anchor LADDER, so the
oracle premium and the release-detection premium are separated:

  oracle        GT MER                       ceiling: pure viewpoint geometry
  proxy_gt      GT release - k               + the proxy assumption, no detector
  proxy_ruled   detected release - k         + detection under release_view()
  proxy_side    side-detected release - k    routing sensitivity
  proxy_frontal frontal-detected release - k routing sensitivity

k = 11 frames @360 Hz (rel - mer: median 11, SD 1.1, n=401).

Definitions are imported, never re-implemented: the observables come from
rejected_gt_full_sweep.observables and the scoring from gate_map.score_cell, so a
cell here is comparable to the same cell in gate_map.csv by construction.

Population: gt_clean, intersected with the pitch ids the frozen map actually used
(read from the two pair dumps) so cell counts are comparable row for row.

Outputs (OBP_VALIDATION_DIR):
  mer_proxy_map.csv         metric x az x el x variant -> full score_cell record
  mer_proxy_map_pairs.csv.gz (--dump)

Run:  cd src\\analysis
      python mer_proxy_map.py --limit 10     (smoke test)
      python mer_proxy_map.py
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
from gt_landmark_outlier_effect import outlier_pitches
from angle_zone_sweep import AZ, EL, release_view
from rejected_gt_full_sweep import CANDS, observables
from gate_map import score_cell

# The five DISTINCT MER quantities on the map. Four of the six mer_true rows are
# these reached through a second code path (the adopted labels), and the two
# "mer_proxy" rows are the same estimators again -- so scoring the quantity once
# covers every row instance.
MER_COLS = ["elbow_flexion_mer", "torso_lateral_tilt_mer", "torso_anterior_tilt_mer",
            "torso_rotation_mer", "glove_shoulder_abduction_mer"]
ADOPTED_TWIN = {"elbow_flexion_mer": "Elbow Flex @MER [O]",
                "torso_lateral_tilt_mer": "Torso Lat Tilt @MER [O]",
                "glove_shoulder_abduction_mer": "Glove Sh Abd @MER [O]",
                "torso_anterior_tilt_mer": "", "torso_rotation_mer": ""}
VARIANTS = ["oracle", "proxy_gt", "proxy_ruled", "proxy_side", "proxy_frontal"]
K_F360 = 11.0


def map_population():
    """The pitch ids the frozen gate map scored, so counts are comparable."""
    V = config.OBP_VALIDATION_DIR
    keep = None
    for f in ("angle_zone_pairs_gt.csv.gz", "rejected_gt_pairs.csv.gz"):
        p = os.path.join(V, f)
        if not os.path.exists(p):
            continue
        s = set(pd.read_csv(p, usecols=["session_pitch"]).session_pitch.unique())
        keep = s if keep is None else (keep & s)
    bad = outlier_pitches()
    return None if keep is None else (keep - bad)


def collect(limit=None):
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    root = os.path.join(config.OBP_DATA_DIR, "c3d")
    gt = load_gt_events()
    pop = map_population()
    print(f"map population: {'unavailable' if pop is None else len(pop)} pitches")

    est = {(c, az, el, v): [] for c in MER_COLS for az in AZ for el in EL
           for v in VARIANTS}
    tru = {c: [] for c in MER_COLS}
    sps, users = [], []
    done = fail = 0
    for i, r in enumerate(md.itertuples(index=False)):
        if limit and i >= limit:
            break
        sp = r.session_pitch
        if pop is not None and sp not in pop:
            continue
        g = gt.get(sp)
        if sp not in poi.index or not g or not {"rel", "mer"} <= set(g):
            fail += 1; continue
        path = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = load_feet(path)
            arm = O.detect_throwing_arm(joints, fps)
        except Exception:
            fail += 1; continue

        k = int(round(K_F360 / 360.0 * fps))
        g_mer, g_rel = int(g["mer"]), int(g["rel"])
        sps.append(sp); users.append(r.user)
        for c in MER_COLS:
            tru[c].append(poi.loc[sp, c] if c in poi.columns else np.nan)

        for az in AZ:
            for el in EL:
                frames = {}
                try:
                    df = project_cam(joints, az, el)
                    o, _ = observables(df, fps)
                    rel_s = int(M.release_frame(df, arm, fps, M.JOINTS, view="side"))
                    rel_f = int(M.release_frame(df, arm, fps, M.JOINTS,
                                                view="frontal"))
                    rel_r = rel_f if release_view(az, el) == "frontal" else rel_s
                    frames = {"oracle": g_mer, "proxy_gt": g_rel - k,
                              "proxy_ruled": rel_r - k, "proxy_side": rel_s - k,
                              "proxy_frontal": rel_f - k}
                except Exception:
                    o = None
                for c in MER_COLS:
                    okey = CANDS[c][0]
                    ser = o.get(okey) if o is not None else None
                    for v in VARIANTS:
                        f = frames.get(v, -1)
                        val = (float(ser[f]) if ser is not None
                               and 0 <= f < len(ser) else np.nan)
                        est[(c, az, el, v)].append(val)
        done += 1
        if done % 25 == 0:
            print(f"  ...{done} processed")
    print(f"processed {done} / failed {fail}\n")
    return est, tru, np.asarray(sps), np.asarray(users)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--floor", type=float, default=0.75)
    ap.add_argument("--strong", type=float, default=0.80)
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()

    est, tru, sps, users = collect(a.limit)
    codes = pd.Series(users).astype("category").cat.codes.to_numpy()

    rows = []
    for c in MER_COLS:
        t = np.asarray(tru[c], float)
        for v in VARIANTS:
            for az in AZ:
                for el in EL:
                    e = np.asarray(est[(c, az, el, v)], float)
                    s = score_cell(e, t, codes, a.floor, a.strong)
                    if s is None:
                        continue
                    rows.append(dict(metric=c, adopted_twin=ADOPTED_TWIN[c],
                                     variant=v, az=az, el=el, **s))
    out = pd.DataFrame(rows)
    p = os.path.join(config.OBP_VALIDATION_DIR, "mer_proxy_map.csv")
    out.to_csv(p, index=False, float_format="%.6g")
    print(f"saved -> {p}   ({len(out):,} rows)")

    if a.dump:
        recs = []
        for c in MER_COLS:
            for az in AZ:
                for el in EL:
                    for v in VARIANTS:
                        recs.append(pd.DataFrame(
                            {"metric": c, "az": az, "el": el, "variant": v,
                             "session_pitch": sps, "est": est[(c, az, el, v)],
                             "truth": tru[c]}))
        dp = os.path.join(config.OBP_VALIDATION_DIR, "mer_proxy_map_pairs.csv.gz")
        pd.concat(recs, ignore_index=True).to_csv(dp, index=False,
                                                  float_format="%.6g")
        print(f"saved -> {dp}")

    # ---- report ----------------------------------------------------------
    pd.set_option("display.width", 200)
    print(f"\npitches scored: {len(sps)}   pitchers: {len(set(users))}\n")
    print("=" * 92)
    print("ORACLE (GT MER) vs DEPLOYABLE (detected release - k), per metric")
    print("=" * 92)
    hdr = (f"{'metric':<30}{'variant':<14}{'strong':>7}{'moder':>7}"
           f"{'best CCC':>10}{'best view':>11}")
    print(hdr); print("-" * len(hdr))
    for c in MER_COLS:
        for v in VARIANTS:
            s = out[(out.metric == c) & (out.variant == v)]
            if not len(s):
                continue
            st = int((s.grade == "strong").sum())
            mo = int((s.grade == "moderate").sum())
            b = s.loc[s.ccc.idxmax()] if s.ccc.notna().any() else None
            bv = f"{int(b.az)}/{int(b.el)}" if b is not None else "-"
            bc = f"{b.ccc:.3f}" if b is not None else "-"
            print(f"{c if v == VARIANTS[0] else '':<30}{v:<14}{st:>7}{mo:>7}"
                  f"{bc:>10}{bv:>11}")
        print()

    print("=" * 92)
    print("CELL SURVIVAL: oracle strong cells that stay strong on the proxy")
    print("=" * 92)
    for c in MER_COLS:
        orc = out[(out.metric == c) & (out.variant == "oracle")
                  & (out.grade == "strong")]
        cells = set(zip(orc.az, orc.el))
        line = f"  {c:<32} oracle {len(cells):>3}"
        for v in VARIANTS[1:]:
            s = out[(out.metric == c) & (out.variant == v) & (out.grade == "strong")]
            keep = cells & set(zip(s.az, s.el))
            line += f"   {v} {len(keep):>3}"
        print(line)

    print("\nMedian CCC drop at the ORACLE strong cells (oracle -> variant):")
    for c in MER_COLS:
        orc = out[(out.metric == c) & (out.variant == "oracle")
                  & (out.grade == "strong")].set_index(["az", "el"]).ccc
        if not len(orc):
            continue
        line = f"  {c:<32}"
        for v in VARIANTS[1:]:
            s = out[(out.metric == c) & (out.variant == v)
                    ].set_index(["az", "el"]).ccc.reindex(orc.index)
            line += f"  {v} {(s - orc).median():+.3f}"
        print(line)


if __name__ == "__main__":
    main()
