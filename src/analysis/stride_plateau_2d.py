"""
Diamond - Stride plateau, part 2: does the 3D verdict survive projection?

stride_plateau_candidates.py showed that in 3D the settling instant does not
matter: 27 criteria plus two references land within dr2 <= 0.007, while the
instants they pick differ by up to 16 frames. If that flatness survives the
camera, the plateau is not an event at all -- it is a window read.

The reason the question is open is the 2D record: reading the SAME quantity at
the GT foot plant scored r2 0.391 in 2D but 0.956 in 3D. Something the camera
adds, not the timing, produced that. So the test is run where the metric lives:
the 30 cells `Stride (anchor)` holds on the graded map.

Variants (all read the SAME estimator, only the frame differs):
  fp_single    the GT foot-plant frame, one frame, no aggregation
  settled80    the adopted metrics.stride_settled_2d (80 ms release-anchored median)
  C1_pos       position stability, eps = 0.005 m expressed in statures, P = 20
  C2_vel       velocity stability, tau = 0.05, P = 10
  C3_mix       both, eps = 0.005 m, tau = 0.10, P = 20
  plateau_pt   C1's frame read as a SINGLE frame (no median) -- separates
               "which instant" from "averaging over a window", which is the only
               way to tell a timing effect from a noise-aggregation effect

Two more variants isolate the REFERENCE POSITION, because the frame and the origin
were changed together in 2026-07-24 and only the frame got the credit:
  fp_trailankle  |x_lead(fp) - x_trail(fp)|  -- the superseded est_stride_fp: the
                 trail ankle AT foot plant, by which time it has been dragged forward
  fp_anchor_fp   trail_anchor_x searched only up to fp (the superseded
                 est_stride_at_fp) instead of up to release
Everything else is held fixed, so whatever moves is the cause.

Scoring is gate_map.score_cell, so a cell here is directly comparable to the same
cell in gate_map.csv. Population = gt_clean (n=394), events = OBP GT landmarks.

Outputs: stride_plateau_2d.csv
Run:  cd src\\analysis
      python stride_plateau_2d.py --limit 20
      python stride_plateau_2d.py
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
from mer_proxy_map import map_population
from gate_map import score_cell

CELLS = [(0, 0), (0, 15), (0, 30), (0, 75), (0, 85), (15, 0), (15, 15),
         (150, 0), (150, 15), (150, 30), (150, 45), (165, 0), (165, 15),
         (165, 30), (165, 45), (180, 0), (180, 15), (180, 30), (195, 0),
         (195, 15), (330, 0), (330, 15), (330, 30), (330, 75), (330, 85),
         (345, 0), (345, 15), (345, 30), (345, 75), (345, 85)]
VARIANTS = ["fp_single", "settled80", "C1_pos", "C2_vel", "C3_mix", "plateau_pt",
            "fp_trailankle", "fp_anchor_fp"]
EPS_M, TAU, P_POS, P_VEL, P_MIX = 0.005, 0.05, 20, 10, 20
SETTLE_S = 0.08


def hold(ok, t, P):
    """criterion array `ok` satisfied continuously over [t, t+P]."""
    return t + P < len(ok) and bool(np.all(ok[t:t + P + 1]))


def plateau_2d(x, v, vpk, fp, rel, eps_px, kind):
    """First frame in [fp, rel] whose criterion holds for P frames. eps_px is the
    metre threshold converted through the clip's own pixel stature, so the test is
    scale-free."""
    n = len(x)
    for t in range(int(fp), int(rel) + 1):
        P = {"C1_pos": P_POS, "C2_vel": P_VEL, "C3_mix": P_MIX}[kind]
        if t + P >= n:
            break
        ok = True
        if kind in ("C1_pos", "C3_mix"):
            seg = x[t:t + P + 1]
            if not np.all(np.isfinite(seg)) or np.nanmax(np.abs(seg - x[t])) > eps_px:
                ok = False
        if ok and kind in ("C2_vel", "C3_mix"):
            seg = np.abs(v[t:t + P + 1])
            if not np.all(np.isfinite(seg)) or np.nanmax(seg) > TAU * vpk:
                ok = False
        if ok:
            return t
    return None


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
    # the frozen map's 394 ids, not gt_clean alone: angle_zone_sweep also
    # requires pkh and rel > fp+1, which drops one more pitch. Official
    # numbers must be on the same population as gate_map.csv.
    pop = map_population()

    est = {(v, c): [] for v in VARIANTS for c in CELLS}
    frm = {(v, c): [] for v in VARIANTS for c in CELLS}
    tru, users = [], []
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
        lan = "l_an" if lead == "left" else "r_an"
        tan = "r_an" if lead == "left" else "l_an"
        fp, rel = int(g["fp"]), int(g["rel"])
        h = float(r.session_height_m)

        tru.append(float(poi.loc[sp, "stride_length"]))
        users.append(r.user)
        for c in CELLS:
            vals = {}
            frames = {}
            try:
                df = project_cam(joints, c[0], c[1])
                x = df[f"{M.JOINTS[lan]}_x"].to_numpy(float)
                xt = df[f"{M.JOINTS[tan]}_x"].to_numpy(float)
                stat = M.pixel_stature(df, M.JOINTS)
                anc = M.trail_anchor_x(xt, rel, fps)
                v = np.gradient(x) * fps
                vpk = float(np.nanmax(np.abs(v[:rel + 1])))
                eps_px = EPS_M / h * stat          # metres -> statures -> pixels
                rd = lambda f: abs(x[int(f)] - anc) / stat

                vals["fp_single"] = rd(fp); frames["fp_single"] = fp
                # reference-position controls: same frame, different origin
                vals["fp_trailankle"] = abs(x[fp] - xt[fp]) / stat
                frames["fp_trailankle"] = fp
                vals["fp_anchor_fp"] = (abs(x[fp] - M.trail_anchor_x(xt, fp, fps))
                                        / stat)
                frames["fp_anchor_fp"] = fp
                vals["settled80"] = M.stride_settled_2d(df, lead, rel, fps, M.JOINTS)
                frames["settled80"] = rel
                for kind in ("C1_pos", "C2_vel", "C3_mix"):
                    t = plateau_2d(x, v, vpk, fp, rel, eps_px, kind)
                    frames[kind] = t
                    if t is None:
                        vals[kind] = np.nan
                        continue
                    w = max(1, int(round(SETTLE_S * fps)))
                    seg = x[max(0, t - w // 2):t + w // 2 + 1]
                    vals[kind] = abs(float(np.nanmedian(seg)) - anc) / stat
                    if kind == "C1_pos":
                        vals["plateau_pt"] = rd(t); frames["plateau_pt"] = t
            except Exception:
                pass
            for vv in VARIANTS:
                est[(vv, c)].append(vals.get(vv, np.nan))
                frm[(vv, c)].append(frames.get(vv, np.nan))
        done += 1
        if done % 50 == 0:
            print(f"  ...{done} processed")
    print(f"processed {done} / failed {fail}\n")

    t = np.asarray(tru, float)
    codes = pd.Series(users).astype("category").cat.codes.to_numpy()
    rows = []
    for vv in VARIANTS:
        for c in CELLS:
            e = np.asarray(est[(vv, c)], float)
            s = score_cell(e, t, codes, a.floor, a.strong)
            if s is None:
                continue
            f = np.asarray(frm[(vv, c)], float)
            rows.append(dict(variant=vv, az=c[0], el=c[1],
                             frame_med=np.nanmedian(f), found=np.mean(np.isfinite(e)),
                             **s))
    out = pd.DataFrame(rows)
    p = os.path.join(config.OBP_VALIDATION_DIR, "stride_plateau_2d.csv")
    out.to_csv(p, index=False, float_format="%.6g")

    pd.set_option("display.width", 200)
    print("=" * 96)
    print(f"STRIDE PLATEAU IN 2D -- {len(CELLS)} Stride (anchor) cells, "
          f"n={len(t)} pitches, {len(set(users))} pitchers")
    print("=" * 96)
    hdr = (f"{'variant':<12}{'cells':>6}{'strong':>8}{'moder':>7}{'r2 med':>9}"
           f"{'CCC med':>9}{'CCC min':>9}{'CCC max':>9}{'found':>8}")
    print(hdr); print("-" * len(hdr))
    for vv in VARIANTS:
        s = out[out.variant == vv]
        if not len(s):
            print(f"{vv:<12}  (no scorable cell)"); continue
        print(f"{vv:<12}{len(s):>6}{int((s.grade=='strong').sum()):>8}"
              f"{int((s.grade=='moderate').sum()):>7}{s.r2.median():>9.3f}"
              f"{s.ccc.median():>9.3f}{s.ccc.min():>9.3f}{s.ccc.max():>9.3f}"
              f"{s.found.min():>8.3f}")

    print("\nper cell, CCC (rows = cell, columns = variant):")
    pv = out.pivot_table(index=["az", "el"], columns="variant", values="ccc")
    print(pv[VARIANTS].round(3).to_string())

    base = out[out.variant == "settled80"].set_index(["az", "el"]).ccc
    print("\nCCC difference vs the adopted settled80, median over cells:")
    for vv in VARIANTS:
        if vv == "settled80":
            continue
        s = out[out.variant == vv].set_index(["az", "el"]).ccc.reindex(base.index)
        print(f"  {vv:<12} {(s - base).median():+.4f}   "
              f"(min {(s - base).min():+.4f}, max {(s - base).max():+.4f})")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
