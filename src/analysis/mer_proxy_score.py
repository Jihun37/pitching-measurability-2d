"""Can the three MER-anchored metrics run on a PROXY event instead of GT MER?

2D cannot see maximum external rotation, so `Elbow Flex @MER`, `Torso Lat Tilt
@MER` and `Glove Sh Abd @MER` are GT-only today. But MER sits at a nearly fixed
lag before ball release (scratch/mer_timing_probe.py: rel - mer = median 11 f,
SD 1.1 f, n=401), so a release detection plus a constant might place it well
enough. "Well enough" is a per-metric question -- knee extension velocity lost
r2 0.13 to a 2-frame error -- so this scores each metric at every offset.

Three layers, deliberately separated:
  GT-MER      the ceiling (what the adoption probe measured)
  rel_gt - j  the proxy assumption alone, release error excluded
  rel_det - j the full deployment chain: release detected IN THE ANCHOR VIEW,
              which for these three is an ELEVATED view where our release
              detector has never been validated

Truth = the OBP column, RAW pooled clean projection, n as reported. LOCO CCC at
the best offset is reported so the result can be read against the same gate the
metrics were adopted under (CCC >= 0.80, tier1_adoption_probe.py).

Run:  conda activate diamond; cd src\\analysis; python mer_proxy_score.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config, metrics as M, obp_project as O
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from obp_gt_events import load_gt_events
from tier1_adoption_probe import observe_at, fit_apply, score

# metric -> (observable, truth column, adopted anchor, release strategy at that view)
# strategy follows the deployment routing rule (deploy/release_offset): frontal
# inside the az 60-120 wedge at el <= 60, side everywhere else.
CANDS = {
    "Elbow Flex @MER":     ("elbow_flex", "elbow_flexion_mer",            (330, 60), "side"),
    "Torso Lat Tilt @MER": ("trunk_lean", "torso_lateral_tilt_mer",       (90, 30),  "frontal"),
    "Glove Sh Abd @MER":   ("abd_glove",  "glove_shoulder_abduction_mer", (75, 15),  "frontal"),
}
# BOTH strategies are scored at every anchor: event_error_sweep --gt showed the
# deployment routing rule (frontal iff az 60-120 and el<=60) sends az 90-120 at
# el 30-60 to frontal, where the release IQR is ~200 ms while side is 3 ms.
# The strategy column above is what the rule picks TODAY; "best" is what the
# data says it should pick.
STRATS = ("side", "frontal")
OFFSETS = list(range(0, 21))          # frames BEFORE the release frame
MODELS = ("ratio", "offset", "linear")


def ccc_r2(e, t):
    e, t = np.asarray(e, float), np.asarray(t, float)
    m = np.isfinite(e) & np.isfinite(t)
    e, t = e[m], t[m]
    if len(e) < 5 or e.std() < 1e-9:
        return np.nan, np.nan
    r = np.corrcoef(e, t)[0, 1]
    cov = ((e - e.mean()) * (t - t.mean())).mean()
    ccc = 2 * cov / (e.var() + t.var() + (e.mean() - t.mean()) ** 2)
    return float(r * r), float(ccc)


def loco_best(e, t, u):
    e, t, u = np.asarray(e, float), np.asarray(t, float), np.asarray(u)
    m = np.isfinite(e) & np.isfinite(t)
    e, t, u = e[m], t[m], u[m]
    out = {}
    for mo in MODELS:
        pred = np.full(len(e), np.nan)
        for uu in np.unique(u):
            te = u == uu; tr = ~te
            if tr.sum() < 5:
                continue
            pred[te] = fit_apply(mo, e[tr], t[tr], e[te])
        k = np.isfinite(pred)
        out[mo] = score(pred[k], t[k], u[k])
    mo = max(out, key=lambda x: out[x]["ccc"])
    return mo, out[mo]


def main():
    gt = load_gt_events()
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    root = os.path.join(config.OBP_DATA_DIR, "c3d")

    # per metric: truth, user, value at GT MER, and value at rel_gt-j / rel_det-j
    D = {c: dict(t=[], u=[], mer=[], gtj={j: [] for j in OFFSETS},
                 dtj={st: {j: [] for j in OFFSETS} for st in STRATS},
                 drel={st: [] for st in STRATS}) for c in CANDS}
    done = 0
    for r in md.itertuples(index=False):
        sp = r.session_pitch
        g = gt.get(sp)
        if sp not in poi.index or not g or not {"mer", "rel"} <= set(g):
            continue
        path = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            continue
        try:
            joints, fps = load_feet(path)
        except Exception:
            continue
        arm = O.detect_throwing_arm(joints, fps)
        mer_gt, rel_gt = int(g["mer"]), int(g["rel"])
        for c, (okey, col, (az, el), strat) in CANDS.items():
            if col not in poi.columns:
                continue
            try:
                df = project_cam(joints, az, el)
            except Exception:
                continue
            n = len(df)
            rel_d = {}
            for st in STRATS:
                try:
                    rel_d[st] = int(M.release_frame(df, arm, fps, M.JOINTS, view=st))
                except Exception:
                    rel_d[st] = -1

            def at(f):
                if not (0 <= f < n):
                    return np.nan
                try:
                    return float(observe_at(df, okey, f, fps))
                except Exception:
                    return np.nan

            D[c]["t"].append(poi.loc[sp, col]); D[c]["u"].append(int(r.user))
            D[c]["mer"].append(at(mer_gt))
            for st in STRATS:
                D[c]["drel"][st].append(rel_d[st] - rel_gt)
                for j in OFFSETS:
                    D[c]["dtj"][st][j].append(
                        at(rel_d[st] - j) if rel_d[st] >= 0 else np.nan)
            for j in OFFSETS:
                D[c]["gtj"][j].append(at(rel_gt - j))
        done += 1
        if done % 100 == 0:
            print(f"  ...{done}")
    print(f"processed {done} pitches\n")

    rows = []
    for c, (okey, col, (az, el), rule_strat) in CANDS.items():
        t = np.asarray(D[c]["t"], float); u = np.asarray(D[c]["u"])
        r2m, cccm = ccc_r2(D[c]["mer"], t)
        print("=" * 78)
        print(f"{c}   anchor az{az}/el{el}   (rule picks: {rule_strat})")
        print(f"  GT MER ceiling            r2 {r2m:.3f}   raw CCC {cccm:.3f}")

        # the proxy assumption alone
        bj, bv = max(((j, ccc_r2(D[c]["gtj"][j], t)[0]) for j in OFFSETS),
                     key=lambda p: (p[1] if np.isfinite(p[1]) else -1))
        mo_g, sg = loco_best(D[c]["gtj"][bj], t, u)
        print(f"  proxy off GT release      best -{bj:>2}f  r2 {bv:.3f}  "
              f"LOCO CCC {sg['ccc']:.3f} ({mo_g})")

        row = dict(metric=c, az=az, el=el, rule_strategy=rule_strat,
                   r2_gtmer=r2m, ccc_gtmer=cccm,
                   best_off_gt=-bj, r2_gt=bv, loco_ccc_gt=sg["ccc"])
        for st in STRATS:
            dr = np.asarray(D[c]["drel"][st], float)
            dr = dr[np.isfinite(dr) & (np.abs(dr) < 200)]
            q = np.percentile(dr, [25, 75]) if len(dr) else (np.nan, np.nan)
            dj, dv = max(((j, ccc_r2(D[c]["dtj"][st][j], t)[0]) for j in OFFSETS),
                         key=lambda p: (p[1] if np.isfinite(p[1]) else -1))
            mo_d, sd_ = loco_best(D[c]["dtj"][st][dj], t, u)
            tag = "  <-- rule" if st == rule_strat else ""
            print(f"  detected release [{st:>7}]  err med {np.median(dr):+5.1f} f "
                  f"IQR {q[1]-q[0]:>5.1f} f | best -{dj:>2}f  r2 {dv:.3f}  "
                  f"LOCO CCC {sd_['ccc']:.3f}{tag}")
            row[f"err_med_{st}"] = float(np.median(dr))
            row[f"err_iqr_{st}"] = float(q[1] - q[0])
            row[f"best_off_{st}"] = -dj
            row[f"r2_{st}"] = dv
            row[f"loco_ccc_{st}"] = sd_["ccc"]
        print()
        rows.append(row)

    out = os.path.join(config.OBP_VALIDATION_DIR, "mer_proxy_score.csv")
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print(f"saved -> {out}")
    print("\nAdoption gate for reference: LOCO CCC >= 0.80 (tier1_adoption_probe).")


if __name__ == "__main__":
    main()
