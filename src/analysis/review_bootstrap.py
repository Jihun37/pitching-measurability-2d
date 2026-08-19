"""A4 -- pitcher-level cluster bootstrap of the graded map (audit item ST2).

WHY. Every cell is graded on a CCC POINT ESTIMATE, and a row is retained if any one of its
168 cells clears the contour. Cells and rows near a contour can therefore change grade
under sampling variation, and the headline counts (30 retained, 12 non-retained, 1,151
graded) carry no uncertainty at all as published.

The right resampling unit is the PITCHER, not the pitch: pitches within a pitcher are not
independent, and the calibration is fitted across pitchers. So resample pitchers with
replacement, keep every pitch of a drawn pitcher together, and treat each draw as its own
cluster -- the standard cluster bootstrap. Then refit the leave-one-pitcher-out
calibration, recompute CCC and re-grade, exactly as `gate_map.score_cell` does.

NO p-VALUES AND NO MULTIPLICITY CORRECTION. This is a selection-uncertainty question, not
a hypothesis test; the audit explicitly asked for stability rather than Bonferroni.

Reports  P(graded) and P(strong) per cell, and the bootstrap distribution of the headline
counts. The 12 non-retained rows peak at CCC 0.6802, well below the 0.75 contour, so this
is expected to STRENGTHEN the published result rather than qualify it.

⚠ ADOPTS NOTHING. Writes new files; `gate_map.csv` is untouched. The published map remains
the point estimate -- this quantifies its stability.

Output: review_bootstrap_cells.csv       per cell: p_graded, p_strong, ccc percentiles
        review_bootstrap_replicates.csv  per replicate: headline counts
Run:  conda activate diamond; cd src\\analysis
      python review_bootstrap.py --B 200
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("", "..", "../stage2"):
    sys.path.insert(0, os.path.join(HERE, p) if p else HERE)
import config
import gate_map as GM
from angle_map_2d import CIRCULAR, unwrap_circular
from gt_landmark_outlier_effect import outlier_pitches

VALID = config.OBP_VALIDATION_DIR
STRONG, MODERATE = 0.80, 0.75


def load_cells():
    """Build the per-cell (e, t, pitcher-code) arrays once, exactly as gate_map does."""
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    user_of = dict(zip(md.session_pitch, md.user))
    drop = outlier_pitches()
    keep = {sp for sp in md.session_pitch if sp not in drop}

    parts = [p for p in (GM.load_pairs(GM.ADOPTED_PAIRS, "adopted", keep),
                         GM.load_pairs(GM.SCREEN_PAIRS, "screened", keep))
             if p is not None]
    common = set.intersection(*[set(p.session_pitch) for p in parts])
    d = pd.concat([p[p.session_pitch.isin(common)] for p in parts], ignore_index=True)
    d["user"] = d.session_pitch.map(user_of)
    d = d[d.user.notna()]

    reg = pd.read_csv(os.path.join(VALID, "paper_registry.csv"))
    d = d[d.metric.isin(set(reg.metric_id))]
    assert d.metric.nunique() == 42, f"{d.metric.nunique()} metrics, expected 42"

    cells = []
    for metric, gm in d.groupby("metric", sort=True):
        circ = metric.strip() in CIRCULAR
        e_all = gm.est.to_numpy(float); t_all = gm.truth.to_numpy(float)
        az_all = gm.az.to_numpy(int); el_all = gm.el.to_numpy(int)
        code_all = gm.user.astype("category").cat.codes.to_numpy()
        for az in np.unique(az_all):
            ma = az_all == az
            for el in np.unique(el_all[ma]):
                m = ma & (el_all == el)
                e = unwrap_circular(e_all[m]) if circ else e_all[m]
                t, g = t_all[m], code_all[m]
                ok = np.isfinite(e) & np.isfinite(t)
                cells.append((metric, int(az), int(el), e[ok], t[ok], g[ok]))
    return cells


def score(e, t, g):
    """CCC of the best of the three LOPO models -- gate_map's own selection rule."""
    if len(e) < 30 or e.std() < 1e-9 or t.std() < 1e-9:
        return np.nan
    _, g = np.unique(g, return_inverse=True)
    preds = GM.loco_predictions(e, t, g)
    best = -9.0
    for p in preds.values():
        m = np.isfinite(p) & np.isfinite(t)
        if m.sum() < 10:
            continue
        c = GM.ccc(p[m], t[m])
        if np.isfinite(c) and c > best:
            best = c
    return best if best > -9 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=200, help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dup-mode", choices=("same", "distinct"), default="same",
                    help="how a pitcher drawn twice is labelled for the leave-one-"
                         "pitcher-out fit. 'same' (default, CORRECT): both copies share "
                         "the pitcher's identity, so holding that pitcher out removes "
                         "BOTH. 'distinct': each draw is its own cluster -- this is the "
                         "naive cluster bootstrap and it LEAKS, because the held-out "
                         "copy's twin stays in the training set and the correction is "
                         "then fitted on the held-out pitcher's own data. Provided only "
                         "to quantify that bias.")
    a = ap.parse_args()
    print(f"dup-mode = {a.dup_mode}"
          + ("" if a.dup_mode == "same" else "   ** LEAKY, diagnostic only **"))

    t0 = time.time()
    cells = load_cells()
    print(f"{len(cells)} cells loaded in {time.time() - t0:.1f}s", flush=True)

    # per-cell pitcher index, built once
    prepped = []
    for metric, az, el, e, t, g in cells:
        gu, gi = np.unique(g, return_inverse=True)
        idx = [np.where(gi == k)[0] for k in range(len(gu))]
        prepped.append((metric, az, el, e, t, idx))
    G = max(len(p[5]) for p in prepped)
    print(f"up to {G} pitchers per cell", flush=True)

    rng = np.random.default_rng(a.seed)
    n_cells = len(prepped)
    hit_graded = np.zeros(n_cells, int)
    hit_strong = np.zeros(n_cells, int)
    valid = np.zeros(n_cells, int)
    ccc_draws = np.full((n_cells, a.B), np.nan)
    reps = []

    t0 = time.time()
    for b in range(a.B):
        out = np.full(n_cells, np.nan)
        for ci, (metric, az, el, e, t, idx) in enumerate(prepped):
            k = len(idx)
            draw = rng.integers(0, k, k)                 # resample pitchers, with repl.
            take = np.concatenate([idx[j] for j in draw])
            if a.dup_mode == "same":
                # duplicate draws keep the PITCHER's identity, so leave-one-pitcher-out
                # removes every copy of that pitcher and stays pitcher-blind
                grp = np.concatenate([np.full(len(idx[j]), j) for j in draw])
            else:
                grp = np.concatenate([np.full(len(idx[j]), r)
                                      for r, j in enumerate(draw)])
            out[ci] = score(e[take], t[take], grp)
        ccc_draws[:, b] = out
        fin = np.isfinite(out)
        valid += fin
        hit_graded += (fin & (out >= MODERATE))
        hit_strong += (fin & (out >= STRONG))

        mets = np.array([p[0] for p in prepped])
        gmask = fin & (out >= MODERATE)
        reps.append(dict(rep=b, graded=int(gmask.sum()),
                         strong=int((fin & (out >= STRONG)).sum()),
                         moderate=int((fin & (out >= MODERATE) & (out < STRONG)).sum()),
                         retained_rows=int(len(np.unique(mets[gmask])))))
        if (b + 1) % 10 == 0 or b == 0:
            el_s = time.time() - t0
            print(f"  replicate {b + 1}/{a.B}  {el_s:.0f}s elapsed, "
                  f"~{el_s / (b + 1) * (a.B - b - 1):.0f}s left  "
                  f"[graded {reps[-1]['graded']}, rows {reps[-1]['retained_rows']}]",
                  flush=True)

    R = pd.DataFrame(reps)
    R.to_csv(os.path.join(VALID, "review_bootstrap_replicates.csv"), index=False)

    pub = pd.read_csv(os.path.join(VALID, "gate_map.csv"))
    key = pd.DataFrame([(p[0], p[1], p[2]) for p in prepped],
                       columns=["metric", "az", "el"])
    key["p_graded"] = hit_graded / np.maximum(valid, 1)
    key["p_strong"] = hit_strong / np.maximum(valid, 1)
    key["ccc_p025"] = np.nanpercentile(ccc_draws, 2.5, axis=1)
    key["ccc_p500"] = np.nanpercentile(ccc_draws, 50.0, axis=1)
    key["ccc_p975"] = np.nanpercentile(ccc_draws, 97.5, axis=1)
    key = key.merge(pub[["metric", "az", "el", "ccc", "grade"]],
                    on=["metric", "az", "el"], how="left")
    key.to_csv(os.path.join(VALID, "review_bootstrap_cells.csv"), index=False)

    print("\n" + "=" * 74)
    print(f"pitcher cluster bootstrap, B = {a.B}")
    for c in ("graded", "strong", "moderate", "retained_rows"):
        print(f"  {c:<15} median {R[c].median():8.1f}   "
              f"95% CI [{R[c].quantile(.025):.0f}, {R[c].quantile(.975):.0f}]   "
              f"published {{1151 819 332 30}}")
        break
    print(f"  graded        median {R.graded.median():7.0f}  "
          f"CI [{R.graded.quantile(.025):.0f}, {R.graded.quantile(.975):.0f}]   published 1151")
    print(f"  strong        median {R.strong.median():7.0f}  "
          f"CI [{R.strong.quantile(.025):.0f}, {R.strong.quantile(.975):.0f}]   published 819")
    print(f"  moderate      median {R.moderate.median():7.0f}  "
          f"CI [{R.moderate.quantile(.025):.0f}, {R.moderate.quantile(.975):.0f}]   published  400")
    print(f"  retained rows median {R.retained_rows.median():7.0f}  "
          f"CI [{R.retained_rows.quantile(.025):.0f}, "
          f"{R.retained_rows.quantile(.975):.0f}]   published   35")

    g = key[key.grade.isin(["strong", "moderate"])]
    ng = key[~key.grade.isin(["strong", "moderate"])]
    print(f"\n  published-graded cells   : P(graded) median {g.p_graded.median():.3f}, "
          f"{(g.p_graded >= 0.95).sum()} of {len(g)} at >= 0.95, "
          f"{(g.p_graded < 0.50).sum()} below 0.50")
    print(f"  published-limited cells  : P(graded) median {ng.p_graded.median():.3f}, "
          f"{(ng.p_graded >= 0.50).sum()} of {len(ng)} at >= 0.50")
    reg = pd.read_csv(os.path.join(VALID, "paper_registry.csv"))
    nonret = set(reg.loc[~reg.retained.astype(bool), "metric_id"])
    nr = key[key.metric.isin(nonret)]
    print(f"  the 12 non-retained rows : {len(nr)} cells, "
          f"max P(graded) {nr.p_graded.max():.3f}, "
          f"cells ever graded {(nr.p_graded > 0).sum()}")
    print("=" * 74)
    print("wrote review_bootstrap_cells.csv / review_bootstrap_replicates.csv")
    print("gate_map.csv NOT modified")


if __name__ == "__main__":
    main()
