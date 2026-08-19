"""A1b -- NESTED correction-model selection (2026-08-07, audit item ST1).

WHY. `gate_map.score_cell` fits offset/ratio/linear leave-one-PITCHER-out, then picks the
winner by the argmax of the out-of-fold CCC taken over ALL pitchers -- the held-out
pitcher's own out-of-fold prediction included. Parameter fitting is pitcher-blind; MODEL
SELECTION IS NOT. `review_sensitivity.py` (A1) bounded the effect at 96 graded cells with
one row crossing the retained line, which fired the pre-registered escalation rule.

WHAT THIS DOES. Fully nested, per cell:

    outer loop   hold out pitcher p entirely
    inner loop   leave-one-pitcher-out WITHIN the remaining pitchers, score each of the
                 three models by CCC on those inner out-of-fold predictions, take the
                 argmax  ->  m*(p)
    apply        fit m*(p) on ALL pitchers except p, predict p

p never influences the choice of m*(p). The cell's CCC is then taken over the assembled
predictions exactly as `score_cell` does.

WHY IT IS AFFORDABLE. All three models are closed form, so the inner leave-two-out fits
are rank-1 downdates of the same per-pitcher sufficient statistics
(n, sum e, sum t, sum e^2, sum e t). Everything is vectorised over the G x G pitcher grid.

FIDELITY. `ccc`, `stats`, `grade_of` and `loco_predictions` are IMPORTED from gate_map
rather than reimplemented, and the population, the pair dumps, the unwrapping of circular
metrics and the cell-inclusion gate are taken from the same sources gate_map uses. A
reimplementation that differed in an edge case would make the comparison meaningless.

⚠ THIS SCRIPT ADOPTS NOTHING. It writes new files and leaves `gate_map.csv` untouched.
Whether the nested map replaces the published one is the author's decision.

Output: review_nested_cells.csv  (per cell)
        review_nested_summary.csv
Run:  conda activate diamond; cd src\\analysis; python review_nested_selection.py
"""
import os
import sys

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
MODELS = ("offset", "ratio", "linear")


def nested_prediction(e, t, g):
    """Nested per-cell prediction. g = integer pitcher codes, 0..G-1, dense.
    Returns (pred, chosen_model_per_fold) or (None, None) if the cell is degenerate."""
    G = int(g.max()) + 1
    ng = np.bincount(g, minlength=G).astype(float)
    se = np.bincount(g, weights=e, minlength=G)
    st = np.bincount(g, weights=t, minlength=G)
    see = np.bincount(g, weights=e * e, minlength=G)
    set_ = np.bincount(g, weights=e * t, minlength=G)
    N, Se, St, See, Set = float(len(e)), se.sum(), st.sum(), see.sum(), set_.sum()

    # leave-TWO-out aggregates: exclude outer pitcher p (rows) and inner pitcher q (cols)
    with np.errstate(divide="ignore", invalid="ignore"):
        on = N - ng[:, None] - ng[None, :]
        oe = Se - se[:, None] - se[None, :]
        ot = St - st[:, None] - st[None, :]
        oee = See - see[:, None] - see[None, :]
        oet = Set - set_[:, None] - set_[None, :]

        mean_e, mean_t = oe / on, ot / on
        b_off = mean_t - mean_e
        k_rat = np.where(np.abs(mean_e) > 1e-12, ot / oe, np.nan)
        den = oee - oe * oe / on
        slope = np.where(np.abs(den) > 1e-12, (oet - oe * ot / on) / den, np.nan)
        icept = mean_t - slope * mean_e

    bad = on < 5                                   # same guard as gate_map
    for arr in (b_off, k_rat, slope, icept):
        arr[bad] = np.nan
    np.fill_diagonal(b_off, np.nan); np.fill_diagonal(k_rat, np.nan)
    np.fill_diagonal(slope, np.nan); np.fill_diagonal(icept, np.nan)

    # inner out-of-fold predictions: entry [i, p] = prediction for pitch i when pitcher p
    # is the OUTER hold-out and pitch i's own pitcher is the inner hold-out
    q = g                                          # each pitch's own pitcher
    inner = {"offset": e[:, None] + b_off[:, q].T,
             "ratio": k_rat[:, q].T * e[:, None],
             "linear": slope[:, q].T * e[:, None] + icept[:, q].T}

    valid = (g[:, None] != np.arange(G)[None, :])  # pitch i is in outer-training for p
    tt = np.where(valid, t[:, None], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        mt = np.nanmean(tt, axis=0)
        vt = np.nanvar(tt, axis=0)
        inner_ccc = np.full((len(MODELS), G), -9.0)
        for mi, mo in enumerate(MODELS):
            pp = np.where(valid, inner[mo], np.nan)
            mp = np.nanmean(pp, axis=0)
            vp = np.nanvar(pp, axis=0)
            cov = np.nanmean((pp - mp) * (tt - mt), axis=0)
            dd = vp + vt + (mp - mt) ** 2
            c = np.where(dd > 0, 2 * cov / dd, np.nan)
            inner_ccc[mi] = np.where(np.isfinite(c), c, -9.0)

    pick = np.argmax(inner_ccc, axis=0)            # m*(p) for each outer pitcher
    if not np.isfinite(inner_ccc.max(axis=0)).all():
        pass                                       # -9 sentinel already handles it

    # outer application: fit m*(p) on everything except p -- the ordinary LOPO params
    outer = GM.loco_predictions(e, t, g)
    pred = np.empty(len(e))
    for mi, mo in enumerate(MODELS):
        sel = pick[g] == mi
        pred[sel] = outer[mo][sel]
    return pred, pick


def main():
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    user_of = dict(zip(md.session_pitch, md.user))
    drop = outlier_pitches()
    keep = {sp for sp in md.session_pitch if sp not in drop}

    parts = []
    for path, src in ((GM.ADOPTED_PAIRS, "adopted"), (GM.SCREEN_PAIRS, "screened")):
        p = GM.load_pairs(path, src, keep)
        if p is not None:
            parts.append(p)
    common = set.intersection(*[set(p.session_pitch) for p in parts])
    parts = [p[p.session_pitch.isin(common)] for p in parts]
    d = pd.concat(parts, ignore_index=True)
    d["user"] = d.session_pitch.map(user_of)
    d = d[d.user.notna()]
    print(f"pairs {len(d):,}  metrics {d.metric.nunique()}  pitches "
          f"{d.session_pitch.nunique()}")

    reg = pd.read_csv(os.path.join(VALID, "paper_registry.csv"))
    rows = set(reg.metric_id)
    d = d[d.metric.isin(rows)]
    assert d.metric.nunique() == 42, f"{d.metric.nunique()} metrics, expected 42"

    out = []
    for (metric, src), gm in d.groupby(["metric", "source"], sort=True):
        codes_all = gm.user.astype("category").cat.codes.to_numpy()
        e_all = gm.est.to_numpy(float); t_all = gm.truth.to_numpy(float)
        az_all = gm.az.to_numpy(int); el_all = gm.el.to_numpy(int)
        circ = metric.strip() in CIRCULAR
        for az in np.unique(az_all):
            ma = az_all == az
            for el in np.unique(el_all[ma]):
                m = ma & (el_all == el)
                e_c = unwrap_circular(e_all[m]) if circ else e_all[m]
                t_c, g_c = t_all[m], codes_all[m]
                ok = np.isfinite(e_c) & np.isfinite(t_c)
                e_c, t_c, g_c = e_c[ok], t_c[ok], g_c[ok]
                if len(e_c) < 30 or e_c.std() < 1e-9 or t_c.std() < 1e-9:
                    continue
                _, g_c = np.unique(g_c, return_inverse=True)
                pred, pick = nested_prediction(e_c, t_c, g_c)
                s = GM.stats(pred, t_c, g_c)
                cnt = np.bincount(pick, minlength=3)
                out.append(dict(metric=metric, az=int(az), el=int(el),
                                n=len(e_c), n_pitcher=int(g_c.max()) + 1,
                                nested_ccc=s["ccc"], nested_mae=s["mae"],
                                nested_pbsd=s["pbsd"],
                                nested_grade=GM.grade_of(s["ccc"], STRONG, MODERATE),
                                folds_offset=int(cnt[0]), folds_ratio=int(cnt[1]),
                                folds_linear=int(cnt[2])))
        print(f"  {metric:<52} done", flush=True)

    nz = pd.DataFrame(out)
    nz.to_csv(os.path.join(VALID, "review_nested_cells.csv"), index=False)

    g0 = pd.read_csv(os.path.join(VALID, "gate_map.csv"))
    g0 = g0[g0.metric.isin(rows)]
    j = g0.merge(nz, on=["metric", "az", "el"], how="left", suffixes=("", "_n"))
    assert j.nested_ccc.notna().all(), "a published cell has no nested counterpart"

    def counts(c):
        gr = c >= MODERATE
        return int(gr.sum()), int((c >= STRONG).sum()), \
            int(((c >= MODERATE) & (c < STRONG)).sum()), \
            j.loc[gr, "metric"].nunique()

    pub = counts(j.ccc)
    nes = counts(j.nested_ccc)
    lin = counts(j.ccc_linear)
    print("\n" + "=" * 74)
    print(f"{'protocol':<28}{'graded':>8}{'strong':>8}{'moder':>8}{'rows':>7}")
    for lab, v in (("published (selected)", pub), ("NESTED selection", nes),
                   ("prespecified linear", lin)):
        print(f"{lab:<28}{v[0]:>8}{v[1]:>8}{v[2]:>8}{v[3]:>7}")
    print("=" * 74)
    print(f"nested - published : {nes[0] - pub[0]:+d} graded cells "
          f"({100.0 * (nes[0] - pub[0]) / pub[0]:+.1f}%), "
          f"{nes[3] - pub[3]:+d} rows")
    d_ccc = j.nested_ccc - j.ccc
    print(f"per-cell dCCC      : median {d_ccc.median():+.5f}  "
          f"mean {d_ccc.mean():+.5f}  p5 {d_ccc.quantile(.05):+.4f}  "
          f"p95 {d_ccc.quantile(.95):+.4f}")
    pub_rows = set(j.loc[j.ccc >= MODERATE, "metric"])
    nes_rows = set(j.loc[j.nested_ccc >= MODERATE, "metric"])
    print(f"rows lost by nesting : {sorted(pub_rows - nes_rows)}")
    print(f"rows gained          : {sorted(nes_rows - pub_rows)}")

    pd.DataFrame([dict(protocol=l, graded=v[0], strong=v[1], moderate=v[2], rows=v[3])
                  for l, v in (("published_selected", pub), ("nested", nes),
                               ("prespecified_linear", lin))]
                 ).to_csv(os.path.join(VALID, "review_nested_summary.csv"), index=False)
    print("\nwrote review_nested_cells.csv / review_nested_summary.csv")
    print("gate_map.csv NOT modified -- adoption is the author's decision")


if __name__ == "__main__":
    main()
