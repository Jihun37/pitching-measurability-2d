"""
Diamond - FROZEN viewpoint-classifier candidate evaluation harness.

Decision 2026-07-12 (user): cal30-idw is FROZEN as an unwired candidate.
This harness compares the CURRENT WIRING vs the frozen candidate on any
labeled clip set, so the next labeled set decides wiring with one command.
No thresholds, constants, or features may be tuned here - the candidate
is exactly the design validated in tests/viewpoint_fold30_probe.py.

Frozen candidate "cal30-idw":
  static6 features - dev-15 global offset (artifact, see --freeze-offset)
  -> kNN k=25 -> NATIVE 30-deg folded-class vote {0,30,60,90} with
  inverse-feature-distance weights, fold 15/45/75 neighbors split 0.5/0.5
  -> unfold with the frozen sign bits (deploy.viewpoint_fold).

Feature extension point (structure only - NO new features before the
fresh-set decision): register extractors in FEATURE_EXTRACTORS below,
  name -> fn(df, arm, fps) -> {feature: value}
A future temporal / arm-direction extractor plugs in there; the OBP bank
(viewpoint_zone_features.csv) must then be regenerated with the SAME
extractor on the projections (analysis/viewpoint_zone_classify.py) before
any config can reference it.

EVERYTHING release-side is FROZEN for this validation (user 2026-07-12):
the release map (release_method_map.csv), the detectors, the stored
evidence rows, all constants. This harness NEVER writes to the evidence
ledger and NEVER rebuilds the map — a fresh set only judges the azimuth
classifier. For fresh clips absent from the ledger, the per-method frames
are computed IN MEMORY with the frozen detectors (same true-az convention
as the stored ledger rows, via release_method_update.score_clip) and
discarded after scoring. Rebuilding the map with new data is a SEPARATE,
explicitly-decided experiment — never a side effect of this one.

Run (diamond env, from src\tests):
  python viewpoint_candidate_eval.py --freeze-offset   (one-time artifact)
  python viewpoint_candidate_eval.py                   (holdout-60 check)
  python viewpoint_candidate_eval.py --gt new_set.csv  (fresh labeled set:
      columns clip, nominal_az [, true_release]; poses already extracted.
      Without true_release the release columns are skipped, nrel=0.)
"""
import os, sys, argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("..", "../stage2", "../analysis", "../deploy"):
    sys.path.insert(0, os.path.join(_HERE, p))

import config
from station_classify import features, FEATS
from real_station_test import detect_arm_2d
from measure_auto import load_clip
from viewpoint_fold import fold, unfold, sign_bits
from viewpoint_fold_probe import video_fps
from scipy.spatial import cKDTree

K = 25
OUT = os.path.join(config.ROOT, "data", "outputs")
OFFSET_CSV = os.path.join(OUT, "viewpoint_calib_offset.csv")


# ---- feature extractors (extension point - structure only) -------------
def static6(df, arm, fps):
    return features(df, arm)


FEATURE_EXTRACTORS = {
    "static6": static6,
    # reserved (NOT implemented - no feature work before the fresh-set
    # decision, user 2026-07-12): "temporal", "arm_direction"
}


def circ(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def arc30(az):
    """Snap to the release-map 30-deg arc grid (half-up, predefined)."""
    return int((az + 15) // 30 * 30) % 360


def freeze_offset(force=False):
    """One-time artifact: dev-15 global domain-shift offset (raw units).
    Identical computation to tests/viewpoint_calib_probe.py."""
    if os.path.exists(OFFSET_CSV) and not force:
        sys.exit(f"offset artifact already frozen: {OFFSET_CSV} "
                 "(use --force to regenerate)")
    bank = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR,
                                    "viewpoint_zone_features.csv")
                       ).dropna(subset=FEATS)
    shelf0 = bank.assign(azf=bank.az.map(fold))
    shelf0 = shelf0[shelf0.el == 0].groupby("azf")[FEATS].mean()
    gt15 = pd.read_csv(os.path.join(OUT, "release_gt_real15.csv"))
    deltas = []
    for r in gt15.itertuples():
        df, _ = load_clip(r.clip, "_rtmp")
        f = features(df, detect_arm_2d(df))
        azf = min(shelf0.index, key=lambda b: abs(b - fold(int(r.nominal_az))))
        deltas.append([f[k] - shelf0.loc[azf][k] for k in FEATS])
    off = pd.DataFrame({"feature": FEATS, "offset": np.mean(deltas, axis=0),
                        "estimated_from": "release_gt_real15",
                        "frozen": "2026-07-12"})
    off.to_csv(OFFSET_CSV, index=False)
    print(off.to_string(index=False))
    print(f"frozen -> {OFFSET_CSV}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=None,
                    help="labeled set csv (clip, nominal_az"
                         " [, true_release]); default = holdout 60")
    ap.add_argument("--freeze-offset", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.freeze_offset:
        freeze_offset(a.force)
        return
    if not os.path.exists(OFFSET_CSV):
        sys.exit("no frozen offset artifact - run --freeze-offset first")
    off = pd.read_csv(OFFSET_CSV).set_index("feature").offset.reindex(
        FEATS).to_numpy(float)

    bank = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR,
                                    "viewpoint_zone_features.csv")
                       ).dropna(subset=FEATS).reset_index(drop=True)
    X = bank[FEATS].to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    tree = cKDTree((X - mu) / sd)
    naz_b = bank.az.to_numpy(int)
    nel_b = bank.el.to_numpy(int)

    gt_release = {}
    if a.gt:
        gt = pd.read_csv(a.gt if os.path.exists(a.gt)
                         else os.path.join(OUT, a.gt))
        clips = [(r.clip, int(r.nominal_az)) for r in gt.itertuples()]
        if "true_release" in gt.columns:
            gt_release = dict(zip(gt.clip, gt.true_release.astype(int)))
        label = os.path.basename(a.gt)
    else:
        clips = [(f"angle{i:02d}_{p:02d}", (30 * i) % 360)
                 for i in range(12) for p in range(5)]
        label = "holdout 60 (reproduction check: baseline 29, cal30-idw 35)"

    # FROZEN release assets: map + stored evidence are read-only; fresh
    # clips get in-memory frames from the frozen detectors (not persisted)
    mp = pd.read_csv(os.path.join(OUT, "release_method_map.csv")
                     ).set_index("az")
    ev = pd.read_csv(os.path.join(OUT, "release_method_evidence.csv")
                     ).set_index("clip")

    def method_frames(name, az):
        """Per-method frames + GT release, ledger row if stored, else
        frozen detectors in memory (same true-az convention)."""
        if name in ev.index:
            r = ev.loc[name]
            return {m: r[m] for m in
                    ("side", "side_off", "frontal", "deploy", "ballsep")}, \
                int(r["true_release"])
        if name not in gt_release:
            return None, None
        from release_method_update import score_clip
        return score_clip(name, arc30(az)), gt_release[name]

    def wired_baseline(xv, dx, sho):
        _, idx = tree.query((xv - mu) / sd, k=K)
        naz = np.array([unfold(az, dx, sho) for az in naz_b[idx]])
        pairs, cnts = np.unique(list(zip(naz, nel_b[idx])), axis=0,
                                return_counts=True)
        return int(pairs[np.argmax(cnts)][0])

    def cal30_idw(xv, dx, sho):
        d, idx = tree.query((xv - off - mu) / sd, k=K)
        acc = {0: 0.0, 30: 0.0, 60: 0.0, 90: 0.0}
        for f, wi in zip((fold(az) for az in naz_b[idx]), 1.0 / (d + 1e-9)):
            if f % 30 == 0:
                acc[f] += wi
            else:
                acc[f - 15] += wi / 2
                acc[f + 15] += wi / 2
        c = max(sorted(acc), key=lambda k: acc[k])
        return int(unfold(c, dx, sho))

    print(f"=== {label} ===")
    preds = {"wired_baseline": {}, "cal30_idw": {}}
    for name, _az in clips:
        df, raw = load_clip(name, "_rtmp")
        arm = detect_arm_2d(df)
        fps = video_fps(name)
        dx, sho = sign_bits(df, raw, arm, fps)
        xv = np.array([static6(df, arm, fps)[k] for k in FEATS], float)
        preds["wired_baseline"][name] = wired_baseline(xv, dx, sho)
        preds["cal30_idw"][name] = cal30_idw(xv, dx, sho)

    frames_cache = {name: method_frames(name, az) for name, az in clips}

    print(f"{'config':16s}{'class':>7}{'off30':>6}{'>=60':>6}{'med':>6}"
          f"{'mean':>7}{'0/180':>7}{'90/270':>8}{'mapAgr':>8}"
          f"{'rel<=1f':>8}{'rel<=2f':>8}{'nrel':>6}")
    for cfg, pv in preds.items():
        errs, e018, e97, agree, r1, r2, n = [], [], [], 0, 0, 0, 0
        for name, az in clips:
            pa = arc30(pv[name])          # both scored at class granularity
            azc = arc30(az)
            errs.append(circ(pa, azc))
            if azc in (0, 180):
                e018.append(errs[-1])
            if azc in (90, 270):
                e97.append(errs[-1])
            m_pred = mp.loc[pa].method if pa in mp.index else None
            m_gt = mp.loc[azc].method if azc in mp.index else None
            agree += (m_pred == m_gt) and m_pred is not None
            fr_row, tr = frames_cache[name]
            if fr_row is not None and m_pred is not None:
                fr = fr_row[m_pred]
                if pd.isna(fr):
                    fr = fr_row["deploy"]
                if not pd.isna(fr):
                    n += 1
                    r1 += abs(int(fr) - tr) <= 1
                    r2 += abs(int(fr) - tr) <= 2
        errs = np.array(errs)
        nn = len(clips)
        sub = lambda e: (f"{int((np.array(e) == 0).sum())}/{len(e)}"
                         if e else "-")
        print(f"{cfg:16s}{int((errs == 0).sum()):>4}/{nn:<3}"
              f"{int((errs == 30).sum()):>5}"
              f"{int((errs >= 60).sum()):>6}{np.median(errs):>6.0f}"
              f"{errs.mean():>7.1f}{sub(e018):>7}{sub(e97):>8}"
              f"{agree:>5}/{nn:<3}{r1:>7}{r2:>8}{n:>6}")

    dump = pd.DataFrame([{"clip": name, "az_true": az,
                          **{c: pv[name] for c, pv in preds.items()}}
                         for name, az in clips])
    out = os.path.join(OUT, f"viewpoint_candidate_eval_"
                            f"{label.split()[0].replace('.csv', '')}.csv")
    dump.to_csv(out, index=False)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
