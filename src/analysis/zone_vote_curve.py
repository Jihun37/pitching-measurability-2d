"""
Diamond - vote-based zone membership + deferral threshold curve.

The real-video spot check showed the kNN vote SPREAD carries confidence
(overhead clips concentrate 0.56 on one bin; a casual side clip spreads 0.12).
Instead of zone-looking-up only the argmax bin, decide per metric from the
neighbor votes directly:

    p_in(metric) = fraction of the k neighbors whose (az, el) bin lies in
                   the metric's claimed usable zone

and accept the metric when p_in >= t. Sweeping t gives the safety/coverage
trade: false-accept (measured when truly out - garbage output) vs
false-reject (refused when truly in - lost coverage).

Runs on the saved clean feature bank (viewpoint_zone_features.csv), GroupKFold
by session_pitch, clean rows evaluated. No projection rebuild.

Run:  cd src\\analysis
      python zone_vote_curve.py
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import config
from station_classify import FEATS, gkf
from viewpoint_zone_classify import zone_masks, ZONE_METRICS

K = 25
THRESH = [round(t, 2) for t in np.arange(0.2, 1.001, 0.1)]   # fine curve


def main():
    bank = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR,
                                    "viewpoint_zone_features.csv")
                       ).dropna(subset=FEATS).reset_index(drop=True)
    masks = zone_masks()
    X = bank[FEATS].to_numpy(float)
    az = bank["az"].to_numpy(int); el = bank["el"].to_numpy(int)
    groups = bank["session_pitch"].to_numpy()
    clean = (bank["aug"] == 0).to_numpy()

    # per-metric in-zone flag of every bank row's TRUE bin
    in_true = {m: np.array([masks[m][(a, e)] for a, e in zip(az, el)])
               for m in ZONE_METRICS}

    p_in = {m: np.full(len(bank), np.nan) for m in ZONE_METRICS}
    for tr, te in gkf(groups, k=5):
        te = te[clean[te]]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        tree = cKDTree((X[tr] - mu) / sd)
        _, idx = tree.query((X[te] - mu) / sd, k=K, workers=-1)
        for m in ZONE_METRICS:
            flags = in_true[m][tr][idx]            # (n_te, K) bool
            p_in[m][te] = flags.mean(axis=1)

    ev = clean & np.isfinite(p_in[ZONE_METRICS[0]])
    print(f"[vote-based zone membership]  k={K}  n_eval={ev.sum()}\n")
    print(f"{'metric':24s}{'t':>6s}{'f-accept':>10s}{'f-reject':>10s}"
          f"{'coverage':>10s}")
    print("-" * 60)
    out = []
    for m in ZONE_METRICS:
        t_in = in_true[m][ev]
        p = p_in[m][ev]
        for t in THRESH:
            acc = p >= t
            fa = float(np.mean(acc[~t_in])) if (~t_in).any() else np.nan
            fr = float(np.mean(~acc[t_in])) if t_in.any() else np.nan
            cov = float(np.mean(acc[t_in])) if t_in.any() else np.nan
            print(f"{m:24s}{t:>6.1f}{fa:>10.3f}{fr:>10.3f}{cov:>10.3f}")
            out.append({"metric": m, "t": t, "f_accept": fa,
                        "f_reject": fr, "coverage": cov})
    outp = os.path.join(config.OBP_VALIDATION_DIR, "zone_vote_curve.csv")
    pd.DataFrame(out).to_csv(outp, index=False)
    print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
