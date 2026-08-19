"""Figure: station classifier evaluation (patent 구성 B + F).
(a) 4-class confusion matrix (grouped CV), (b) safety tradeoff: false-accept vs
over-caution vs confidence threshold tau. Loads the cached OBP feature table."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import config
from station_classify import gkf, knn, FEATS, CLASSES, USABLE, build

INK = "#0E1B33"
CACHE = os.path.join(config.OBP_VALIDATION_DIR, "station_train_features.csv")
PRED = os.path.join(config.OBP_VALIDATION_DIR, "station_cv_predictions.csv")


def main():
    if os.path.exists(PRED):
        # preferred: render from the SAME CV run whose numbers go in the text
        # (produced by station_classify.py main)
        d = pd.read_csv(PRED)
        y, pred, share = d["y"].to_numpy(), d["pred"].to_numpy(), d["share"].to_numpy()
        print(f"using saved CV predictions ({len(d)} rows)")
    else:
        pool = (pd.read_csv(CACHE) if os.path.exists(CACHE)
                else build(300, 0.0)).dropna(subset=FEATS).reset_index(drop=True)
        if "aug" not in pool.columns:   # legacy cache without augmentation
            pool["aug"] = 0
        # test folds contain only CLEAN rows; augmented copies train with
        # their own group (same protocol as station_classify.main)
        data = pool[pool.aug == 0].reset_index(drop=True)
        X = data[FEATS].to_numpy(float); y = data["y"].to_numpy()
        groups = data["session_pitch"].to_numpy()
        pred = np.empty(len(data), dtype=y.dtype); share = np.empty(len(data))
        for tr, te in gkf(groups, 5):
            te_groups = set(groups[te].tolist())
            trp = pool[~pool.session_pitch.isin(te_groups)]
            pred[te], share[te] = knn(trp[FEATS].to_numpy(float),
                                      trp["y"].to_numpy(), X[te], 25)

    idx = {c: i for i, c in enumerate(CLASSES)}
    cm = np.zeros((4, 4), int)
    for t, p in zip(y, pred):
        cm[idx[t], idx[p]] += 1
    cmn = cm / cm.sum(1, keepdims=True)

    rej = y == "reject"; us = np.isin(y, list(USABLE))
    taus = np.linspace(0, 0.96, 25)
    fa, oc, acc = [], [], []
    for t in taus:
        adj = np.where((share >= t) | (pred == "reject"), pred, "reject")
        fa.append(np.isin(adj[rej], list(USABLE)).mean())
        oc.append((adj[us] == "reject").mean())
        acc.append((adj == y).mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    im = ax1.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax1.set_xticks(range(4)); ax1.set_yticks(range(4))
    ax1.set_xticklabels(CLASSES, rotation=20, ha="right"); ax1.set_yticklabels(CLASSES)
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, f"{cmn[i,j]:.2f}\n({cm[i,j]})", ha="center", va="center",
                     fontsize=9, color="white" if cmn[i, j] > 0.5 else INK)
    ax1.set_xlabel("predicted"); ax1.set_ylabel("true")
    ax1.set_title("(a) Station confusion (grouped CV)", fontsize=12, color=INK, weight="bold")

    ax2.plot(taus, fa, marker="o", ms=3, color="#E0533D", label="false-accept (dead→usable)")
    ax2.plot(taus, oc, marker="s", ms=3, color="#2563EB", label="over-caution (usable→re-shoot)")
    ax2.plot(taus, acc, marker="^", ms=3, color="#16A34A", label="overall accuracy")
    ax2.axvline(0.72, ls="--", color="0.5", lw=1)
    ax2.text(0.72, 0.02, " recommended\n $\\tau\\approx0.72$", fontsize=9, color="0.35")
    ax2.set_xlabel("confidence threshold  $\\tau$"); ax2.set_ylabel("rate")
    ax2.set_ylim(0, 1); ax2.set_title("(b) Safety tradeoff", fontsize=12, color=INK, weight="bold")
    ax2.legend(frameon=False, fontsize=9, loc="center right")
    ax2.grid(True, alpha=0.25)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    out = os.path.join(config.OBP_VALIDATION_DIR, "fig_station_eval.png")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved -> {out}  (acc@tau0={acc[0]:.3f})")


if __name__ == "__main__":
    main()
