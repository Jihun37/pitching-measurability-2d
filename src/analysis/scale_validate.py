"""
Scale standardisation: our stride against the release's stride_length.

Our stride_pct_height is stride_px over pixel stature; the release reports
stride_length as a percentage of body height. This asks not only whether they
rank pitches alike (r-squared) but whether they AGREE in absolute terms, so it
reports bias and RMSE and derives the correction constant.

Only stride is in scope. The angles are dimensionless and need no scale.
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

FEATURES = os.path.join(config.OBP_VALIDATION_DIR, "candidate_features_obp.csv")
OBP_DATA = config.OBP_DATA_DIR


def main():
    feat = pd.read_csv(FEATURES)
    poi = pd.read_csv(os.path.join(OBP_DATA, "poi", "poi_metrics.csv"))
    df = feat.merge(
        poi[["session_pitch", "stride_length"]].rename(columns={"stride_length": "obp_stride"}),
        on="session_pitch", how="inner")
    d = df[["stride_pct_height", "obp_stride"]].dropna()
    our = d["stride_pct_height"].to_numpy()
    obp = d["obp_stride"].to_numpy()
    print(f"matched {len(d)}\n")

    r = np.corrcoef(our, obp)[0, 1]
    print("[scale check]  our stride_pct_height  vs  OBP stride_length (% height)")
    print("-" * 60)
    print(f"  correlation r = {r:.3f}   r2 = {r*r:.3f}")
    print(f"  ours:  mean {our.mean():.3f}  std {our.std():.3f}")
    print(f"  OBP:   mean {obp.mean():.3f}  std {obp.std():.3f}")
    print(f"  bias before correction (ours - OBP) = {(our-obp).mean():+.3f}")
    print(f"  RMSE before correction              = {np.sqrt(((our-obp)**2).mean()):.3f}")

    # Correct by a plain ratio k = OBP / ours, which absorbs the bias
    k = obp.mean() / our.mean()
    corr = our * k
    print(f"\n  correction constant k = {k:.3f}  (stride_pct_height x k = stature-normalised stride)")
    print(f"  bias after correction = {(corr-obp).mean():+.3f}")
    print(f"  RMSE after correction = {np.sqrt(((corr-obp)**2).mean()):.3f}")
    print(f"  -> after correction, a mean of {corr.mean()*100:.0f}% of stature, "
          f"error +/-{np.sqrt(((corr-obp)**2).mean())*100:.1f} points (1 sigma)")

    # A linear fit as well, for reference
    slope, icpt = np.polyfit(our, obp, 1)
    print()
    print(f"  (reference) linear fit: OBP ~ {slope:.3f} x ours + {icpt:+.3f}")


if __name__ == "__main__":
    main()