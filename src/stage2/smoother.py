"""Noise removal for extracted joint coordinates.

A pose estimator returns a position per frame independently, so its output jitters even
where the limb does not. A Savitzky-Golay filter is used rather than a moving average
because it fits a low-order polynomial over the window and therefore preserves the peaks
that several estimators read: an average flattens the extremum the measurement is.

Projected coordinates are exact and are never smoothed. This runs on video only.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def smooth_coordinates(df, window=7, polyorder=2, visibility_threshold=0.5):
    """Savitzky-Golay over every joint coordinate column.

    Parameters
    ----------
    df                   : DataFrame  coordinates as the extractor wrote them
    window               : int        filter width in frames, odd; larger is smoother
    polyorder            : int        polynomial order fitted inside the window
    visibility_threshold : float      a coordinate whose confidence falls below this is
                                      treated as a detection failure rather than a
                                      position, and is interpolated instead of trusted

    Returns
    -------
    DataFrame with the same columns, coordinates filtered.
    """
    smoothed = df.copy()

    # A low-confidence frame is not a measurement. Dropping it to NaN first means the
    # interpolation below spans it, whereas filtering over it would drag the spike into
    # its neighbours.
    joint_names = set(c[:-2] for c in df.columns if c.endswith("_x"))
    for joint in joint_names:
        vcol = f"{joint}_v"
        if vcol in df.columns:
            bad = df[vcol] < visibility_threshold
            smoothed.loc[bad, f"{joint}_x"] = np.nan
            smoothed.loc[bad, f"{joint}_y"] = np.nan

    coord_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]

    for col in coord_cols:
        series = smoothed[col].values

        if np.isnan(series).any():
            series = pd.Series(series).interpolate(
                method="linear", limit_direction="both"
            ).values

        # The window cannot exceed the series, and savgol needs it odd and wider than
        # the polynomial. A clip too short for either is returned interpolated only.
        win = min(window, len(series) if len(series) % 2 == 1 else len(series) - 1)
        if win < polyorder + 2:
            smoothed[col] = series
            continue

        smoothed[col] = savgol_filter(series, window_length=win, polyorder=polyorder)

    return smoothed


def smoothing_report(df_raw, df_smooth, joint="right_wrist"):
    """Frame-to-frame jitter before and after, as a standard deviation of the
    first difference. A diagnostic, not a quality criterion."""
    for axis in ["x", "y"]:
        col = f"{joint}_{axis}"
        raw_jitter = np.nanstd(np.diff(df_raw[col].values))
        smooth_jitter = np.nanstd(np.diff(df_smooth[col].values))
        print(f"{col}: jitter {raw_jitter:.2f} -> {smooth_jitter:.2f} "
              f"({(1 - smooth_jitter / raw_jitter) * 100:.0f}% lower)")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config

    df_raw = pd.read_csv(config.COORDS_CSV)
    df_smooth = smooth_coordinates(df_raw)
    df_smooth.to_csv(config.SMOOTHED_CSV, index=False)

    print("smoothing done")
    smoothing_report(df_raw, df_smooth)
