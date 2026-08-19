"""
Diamond - Release-free HSS dip detection, validated on OBP.

Motivation (2026-07-04): the real-video HSS recipe anchors its search window
on the release frame (wrist-speed argmax). That anchor turned out to be the
dominant error source: backbone-dependent (+/-15 frames..1.2 s drift), arm
misdetection anchors on the glove arm, slow-mo needs manual win_pre. Release
has no physical relation to HSS (the dip peaks near foot plant) — it is only
a window locator. This script tests whether the dip can be located from the
sep(t) series ITSELF, using the documented signature:

    windup ~0 -> counter-rotation (+) -> DIP (-) -> follow-through rebound (+)
    (signs flip with handedness/camera; only the pattern is used)

Candidate detectors (both torso-only, no wrist/arm/release/fps-scaling):
  sig_A : rebound = global |sep| argmax; dip = extremum of the last
          opposite-sign run immediately before the rebound.
  sig_B : the dip->rebound transition is the fastest zero-crossing of the
          whole motion; t* = argmax |d(sep)|; dip = extremum of the
          opposite-sign run immediately before t*.

Compared against, on identical medfilt'd series (90 ms kernel, same as the
real-video recipe):
  base_max : whole-clip |sep| max            (existing OBP result, r2~0.61)
  rel_win  : windowed peak ENDING at release (the current real-video recipe,
             release re-detected on the same projected view)

Truth: OBP poi column max_rotation_hip_shoulder_separation (3D).
Views: el in {85, 90} x az in {0, 90} (HSS is azimuth-invariant up high).

Run:  cd src\analysis
      python hss_signature_validate.py            # all pitches
      python hss_signature_validate.py --limit 60
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from scipy.signal import medfilt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import config
import obp_project as O
import metrics as M
from hss_elevation_test import project_cam, hss_sep_series, r2

VIEWS = [(0, 85), (90, 85), (0, 90), (90, 90)]   # (az, el)
WIN_PRE_S = 0.6


def _run_before(sep, i, sign_opp):
    """Extremum of the contiguous run of sign_opp-signed samples ending
    right before index i (skipping any same-sign/zero gap of a few frames)."""
    j = i - 1
    # walk back over the transition until we are inside the opposite-sign run
    while j > 0 and np.sign(sep[j]) != sign_opp:
        j -= 1
    if j <= 0:
        return None
    start = j
    while start > 0 and np.sign(sep[start - 1]) == sign_opp:
        start -= 1
    seg = sep[start:j + 1]
    return start + int(np.nanargmax(np.abs(seg)))


def sig_a(sep):
    """Dip = extremum of the opposite-sign run before the global |sep| max."""
    i_reb = int(np.nanargmax(np.abs(sep)))
    if sep[i_reb] == 0:
        return None
    return _run_before(sep, i_reb, -np.sign(sep[i_reb]))


def steepest_transition(sep, fps, span_s=0.15, bounds=None):
    """Start frame and direction of the largest SUSTAINED swing (~span_s).
    Single-frame diff is glitch-bait on real video (1-3 frame landmark spikes
    survive the medfilt boundary); slow warmup twists are large but not fast.
    The dip->rebound whip is a large MONOTONIC swing over 0.15-0.3 s; longer
    spans also see through transient glitch humps (up-and-down inside the
    span cancels out). bounds=(lo,hi) restricts the search (coarse pitch
    localization) — indices returned are absolute."""
    w = max(1, int(span_s * fps))
    if len(sep) <= w:
        return None, 0
    swing = sep[w:] - sep[:-w]
    off = 0
    if bounds is not None:
        lo, hi = max(0, bounds[0]), min(len(swing) - 1, bounds[1])
        if hi <= lo:
            return None, 0
        swing = swing[lo:hi + 1]
        off = lo
    i0 = off + int(np.nanargmax(np.abs(swing)))
    return i0, np.sign(swing[i0 - off])


# Promoted to metrics.py (2026-07-04). Re-exported for existing importers.
robust_transition = M.hss_transition


# Promoted to metrics.py (2026-07-04). Re-exported for existing importers.
chord_valid_mask = M.hss_chord_valid


def sig_win_value(sep, fps, span_s, bounds=None, win_pre_s=0.6):
    """Windowed |max| anchored at the sustained-swing transition."""
    i0, sgn = steepest_transition(sep, fps, span_s, bounds)
    if i0 is None or sgn == 0:
        return np.nan
    w = max(1, int(span_s * fps))
    hi = min(len(sep) - 1, i0 + w // 4)
    lo = max(0, i0 - int(win_pre_s * fps))
    return float(np.nanmax(np.abs(sep[lo:hi + 1])))


def sig_b(sep, fps):
    """Dip = extremum of the run before the sustained-swing transition."""
    i0, sign_reb = steepest_transition(sep, fps)
    if i0 is None or sign_reb == 0:
        return None
    return _run_before(sep, i0 + 1, -sign_reb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    J = M.JOINTS

    rows = []
    done = fail = 0
    for r in md.itertuples(index=False):
        if a.limit and done >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = O.load_c3d_joints(path)
            arm = O.detect_throwing_arm(joints, fps)
        except Exception:
            fail += 1; continue

        k = max(3, int(0.09 * fps) // 2 * 2 + 1)
        for az, el in VIEWS:
            try:
                df = project_cam(joints, az, el)
                sep = medfilt(np.nan_to_num(hss_sep_series(df, J), nan=0.0),
                              kernel_size=k)
                row = {"session_pitch": r.session_pitch, "az": az, "el": el,
                       "base_max": float(np.nanmax(np.abs(sep)))}

                # current real-video recipe: window ends at release,
                # release re-detected on this same projected view
                try:
                    rel = M.release_frame(df, arm, fps, J, view="side")
                    lo = max(0, rel - int(WIN_PRE_S * fps))
                    row["rel_win"] = float(np.nanmax(np.abs(sep[lo:rel + 1])))
                except Exception:
                    row["rel_win"] = np.nan

                # span sweep for the signature anchor, with and without
                # coarse localization around the wrist-speed peak (+/-1.5 s)
                nb = None
                try:
                    rel_c = M.release_frame(df, arm, fps, J, view="side")
                    nb = (max(0, rel_c - int(1.5 * fps)),
                          min(len(sep) - 1, rel_c + int(1.5 * fps)))
                except Exception:
                    pass
                for s in (0.15, 0.20, 0.25, 0.30):
                    c = f"{int(s*100):02d}"
                    row[f"win{c}"] = sig_win_value(sep, fps, s)
                    row[f"loc{c}"] = (sig_win_value(sep, fps, s, bounds=nb)
                                      if nb else np.nan)

                # two-stage: anchor searched on a HEAVILY median-filtered
                # series (0.25 s kernel kills transient glitch humps that
                # survive the 90 ms one), value still read off the 90 ms
                # series so the dip magnitude is not blunted
                # persistence-checked anchor + win_pre sweep for the value
                i0, sgn = robust_transition(sep, fps, 0.25, bounds=nb)
                w = max(1, int(0.25 * fps))
                for wp in (0.30, 0.45, 0.60):
                    c = f"pers{int(wp*100)}"
                    if i0 is None or sgn == 0:
                        row[c] = np.nan
                        continue
                    hh = min(len(sep) - 1, i0 + w // 4)
                    ll = max(0, i0 - int(wp * fps))
                    row[c] = float(np.nanmax(np.abs(sep[ll:hh + 1])))

                # chord-validity gate (geometric) + persistence anchor
                valid = chord_valid_mask(df, J)
                sep_g = medfilt(np.where(valid, np.nan_to_num(
                    hss_sep_series(df, J), nan=0.0), 0.0), kernel_size=k)
                ig, sg = robust_transition(sep_g, fps, 0.25, bounds=nb)
                if ig is not None and sg != 0:
                    hh = min(len(sep_g) - 1, ig + w // 4)
                    ll = max(0, ig - int(0.45 * fps))
                    row["gate45"] = float(np.nanmax(np.abs(sep_g[ll:hh + 1])))
                    row["gate_frac"] = float(1.0 - valid.mean())
                else:
                    row["gate45"] = np.nan
                    row["gate_frac"] = float(1.0 - valid.mean())
                rows.append(row)
            except Exception:
                pass
        done += 1
        if done % 100 == 0:
            print(f"  ...{done} pitches", flush=True)
    print(f"processed {done} / {fail} missing\n")

    feat = pd.DataFrame(rows)
    truth = "max_rotation_hip_shoulder_separation"
    m = feat.merge(poi[["session_pitch", truth]], on="session_pitch", how="inner")

    cols = ["base_max", "rel_win"] + [f"{p}{int(s*100):02d}"
            for p in ("win", "loc") for s in (0.15, 0.20, 0.25, 0.30)] + [
            "pers30", "pers45", "pers60", "gate45"]
    print(f"r2 vs {truth}")
    print("-" * 110)
    print(f"{'view':>10s} {'n':>4s} " + "".join(f"{c:>9s}" for c in cols))
    for az, el in VIEWS:
        g = m[(m.az == az) & (m.el == el)]
        line = f"az{az:>3d} el{el:>2d} {len(g):>4d} "
        for c in cols:
            line += f" {r2(g[c], g[truth]):>8.2f}"
        print(line)

    out = os.path.join(config.OBP_VALIDATION_DIR, "hss_signature_features.csv")
    feat.to_csv(out, index=False)
    print(f"\nfeatures saved -> {out}")


if __name__ == "__main__":
    main()
