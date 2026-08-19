"""
Diamond - Release frame detection validation
release_check.py

Question: does the new view/slot-invariant release detector find the SAME
frame from the lateral (0 deg) and frontal (90 deg) projections of the
same pitch?

New detector (slot-invariant, view-invariant):
  1) upper bound  = wrist speed peak frame (follow-through; release is before it)
  2) lower bound  = motion start (wrist speed first exceeds 30% of peak)
  3) release      = within [start, peak], first frame where shoulder->wrist
                    projected extension reaches 97% of its window maximum

Comparisons reported (per pitch, frame diff converted to ms):
  A) new(0)  vs new(90)   -> main consistency metric (same pitch, two views)
  B) old(90) vs old(0)    -> size of the current bug (speed argmax frontally)
  C) new(90) vs old(0)    -> new frontal estimate vs previously trusted lateral ref
Stratified by arm slot (3D frontal-plane truth):
  overhand < 40 deg / three-quarter 40-60 deg / sidearm >= 60 deg

Usage:
    python release_check.py            # full OBP batch
    python release_check.py --limit 50 # quick run
"""
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))

import config
import obp_project as O
import metrics as M

OBP_DATA = config.OBP_DATA_DIR
SLOT_BINS = [(-np.inf, 40, "overhand"), (40, 60, "three-quarter"), (60, np.inf, "sidearm")]


# ── New detector (candidate for metrics.release_frame) ──────────
def release_frame_new(df, arm, fps, J=M.JOINTS, win_s=0.20):
    """Frontal release detection: arm extension argmax within a short
    window ending at the wrist speed peak (follow-through).
    Release precedes the frontal speed peak by ~30ms (clean data),
    so a 200ms window safely contains it while excluding the
    early extended-arm phases that caused the early-fire tail."""
    wkey = "r_wr" if arm == "right" else "l_wr"
    skey = "r_sh" if arm == "right" else "l_sh"
    wx, wy = M._xy(df, wkey, J)
    sx, sy = M._xy(df, skey, J)
    spd = M._speed(wx, wy, fps)
    pk = int(np.nanargmax(spd))
    lo = max(0, pk - int(win_s * fps))
    ext = np.hypot(wx - sx, wy - sy)
    seg = ext[lo:pk + 1]
    if np.all(np.isnan(seg)):
        return pk
    return int(lo + np.nanargmax(seg))


def release_frame_old(df, arm, fps, J=M.JOINTS):
    """Current method: wrist speed argmax (known to break frontally)."""
    wkey = "r_wr" if arm == "right" else "l_wr"
    x, y = M._xy(df, wkey, J)
    return int(np.nanargmax(M._speed(x, y, fps)))


def truth_3d_frontal_slot(joints, arm, rel):
    """3D frontal-plane arm slot truth at the given frame (deg, vertical=0)."""
    S = joints[f"{arm}_shoulder"][:, rel]
    W = joints[f"{arm}_wrist"][:, rel]
    vec = W - S
    return float(np.degrees(np.arctan2(abs(vec[1]), vec[2])))


def slot_bin(deg):
    for lo, hi, name in SLOT_BINS:
        if lo <= deg < hi:
            return name
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(OBP_DATA, "metadata.csv"))
    c3d_root = os.path.join(OBP_DATA, "c3d")

    rows, fail = [], 0
    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1
            continue
        try:
            joints, fps = O.load_c3d_joints(path)
            arm = O.detect_throwing_arm(joints, fps)
            df0 = O.project_view(joints, azimuth_deg=0.0)
            df90 = O.project_view(joints, azimuth_deg=90.0)

            new0 = release_frame_new(df0, arm, fps)
            new90 = release_frame_new(df90, arm, fps)
            old0 = release_frame_old(df0, arm, fps)
            old90 = release_frame_old(df90, arm, fps)

            slot = truth_3d_frontal_slot(joints, arm, new0)
            ms = 1000.0 / fps
            rows.append({
                "session_pitch": r.session_pitch, "arm": arm, "fps": fps,
                "arm_slot_deg": slot, "slot_bin": slot_bin(slot),
                "new0": new0, "new90": new90, "old0": old0, "old90": old90,
                "A_new0_vs_new90_ms": (new90 - new0) * ms,
                "B_old90_vs_old0_ms": (old90 - old0) * ms,
                "C_new90_vs_old0_ms": (new90 - old0) * ms,
            })
        except Exception:
            fail += 1
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1} processed")

    df = pd.DataFrame(rows)
    print(f"processed {len(df)} / failed {fail}\n")
    if df.empty:
        print("no results - check OBP paths")
        return

    def summarize(sub, label):
        print(f"--- {label}  (n={len(sub)}) ---")
        for col, desc in [
            ("A_new0_vs_new90_ms", "A) NEW lateral vs frontal (consistency)"),
            ("B_old90_vs_old0_ms", "B) OLD frontal vs lateral (current bug)"),
            ("C_new90_vs_old0_ms", "C) NEW frontal vs OLD lateral ref"),
        ]:
            d = sub[col].to_numpy(float)
            ad = np.abs(d)
            within = (ad <= 33.4).mean() * 100   # within ~1 frame @30fps
            print(f"  {desc}")
            print(f"    mean|d|={np.mean(ad):6.1f}ms  median|d|={np.median(ad):6.1f}ms"
                  f"  bias={np.mean(d):+6.1f}ms  within 33ms: {within:5.1f}%")
        print()

    summarize(df, "ALL pitches")
    print("[arm slot stratification]")
    for col, desc in [("A_new0_vs_new90_ms", "A) new0 vs new90"),
                      ("C_new90_vs_old0_ms", "C) new90 vs old0 (key check)")]:
        print(f"\n  {desc}")
        print(f"  {'bin':14s}{'n':>5s}{'mean|d| ms':>12s}{'median ms':>11s}{'<=33ms %':>10s}")
        print("  " + "-" * 52)
        for _, _, name in SLOT_BINS:
            sub = df[df['slot_bin'] == name]
            if sub.empty:
                continue
            d = np.abs(sub[col].to_numpy(float))
            print(f"  {name:14s}{len(sub):>5d}{np.mean(d):>12.1f}"
                  f"{np.median(d):>11.1f}{(d <= 33.4).mean() * 100:>10.1f}")

    out = os.path.join(config.OBP_VALIDATION_DIR, "release_check_results.csv")
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()