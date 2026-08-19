"""
Diamond - Is the overhead hip-line collapse GEOMETRIC or a detection artifact?
overhead_hip_collapse_test.py

Real overhead video: the hip line collapses at the throw, breaking pelvis rot-velo.
Question that decides feasibility of "get good overhead pose": is that collapse
geometric (pelvis genuinely edge-on to an overhead camera -> hips project to one
point -> unrecoverable by ANY 2D method), or a detection/occlusion failure (arm/
torso hides the hips -> recoverable with a better/temporal pose)?

OBP answers it: clean mocap NEVER has detection failure, so if the CLEAN projected
hip line stays healthy at the pelvis-rotation peak (el=85), the real-video collapse
is a detection artifact (recoverable). If the clean line ALSO collapses, it is
geometric (a wall).

Reports, over N pitches at el=85: projected hip-line length at the rotation peak
as a fraction of its pitch-median, and how often it drops below the 45% gate.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import config
import obp_project as O
import metrics as M
import truth_redefine_gate as T

EL = 85.0
N = 120


def main():
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    ratios, min_ratios, below = [], [], 0
    n = fail = 0
    for i, r in enumerate(md.itertuples(index=False)):
        if n >= N:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1
            continue
        try:
            mk, fps = T.load_markers(path, T.PEL_MK)
            joints, _ = O.load_c3d_joints(path)
            arm = O.detect_throwing_arm(joints, fps)
            rel = M.release_frame(O.project_view(joints, azimuth_deg=0.0), arm, fps, M.JOINTS)
            nf = joints["left_hip"].shape[1]
            lo, hi = max(0, rel - int(0.40 * fps)), min(nf - 1, rel + int(0.05 * fps))

            # pelvis rotation peak (omega_ax), the moment we most need the hips
            Rp, sp = T.pelvis_frame(mk)
            w = np.degrees(np.sum(T.omega_world(Rp, fps) * sp, axis=0))
            win = max(5, int(round(0.05 * fps)) | 1)
            w = savgol_filter(w, min(win, nf - (nf % 2 == 0)), 3, mode="interp")
            pk = lo + int(np.nanargmax(np.abs(w[lo:hi + 1])))

            # projected hip-line length through the pitch at el=85 (fixed projector)
            df = O.project_view(joints, azimuth_deg=0.0, elevation_deg=EL)
            hlen = np.hypot(df["right_hip_x"] - df["left_hip_x"],
                            df["right_hip_y"] - df["left_hip_y"]).to_numpy()
            med = np.nanmedian(hlen)
            ratio_pk = hlen[pk] / med
            min_ratio = np.nanmin(hlen[lo:hi + 1]) / med
            ratios.append(ratio_pk)
            min_ratios.append(min_ratio)
            below += int(ratio_pk < 0.45)
            n += 1
        except Exception:
            fail += 1

    ratios = np.array(ratios); min_ratios = np.array(min_ratios)
    print(f"OBP clean projection at el={EL:.0f}, n={n} (fail {fail})\n")
    print("hip-line length AT the pelvis-rotation peak, / pitch-median:")
    print(f"  median {np.median(ratios):.2f}   p10 {np.percentile(ratios,10):.2f}   "
          f"min {ratios.min():.2f}")
    print(f"  fraction below the 0.45 collapse gate: {below/n*100:.1f}%")
    print("\nMIN hip-line length in the throw window, / median (worst instant):")
    print(f"  median {np.median(min_ratios):.2f}   p10 {np.percentile(min_ratios,10):.2f}   "
          f"min {min_ratios.min():.2f}")
    print("\nVERDICT:")
    if np.median(ratios) > 0.7 and below / n < 0.10:
        print("  Clean hip line STAYS HEALTHY at the rotation peak -> the real-video")
        print("  collapse is a DETECTION/OCCLUSION artifact, NOT geometric.")
        print("  => 'get good overhead pose' is FEASIBLE (temporal recovery / better")
        print("     hip detector can bridge the arm/torso occlusion at release).")
    else:
        print("  Clean hip line ALSO collapses at the peak -> the collapse is")
        print("  GEOMETRIC (pelvis edge-on from overhead). No single-2D pose fixes it.")


if __name__ == "__main__":
    main()
