"""Elbow Flex @MER: can it be read WITHOUT locating MER?

mer_proxy_score showed a fixed-lag proxy caps at r2 0.550 against a 0.890 ceiling
even with a perfect release -- elbow flexion moves too fast near MER for any
constant offset. Three escapes are tried here, in increasing order of how much
they give up:

  A. WINDOW EXTREMUM (the HSS trick). If flexion has a local extremum near MER,
     read the extremum VALUE over a release-anchored window and the frame stops
     mattering. `metrics.hss_peak_overhead` is the precedent: it made HSS
     release-free and kept the r2.
  B. SIGNATURE ANCHOR. Locate MER from a 2D signature instead of a clock: the
     lay-back instant is where the wrist sits furthest BEHIND the elbow along the
     throwing direction, which is visible in the image.
  C. EVENT-ROBUST CELL. The metric has 24 usable cells; the anchor was chosen for
     the highest r2 at the exact MER frame, not for tolerance to a wrong frame. A
     cell with a flatter flexion curve can beat a better cell read late.

Truth = OBP elbow_flexion_mer, RAW pooled clean projection. Reported against the
adoption gate (LOCO CCC >= 0.80) and the fixed-lag baseline (0.550 / 0.737).

Run:  conda activate diamond; cd src\\analysis; python elbow_mer_rescue.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)
import numpy as np, pandas as pd
import config, metrics as M, obp_project as O
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from obp_gt_events import load_gt_events
from tier1_adoption_probe import fit_apply, score
from mer_proxy_score import ccc_r2, loco_best

TRUTH = "elbow_flexion_mer"
ANCHOR = (330, 60)
# cells to test for C: the anchor plus its zone neighbours and the other cells
# that cleared 0.60 in the GT map (kept small so this stays a probe, not a sweep)
CELLS = [(330, 60), (330, 45), (315, 60), (345, 60), (330, 75), (0, 45), (0, 60),
         (315, 45), (345, 45), (300, 60), (15, 60), (0, 30)]
WINDOWS_F = [10, 15, 20, 25, 30, 40]     # release-anchored window lengths
# A 2D landmark that is repeatable but systematically early/late is still a good
# anchor -- our release detector is exactly that (-4 f definition offset, SD 0.9).
# So each candidate FRAME is also read at a constant offset, scanned here.
LANDMARK_OFF = list(range(-20, 16))   # must span BOTH signs: the flexion-peak
                                      # landmark sits ~+11 f AFTER MER, so it
                                      # needs a NEGATIVE offset to read back


def flex_curve(df, arm):
    """Elbow flexion (deg) for every frame, same definition as
    metrics.elbow_flexion_2d but vectorised over the clip."""
    J = M.JOINTS
    def xy(k):
        return (df[f"{J[k]}_x"].to_numpy(float), df[f"{J[k]}_y"].to_numpy(float))
    sx, sy = xy("r_sh" if arm == "right" else "l_sh")
    ex, ey = xy("r_el" if arm == "right" else "l_el")
    wx, wy = xy("r_wr" if arm == "right" else "l_wr")
    return M._angle(sx, sy, ex, ey, wx, wy), (sx, sy, ex, ey, wx, wy)


def main():
    gt = load_gt_events()
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    root = os.path.join(config.OBP_DATA_DIR, "c3d")

    keys = ["truth", "user", "at_mer", "lag16"]
    keys += [f"min_w{w}" for w in WINDOWS_F] + [f"max_w{w}" for w in WINDOWS_F]
    keys += ["sig_frame_err", "at_sig", "extr_frame_err"]
    keys += [f"sig{k:+d}" for k in LANDMARK_OFF]
    keys += [f"ext{k:+d}" for k in LANDMARK_OFF]
    D = {c: {k: [] for k in keys} for c in CELLS}
    done = 0
    for r in md.itertuples(index=False):
        sp = r.session_pitch
        g = gt.get(sp)
        if sp not in poi.index or not g or not {"mer", "rel", "fp"} <= set(g):
            continue
        path = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            continue
        try:
            joints, fps = load_feet(path)
        except Exception:
            continue
        arm = O.detect_throwing_arm(joints, fps)
        mer, rel, fp = int(g["mer"]), int(g["rel"]), int(g["fp"])
        for (az, el) in CELLS:
            try:
                df = project_cam(joints, az, el)
            except Exception:
                continue
            n = len(df)
            if not (0 <= fp < mer < rel < n):
                continue
            fx, (sx, sy, ex, ey, wx, wy) = flex_curve(df, arm)
            d = D[(az, el)]
            d["truth"].append(poi.loc[sp, TRUTH]); d["user"].append(int(r.user))
            d["at_mer"].append(fx[mer])
            d["lag16"].append(fx[rel - 16] if rel - 16 >= 0 else np.nan)

            # A: release-anchored window extremum
            for w in WINDOWS_F:
                seg = fx[max(0, rel - w):rel + 1]
                seg = seg[np.isfinite(seg)]
                d[f"min_w{w}"].append(float(seg.min()) if seg.size else np.nan)
                d[f"max_w{w}"].append(float(seg.max()) if seg.size else np.nan)

            # A': where IS the in-window extremum, relative to MER?
            seg = fx[max(0, rel - 40):rel + 1]
            if np.isfinite(seg).any():
                f_ext = int(max(0, rel - 40) + np.nanargmax(seg))
                d["extr_frame_err"].append(f_ext - mer)
            else:
                f_ext = -1
                d["extr_frame_err"].append(np.nan)
            for k in LANDMARK_OFF:
                ff = f_ext + k
                d[f"ext{k:+d}"].append(fx[ff] if f_ext >= 0 and 0 <= ff < n
                                       else np.nan)

            # B: signature anchor = wrist furthest BEHIND the elbow along the
            # throwing direction (the lay-back instant), searched over [fp, rel]
            dirn = np.sign(wx[rel] - wx[fp]) or 1.0
            proj = (wx - ex) * dirn
            lo, hi = fp, rel
            seg = proj[lo:hi + 1]
            if np.isfinite(seg).any():
                f_sig = lo + int(np.nanargmin(seg))
                d["sig_frame_err"].append(f_sig - mer)
                d["at_sig"].append(fx[f_sig])
            else:
                f_sig = -1
                d["sig_frame_err"].append(np.nan); d["at_sig"].append(np.nan)
            for k in LANDMARK_OFF:
                ff = f_sig + k
                d[f"sig{k:+d}"].append(fx[ff] if f_sig >= 0 and 0 <= ff < n
                                       else np.nan)
        done += 1
        if done % 100 == 0:
            print(f"  ...{done}")
    print(f"processed {done} pitches\n")

    rows = []
    for (az, el) in CELLS:
        d = D[(az, el)]
        t = np.asarray(d["truth"], float); u = np.asarray(d["user"])
        if len(t) < 50:
            continue
        base_r2, _ = ccc_r2(d["at_mer"], t)
        lag_r2, _ = ccc_r2(d["lag16"], t)
        sig_err = np.asarray(d["sig_frame_err"], float)
        sig_err = sig_err[np.isfinite(sig_err)]
        ext_err = np.asarray(d["extr_frame_err"], float)
        ext_err = ext_err[np.isfinite(ext_err)]
        print("=" * 74)
        print(f"cell az{az}/el{el}   n={len(t)}   GT-MER r2 {base_r2:.3f}   "
              f"fixed-lag(-16f) r2 {lag_r2:.3f}")
        print(f"  signature frame - MER : median {np.median(sig_err):+.0f} f, "
              f"IQR {np.subtract(*np.percentile(sig_err, [75, 25])):.0f} f")
        print(f"  window-max frame - MER: median {np.median(ext_err):+.0f} f, "
              f"IQR {np.subtract(*np.percentile(ext_err, [75, 25])):.0f} f")
        best = ("", -1.0, None)
        cands = {}
        for w in WINDOWS_F:
            cands[f"min_w{w}(A)"] = d[f"min_w{w}"]
        # landmark + constant offset: keep only each landmark's best offset in
        # the printout, but score every one
        for tag, lab in (("sig", "B:layback"), ("ext", "B:flexpeak")):
            bo, bv = max(((k, ccc_r2(d[f"{tag}{k:+d}"], t)[0])
                          for k in LANDMARK_OFF),
                         key=lambda p: (p[1] if np.isfinite(p[1]) else -1))
            cands[f"{lab}{bo:+d}f"] = d[f"{tag}{bo:+d}"]
        for nm, v in cands.items():
            r2v, cccv = ccc_r2(v, t)
            if np.isfinite(r2v) and r2v > best[1]:
                best = (nm, r2v, v)
            flag = " <<<" if np.isfinite(r2v) and r2v > 0.60 else ""
            print(f"    {nm:<12} r2 {r2v:>6.3f}  raw CCC {cccv:>6.3f}{flag}")
        if best[2] is not None:
            mo, s = loco_best(best[2], t, u)
            print(f"  BEST {best[0]}: r2 {best[1]:.3f}, LOCO CCC {s['ccc']:.3f} ({mo})")
            rows.append(dict(az=az, el=el, n=len(t), r2_gtmer=base_r2,
                             r2_lag16=lag_r2, best=best[0], r2_best=best[1],
                             loco_ccc_best=s["ccc"],
                             sig_err_med=float(np.median(sig_err)),
                             extr_err_med=float(np.median(ext_err))))
        print()

    out = os.path.join(config.OBP_VALIDATION_DIR, "elbow_mer_rescue.csv")
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print(f"saved -> {out}")
    print("Baseline to beat: fixed-lag proxy r2 0.550 / LOCO CCC 0.737; "
          "gate 0.80; GT-MER ceiling r2 0.890.")


if __name__ == "__main__":
    main()
