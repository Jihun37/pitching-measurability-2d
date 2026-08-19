"""Method-robustness re-check of the below-floor KINEMATIC columns.

The concern (the user's, and a real one): a column can score low not because the
metric is unmeasurable but because the SCREEN read it with a naive method -- the
exact thing that once put stride at 0.39 (value at the single foot-plant frame)
when the settled-window read was 0.84. The full sweep sweeps view (az x el) and
uses the correct GT event, but it reads each observable at a SINGLE event frame
with ONE fixed observable. So a below-floor result could still be a method
artefact of (a) a plane-mismatched observable, or (b) a single-instant read of a
quantity whose signal sits in a nearby window / at an event offset.

This probe re-screens only the below-floor kinematic columns (kinetics and
limb-long-axis rotations are excluded -- no method can measure force or axial
rotation) with, per column, applied IDENTICALLY to every pitch and every view
(definition-level, not per-pitch tuning):

  * offset scan   value at event + k, k in +-12 frames  (a definitional offset,
                  the same kind that the knee_ext_velo -4f offset revealed)
  * window median median over [event-w, event+w]        (the stride settled read)
  * pelvis_anterior_tilt only: SAGITTAL proxies (trunk-thigh angle) instead of the
                  coronal hip-obliquity observable the sweep used -- a genuine
                  plane mismatch flagged in the audit.

For each column it prints the sweep baseline (screen observable, single frame) vs
the best method found, and whether any method crosses the 0.60 usable floor. RAW
clean-projection r2, pooled, GT events, n up to 403.

Run: conda activate diamond; cd src\\analysis; python method_robustness_probe.py
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
from rejected_gt_full_sweep import observables, r2, AZ, EL, FLOOR

# column -> (event, [observable keys to try]). First key = the screen's own
# observable (baseline). Extra keys for pelvis anterior tilt are the sagittal
# proxies computed below.
PROBE = {
    "pelvis_anterior_tilt_fp":  ("fp", ["hip_lean", "trunk_thigh_lead", "trunk_thigh_rear"]),
    "pelvis_lateral_tilt_fp":   ("fp", ["hip_line"]),
    "torso_lateral_tilt_fp":    ("fp", ["trunk_lean"]),
    "shoulder_abduction_fp":    ("fp", ["abd_throw"]),
    "glove_shoulder_abduction_fp": ("fp", ["abd_glove"]),
    "shoulder_horizontal_abduction_fp":       ("fp", ["hz_abd_throw"]),
    "glove_shoulder_horizontal_abduction_fp": ("fp", ["hz_abd_glove"]),
    "lead_knee_extension_angular_velo_fp":    ("fp", ["knee_ext_velo_at"]),
}

OFFS = [-12, -9, -6, -3, 3, 6, 9, 12]     # frames (360 Hz -> up to 33 ms)
WINS = [3, 5, 9]                           # median half-width -> windows 7/11/19 f


def variants():
    yield ("base", None)
    for w in WINS:
        yield (f"win{w}", ("win", w))
    for k in OFFS:
        yield (f"off{k:+d}", ("off", k))


def read(series, f, spec):
    """Apply a read rule to a per-frame series at event frame f."""
    n = len(series)
    if not (0 <= f < n):
        return np.nan
    if spec is None:
        return float(series[f])
    kind, v = spec
    if kind == "off":
        j = f + v
        return float(series[j]) if 0 <= j < n else np.nan
    lo, hi = max(0, f - v), min(n, f + v + 1)
    seg = series[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(np.nanmedian(seg)) if seg.size else np.nan


def sag_proxies(df, lead):
    """Sagittal pelvis-tilt proxies from sparse joints: trunk-thigh angle at the
    hip (shoulder_mid - hip_mid - knee), lead and rear leg."""
    def c(name):
        return df[f"{name}_x"].to_numpy(float), df[f"{name}_y"].to_numpy(float)
    lsx, lsy = c("left_shoulder"); rsx, rsy = c("right_shoulder")
    lhx, lhy = c("left_hip"); rhx, rhy = c("right_hip")
    smx, smy = (lsx + rsx) / 2, (lsy + rsy) / 2
    hmx, hmy = (lhx + rhx) / 2, (lhy + rhy) / 2
    rear = "right" if lead == "left" else "left"
    klx, kly = c(f"{lead}_knee"); krx, kry = c(f"{rear}_knee")
    return {"trunk_thigh_lead": M._angle(smx, smy, hmx, hmy, klx, kly),
            "trunk_thigh_rear": M._angle(smx, smy, hmx, hmy, krx, kry)}


def main():
    gt = load_gt_events()
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    root = os.path.join(config.OBP_DATA_DIR, "c3d")

    cols = [c for c in PROBE if c in poi.columns]
    vlist = list(variants())
    est = {(c, ob, vn, az, el): []
           for c in cols for ob in PROBE[c][1] for vn, _ in vlist
           for az in AZ for el in EL}
    tru = {c: [] for c in cols}
    done = fail = 0

    for r in md.itertuples(index=False):
        sp = r.session_pitch
        g = gt.get(sp)
        if sp not in poi.index or not g or "fp" not in g:
            fail += 1; continue
        path = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = load_feet(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
        except Exception:
            fail += 1; continue

        for c in cols:
            tru[c].append(poi.loc[sp, c])
        f_fp = int(g["fp"])

        for az in AZ:
            for el in EL:
                try:
                    df = project_cam(joints, az, el)
                    o, _ = observables(df, fps)
                    # lead-knee extension velocity series (per-frame)
                    def kxy(k):
                        return (df[f"{k}_x"].to_numpy(float),
                                df[f"{k}_y"].to_numpy(float))
                    hx, hy = kxy(f"{lead}_hip"); kx, ky = kxy(f"{lead}_knee")
                    axx, ay = kxy(f"{lead}_ankle")
                    o["knee_ext_velo_at"] = np.gradient(
                        M._angle(hx, hy, kx, ky, axx, ay)) * fps
                    o.update(sag_proxies(df, lead))
                except Exception:
                    for c in cols:
                        for ob in PROBE[c][1]:
                            for vn, _ in vlist:
                                est[(c, ob, vn, az, el)].append(np.nan)
                    continue

                for c in cols:
                    for ob in PROBE[c][1]:
                        s = o.get(ob)
                        for vn, spec in vlist:
                            v = read(s, f_fp, spec) if s is not None else np.nan
                            est[(c, ob, vn, az, el)].append(v)
        done += 1
        if done % 50 == 0:
            print(f"  ...{done} processed")

    print(f"processed {done} / failed {fail}\n")

    rows = []
    for c in cols:
        t = tru[c]
        best = {"r2": -1, "ob": None, "vn": None, "az": None, "el": None}
        base = {"r2": -1, "az": None, "el": None}
        base_ob = PROBE[c][1][0]
        for ob in PROBE[c][1]:
            for vn, _ in vlist:
                for az in AZ:
                    for el in EL:
                        v = r2(est[(c, ob, vn, az, el)], t)
                        if np.isfinite(v):
                            if v > best["r2"]:
                                best = {"r2": v, "ob": ob, "vn": vn, "az": az, "el": el}
                            if ob == base_ob and vn == "base" and v > base["r2"]:
                                base = {"r2": v, "az": az, "el": el}
        # count usable cells for the winning (ob, vn)
        cells = sum(1 for az in AZ for el in EL
                    if (r2(est[(c, best["ob"], best["vn"], az, el)], t) or 0) >= FLOOR)
        rows.append(dict(column=c, base_r2=round(base["r2"], 3),
                         base_view=f"{base['az']}/{base['el']}",
                         best_r2=round(best["r2"], 3), best_ob=best["ob"],
                         best_read=best["vn"], best_view=f"{best['az']}/{best['el']}",
                         best_cells=cells, gain=round(best["r2"] - base["r2"], 3),
                         crosses_floor=bool(best["r2"] >= FLOOR)))
    res = pd.DataFrame(rows).sort_values("best_r2", ascending=False)
    out = os.path.join(config.OBP_VALIDATION_DIR, "method_robustness_probe.csv")
    res.to_csv(out, index=False)

    print("=" * 108)
    print("METHOD-ROBUSTNESS re-check of below-floor kinematic columns "
          "(GT events, offset scan + window + sagittal proxy)")
    print("  base = screen observable at the single event frame; best = any "
          "observable/read; gain = best - base")
    print("=" * 108)
    print(f"{'column':<42}{'base r2':>8}{'best r2':>9}{'  read':>8}"
          f"{'  obs':>18}{'  view':>9}{'cells':>6}{'gain':>7}  floor")
    for r in res.itertuples(index=False):
        flag = "  CROSSES 0.60" if r.crosses_floor else ""
        print(f"{r.column:<42}{r.base_r2:>8.3f}{r.best_r2:>9.3f}"
              f"{r.best_read:>8}{r.best_ob:>18}{f'  {r.best_view}':>9}"
              f"{r.best_cells:>6d}{r.gain:>+7.3f}{flag}")

    lifted = res[(res.crosses_floor) & (res.base_r2 < FLOOR)]
    print(f"\n{len(lifted)} column(s) lifted from below-floor to >= 0.60 by a "
          f"better method (the stride-style artefact, if any):")
    for r in lifted.itertuples(index=False):
        print(f"  {r.column}: {r.base_r2:.3f} -> {r.best_r2:.3f} "
              f"via {r.best_read} on {r.best_ob} @ {r.best_view}")
    if len(lifted) == 0:
        print("  (none) -- every below-floor kinematic column stays below 0.60 "
              "under offset/window/plane-corrected reads: a measurability wall, "
              "not a method artefact.")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
