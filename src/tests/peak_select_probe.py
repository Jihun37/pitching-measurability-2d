"""
Diamond - PEAK-SELECTION probe for the open-rear release arc (az ~210-255).

Open task (c) in docs/EVENT_REDETECT_SWEEP_HANDOFF.md: on real clips 03/15
the wrist-speed signal is ALIVE at release (healthy release-adjacent peak)
but the global argmax is stolen by the follow-through peak (03: 4479 vs
7630 px/s; 15: 5174 vs 5567). This probe asks whether a peak-SELECTION rule
(instead of global argmax) fixes that arc without breaking the views where
argmax already works.

Design data = OBP projections (GT release = the adopted el0/az0 detection,
Level-A validated). Candidate rules, evaluated per (az, el):
  argmax       current side strategy (baseline)
  early_qXX    earliest peak with height >= q * global max
  ext_qXX      among peaks >= q * max, the one with max shoulder-wrist
               extension at the peak frame (release = arm extended;
               follow-through = arm wrapping across the body)
  score        argmax of height * extension over peaks >= 0.3 * max
  extgate      earliest peak >= 0.4 * max with extension >= 0.85 * clip max

Then --real applies the same rules to the 15 user-filmed clips and compares
against data/outputs/release_gt_real15.csv (raw-refine of the chosen peak
mirrors metrics.release_frame's two-stage behavior).

NOT wired into metrics/deploy - a read-only study. Adoption is a separate
user decision.

Run:  cd src\tests
      python peak_select_probe.py --limit 60          (OBP design pass)
      python peak_select_probe.py                     (OBP full n=408)
      python peak_select_probe.py --real              (15 real clips)
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

_HERE = os.path.dirname(__file__)
for p in ("..", "../stage2", "../stage3", "../analysis", "../deploy"):
    sys.path.insert(0, os.path.join(_HERE, p))

import config
import obp_project as O
import metrics as M

AZ = list(range(0, 360, 15))
EL = [0, 15, 30]

# thresholds in ms so OBP (mocap fps) and real video (120 fps) compare:
# 25 ms ~ 3 frames @120, 83 ms ~ 10 frames (catastrophic).
OK_MS, CATA_MS = 25.0, 83.0


def peak_candidates(spd, fps, floor=0.25):
    """Prominent local maxima of the wrist-speed trace.
    Returns (frames, heights); global argmax is always included."""
    s = np.nan_to_num(spd, nan=0.0)
    hmax = float(np.max(s))
    if hmax <= 0:
        return np.array([], int), np.array([])
    pk, _ = find_peaks(s, height=floor * hmax,
                       distance=max(2, int(0.03 * fps)))
    gm = int(np.argmax(s))
    if gm not in pk:
        pk = np.sort(np.append(pk, gm))
    return pk, s[pk]


def rules(pk, h, ext, fps):
    """All selection rules -> {name: chosen_frame}. ext = extension trace.

    race_qQQ_tTT: PEAK-RACE rule. Keep argmax unless an earlier peak
    >= q * max sits at least T ms BEFORE it - the follow-through-steal
    signature (real clips 03/15: true peak ~100 ms before the stolen
    argmax) - in which case take the earliest such peak. OBP's genuine
    pre-release shoulder peaks are only ~6-22 ms early, so a T gap
    separates the two regimes."""
    out = {}
    if pk.size == 0:
        return out
    hmax = float(h.max())
    epk = ext[pk]
    emax = float(np.nanmax(ext))
    gm = int(pk[np.argmax(h)])
    out["argmax"] = gm
    for q in (0.4, 0.5, 0.6, 0.7):
        sel = pk[h >= q * hmax]
        out[f"early_q{int(q*100)}"] = int(sel[0])
    for q in (0.5, 0.6):
        sel = pk[h >= q * hmax]
        e = int(sel[0])
        for t in (40, 60, 80, 100):
            out[f"race_q{int(q*100)}_t{t}"] = (
                e if (gm - e) / fps * 1000.0 >= t else gm)
    for q in (0.4, 0.5):
        m = h >= q * hmax
        sub, esub = pk[m], epk[m]
        out[f"ext_q{int(q*100)}"] = (int(sub[np.nanargmax(esub)])
                                     if np.any(np.isfinite(esub))
                                     else out["argmax"])
    m = h >= 0.3 * hmax
    sc = h[m] * np.nan_to_num(epk[m], nan=0.0)
    out["score"] = int(pk[m][np.argmax(sc)]) if sc.size else out["argmax"]
    m = (h >= 0.4 * hmax) & (np.nan_to_num(epk, nan=0.0) >= 0.85 * emax)
    out["extgate"] = int(pk[m][0]) if np.any(m) else out["argmax"]
    return out


def wrist_ext(df, arm, J):
    wkey, skey = ("r_wr", "r_sh") if arm == "right" else ("l_wr", "l_sh")
    wx, wy = M._xy(df, wkey, J)
    sx, sy = M._xy(df, skey, J)
    return wx, wy, np.hypot(wx - sx, wy - sy)


def run_obp(limit):
    from master_angle_table import load_feet
    from hss_elevation_test import project_cam

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    recs = []
    done = fail = 0
    for i, r in enumerate(md.itertuples(index=False)):
        if limit and done >= limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = load_feet(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
            df0 = O.project_view(joints, azimuth_deg=0.0)
            rel = M.release_frame(df0, arm, fps, M.JOINTS)
            fp = M.foot_plant_frame(df0, lead, fps, M.JOINTS, rel)
            if rel <= fp + 1 or fp < 3:
                fail += 1; continue
            for az in AZ:
                for el in EL:
                    df = project_cam(joints, az, el)
                    wx, wy, ext = wrist_ext(df, arm, M.JOINTS)
                    spd = M._speed(wx, wy, fps)
                    pk, h = peak_candidates(spd, fps)
                    near = (np.abs(pk - rel) / fps * 1000.0 <= OK_MS)
                    for name, f in rules(pk, h, ext, fps).items():
                        recs.append({"az": az, "el": el, "rule": name,
                                     "err_ms": (f - rel) / fps * 1000.0,
                                     "alive": bool(near.any())})
            done += 1
        except Exception:
            fail += 1
        if done and done % 50 == 0:
            print(f"  ...{done} pitches")
    print(f"processed {done} / failed {fail}\n")

    t = pd.DataFrame(recs)
    out = os.path.join(config.OBP_VALIDATION_DIR, "peak_select_probe.csv")
    t.to_csv(out, index=False)
    print(f"raw rows -> {out}\n")

    g = t.groupby(["az", "el", "rule"])["err_ms"]
    summ = pd.DataFrame({
        "med": g.median(),
        "iqr": g.quantile(0.75) - g.quantile(0.25),
        "ok": g.apply(lambda e: float(np.mean(np.abs(e) <= OK_MS))),
        "cata": g.apply(lambda e: float(np.mean(np.abs(e) > CATA_MS))),
    }).reset_index()
    souf = os.path.join(config.OBP_VALIDATION_DIR,
                        "peak_select_probe_summary.csv")
    summ.to_csv(souf, index=False)
    print(f"summary  -> {souf}\n")

    alive = t[t.rule == "argmax"].groupby(["az", "el"])["alive"].mean()
    print("signal-alive rate (a candidate peak within +-25ms of GT), el rows:")
    for el in EL:
        line = "  el=%2d  " % el
        line += " ".join(f"{alive.get((az, el), np.nan):.2f}" for az in AZ)
        print(line)
    print("        " + " ".join(f"{az:>4d}" for az in AZ))
    print()

    def show(el, azs):
        sub = summ[(summ.el == el) & (summ.az.isin(azs))]
        print(f"-- el={el}, az={azs} : %ok(+-25ms) / %cata(>83ms) / med_ms --")
        pv_ok = sub.pivot_table(index="rule", columns="az", values="ok")
        pv_ca = sub.pivot_table(index="rule", columns="az", values="cata")
        pv_md = sub.pivot_table(index="rule", columns="az", values="med")
        for name in ("argmax", "early_q50", "early_q60", "early_q70",
                     "race_q50_t40", "race_q50_t60", "race_q50_t80",
                     "race_q50_t100", "race_q60_t60", "race_q60_t80",
                     "score", "extgate"):
            if name not in pv_ok.index:
                continue
            cells = "  ".join(
                f"{pv_ok.loc[name, az]:.2f}/{pv_ca.loc[name, az]:.2f}/"
                f"{pv_md.loc[name, az]:+5.0f}" for az in azs)
            print(f"  {name:<10} {cells}")
        print()

    show(0, [210, 225, 240, 255])       # the open-rear failure arc
    show(15, [210, 225, 240, 255])
    show(30, [210, 225, 240, 255])
    show(0, [0, 90, 165, 180, 330])     # no-regression controls
    # global no-regression: worst-view %ok per rule
    w = summ.groupby("rule").agg(worst_ok=("ok", "min"),
                                 mean_ok=("ok", "mean"),
                                 mean_cata=("cata", "mean"))
    print("-- all 72 views: worst / mean %ok, mean %cata --")
    print(w.sort_values("mean_ok", ascending=False).round(3).to_string())


def run_real():
    from measure_auto import load_clip
    from real_station_test import detect_arm_2d
    import cv2

    gt = pd.read_csv(os.path.join(config.ROOT, "data", "outputs",
                                  "release_gt_real15.csv"))
    print(f"{'clip':<22}{'GT':>5} {'argmax':>7}", end="")
    names = None
    rows = []
    for r in gt.itertuples(index=False):
        name = r.clip
        try:
            df, raw = load_clip(name, "_rtmp")
        except FileNotFoundError:
            print(f"\n{name}: csv missing - skipped")
            continue
        arm = detect_arm_2d(df)
        vp = None
        for ext in (".mp4", ".MOV", ".mov"):
            p = os.path.join(config.ROOT, "data", "videos", name + ext)
            if os.path.exists(p):
                vp = p; break
        cap = cv2.VideoCapture(vp); fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        wx, wy, ext = wrist_ext(df, arm, M.JOINTS)
        spd = M._speed(wx, wy, fps)
        pk, h = peak_candidates(spd, fps)
        picks = rules(pk, h, ext, fps)

        # raw-refine each pick within +-3f (mirrors metrics.release_frame)
        if raw is not None:
            rwx, rwy, _ = wrist_ext(raw, arm, M.JOINTS)
            rspd = M._speed(rwx, rwy, fps)
            for k, f in picks.items():
                lo, hi = max(0, f - 3), min(len(rspd) - 1, f + 3)
                if hi >= lo and not np.all(np.isnan(rspd[lo:hi + 1])):
                    picks[k] = lo + int(np.nanargmax(rspd[lo:hi + 1]))

        if names is None:
            names = [k for k in picks if k != "argmax"]
            print(" " + " ".join(f"{k:>9}" for k in names))
        errs = {k: f - int(r.true_release) for k, f in picks.items()}
        rows.append({"clip": name, **errs})
        print(f"{name:<22}{int(r.true_release):>5} {errs['argmax']:>+7d} "
              + " ".join(f"{errs[k]:>+9d}" for k in names))

    t = pd.DataFrame(rows)
    print("\n-- per-rule score over judged clips (|err| frames) --")
    for k in ["argmax"] + (names or []):
        e = t[k].abs()
        print(f"  {k:<10} <=3f {int((e <= 3).sum()):>2}/{len(e)}   "
              f"<=5f {int((e <= 5).sum()):>2}/{len(e)}   "
              f">10f {int((e > 10).sum()):>2}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    if a.real:
        run_real()
    else:
        run_obp(a.limit)
