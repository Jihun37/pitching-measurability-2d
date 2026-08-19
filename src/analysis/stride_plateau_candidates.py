"""
Diamond - Stride plateau: candidate comparison in 3D (2026-07-27).

Step 2 of the event-axis rebuild. `Stride (anchor)` does not read foot plant; it
reads the lead ankle once the stride has SETTLED, and today it dodges the event by
taking a median over a release-anchored 80 ms window. That dodge assumes a delivery
tempo (fp->release ~130 ms median, ~110 ms at the 5th pct) and it has no GT, so the
settling instant is an unnamed event with no error number.

This file does NOT freeze a definition. It scores three families against each other
on 3D marker data, where the projection cannot be blamed:

  C1 position stability  |X(t+i) - X(t)| <= eps for i = 1..P        (release-free)
  C2 velocity stability  |vx(t+i)| <= tau * peak|vx| for i = 0..P   (release-free)
  C3 mixed               both at once

Deliberately release-free: a criterion written against "the value just before
release" (|X - X_settled| < eps) inherits release detection AND re-uses the very
stride length it is supposed to time, so it is excluded by construction.

Comparison axes (not SD alone -- a definition that fires late and rarely can have
the smallest spread):
  detect      share of pitches where the criterion fires before release
  delay       fp -> plateau, median / IQR, and within- vs between-pitcher SD
  agreement   stride read at the plateau vs the OBP stride_length column (r2/CCC)
  drift       plateau reading minus the settled reading, in % of stride
  vs current  the same for the adopted 80 ms release-anchored median

Coordinates: c3d world X is the pitching direction, units metres; heights come from
metadata.session_height_m, so eps is an absolute distance and never a fraction of
the quantity being measured.

NOTE OBP carries no delivery-type label (metadata has playing_level only), so the
set/quick-pitch stability axis cannot be answered here -- it needs our own set
clips (scratch/fp_set_headtohead.py).

Outputs (OBP_VALIDATION_DIR):
  stride_plateau_candidates.csv       per (variant, pitch)
  stride_plateau_summary.csv          per variant

Run:  cd src\\analysis
      python stride_plateau_candidates.py --limit 20
      python stride_plateau_candidates.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)

import config
from master_angle_table import load_feet
from obp_gt_events import load_gt_events
from gt_landmark_outlier_effect import outlier_pitches
from mer_proxy_map import map_population

EPS = [0.005, 0.010, 0.020]        # metres
TAU = [0.05, 0.10, 0.20]           # fraction of the lead ankle's own peak speed
PERS = [5, 10, 20]                 # frames held @360 Hz = 14 / 28 / 56 ms
SETTLE_S = 0.08                    # the adopted release-anchored window


def variants():
    v = []
    for P in PERS:
        for e in EPS:
            v.append(("C1_pos", e, np.nan, P))
        for t in TAU:
            v.append(("C2_vel", np.nan, t, P))
        for e in EPS:
            v.append(("C3_mix", e, 0.10, P))
    return v


def plateau_frame(kind, X, V, vpk, fp, rel, eps, tau, P):
    """First frame at/after fp that satisfies the criterion and HOLDS it for P
    frames. Returns None when nothing qualifies before release."""
    hi = int(rel)
    for t in range(int(fp), hi + 1):
        if t + P > len(X) - 1:
            break
        ok = True
        if kind in ("C1_pos", "C3_mix"):
            seg = X[t:t + P + 1]
            if not np.all(np.isfinite(seg)) or np.nanmax(np.abs(seg - X[t])) > eps:
                ok = False
        if ok and kind in ("C2_vel", "C3_mix"):
            seg = np.abs(V[t:t + P + 1])
            if not np.all(np.isfinite(seg)) or np.nanmax(seg) > tau * vpk:
                ok = False
        if ok:
            return t
    return None


def ccc_r2(e, t):
    e, t = np.asarray(e, float), np.asarray(t, float)
    m = np.isfinite(e) & np.isfinite(t)
    e, t = e[m], t[m]
    if len(e) < 5 or e.std() < 1e-9:
        return np.nan, np.nan
    r = float(np.corrcoef(e, t)[0, 1])
    cov = float(((e - e.mean()) * (t - t.mean())).mean())
    return r ** 2, 2 * cov / (e.var() + t.var() + (e.mean() - t.mean()) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    root = os.path.join(config.OBP_DATA_DIR, "c3d")
    gt = load_gt_events()
    # the frozen map's 394 ids, not gt_clean alone: angle_zone_sweep also
    # requires pkh and rel > fp+1, which drops one more pitch. Official
    # numbers must be on the same population as gate_map.csv.
    pop = map_population()
    VS = variants()

    recs = []
    done = fail = 0
    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        sp = r.session_pitch
        if sp not in pop or sp not in poi.index:
            continue
        g = gt.get(sp)
        if not g or not {"fp", "rel"} <= set(g):
            fail += 1; continue
        path = os.path.join(root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            j, fps = load_feet(path)
        except Exception:
            fail += 1; continue

        # lead = the leg opposite the throwing arm; the loader has already
        # reflected every LHP to RHP, so lead is always LEFT here. Keys are
        # obp_project.MARKER_MAP names, not the 2D JOINTS short codes.
        lead, trail = "left_ankle", "right_ankle"
        if lead not in j or trail not in j:
            fail += 1; continue
        X = j[lead][0].astype(float)          # world X = pitching direction, metres
        XT = j[trail][0].astype(float)
        if np.nanmedian(np.diff(X[:int(g["rel"])])) < 0:   # travel must be +X
            X, XT = -X, -XT
        V = np.gradient(X) * fps
        vpk = float(np.nanmax(np.abs(V[:int(g["rel"]) + 1])))
        if not np.isfinite(vpk) or vpk <= 1e-9:
            fail += 1; continue

        fp, rel = int(g["fp"]), int(g["rel"])
        h = float(r.session_height_m)
        anchor = float(np.nanmedian(XT[:max(3, int(0.2 * fps))]))
        w = max(1, int(round(SETTLE_S * fps)))
        x_settled = float(np.nanmedian(X[max(0, rel - w):rel + 1]))
        stride_settled = abs(x_settled - anchor) / h
        stride_at_fp = abs(X[fp] - anchor) / h
        truth = float(poi.loc[sp, "stride_length"])

        for kind, eps, tau, P in VS:
            t = plateau_frame(kind, X, V, vpk, fp, rel, eps, tau, P)
            s = abs(X[t] - anchor) / h if t is not None else np.nan
            recs.append(dict(
                sp=sp, user=r.user, variant=f"{kind}|eps={eps}|tau={tau}|P={P}",
                kind=kind, eps=eps, tau=tau, P=P,
                found=t is not None, t_plateau=t,
                delay_f=(t - fp) if t is not None else np.nan,
                delay_ms=((t - fp) / fps * 1000) if t is not None else np.nan,
                lead_ms=((rel - t) / fps * 1000) if t is not None else np.nan,
                stride=s, stride_settled=stride_settled, stride_at_fp=stride_at_fp,
                truth=truth, fps=fps))
        done += 1
        if done % 50 == 0:
            print(f"  ...{done} processed")

    print(f"processed {done} / failed {fail}\n")
    d = pd.DataFrame(recs)
    p1 = os.path.join(config.OBP_VALIDATION_DIR, "stride_plateau_candidates.csv")
    d.to_csv(p1, index=False, float_format="%.6g")

    # ---- summary ---------------------------------------------------------
    rows = []
    for v, s in d.groupby("variant"):
        f = s[s.found]
        r2, cc = ccc_r2(f.stride, f.truth)
        drift = (f.stride - f.stride_settled) / f.stride_settled * 100
        wit = f.groupby("user").delay_f.std().median()
        rows.append(dict(
            variant=v, kind=s.kind.iloc[0], eps=s.eps.iloc[0], tau=s.tau.iloc[0],
            P=int(s.P.iloc[0]),
            detect=len(f) / len(s), n=len(f),
            delay_med_f=f.delay_f.median(), delay_iqr_f=f.delay_f.quantile(.75)
            - f.delay_f.quantile(.25), delay_p90_f=f.delay_f.quantile(.9),
            lead_med_ms=f.lead_ms.median(),
            within_sd_f=wit, between_sd_f=f.groupby("user").delay_f.median().std(),
            r2=r2, ccc=cc,
            drift_med_pct=drift.median(), drift_absmed_pct=drift.abs().median(),
            drift_p90_pct=drift.abs().quantile(.9)))
    S = pd.DataFrame(rows).sort_values(["kind", "P", "eps", "tau"])
    ref = d.drop_duplicates("sp")
    r2s, ccs = ccc_r2(ref.stride_settled, ref.truth)
    r2f, ccf = ccc_r2(ref.stride_at_fp, ref.truth)
    p2 = os.path.join(config.OBP_VALIDATION_DIR, "stride_plateau_summary.csv")
    S.to_csv(p2, index=False, float_format="%.4g")

    pd.set_option("display.width", 220, "display.max_rows", 60)
    print("=" * 112)
    print(f"STRIDE PLATEAU CANDIDATES -- 3D, n={ref.sp.nunique()} pitches, "
          f"{ref.user.nunique()} pitchers")
    print("=" * 112)
    print(f"REFERENCE (no plateau event):")
    print(f"  adopted 80 ms release-anchored median : r2 {r2s:.3f}  CCC {ccs:.3f}")
    print(f"  read at the GT foot plant             : r2 {r2f:.3f}  CCC {ccf:.3f}")
    print(f"  (the gap is exactly why the settling instant matters)\n")
    cols = ["kind", "eps", "tau", "P", "detect", "delay_med_f", "delay_iqr_f",
            "delay_p90_f", "lead_med_ms", "within_sd_f", "between_sd_f",
            "r2", "ccc", "drift_absmed_pct", "drift_p90_pct"]
    print(S[cols].round(3).to_string(index=False))
    print("\ndetect = fires before release | delay = fp->plateau in c3d frames @360Hz")
    print("lead_med_ms = plateau -> release margin | drift = plateau vs settled, "
          "% of stride")
    print(f"\nsaved -> {p1}\nsaved -> {p2}")

    print("\n--- shortlist: detect >= 0.98 and |drift| median <= 1 % ---")
    sl = S[(S.detect >= 0.98) & (S.drift_absmed_pct <= 1.0)]
    if len(sl):
        print(sl[cols].sort_values("ccc", ascending=False).round(3).to_string(index=False))
    else:
        print("  none -- relax a bound and read the full table")


if __name__ == "__main__":
    main()
