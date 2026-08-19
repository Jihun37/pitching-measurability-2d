"""Event-anchor tolerance for the ADOPTED map rows, in isolation from the frozen
paper-table generators.

WHY A SEPARATE SCRIPT. event_tolerance.py reads the SCREENED offset dump written by
`rejected_gt_full_sweep.py --event-offsets=...`, so only the 23 screened map rows
have a tolerance number; the 12 adopted rows have none. The obvious fix, adding
`--event-offsets` to angle_zone_sweep.py, was rejected (user decision 2026-07-29):
that file generates a frozen paper table, and giving it a second code path risks
overwriting the official dump even when the defaults match. This script imports the
same estimator registry and the same scoring routine but writes only to
`adopted_tolerance_*`.

WHAT A UNIFORM SHIFT MEANS HERE. Every external ground-truth anchor the row reads
(release, foot plant, peak knee height, MER) is displaced by the SAME k frames, so a
row whose window is bounded by two events keeps its window length and slides it.
This matches the screened sweep's rule for window observables.

APPLICABILITY IS DERIVED, NOT DECLARED. A row is `applicable` only if shifting the
anchors actually changes its output. `--probe` measures that per row on a sample of
pitches rather than trusting a hand-written table: two adopted rows read no external
event at all (`Wrist Speed [O]` takes a whole-clip maximum, `Hip-Shoulder Sep [O]`
locates its own signature anchor), and for those a tolerance number would be
meaningless, not favourable. They are recorded as not applicable and excluded from
the denominator.

Run:  conda activate diamond
      cd src\\analysis
      python adopted_event_tolerance_sweep.py --probe            # classify only
      python adopted_event_tolerance_sweep.py --offsets 0        # parity run
      python adopted_event_tolerance_sweep.py --offsets -3,-2,-1,0,1,2,3
"""
import os, sys, argparse, gzip
_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)

import numpy as np, pandas as pd
import config
import obp_project as O
import metrics as M
from master_angle_table import load_feet
from hss_elevation_test import project_cam
from angle_map_2d import adopted_rows
from obp_gt_events import load_gt_events
from mer_proxy_map import map_population

AZ_STEP = 15
AZ = list(range(0, 360, AZ_STEP))
EL = [0, 15, 30, 45, 60, 75, 85]

# the anchors a shift can move; ctx keys, in the order they are reported
ANCHORS = ("rel", "fp", "pkh", "mer")
ANCHOR_NAME = {"rel": "release", "fp": "fp", "pkh": "pkh", "mer": "mer"}

V = config.OBP_VALIDATION_DIR
PAIRS_OUT = os.path.join(V, "adopted_tolerance_pairs.csv.gz")
CLASS_OUT = os.path.join(V, "adopted_anchor_classes.csv")


def build_ctx(joints, fps, r, g):
    """The same ctx angle_zone_sweep builds under --gt-events, or None if the
    pitch fails the same admission test."""
    arm = O.detect_throwing_arm(joints, fps)
    lead = "left" if arm == "right" else "right"
    trail = "right" if lead == "left" else "left"
    if not g or not {"rel", "fp", "pkh"} <= set(g):
        return None
    rel, fp, pkh = g["rel"], g["fp"], g["pkh"]
    if rel <= fp + 1 or fp < 3:
        return None
    return {"arm": arm, "lead": lead, "trail": trail,
            "rel": rel, "fp": fp, "pkh": pkh, "fps": fps,
            "height_m": float(r.session_height_m),
            "mer": g.get("mer"), "mir": g.get("mir")}


def shift(ctx, k):
    """Displace every present external anchor by the same k frames."""
    out = dict(ctx)
    for a in ANCHORS:
        if out.get(a) is not None:
            out[a] = int(out[a]) + k
    return out


def safe(estfn, df, ctx):
    try:
        v = estfn(df, ctx)
        return float(v) if v is not None else np.nan
    except Exception:
        return np.nan


def iter_pitches(md, gt, keep, limit=None):
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    n = 0
    for r in md.itertuples(index=False):
        if r.session_pitch not in keep:
            continue
        if limit and n >= limit:
            return
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            continue
        try:
            joints, fps = load_feet(path)
        except Exception:
            continue
        ctx = build_ctx(joints, fps, r, gt.get(r.session_pitch))
        if ctx is None:
            continue
        n += 1
        yield r.session_pitch, joints, ctx


PROBE_SMALL = 3


class TracingCtx(dict):
    """A ctx that records which keys an estimator actually reads.

    Applicability MUST be decided structurally, not by whether a displacement
    happens to change the output. Pelvis Rot Velo takes its peak inside a
    163-frame release-anchored window, so displacing release by 3 frames -- or
    even 60 -- leaves the peak inside both the old and the new window and the
    value identical. An empirical probe therefore reports "reads no anchor",
    which is false: the row IS release-anchored and simply has a wide tolerance.
    Those two are different findings and only key access separates them.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)


def probe(md, gt, keep, rows, n_pitch, views):
    """anchor_type from key access (structural); the +-3 column is the result."""
    used = {lab: set() for lab, _, _ in rows}
    small = {lab: {a: 0 for a in ANCHORS} for lab, _, _ in rows}
    seen = {lab: 0 for lab, _, _ in rows}
    for sp, joints, ctx in iter_pitches(md, gt, keep, limit=n_pitch):
        for az, el in views:
            df = project_cam(joints, az, el)
            for lab, estfn, _ in rows:
                tc = TracingCtx(ctx)
                base = safe(estfn, df, tc)
                used[lab] |= (tc.read & set(ANCHORS))
                if not np.isfinite(base):
                    continue
                seen[lab] += 1
                for a in ANCHORS:
                    if ctx.get(a) is None:
                        continue
                    c = dict(ctx); c[a] = int(c[a]) + PROBE_SMALL
                    v = safe(estfn, df, c)
                    if not np.isfinite(v) or abs(v - base) > 1e-9:
                        small[lab][a] += 1
    out = []
    for lab, _, _ in rows:
        u = [ANCHOR_NAME[a] for a in ANCHORS if a in used[lab]]
        out.append(dict(
            metric=lab, evaluations=seen[lab],
            anchor_type="+".join(u) if u else "none",
            shifted_boundaries="+".join(u),
            applicable=bool(u),
            not_applicable_reason="" if u else
            "reads no external event anchor from the context",
            **{f"reads_{ANCHOR_NAME[a]}": (a in used[lab]) for a in ANCHORS},
            **{f"moved3_{ANCHOR_NAME[a]}": small[lab][a] for a in ANCHORS}))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", default=None,
                    help="comma list of frame offsets, e.g. '-3,-2,-1,0,1,2,3'")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--probe", action="store_true",
                    help="only classify rows by anchor dependency, no sweep")
    ap.add_argument("--probe-pitches", type=int, default=6)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi_p = os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
    if not os.path.exists(poi_p):
        import glob
        poi_p = glob.glob(os.path.join(config.OBP_DATA_DIR, "**",
                                       "poi_metrics.csv"), recursive=True)[0]
    poi = pd.read_csv(poi_p).set_index("session_pitch")
    gt = load_gt_events()
    keep = set(map_population())
    rows = adopted_rows()
    print(f"{len(rows)} adopted rows | population {len(keep)} pitches")

    if a.probe:
        views = [(0, 0), (90, 0), (0, 75), (180, 45)]
        cls = probe(md, gt, keep, rows, a.probe_pitches, views)
        cls.to_csv(CLASS_OUT, index=False)
        print(f"\nprobed {a.probe_pitches} pitches x {len(views)} views\n")
        cols = ["metric", "anchor_type", "applicable",
                "reads_release", "reads_fp", "reads_pkh", "reads_mer",
                "moved3_release", "moved3_fp", "moved3_pkh", "moved3_mer"]
        print(cls[cols].to_string(index=False))
        print(f"\nsaved -> {CLASS_OUT}")
        return

    if not a.offsets:
        sys.exit("give --offsets or --probe")
    offs = [int(x) for x in a.offsets.split(",")]
    if not os.path.exists(CLASS_OUT):
        sys.exit(f"run --probe first: {CLASS_OUT} is the applicability record")
    cls = pd.read_csv(CLASS_OUT)
    use = [(lab, fn, tr) for lab, fn, tr in rows
           if bool(cls.loc[cls.metric == lab, "applicable"].iloc[0])]
    print(f"applicable rows: {len(use)} of {len(rows)}  offsets {offs}")

    truth_cache = {}
    n = 0
    first = True
    with gzip.open(PAIRS_OUT, "wt", newline="") as fh:
        for sp, joints, ctx in iter_pitches(md, gt, keep, a.limit):
            n += 1
            for lab, estfn, tr in use:
                if isinstance(tr, tuple):
                    truth_cache[(sp, lab)] = safe(tr[1], joints, ctx)
                else:
                    truth_cache[(sp, lab)] = (
                        float(poi.loc[sp, tr])
                        if sp in poi.index and tr in poi.columns else np.nan)
            buf = []
            for az in AZ:
                for el in EL:
                    df = project_cam(joints, az, el)
                    for k in offs:
                        c = shift(ctx, k)
                        for lab, estfn, tr in use:
                            buf.append((lab, az, el, sp, k,
                                        safe(estfn, df, c),
                                        truth_cache[(sp, lab)]))
            pd.DataFrame(buf, columns=["metric", "az", "el", "session_pitch",
                                       "offset", "est", "truth"]).to_csv(
                fh, index=False, header=first, float_format="%.6g")
            first = False
            if n % 25 == 0:
                print(f"  {n} pitches", flush=True)

    print(f"\n{n} pitches -> {PAIRS_OUT} "
          f"({os.path.getsize(PAIRS_OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
