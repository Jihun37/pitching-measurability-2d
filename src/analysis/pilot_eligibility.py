"""Real-video pilot ELIGIBILITY audit and re-tally on the corrected population.

No new analysis: this only re-reads the frozen pilot tables, applies a documented
eligibility rule, and re-counts. No detector, threshold, map or LOPO number is
touched.

RULE (2026-07-28, user decision + audit)
  EXCLUDE  a clip that is not a real, complete pitching delivery -- a non-pitch
           diagnostic / engineering stress test. These are kept on disk as
           development records but must never enter a feasibility statistic.
  KEEP     every real pitch, including truncated or low-frame-rate ones. A clip
           that starts after the leg lift, or that was filmed at 30 fps, is a real
           capture condition and its consequence is a reportable FAILURE MODE, not
           grounds for exclusion. Removing them would be selecting the population
           on the outcome.

Writes:  pilot_clips_eligible.csv / pilot_metrics_eligible.csv  (paper population)
         pilot_excluded.csv                                     (audit trail)

Run:  conda activate diamond
      cd src\\analysis
      python pilot_eligibility.py
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import config

PILOT = os.path.join(config.ROOT, "data", "outputs", "realvideo_pilot")

# Non-pitch diagnostic clips / engineering stress tests. Filmed to test the
# set-position (stretch) foot-plant detector, not as pitching-feasibility material.
NON_PITCH = {
    "set_01": "set-position detector stress test, not a full pitching delivery; "
              "throwing arm also mis-detected as left (subject is RHP)",
    "set_02": "set-position detector stress test, not a full pitching delivery",
}

# Externally sourced clips whose provenance / reuse permission for publication is NOT
# established (downloaded reference footage), plus one non-pitch clip. Excluded
# 2026-07-28 on the user's instruction. The paper population must contain only
# self-filmed material or footage with confirmed rights.
# NOTE these are stated as SOURCE FILE names; the audit matches them against the
# clip names actually present in the pilot tables and reports which ones matched, so
# a listed file that never entered the pilot cannot silently change the count.
UNLICENSED = {
    "video_test_01":  "external source, reuse permission for publication not established",
    "video_test_01c": "external source, reuse permission for publication not established",
    "video_test_02":  "external source, reuse permission for publication not established",
    "video_test_02c": "external source, reuse permission for publication not established",
    "video_test_03":  "external source, reuse permission for publication not established",
    "video_test_03c": "external source, reuse permission for publication not established",
    "dl_lIImrJT3wX0": "external source, reuse permission for publication not established",
    "dl_ujGSpUgxjHc": "external source, reuse permission for publication not established",
    "overhead_test":  "external source, reuse permission for publication not established",
    "batting_test":   "not a pitching clip (batting)",
}


def main():
    c = pd.read_csv(os.path.join(PILOT, "pilot_clips.csv"))
    m = pd.read_csv(os.path.join(PILOT, "pilot_metrics.csv"))
    n0 = len(c)

    present = set(c["clip"])
    print("=" * 92)
    print("0. FILENAME MATCH CHECK for the externally-sourced / non-pitch list")
    print("=" * 92)
    matched, absent = [], []
    for k in UNLICENSED:
        (matched if k in present else absent).append(k)
    for k in matched:
        print(f"     PRESENT in the pilot tables : {k}")
    for k in absent:
        print(f"     not in the pilot tables     : {k}  (no extracted pose; "
              f"never entered the population)")
    print(f"     -> {len(matched)} of {len(UNLICENSED)} listed files are actually "
          f"in the population and will be removed")

    reasons = dict(NON_PITCH)
    reasons.update(UNLICENSED)
    c["excluded_reason"] = c["clip"].map(reasons).fillna("")
    ex = c[c.excluded_reason != ""].copy()
    keep = c[c.excluded_reason == ""].copy()
    mk = m[~m["clip"].isin(set(ex["clip"]))].copy()

    print("=" * 92)
    print("A. ELIGIBILITY AUDIT")
    print("=" * 92)
    print(f"  clips in the pilot run            {n0}")
    print(f"  EXCLUDED (not a full delivery)    {len(ex)}")
    for r in ex.itertuples(index=False):
        print(f"     {r.clip:<22s} {r.excluded_reason}")
    print(f"  PAPER POPULATION                  {len(keep)}")

    print("\n  clips reviewed but KEPT (real pitches with a capture limitation --")
    print("  their consequence is a reported failure mode, not an exclusion):")
    flags = keep[keep["flags"].fillna("") != ""]
    for r in flags.itertuples(index=False):
        print(f"     {r.clip:<22s} {r.fps:>6.1f} fps  {r.duration_s:>5.1f} s  "
              f"flags: {r.flags}")
    print(f"     -- {len(flags)} clips carry at least one capture flag; all retained")

    live = keep[keep.release_f.notna()]
    print("\n" + "=" * 92)
    print(f"B. RE-TALLY ON THE CORRECTED POPULATION  (n = {len(keep)} clips)")
    print("=" * 92)

    print("  RELEASE")
    print(f"     detected                       {len(live)}/{len(keep)}  "
          f"({len(live)/len(keep)*100:.1f} %)")
    rej = keep[keep.release_f.isna()]
    print(f"     rejected by the safety gate    {len(rej)}/{len(keep)}  "
          f"({len(rej)/len(keep)*100:.1f} %)"
          + (f"   [{', '.join(rej['clip'])}]" if len(rej) else ""))

    print("\n  FOOT PLANT   (denominator = clips with a release frame)")
    det = int(live.fp_detected.sum())
    print(f"     truly DETECTED                 {det}/{len(live)}  "
          f"({det/len(live)*100:.1f} %)")
    print(f"     FALLBACK (blind rel - 0.13 s)  {len(live)-det}/{len(live)}  "
          f"({(len(live)-det)/len(live)*100:.1f} %)")
    for d, g in live.groupby("fp_detector"):
        print(f"       routed {d:<8s} n={len(g):>3}  detected {int(g.fp_detected.sum()):>3}"
              f" ({g.fp_detected.mean()*100:5.1f} %)")
    for k, n in live[~live.fp_detected].fp_status.value_counts().items():
        print(f"       fallback reason: {k} = {n}")

    print("\n  METRIC COVERAGE   (15 slots x clips)")
    slots = len(mk)
    raw = int(keep.n_metrics_reported_raw.sum())
    val = int(keep.n_metrics_valid.sum())
    print(f"     slots                          {slots}")
    print(f"     raw 'measured'                 {raw}")
    print(f"     VALID after exclusions         {val}  "
          f"({val/raw*100:.1f} % of raw, {val/slots*100:.1f} % of slots)")
    print(f"     per clip: mean {keep.n_metrics_valid.mean():.2f}  "
          f"median {keep.n_metrics_valid.median():.0f}  "
          f"max {int(keep.n_metrics_valid.max())}  "
          f"clips with zero {int((keep.n_metrics_valid==0).sum())}")

    print("\n     per metric (VALID count):")
    piv = mk[mk.status == "measured"].groupby("metric").size().sort_values(
        ascending=False)
    for k, v in piv.items():
        print(f"       {k:<26s} {v:>3}  ({v/len(keep)*100:5.1f} % of clips)")

    print("\n  FAILURE / EXCLUSION REASONS   (metric slots)")
    for k, n in mk[mk.status == "excluded"].exclusion.value_counts().items():
        print(f"     {n:>4}  excluded: {k}")
    n_sign = int((mk.status == "measured_fp_sign_risk").sum())
    print(f"     {n_sign:>4}  measured but fp-sign risk (Trunk Tilt on a fallback fp)")
    for k, n in mk[mk.status.isin(["deferred", "deferred_norel", "out"])
                   ].status.value_counts().items():
        lab = {"deferred": "viewpoint outside the metric's zone",
               "deferred_norel": "release rejected",
               "out": "outside the point recommendation"}[k]
        print(f"     {n:>4}  {k}: {lab}")

    print("\n  CLIP-LEVEL CAPTURE FLAGS")
    from collections import Counter
    fl = Counter(f for s in keep["flags"].fillna("")
                 for f in (s.split(";") if s else []))
    for k, n in fl.most_common():
        print(f"     {n:>4}  {k}")

    keep.drop(columns=["excluded_reason"]).to_csv(
        os.path.join(PILOT, "pilot_clips_eligible.csv"), index=False)
    mk.to_csv(os.path.join(PILOT, "pilot_metrics_eligible.csv"), index=False)
    ex[["clip", "fps", "duration_s", "excluded_reason"]].to_csv(
        os.path.join(PILOT, "pilot_excluded.csv"), index=False)
    print(f"\nwrote pilot_clips_eligible.csv ({len(keep)}), "
          f"pilot_metrics_eligible.csv ({len(mk)}), pilot_excluded.csv ({len(ex)})")


if __name__ == "__main__":
    main()
