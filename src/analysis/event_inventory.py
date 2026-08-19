"""
Diamond - temporal-reference inventory of the gate map (2026-07-27).

Step 1 of the event-axis rebuild. The question is NOT "which of our four detectors
does this metric use" -- that framing silently dropped every temporal reference we
never gave a name to. It is: what instants and intervals does each row's DEFINITION
actually contain, one row at a time, including the ones with no detector, no GT and
no name.

Scope = all 40 rows of the graded map (34 strong-capable + 6 moderate-only), not the
strong tier alone: Hip-Shoulder Sep is moderate-only and carries a bespoke anchor of
its own, so a strong-only inventory would miss an entire event class.

Taxonomy (user, 2026-07-27):
  direct      Release/BR, Foot Plant, PKH            - detected from the image
  derived     MER proxy (rel - k)                    - an offset from a direct event
  to_formalize  (empty as of 2026-07-27 -- all three candidates were tested and
                none of them is an event)
  setup_anchor  Motion onset. Closed 2026-07-27: a reference POSITION, not an
                instant -- 12 threshold x persistence variants move the final CCC
                by <= 0.002; only the deferral rate responds.
  window_read   Stride plateau. Closed 2026-07-27: the quantity is flat in time, so
                the release-anchored window needs no settling detector.
  internal    the estimator's OWN argmax inside its window. NOT a detector to build:
              its error is peak MIS-SELECTION, not frame offset.
  composite   an interval whose two ends move independently (FP-BR)
  gt_target   fp_10 vs fp_100: which landmark our detector is actually closest to.
              A target-definition check, not a new event.

Every (metric, reference, role) below is read off the estimator source -- see
`--show` to print the source next to the assignment.

Outputs: event_inventory.csv  (metric x reference x role, with cell counts)
Run:  cd src\\analysis
      python event_inventory.py
      python event_inventory.py --show
"""
import os, sys, argparse, inspect
import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)

import config
import metrics as M
import angle_map_2d as A
import rejected_gt_full_sweep as R

# ---------------------------------------------------------------- taxonomy ---
CLASS = {
    "rel":            "direct",
    "fp":             "direct",
    "pkh":            "direct",
    "mer_proxy":      "derived",
    "mer_true":       "gt_only",
    # 2026-07-27: demoted from to_formalize. Settled by measurement, not by taste --
    # see stride_plateau_candidates.py (3D) and stride_plateau_2d.py (30 cells).
    "stride_plateau": "window_read",
    # 2026-07-27: reclassified. Not an event -- a setup quiet-window ANCHOR whose
    # only sensitive axis is the deferral rate (motion_onset_candidates.py).
    "motion_onset":   "setup_anchor",
    # 2026-07-27: NOT formalised. t* is a search-window placement device, not an
    # event, and it is not the binding constraint on HSS (hss_anchor_probe.py).
    "hss_anchor":     "internal",
    "internal_peak":  "internal",
}
STATUS = {
    "rel": "detector + per-cell error map",
    "fp": "two detectors + per-cell error map; no routing rule",
    "pkh": "detector + per-cell error map (fp-bounded)",
    "mer_proxy": ("rel - k, k=11f@360. DEPLOYMENT anchor only -- the map itself is "
                  "oracle. Scored per cell 2026-07-27 (mer_proxy_map.py): 131 oracle "
                  "strong cells -> 68 deployable"),
    "mer_true": ("NOT detectable in 2D; GT landmark only. EVERY MER row in the "
                 "graded map is anchored here, including the two labelled 'proxy' "
                 "-- mer_frame() prefers ctx['mer'] and --gt-events supplies it"),
    "stride_plateau": ("NOT AN EVENT (2026-07-27). 27 criteria in 3D land within "
                       "dr2 0.007 of each other and of both references, while the "
                       "instants they pick differ by up to 16 frames; in 2D over "
                       "the 30 Stride (anchor) cells every variant scores 20 strong "
                       "/ 10 moderate, |dCCC| <= 0.006. The quantity is flat there, "
                       "so the window read needs no settling detector"),
    "motion_onset": ("SETUP ANCHOR, not an event (2026-07-27). The ORIGIN it "
                     "supplies is what carries Stride (anchor): replacing it with "
                     "the trail ankle at foot plant costs 0.148 CCC and all 20 "
                     "strong cells. But WHERE the window ends barely matters -- 12 "
                     "threshold x persistence variants give stride CCC 0.829-0.832 "
                     "(20 strong in every one) and Release Ext 0.845-0.858. The one "
                     "sensitive axis is the DEFERRAL rate: quiet-window success runs "
                     "0.823 (10 %, no guard) to 0.977 (30 %, guard 10) against a GT "
                     "stillness rate of 0.909"),
    "hss_anchor": ("SEARCH-WINDOW ANCHOR, not an event (2026-07-27, "
                   "hss_anchor_probe.py, n=394 x 40 cells). Not a proxy either: its "
                   "tightest tie is release - 52 f at SD 6.9 f, and only 40 % of "
                   "pitches sit within +-3 f of that constant. But swapping its "
                   "511 ms window for GT-event windows costs at most 0.023 CCC "
                   "([fp, rel]) and a wide [pkh, rel+0.1 s] window costs 0.007 -- "
                   "and NO window reaches CCC 0.80 anywhere (native 0.780, gt_mer "
                   "0.777, wide 0.774, gt_fp_rel 0.753). The anchor is not what "
                   "limits HSS. Its real value is that it needs no release or foot "
                   "plant at all, which is why it survives on real video"),
    "internal_peak": "inside the estimator; needs mis-selection rate, not a detector",
}

# metric -> [(reference, role, note)]   role: anchor / window / reference_pos / value
# Grounded in the estimator sources (python event_inventory.py --show).
REFS = {
    # ---- adopted -----------------------------------------------------------
    "Stride Angle [O]": [
        ("fp", "anchor", "stride_angle_2d reads the ankle line AT foot plant")],
    "COG Velo @PKH [O]": [
        ("fp", "window", "bounds the pkh argmin search"),
        ("pkh", "anchor", "SG derivative of COM_x evaluated at pkh")],
    "Arm Slot [O]": [("rel", "anchor", "shoulder->wrist image angle at release")],
    "Release Height [O]": [("rel", "anchor", "wrist y at release")],
    "Lead Knee Angle [O]": [("rel", "anchor", "hip-knee-ankle angle at release")],
    "Pelvis Rot Velo [O]": [
        ("rel", "window", "search window [rel-0.40s, rel+0.05s]"),
        ("internal_peak", "value", "argmax |d(hip-line)/dt| inside that window")],
    "Wrist Speed [O]": [
        ("internal_peak", "value", "argmax of wrist speed over the WHOLE clip; "
                                   "no window, so nothing anchors it")],
    "Trunk Tilt (ant) [O]": [("rel", "anchor", "trunk image angle at release")],
    "Torso Rot @BR [O]": [("rel", "anchor", "shoulder-line image angle at release")],
    # Every MER row carries TWO anchors: the one the map scored (GT MER) and the
    # one deployment can actually reach (rel - k). Listing only one of them is what
    # hid the oracle leak until 2026-07-27.
    "Torso Lat Tilt @MER [O]": [
        ("mer_true", "map_anchor", "mer_frame() returns ctx['mer'] under --gt-events"),
        ("mer_proxy", "deploy_anchor", "rel - k; 27 of 37 oracle cells survive")],
    "Glove Sh Abd @MER [O]": [
        ("mer_true", "map_anchor", "same: the map cell is GT-anchored"),
        ("mer_proxy", "deploy_anchor", "rel - k; 7 of 10 oracle cells survive")],
    "Elbow Flex @MER [O]": [
        ("mer_true", "map_anchor", "GT MER: 4.32 deg/frame"),
        ("mer_proxy", "deploy_anchor", "rel - k FAILS: 0 of 20, and 0 even with a "
                                       "PERFECT release, so it is the proxy, not "
                                       "the detector")],
    "Release Ext [O]": [
        ("motion_onset", "reference_pos", "trail_anchor_x = pre-motion quiet window; "
                                          "NaN below min_quiet_s=0.10"),
        ("rel", "anchor", "wrist x at release"),
        ("fp", "window", "passed to release_extension for the quiet-window search")],
    "Stride (anchor) [O]": [
        ("motion_onset", "reference_pos", "trail_anchor_x supplies the origin"),
        ("stride_plateau", "value", "median of ankle_x over [rel-0.08s, rel]; the "
                                    "settling instant is NOT an event -- measured "
                                    "flat in 3D and in 2D (2026-07-27)"),
        ("rel", "window", "the plateau window is release-anchored")],
    "COG Fwd Velo [O]": [
        ("rel", "window", "search window [0, rel]"),
        ("internal_peak", "value", "argmax of |d(COM_x)/dt| inside it")],
    "Hip-Shoulder Sep [O]": [
        ("hss_anchor", "anchor", "chord-validity gate -> 90 ms medfilt -> "
                                 "persistence-checked sustained-swing anchor t*"),
        ("internal_peak", "value", "windowed |max| over [t*-win_pre, t*+span/4] "
                                   "in MOTION time, release-free")],
    # ---- screened ----------------------------------------------------------
    "arm_slot": [("rel", "anchor", "forearm image angle at release")],
    "torso_rotation_br": [("rel", "anchor", "= Torso Rot @BR, second code path")],
    "torso_lateral_tilt_br": [("rel", "anchor", "trunk_lean at release")],
    "torso_anterior_tilt_mer": [
        ("mer_true", "map_anchor", "trunk_lean at GT MER"),
        ("mer_proxy", "deploy_anchor", "rel - k; 30 of 38 oracle cells survive")],
    "torso_lateral_tilt_mer": [
        ("mer_true", "map_anchor", "= Torso Lat Tilt @MER path"),
        ("mer_proxy", "deploy_anchor", "rel - k; 27 of 37")],
    "torso_rotation_mer": [
        ("mer_true", "map_anchor", "shoulder_line at GT MER"),
        ("mer_proxy", "deploy_anchor", "rel - k; 4 of 26 -- release-detection "
                                       "limited (raw CCC 0.10, calibrated 0.69)")],
    "elbow_flexion_mer": [
        ("mer_true", "map_anchor", "= Elbow Flex @MER path"),
        ("mer_proxy", "deploy_anchor", "rel - k; 0 of 20")],
    "glove_shoulder_abduction_mer": [
        ("mer_true", "map_anchor", "= Glove Sh Abd path"),
        ("mer_proxy", "deploy_anchor", "rel - k; 7 of 10")],
    "pelvis_rotation_fp": [("fp", "anchor", "hip-line image angle at foot plant")],
    "pelvis_lateral_tilt_fp": [("fp", "anchor", "hip-line at foot plant")],
    "torso_anterior_tilt_fp": [("fp", "anchor", "trunk_lean at foot plant")],
    "torso_lateral_tilt_fp": [("fp", "anchor", "trunk_lean at foot plant")],
    "torso_rotation_fp": [("fp", "anchor", "shoulder_line at foot plant")],
    "shoulder_abduction_fp": [("fp", "anchor", "elbow-shoulder-hip angle at fp")],
    "glove_shoulder_abduction_fp": [("fp", "anchor", "glove-side, at fp")],
    "shoulder_horizontal_abduction_fp": [("fp", "anchor", "upper arm vs shoulder line")],
    "glove_shoulder_horizontal_abduction_fp": [("fp", "anchor", "glove side, at fp")],
    "elbow_flexion_fp": [("fp", "anchor", "elbow angle at foot plant")],
    "rotation_hip_shoulder_separation_fp": [
        ("fp", "anchor", "shoulder_line - hip_line at foot plant")],
    "lead_knee_extension_angular_velo_fp": [
        ("fp", "anchor", "knee angular velocity SAMPLED at fp (not a window max)")],
    "lead_knee_extension_from_fp_to_br": [
        ("fp", "composite_end", "kang[rel] - kang[fp]: both ends move independently"),
        ("rel", "composite_end", "same")],
    "max_pelvis_rotational_velo": [
        ("fp", "window", "window [fp, rel]"), ("rel", "window", "window [fp, rel]"),
        ("internal_peak", "value", "argmax |d(hip-line)/dt| inside the window")],
    "max_elbow_flexion": [
        ("fp", "window", "window [fp, rel]"), ("rel", "window", "window [fp, rel]"),
        ("internal_peak", "value", "argmax elbow flexion inside the window")],
    # 2026-07-27: window widened to [fp, rel+12f@360Hz]. The +12 is a DERIVED
    # BOUNDARY off the release anchor (from the MIR-BR distribution), NOT an event --
    # it deliberately has no entry of its own here, exactly like the rel-11 MER proxy.
    "lead_knee_extension_angular_velo_max": [
        ("fp", "window", "window start [fp, ...]"),
        ("rel", "window", "window end [..., rel+12f]: derived bound, not an event"),
        ("internal_peak", "value", "argmax knee ext velocity inside the window")],
}


def load_rows():
    V = config.OBP_VALIDATION_DIR
    gm = pd.read_csv(os.path.join(V, "gate_map.csv"))
    g = gm[gm.gate_pass]
    t = (g.assign(strong=(g.grade == "strong").astype(int))
          .groupby(["metric", "source"])
          .agg(cells=("grade", "size"), strong=("strong", "sum"),
               el_lo=("el", "min"), el_hi=("el", "max"),
               best_ccc=("ccc", "max")).reset_index())
    t["tier"] = np.where(t.strong > 0, "strong-capable", "moderate-only")
    return t.sort_values("cells", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true",
                    help="print each estimator's source next to its assignment")
    a = ap.parse_args()

    rows = load_rows()
    missing = set(rows.metric) - set(REFS)
    extra = set(REFS) - set(rows.metric)
    if missing:
        print("!! map rows with no inventory entry:", sorted(missing))
    if extra:
        print("!! inventory entries not on the map:", sorted(extra))

    recs = []
    for r in rows.itertuples(index=False):
        for ref, role, note in REFS.get(r.metric, []):
            recs.append(dict(metric=r.metric, source=r.source, tier=r.tier,
                             cells=r.cells, strong=r.strong,
                             el_lo=r.el_lo, el_hi=r.el_hi,
                             reference=ref, ref_class=CLASS[ref], role=role,
                             note=note))
    inv = pd.DataFrame(recs)
    out = os.path.join(config.OBP_VALIDATION_DIR, "event_inventory.csv")
    inv.to_csv(out, index=False)

    pd.set_option("display.width", 220, "display.max_rows", 300,
                  "display.max_colwidth", 60)
    print("=" * 104)
    print(f"TEMPORAL-REFERENCE INVENTORY -- {len(rows)} map rows "
          f"({(rows.tier=='strong-capable').sum()} strong-capable + "
          f"{(rows.tier=='moderate-only').sum()} moderate-only), "
          f"{int(rows.cells.sum()):,} cells")
    print("=" * 104)

    print("\n--- 1. per reference: who depends on it -------------------------------")
    for cls in ("direct", "derived", "gt_only", "to_formalize", "setup_anchor",
                "window_read", "internal"):
        sub = inv[inv.ref_class == cls]
        if not len(sub):
            continue
        print(f"\n[{cls.upper()}]")
        for ref, s in sub.groupby("reference"):
            anchors = s[s.role == "anchor"]
            print(f"  {ref:<16} rows {s.metric.nunique():>2}   "
                  f"cells(any role) {s.cells.sum():>5,}   "
                  f"as ANCHOR {anchors.cells.sum():>5,}")
            print(f"  {'':<16} status: {STATUS[ref]}")
            for role, s2 in s.groupby("role"):
                print(f"  {'':<16}   {role:<14} "
                      f"{', '.join(sorted(s2.metric.unique()))[:150]}")

    print("\n--- 2. per row: every temporal reference it contains -------------------")
    for r in rows.itertuples(index=False):
        rs = REFS.get(r.metric, [])
        tag = "*" if r.tier == "moderate-only" else " "
        print(f"\n{tag}{r.metric:<40} {r.cells:>4} cells "
              f"(strong {r.strong:>3})  el {r.el_lo}-{r.el_hi}")
        for ref, role, note in rs:
            print(f"    {ref:<16} {role:<14} {note}")
        if a.show:
            fn = {l: f for l, f, _ in A.adopted_rows() + A.gt_only_rows()
                  }.get(r.metric)
            if fn is not None:
                src = [l for l in inspect.getsource(fn).splitlines()
                       if l.strip() and not l.strip().startswith(("#", '"""'))]
                for l in src[-6:]:
                    print("      |", l.rstrip())
            elif r.metric in R.CANDS:
                print("      |", f"CANDS: {R.CANDS[r.metric]}")

    print("\n--- 3. what has to be built -------------------------------------------")
    order = [
        ("stride_plateau", "CLOSED 2026-07-27: not an event, the window read stands"),
        ("motion_onset",   "CLOSED 2026-07-27: setup anchor; only the defer rate moves"),
        ("hss_anchor",     "CLOSED 2026-07-27: search-window anchor, not an event"),
        ("composite",      "CLOSED 2026-07-27: matters for 1 of 5 rows (knee velo max)"),
        ("internal_peak",  "CLOSED 2026-07-27: not the binding constraint on 5 of 7 "
                           "rows (internal_peak_sweep.py)"),
        ("gt_target",      "CLOSED 2026-07-27: map wants fp_100 (-123 strong cells "
                           "at fp_10); side detector tracks fp_100 above el0 "
                           "(fp_target_check.py)"),
    ]
    for i, (k, what) in enumerate(order, 1):
        if k == "composite":
            s = inv[inv.role == "composite_end"]
            n_rows, n_cells = s.metric.nunique(), s.cells.sum() // 2
        elif k == "gt_target":
            s = inv[inv.reference == "fp"]
            n_rows, n_cells = s.metric.nunique(), s.cells.sum()
        else:
            s = inv[inv.reference == k]
            n_rows, n_cells = s.metric.nunique(), s.cells.sum()
        print(f"  {i}. {k:<15} {n_rows:>2} rows / {n_cells:>5,} cells   {what}")
    print(f"\nsaved -> {out}  ({len(inv)} metric x reference entries)")


if __name__ == "__main__":
    main()
