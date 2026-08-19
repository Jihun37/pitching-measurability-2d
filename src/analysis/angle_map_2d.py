"""
Diamond - 2D angle map: metric r2 over (azimuth x elevation).

Extends the azimuth-only master table to the elevation axis and adds hip-shoulder
separation (HSS). Answers "from which viewpoint can each metric be trusted?"
across BOTH camera axes, in one table.

Reuse (no re-implementation):
  - estimator fns + 3D-direct truth fns + load_feet from master_angle_table.py
  - robust projector project_cam + hss_sep_series from hss_elevation_test.py
    (project_cam handles elevation=90; obp_project.project_view gimbal-locks there)

Conventions (kept identical to master_angle_table):
  - Pearson r2 (corrcoef^2) vs poi-column or 3D-direct truth.
  - Events (release, foot-plant) detected ONCE on the el=0/az=0 side view and
    reused across viewpoints (same convention as master_angle_table /
    angle_sweep_full). This is the paper-table convention, distinct from the
    deployment rule of per-view re-detection.

Physical note: angle metrics (knee, arm slot, trunk, knee-velo, HSS) compute
cleanly at any elevation. Distance/speed metrics (stride, wrist speed, release
height) use vertical (stature) normalization, which collapses as the camera
rises - their r2 drop at high elevation reflects that real failure.

Run:  cd src\analysis
      python angle_map_2d.py --limit 80
      python angle_map_2d.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

_HERE = os.path.dirname(__file__)
for p in ("", "..", "../stage2", "../stage3"):
    sys.path.insert(0, os.path.join(_HERE, p) if p else _HERE)

import config
import obp_project as O
import metrics as M
from master_angle_table import build_rows, load_feet
from hss_elevation_test import project_cam, hss_sep_series

AZ = [0, 45, 90, 135, 180, 225, 270, 315]   # full 360deg orbit. az==az+180 ONLY at
                                            # el=0 (pure u-flip); at el>0 the elevated
                                            # opposite-side views genuinely differ.
EL = [0, 30, 60, 85]            # overhead row = 85 (validated regime; true 90 is the singularity)

# Metrics whose value is an ANGLE ON A CIRCLE, i.e. -179deg and +179deg are
# neighbours. Pearson r2 is a LINEAR statistic and is meaningless across the
# atan2 branch cut, so these must be recentred before scoring (recentre_circular).
# Only stride angle qualifies: every other angle metric is bounded well away from
# +-180 (knee/HSS come from arccos, trunk and arm slot sit near 0).
CIRCULAR = {"Stride Angle [O]"}


def unwrap_circular(vals):
    """Put a circular sample on ONE contiguous branch: each value is moved by a
    multiple of 360deg onto the branch containing the sample's circular mean, so a
    cluster straddling +-180 stops being split in two.

    Needed because stride_angle_2d returns raw atan2: at the front view the two
    ankles are nearly level, so the image-plane angle sits near +-180 for ~10% of
    pitches and the sample wraps. The failure is silent and view-dependent -- it was
    caught by the el=0 mirror invariant (az and az+180 are the same plane u-flipped,
    so their r2 MUST be equal; the GT-event map gave az90 0.492 vs az270 0.863, and
    unwrapping makes both 0.863). See docs/legacy_pre_dedup/GT_EVENT_MAP_HANDOFF.md 4.2.

    Adds +-360 where needed and NOTHING else -- values that do not wrap come back
    bit-identical. That matters because the same helper feeds absacc_table: a plain
    recentring (subtracting the circular mean) would also give the right r2, since
    r2 is shift-invariant, but it would silently rewrite bias and MAE. Safe to apply
    to any declared-circular metric at every view."""
    a = np.asarray(vals, float)
    m = np.isfinite(a)
    if m.sum() < 2:
        return a
    r = np.radians(a[m])
    mu = np.degrees(np.arctan2(np.nanmean(np.sin(r)), np.nanmean(np.cos(r))))
    return mu + ((a - mu + 180.0) % 360.0 - 180.0)


def est_wrist_abs(df, ctx):
    """Absolute wrist speed (m/s), stature-scaled by the subject's known height.
    (px peak / pixel_stature) * height_m -- the projection scale cancels, so this
    is valid on the scale-free project_cam output. Adopted def (r2~0.82) replaces
    the stature-normalized ratio (r2~0.60). Needs a one-time height input."""
    a = ctx["arm"]
    wx = df[f"{a}_wrist_x"].to_numpy(float); wy = df[f"{a}_wrist_y"].to_numpy(float)
    pk = float(np.nanmax(M._speed(wx, wy, ctx["fps"])))
    return pk / M.pixel_stature(df, M.JOINTS) * ctx["height_m"]


def est_hss(df, ctx):
    """Adopted overhead-HSS recipe (metrics.hss_peak_overhead, OBP r2 0.63).
    Replaces the whole-clip |max| estimator (which the follow-through rebound
    inflates). Overhead-only; degenerate at low elevation."""
    r = M.hss_peak_overhead(df, ctx["fps"], M.JOINTS)
    return r["hss"] if r else np.nan


def est_pelvis_rot(df, ctx):
    """Peak pelvis transverse rotational velocity (deg/s) = SG-derivative of the
    unwrapped hip-line angle, peak within the release window [rel-0.40s, rel+0.05s].
    This is the CLEAN-projection validation estimator (matches
    research/audit_pelvis_projection); it reproduces OBP max_pelvis_rotational_velo
    only overhead (el>=75), degenerate from side/front. NOTE: the deployment fn
    metrics.pelvis_rot_velo_overhead adds occlusion recovery for REAL video and
    over-processes clean projections (r2 0.70 -> 0.41), so it is NOT used here."""
    rhx = df["right_hip_x"].to_numpy(float); rhy = df["right_hip_y"].to_numpy(float)
    lhx = df["left_hip_x"].to_numpy(float);  lhy = df["left_hip_y"].to_numpy(float)
    ang = np.unwrap(np.arctan2(rhy - lhy, rhx - lhx))
    fps = ctx["fps"]; n = len(ang)
    win = max(5, int(round(0.05 * fps))); win += (win % 2 == 0); win = min(win, n - (n % 2 == 0))
    vel = (np.degrees(savgol_filter(ang, win, 3, deriv=1, delta=1.0 / fps, mode="interp"))
           if win > 3 else np.degrees(np.gradient(ang) * fps))
    rel = ctx["rel"]
    lo = max(0, rel - int(0.40 * fps)); hi = min(n - 1, rel + int(0.05 * fps))
    seg = np.abs(vel[lo:hi + 1])
    return float(np.nanmax(seg)) if seg.size and not np.all(np.isnan(seg)) else np.nan


def est_cog_velo(df, ctx):
    """Peak forward (image-x) speed of the Winter segment-mass-weighted whole-body
    COM over [0, release], stature-normalized then scaled to m/s by the subject's
    known height. Adopted def (r2~0.75 @side) replacing the hip-midpoint proxy
    (r2~0.41, ROADMAP batch-1 marginal); the mass-weighted whole-body COM tracks
    OBP max_cog_velo_x far better and the height scaling matches its m/s units.
    Best from the side (forward axis in-plane). The "velocity metric, so
    point-only (no noise-valid zone)" note that used to sit here is RETIRED
    2026-07-29: on the deduplicated CCC map velocity rows hold arcs as wide as
    angle rows, and no noise sweep has been run on it (paper Sec. IV-B).
    Needs the one-time height input, like est_wrist_abs.
    Delegates to metrics.cog_fwd_velo (single source shared with deployment)."""
    return M.cog_fwd_velo(df, ctx["fps"], ctx["rel"], M.JOINTS, ctx["height_m"])


def est_cog_pkh(df, ctx):
    """Forward (image-x) COM velocity AT peak knee height (the balance point),
    stature-normalized then scaled to m/s by the subject's height. GT = OBP
    cog_velo_pkh (instantaneous forward COG velocity at the top of the leg lift,
    mean ~0.32 m/s). Uses the Winter whole-body COM (metrics.body_com); the
    hip-midpoint proxy reads only r2~0.45. Nearly unbiased even raw (the balance
    point is slow and the limbs are still). The instantaneous velocity is taken
    with a Savitzky-Golay derivative over a ~0.05 s window (a local linear fit),
    NOT a 2-frame gradient: this is a better instantaneous estimate even on clean
    data (r2 0.80 -> 0.82) and, critically, it denoises the single-frame value on
    REAL video (where a 2-frame gradient at one frame is jitter-dominated; clean
    projection had no such noise). Needs the peak-knee-height event: taken from
    ctx['pkh'] when present (detected with rel/fp under the run's convention),
    else detected here from ctx['fp']. Delegates to metrics.cog_velo_at_pkh
    (single source shared with deployment)."""
    pkh = ctx.get("pkh")
    if pkh is None:
        pkh = M.peak_knee_height_frame(df, ctx["lead"], ctx["fp"], M.JOINTS)
    return M.cog_velo_at_pkh(df, ctx["fps"], pkh, M.JOINTS, ctx["height_m"])


def est_release_ext(df, ctx):
    """Release extension: forward distance from the setup-foot anchor to the
    release point, stature-normalised then scaled to metres by the subject's
    height. Sagittal -- valid only from a pure side view (az0/az180), r2 falling
    to ~0.01-0.18 by az45-az135. Delegates to metrics.release_extension (single
    source shared with deployment), which DEFERS (NaN) when the delivery has no
    still setup, since the trail-foot anchor is meaningless without one."""
    return M.release_extension(df, ctx["arm"], ctx["rel"], ctx["fp"], ctx["fps"],
                               M.JOINTS, ctx["height_m"])


def t3_release_ext(j, ctx):
    """3D-direct truth: same definition on the raw c3d (X = pitching direction),
    metres. Uses the same trail-anchor rule so the comparison is definitional,
    not a different reference point."""
    trail = "right_ankle" if ctx["arm"] == "right" else "left_ankle"
    wx = j[f"{ctx['arm']}_wrist"][0]
    anc = M.trail_anchor_x(j[trail][0], ctx["fp"], ctx["fps"])
    return float(abs(wx[int(ctx["rel"])] - anc))


def est_stride_angle(df, ctx):
    """Stride angle: image-plane orientation of the lead-to-trail ankle line at
    foot plant. Front-view metric (az90); its sign flips with handedness, so it
    only recovers under the reflecting loader (obp_project.reflect_to_rhp).
    Delegates to metrics.stride_angle_2d. Truth = OBP stride_angle column
    (OBP-column mapping). Adopted 2026-07-23 (12 -> 13)."""
    return M.stride_angle_2d(df, ctx["arm"], ctx["fp"], M.JOINTS)


# MER cannot be seen in 2D, but it sits at a nearly fixed lag before release:
# rel - mer = 11 frames at 360 Hz, SD 1.1 (n=401, scratch/mer_timing_probe.py).
# Metrics that are FLAT in time around MER tolerate that jitter and can therefore
# run off a detected release with no MER detector at all (Torso Lat Tilt LOCO CCC
# 0.884, Glove Sh Abd 0.891 -- analysis/mer_proxy_score.py). Metrics that are
# STEEP there cannot: elbow flexion moves 4.32 deg/frame against a truth SD of
# 8.07 deg, so it stays GT-only (docs/legacy_pre_dedup/GT_EVENT_MAP_HANDOFF.md 6g).
MER_LAG_S = 11.0 / 360.0


def mer_frame(ctx):
    """GT MER when the sweep has it, else the release-lag proxy. Keeping the GT
    branch first is what makes the paper (--gt-events) numbers unchanged."""
    mer = ctx.get("mer")
    if mer is not None:
        return int(mer)
    return int(round(ctx["rel"] - MER_LAG_S * ctx["fps"]))


def est_elbow_flex_mer(df, ctx):
    """Elbow flexion at max external rotation. STAYS GT-only: the release-lag
    proxy caps it at r2 0.55 against a 0.89 ceiling because the angle is too steep
    in time (see MER_LAG_S note). metrics.elbow_flexion_2d."""
    mer = ctx.get("mer")
    return (M.elbow_flexion_2d(df, ctx["arm"], int(mer), M.JOINTS)
            if mer is not None else np.nan)


def est_glove_abd_mer(df, ctx):
    """Glove-arm shoulder abduction at MER, GT event or release-lag proxy.
    metrics.shoulder_abduction_2d."""
    return M.shoulder_abduction_2d(df, "glove", mer_frame(ctx), M.JOINTS)


def est_torso_rot_br(df, ctx):
    """Trunk transverse rotation at release (shoulder-line image angle). Overhead-
    recovered. Release-anchored, so it needs no GT-only event. metrics.torso_rotation_2d."""
    return M.torso_rotation_2d(df, ctx["rel"], M.JOINTS)


def est_torso_lat_tilt_mer(df, ctx):
    """Trunk lateral (coronal) tilt at MER, GT event or release-lag proxy.
    metrics.trunk_lean_2d."""
    return M.trunk_lean_2d(df, mer_frame(ctx), M.JOINTS)


def gt_only_rows():
    """EMPTY since the 2026-07-29 GT dedup (user decision, analysis/dedup_rows.py).

    This held ("Elbow Flex @MER [O]", est_elbow_flex_mer, "elbow_flexion_mer"),
    which measured the same quantity as the screened row `elbow_flexion_mer`
    through a second code path and agreed with it to dump rounding. The
    column-named row is kept; this one was dropped so a row count and a quantity
    count are the same number. `est_elbow_flex_mer` stays defined for other callers.

    Elbow flexion at MER is still measurable but not per-pitch deployable, since
    MER is a rotation instant a single 2D view cannot locate. That is a property
    the detected-anchor layer reports, not a membership flag here."""
    return []


def adopted_rows():
    """The adopted metric set of the 2D map: master-table rows minus rejected
    variants, wrist speed overridden with the absolute (height-scaled) def,
    plus HSS, pelvis rotational velocity, COG forward velocity (peak over the
    delivery), and COG velocity at peak knee height. Shared with angle_zone_sweep.

    The GT-only additions (metrics.elbow_flexion_2d etc.) live in gt_only_rows() and
    are appended by angle_zone_sweep only in --gt-events mode."""
    rows = []
    for label, estfn, truth in build_rows():
        if label.startswith("  "):
            continue
        if label.startswith("Wrist Speed"):
            estfn = est_wrist_abs
        rows.append((label, estfn, truth))
    rows.append(("Hip-Shoulder Sep [O]", est_hss,
                 "max_rotation_hip_shoulder_separation"))
    # DROPPED 2026-07-29 (GT dedup, user decision -- see analysis/dedup_rows.py).
    # These three measured the same quantity as the screened rows
    # glove_shoulder_abduction_mer / torso_rotation_br / torso_lateral_tilt_mer
    # through a second code path and agreed with them to dump rounding, so the
    # quantity was counted twice. The column-named rows are kept.
    # ⚠ The DEPLOYED map artefacts were deliberately left at their pre-dedup state
    # (15 rows here, 40 map rows, 991 cells) and are reconciled separately while the
    # paper is written. Regenerating the detected-event dumps with this file as it
    # now stands will therefore NOT reproduce the stored deployment numbers.
    #   rows.append(("Glove Sh Abd @MER [O]", est_glove_abd_mer,
    #                "glove_shoulder_abduction_mer"))
    #   rows.append(("Torso Rot @BR [O]", est_torso_rot_br, "torso_rotation_br"))
    #   rows.append(("Torso Lat Tilt @MER [O]", est_torso_lat_tilt_mer,
    #                "torso_lateral_tilt_mer"))
    # KEPT: this row and the screened `max_pelvis_rotational_velo` reach the same
    # truth through genuinely DIFFERENT 2D observables, and this one is much the
    # better (52 vs 15 strong cells), so the screened row was dropped instead.
    rows.append(("Pelvis Rot Velo [O]", est_pelvis_rot,
                 "max_pelvis_rotational_velo"))
    rows.append(("COG Fwd Velo [O]", est_cog_velo, "max_cog_velo_x"))
    rows.append(("COG Velo @PKH [O]", est_cog_pkh, "cog_velo_pkh"))
    rows.append(("Release Ext [O]", est_release_ext, ("3d", t3_release_ext)))
    rows.append(("Stride Angle [O]", est_stride_angle, "stride_angle"))
    return rows


def plane_of(label):
    l = label.lower()
    if "arm slot" in l:            return "coronal (front)"
    if "hip-shoulder" in l:        return "transverse (overhead)"
    if "pelvis rot" in l:          return "transverse (overhead)"
    return "sagittal (side)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    rows = adopted_rows()

    est = {(ri, az, el): [] for ri in range(len(rows)) for az in AZ for el in EL}
    tru = {ri: [] for ri in range(len(rows))}
    done = fail = 0

    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        try:
            joints, fps = load_feet(path)
            arm = O.detect_throwing_arm(joints, fps)
            lead = "left" if arm == "right" else "right"
            trail = "right" if lead == "left" else "left"
            df0 = O.project_view(joints, azimuth_deg=0.0)
            rel = M.release_frame(df0, arm, fps, M.JOINTS)
            fp = M.foot_plant_frame(df0, lead, fps, M.JOINTS, rel)
            if rel <= fp + 1 or fp < 3:
                fail += 1; continue
            pkh = M.peak_knee_height_frame(df0, lead, fp, M.JOINTS)
            ctx = {"arm": arm, "lead": lead, "trail": trail,
                   "rel": rel, "fp": fp, "pkh": pkh, "fps": fps,
                   "height_m": float(r.session_height_m)}
            sp = r.session_pitch

            for ri, (label, estfn, truth) in enumerate(rows):
                if isinstance(truth, tuple):
                    tval = truth[1](joints, ctx)
                else:
                    tval = poi.loc[sp, truth] if (sp in poi.index and truth in poi.columns) else np.nan
                tru[ri].append(tval)
                for az in AZ:
                    for el in EL:
                        df = project_cam(joints, az, el)
                        try:
                            est[(ri, az, el)].append(estfn(df, ctx))
                        except Exception:
                            est[(ri, az, el)].append(np.nan)
            done += 1
        except Exception:
            fail += 1
        if done and done % 100 == 0:
            print(f"  ...{done} processed")
    print(f"processed {done} / failed {fail}\n")

    def r2(e, t, circular=False):
        e = np.asarray(e, float); t = np.asarray(t, float)
        if circular:
            e = unwrap_circular(e)
        m = np.isfinite(e) & np.isfinite(t)
        return np.corrcoef(e[m], t[m])[0, 1] ** 2 if m.sum() > 2 else np.nan

    out_rows, summary = [], []
    print("=" * 78)
    print("[2D ANGLE MAP]  r2 by (elevation x azimuth)   0deg az=side, 90=front")
    print("=" * 78)
    for ri, (label, _, truth) in enumerate(rows):
        best = (-1, None, None)
        circ = label.strip() in CIRCULAR
        print(f"\n{label}   [{plane_of(label)}]")
        print("        " + "".join(f"  az={az:>2d}" for az in AZ))
        for el in EL:
            line = f"  el={el:>2d} "
            for az in AZ:
                v = r2(est[(ri, az, el)], tru[ri], circ)
                line += f"  {v:>5.2f}" if pd.notna(v) else "     -"
                if pd.notna(v) and v > best[0]:
                    best = (v, az, el)
                out_rows.append({"metric": label.strip(), "az": az, "el": el, "r2": v})
            print(line)
        print(f"   -> best: az={best[1]}, el={best[2]}  (r2={best[0]:.2f})")
        summary.append({"metric": label.strip(), "plane": plane_of(label),
                        "best_az": best[1], "best_el": best[2], "best_r2": best[0]})

    print("\n" + "=" * 78)
    print("[SUMMARY] which viewpoint unlocks which metric")
    print("=" * 78)
    print(f"{'metric':24s}{'plane':24s}{'best view':16s}{'r2':>6s}")
    print("-" * 70)
    for s in sorted(summary, key=lambda x: -x["best_r2"]):
        view = f"az={s['best_az']},el={s['best_el']}"
        print(f"{s['metric']:24s}{s['plane']:24s}{view:16s}{s['best_r2']:>6.2f}")

    out = os.path.join(config.OBP_VALIDATION_DIR, "angle_map_2d.csv")
    pd.DataFrame(out_rows).to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
