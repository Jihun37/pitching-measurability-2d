"""
The shared estimator engine. Every measured quantity is defined here, once.

A function in this file takes a table of image coordinates and nothing else, so
the same code reads a pitch projected from marker data and a pitch extracted from
video. That is what makes the two comparable at all: if an estimator were
reimplemented for either path, a disagreement between them would no longer be
evidence about the viewpoint. Never reimplement one of these inline; import it.

Each quantity carries a projection-reliability tag, which says what a single
image plane can carry of it:

  CLEAN       the quantity lies in the image plane and is read directly.
  DEPTH       the quantity runs along the viewing axis, so it is foreshortened
              and needs calibration onto the reference scale to mean anything.
  DEGENERATE  the quantity is a rotation about the vertical, which a
              ground-level camera carries no component of.

That tag is a property of the geometry and is fixed per quantity. It is NOT the
paper's grading: strong, moderate and limited are measured per (quantity,
viewpoint) cell from agreement with the 3D truth, and live in the map, not here.
"""
import os, sys
import numpy as np
from scipy.signal import medfilt as _medfilt
from scipy.signal import savgol_filter as _savgol

# Joint names as the extractor writes them. A source that names its joints
# differently is accommodated by passing a different dict, not by editing any
# estimator: every function below takes J and looks its joints up through it.
JOINTS = dict(
    l_sh="left_shoulder",  r_sh="right_shoulder",
    l_hip="left_hip",      r_hip="right_hip",
    l_kn="left_knee",      r_kn="right_knee",
    l_an="left_ankle",     r_an="right_ankle",
    l_wr="left_wrist",     r_wr="right_wrist",
    l_el="left_elbow",     r_el="right_elbow",
    head="head",
)

# The projection-reliability tags described in the module docstring.
CLEAN, DEPTH, DEGEN = "clean", "needs_calibration", "degenerate"


# Low-level helpers.
def _xy(df, key, J):
    j = J[key]
    return df[f"{j}_x"].to_numpy(float), df[f"{j}_y"].to_numpy(float)

def _angle(ax, ay, bx, by, cx, cy):
    """Angle at b, in degrees, one value per frame. Unsigned, on [0, 180]."""
    v1x, v1y = ax-bx, ay-by
    v2x, v2y = cx-bx, cy-by
    dot = v1x*v2x + v1y*v2y
    n = np.hypot(v1x, v1y)*np.hypot(v2x, v2y) + 1e-6
    return np.degrees(np.arccos(np.clip(dot/n, -1, 1)))


def mirror_2d_to_rhp(df):
    """Mirror a LHP clip's 2D pose into the RHP frame (deployment sibling of
    obp_project.reflect_to_rhp, which does the same for OBP 3D at load time).

    Reflects the image horizontally (x -> -x, about the clip's own x-centre so
    coordinates stay in a comparable range) and swaps left_*/right_* columns,
    because the mirror of a left shoulder IS a right shoulder. After this the
    delivery reads as right-handed, so detect_arm returns "right" and every
    downstream stage (sign bits, release detection, estimators, zone lookup)
    runs in ONE frame.

    Why it is needed: the measurability map is handedness-relative
    (az0 = the OPEN side), because the OBP loader reflects LHP. The deployment
    viewpoint chain, run on a raw LHP clip, returns the FIELD azimuth instead
    (verified: LHP matched field az 90/120 vs open-relative 24/120, the 24 being
    only the az90/270 fixed points). Indexing the open-relative map with a field
    azimuth mirrors every non-degenerate LHP lookup. Mirroring the input removes
    the mismatch at the source rather than patching each lookup site.

    Call ONCE, right after pose load, only when the detected arm is "left"."""
    out = df.copy()
    xcols = [c for c in df.columns if c.endswith("_x")]
    if xcols:
        centre = float(np.nanmean(df[xcols].to_numpy(float)))
        for c in xcols:
            out[c] = 2.0 * centre - df[c].to_numpy(float)
    ren = {}
    for c in df.columns:
        if c.startswith("left_"):
            ren[c] = "right_" + c[len("left_"):]
        elif c.startswith("right_"):
            ren[c] = "left_" + c[len("right_"):]
    return out.rename(columns=ren)

def _speed(x, y, fps):
    s = np.hypot(np.diff(x), np.diff(y)) * fps
    return np.concatenate([[0.0], s])


# Normalisation scales, which absorb camera distance and zoom.
def body_scale_px(df, J):
    """Median vertical distance from shoulder mid to hip mid, in pixels.

    The scale every length and speed is divided by, so a measurement does not
    depend on how far the camera stood or how far it zoomed.
    """
    lsx, lsy = _xy(df, "l_sh", J); rsx, rsy = _xy(df, "r_sh", J)
    lhx, lhy = _xy(df, "l_hip", J); rhx, rhy = _xy(df, "r_hip", J)
    torso = np.abs(((lsy+rsy)/2) - ((lhy+rhy)/2))
    s = np.nanmedian(torso)
    return float(s) if s > 1 else 1.0


def pixel_stature(df, J):
    """Standing stature in pixels: the 95th percentile of
    (lowest ankle y - head y).

    A percentile rather than the maximum, because one frame of a mis-detected
    head would otherwise set the scale for the whole clip. Falls back to
    shoulder-to-ankle times 1.25 where the source has no head joint. Stride is
    reported against this, matching the release's stride_length, which is a
    percentage of body height.
    """
    lay = _xy(df, "l_an", J)[1]; ray = _xy(df, "r_an", J)[1]
    ankle = np.maximum(lay, ray)
    if f"{J['head']}_y" in df.columns:
        top = df[f"{J['head']}_y"].to_numpy(float)
    else:
        top = (_xy(df, "l_sh", J)[1] + _xy(df, "r_sh", J)[1]) / 2
        return float(np.nanpercentile((ankle - top), 95) * 1.25)
    h = ankle - top
    s = np.nanpercentile(h, 95)
    return float(s) if s > 1 else 1.0


def body_com(df, J):
    """Winter (1990) segment-mass-weighted whole-body centre of mass, per frame.
    Returns (com_x, com_y) pixel series. Segment mass fractions and proximal->
    distal COM locations are the standard Winter table; the hand is folded into
    the forearm ("forearm+hand"), the foot COM is taken at the ankle (no toe
    marker), and head+neck is placed at the head joint. Weights sum to 1.0. If
    the head joint is absent (some real-video pose backbones) the head+neck
    segment is dropped and the remaining weights renormalised.
    Basis of the adopted COG forward-velocity metric (validated r2~0.75 @side vs
    OBP max_cog_velo_x); the unweighted joint mean is materially worse (r2~0.44)
    because fast-swinging limbs are not mass-discounted, so the weighting matters."""
    def P(key):
        return np.array(_xy(df, key, J))                       # (2, nframes)
    sho_mid = (P("l_sh") + P("r_sh")) / 2
    hip_mid = (P("l_hip") + P("r_hip")) / 2
    segs = [(0.497, sho_mid + 0.50 * (hip_mid - sho_mid))]      # trunk
    for s in ("l", "r"):
        segs.append((0.028, P(f"{s}_sh") + 0.436 * (P(f"{s}_el") - P(f"{s}_sh"))))
        segs.append((0.022, P(f"{s}_el") + 0.682 * (P(f"{s}_wr") - P(f"{s}_el"))))
        segs.append((0.100, P(f"{s}_hip") + 0.433 * (P(f"{s}_kn") - P(f"{s}_hip"))))
        segs.append((0.0465, P(f"{s}_kn") + 0.433 * (P(f"{s}_an") - P(f"{s}_kn"))))
        segs.append((0.0145, P(f"{s}_an")))                    # foot ~ ankle
    if f"{J['head']}_x" in df.columns:
        segs.append((0.081, P("head")))                        # head+neck
    w = sum(m for m, _ in segs)
    com = sum(m * pt for m, pt in segs) / w
    return com[0], com[1]


def peak_knee_height_frame(df, lead, fp, J):
    """Peak-knee-height event: the lead knee at its highest point (min image y)
    during the leg lift, argmin over [0, fp] of the lead-knee vertical position
    (the balance-point / top-of-lift instant, used by OBP's cog_velo_pkh).
    View-stability note: camera AZIMUTH rotates in the horizontal plane, so the
    vertical (image-y) coordinate is unchanged by azimuth -> at elevation 0 this
    event is IDENTICAL from every azimuth, far more view-stable than the
    wrist-speed release or the foot-plant events (it only tilts under camera
    elevation). Bounded by the existing foot-plant frame; no new signal needed
    beyond the lead-knee vertical trajectory."""
    key = "l_kn" if lead == "left" else "r_kn"
    ky = _xy(df, key, J)[1]
    hi = max(3, min(int(fp), len(ky)))
    return int(np.nanargmin(ky[:hi]))


def cog_fwd_velo(df, fps, rel, J, height_m=None):
    """Adopted COG forward-velocity metric: peak forward (image-x) speed of the
    Winter whole-body COM over [0, rel], stature-normalized. Returns m/s when
    height_m is given (the OBP max_cog_velo_x unit), else statures/s. Side/ground
    metric (forward axis in-plane); r2~0.75 vs OBP. Single source of truth shared
    by angle_map_2d.est_cog_velo (validation) and compute_candidates (deployment)."""
    comx, _ = body_com(df, J)
    v = np.abs(np.gradient(comx)) * fps / pixel_stature(df, J)
    seg = v[:int(rel) + 1]
    pk = float(np.nanmax(seg)) if seg.size and not np.all(np.isnan(seg)) else np.nan
    return pk * height_m if height_m else pk


def cog_velo_at_pkh(df, fps, pkh, J, height_m=None):
    """Adopted COG-velocity-at-peak-knee-height metric: forward (image-x) velocity
    of the Winter whole-body COM at the pkh frame, via a Savitzky-Golay derivative
    over a ~0.05 s window (a local linear fit, NOT a 2-frame gradient — this
    denoises the single-frame value on real video and even lifts the clean r2
    0.80 -> 0.82). Returns m/s when height_m is given (OBP cog_velo_pkh unit),
    else statures/s. Shared by angle_map_2d.est_cog_pkh and compute_candidates."""
    comx, _ = body_com(df, J)
    n = len(comx)
    win = max(5, int(round(0.05 * fps))); win += (win % 2 == 0)
    win = min(win, n - (n % 2 == 0))
    v = (_savgol(comx, win, 2, deriv=1, delta=1.0 / fps, mode="interp")
         if win > 3 else np.gradient(comx) * fps)
    val = float(v[int(pkh)]) / pixel_stature(df, J)
    return val * height_m if height_m else val


def _is_frontal(view):
    """One spelling test for every view-aware event detector.

    The canonical string is "frontal" -- that is what angle_zone_sweep.release_view
    returns and what compute_candidates passes around. foot_plant_frame historically
    branched on "front" instead, so `foot_plant_frame(..., view=release_view(az, el))`
    fell through to the SIDE detector silently, and `release_frame(..., view="front")`
    did the same in the other direction. Both spellings are accepted everywhere now
    (2026-07-27); no existing caller changes behaviour, since every release_frame
    caller already passed "frontal" and every foot_plant_frame caller "front"."""
    return str(view).lower() in ("front", "frontal")


# ── Event: release frame ─────────────────────────────────
def release_frame(df, arm, fps, J, view="side", win_s=0.20,
                  raw_df=None, refine_s=0.03):
    """Release frame detection, view-aware.
    side    : wrist speed peak (ball velocity is in-plane laterally;
              slot-invariant, matches all existing Level-A validation).
    frontal : arm extension argmax within a short window ending at the
              wrist speed peak (follow-through). Release precedes the
              frontal speed peak by ~30 ms, so a 200 ms window contains
              it while excluding early extended-arm phases.
              Validated on OBP (n=405): agrees with the lateral speed-peak
              reference within 33 ms for 99.8% of pitches, uniformly
              across overhand / three-quarter / sidearm slots.
    raw_df  : optional UNSMOOTHED coords for real video. SG smoothing on an
              asymmetric peak can shift the argmax by a frame (verified on
              lateral_02: smoothed f225, true release f226 — the raw RTMPose
              peak is exactly f226 while raw MediaPipe's global argmax is a
              jitter spike at f208). Two-stage: the SMOOTHED argmax LOCATES
              the peak spike-robustly, then the raw series REFINES within
              +/-refine_s (~3 frames @120fps). OBP validation passes clean
              projections without raw_df -> behavior and all published
              numbers unchanged.
    """
    wkey = "r_wr" if arm == "right" else "l_wr"
    wx, wy = _xy(df, wkey, J)
    spd = _speed(wx, wy, fps)
    pk = int(np.nanargmax(spd))
    if raw_df is not None:
        rwx, rwy = _xy(raw_df, wkey, J)
        rspd = _speed(rwx, rwy, fps)
        r = max(1, int(refine_s * fps))
        lo, hi = max(0, pk - r), min(len(rspd) - 1, pk + r)
        if hi >= lo and not np.all(np.isnan(rspd[lo:hi + 1])):
            pk = lo + int(np.nanargmax(rspd[lo:hi + 1]))
    if not _is_frontal(view):
        return pk
    skey = "r_sh" if arm == "right" else "l_sh"
    sx, sy = _xy(df, skey, J)
    lo = max(0, pk - int(win_s * fps))
    ext = np.hypot(wx - sx, wy - sy)
    seg = ext[lo:pk + 1]
    if np.all(np.isnan(seg)):
        return pk
    return int(lo + np.nanargmax(seg))


def foot_plant_frame_front(df, lead, fps, J, rel, k=0.80, wfrac=0.10,
                           floor_frac=0.03):
    """FRONTAL foot plant (adopted 2026-07-24). From the front the stride runs
    along the camera axis, so the forward-travel gate the side detector needs can
    never fire: on OBP the side detector falls back to its release-anchored default
    on 84% of frontal pitches (75% at az75) -- at the front it is a fixed offset,
    not a detection.

    The anchor is expressed on the PITCH'S OWN timescale, not in milliseconds.
    Peak knee height and release are both detected per pitch, and the true plant
    sits at a strikingly stable fraction of the interval between them: measured
    against the OBP landmark, median k = 0.839 with IQR 0.820-0.855 (n=403). So
    take prior = pkh + k*(rel - pkh) and refine inside +-wfrac of that same
    interval to the first frame the lead ankle is within floor_frac of its lowest
    image position (the foot is down).

    Do NOT reintroduce an absolute offset here. The earlier version used
    `release - 130 ms`, which is the OBP population's own interval: it scored well
    on OBP and did not transfer -- on the angle03 real clips the plant is >200 ms
    before release, outside that version's reachable window, so it could only ever
    return a late frame. Nor widen the search to all of [pkh, rel]: the ankle
    reaches its floor before the landmark plant, which fires ~22 ms early and costs
    r2 (0.795 vs 0.837 at az90).

    OBP n=411, vs the shipped side detector:
      event error @az90  median -2 -> -1 frame (SD ~16 for both; ~5% of pitches
                         miss by >20 frames in either detector)
      stride_angle r2    az75 .682 -> .717     az90 .799 -> .837
    az>=105 is dead for both (0.00) -- that is release detection breaking down at
    those views, not the foot plant.
    """
    ak = "l_an" if lead == "left" else "r_an"
    _, y = _xy(df, ak, J)
    rel = int(rel)
    try:
        pkh = int(peak_knee_height_frame(df, lead, rel, J))
    except Exception:
        pkh = None
    if pkh is None or rel - pkh < 4:
        return max(0, rel - int(0.13 * fps))
    span = rel - pkh
    prior = int(round(pkh + k * span))
    h = max(2, int(wfrac * span))
    lo, hi = max(pkh + 1, prior - h), min(rel - 1, prior + h)
    if hi <= lo:
        return prior
    seg = y[lo:hi + 1]
    if np.all(~np.isfinite(seg)):
        return prior
    ymax = np.nanmax(seg)
    rng = ymax - np.nanmin(seg) + 1e-9
    for f in range(lo, hi + 1):
        if np.isfinite(y[f]) and (ymax - y[f]) / rng < floor_frac:
            return f
    return prior


def foot_plant_frame_front_speed(df, lead, fps, J, rel, tau=0.25, pkh=None,
                                 back_s=0.5, guard_f=3):
    """FRONTAL foot plant via total ankle-speed collapse (CANDIDATE, 2026-07-26;
    release-anchored guard added and validated on set-position clips 2026-07-26).

    Side-by-side alternative to foot_plant_frame_front (the shipped pkh-anchored
    y-floor detector). The lead foot is planted once its image motion dies just
    before release, a delivery-invariant signature: a planted foot has ~zero image
    velocity regardless of leg-lift height. So unlike the y-floor anchor it does
    NOT assume the windup leg-lift-to-release fraction (k=0.80), which is why it is
    the candidate for set-position / stretch deliveries.

    |v| = hypot(vx, vy) of the lead ankle (SG derivative). Scan back for the last
    frame |v| exceeded tau * its stride peak; the plant is the frame right after.

    RELEASE-ANCHORED GUARD (set_01/set_02, 2026-07-26): the scan is confined to
    [rel - back_s, rel - guard_f]. Without it the raw scan returned the release
    frame itself when the ankle never slowed before release (set_02: |v| still
    rising at BR). guard_f keeps the plant strictly before release; back_s keeps it
    from latching onto mid-lift motion. On the set clips |v|+guard landed on the
    true first-ground-contact (set_01 f219, set_02 f451) while y-floor was 6-10 f
    late. Requires a CORRECT release: a mis-detected early release (set_02 auto
    picked the arm-pump wrist peak) still corrupts it -- that is a release-detector
    limitation, logged separately."""
    ak = "l_an" if lead == "left" else "r_an"
    x, y = _xy(df, ak, J)
    rel = int(rel)
    fallback = max(0, rel - int(0.13 * fps))
    if pkh is None:
        try:
            pkh = int(peak_knee_height_frame(df, lead, rel, J))
        except Exception:
            return fallback
    n = len(x)
    win = max(5, int(round(0.03 * fps)) | 1)
    win = min(win, n if n % 2 else n - 1)
    if win < 5:
        return fallback
    vx = _savgol(x, win, 3, deriv=1, delta=1.0 / fps, mode="interp")
    vy = _savgol(y, win, 3, deriv=1, delta=1.0 / fps, mode="interp")
    v = np.hypot(vx, vy)
    lo = max(1, int(pkh), rel - int(round(back_s * fps)))
    hi = min(n - 1, rel - int(guard_f))
    if hi - lo < 4:
        return fallback
    seg = v[lo:hi + 1]
    if not np.any(np.isfinite(seg)):
        return fallback
    peak = np.nanmax(seg)
    if not np.isfinite(peak) or peak <= 1e-9:
        return lo
    thr = tau * peak
    for f in range(hi - 1, lo - 1, -1):                 # last still-moving frame
        if np.isfinite(v[f]) and v[f] > thr:
            return min(f + 1, hi)
    return lo


def foot_plant_frame(df, lead, fps, J, rel, view="side"):
    """The frame at which the lead foot plants.

    Read in the image plane: the foot has finished travelling forward in x and
    its vertical speed has died away. Where neither is found, the fallback is
    release minus about 130 ms, which is a plausible interval rather than a
    detection and is reported as such.

    view="frontal" (or the legacy "front") routes to foot_plant_frame_front_speed
    (adopted 2026-07-26 over the y-floor foot_plant_frame_front after the
    set-position test): the |v|+guard speed detector is delivery-invariant -- it wins
    on set/stretch clips, at a small windup cost that stays well above the usable
    floor. The forward-travel gate below is unobservable from the front (stride is
    along the camera axis). foot_plant_frame_front is retained for reference /
    rollback. Spelling: see _is_frontal -- this branch used to accept "front" ONLY,
    so release_view()'s "frontal" silently selected the side detector."""
    if _is_frontal(view):
        return foot_plant_frame_front_speed(df, lead, fps, J, rel)
    ak = "l_an" if lead == "left" else "r_an"
    x, y = _xy(df, ak, J)
    end = max(3, int(rel))
    fallback = max(0, int(rel) - int(0.13 * fps))
    fwd = x[:end + 1] - x[0]
    if np.nanmax(np.abs(fwd)) < 1e-6:
        return fallback
    if abs(np.nanmin(fwd)) > abs(np.nanmax(fwd)):   # flip if forward is -x
        fwd = -fwd
    xmax = np.nanmax(fwd)
    # Planted means both at once: the foot is at its lowest in the image and its
    # vertical speed has fallen. Either alone fires during the stride.
    ysub = y[:end + 1]
    ylo, yhi = np.nanmin(ysub), np.nanmax(ysub)
    ynorm = (y - ylo) / (yhi - ylo + 1e-9)          # 1 is the ground
    vy = np.abs(np.concatenate([[0.0], np.diff(y)])) * fps
    vypk = np.nanmax(vy[:end + 1]) + 1e-9
    cand = [f for f in range(3, end)
            if fwd[f] > 0.70 * xmax and ynorm[f] > 0.97 and vy[f] < 0.15 * vypk]
    return cand[0] if cand else fallback


# Setup quiet-window rule. ONE definition, shared by the anchor and the length so
# they can never drift apart (they each carried their own literal 0.20 until
# 2026-07-27). QUIET_GUARD adopted 2026-07-27 (user) from
# analysis/motion_onset_candidates.py: the crossing must HOLD for 5 frames, so a
# single jittery frame no longer ends the window. Measured over 12 threshold x
# guard variants at 49 cells, n=394 -- the quiet-window success rate rises
# 0.954 -> 0.972 while Stride (anchor) and Release Ext move by <= 0.002 CCC and
# neither loses a cell. Motion onset is a reference POSITION, not an instant:
# where the window ends barely matters, that the origin exists is everything
# (using the trail ankle AT foot plant instead costs 0.148 CCC and all 20 strong
# cells -- analysis/stride_plateau_2d.py).
QUIET_THR = 0.20        # fraction of the trail ankle's own peak speed
QUIET_GUARD = 5         # frames the crossing must persist, at the clip's own rate


def _quiet_end(seg, fps, thr=None, guard=None):
    """Index at which the pre-motion quiet window ends: the first frame whose
    speed exceeds thr * peak AND stays above it for `guard` frames. guard <= 1
    reproduces the pre-2026-07-27 behaviour exactly. thr/guard default to the
    module constants AT CALL TIME, so a regression harness can flip them."""
    thr = QUIET_THR if thr is None else thr
    guard = QUIET_GUARD if guard is None else guard
    v = np.abs(np.diff(seg)) * fps
    vpk = np.nanmax(v)
    if not np.isfinite(vpk) or vpk <= 0:
        return None
    over = v > thr * vpk
    if guard <= 1:
        idx = np.where(over)[0]
        return max(3, int(idx[0]) if len(idx) else len(seg) - 1)
    run = 0
    for i, o in enumerate(over):
        run = run + 1 if o else 0
        if run >= guard:
            return max(3, i - guard + 1)
    return max(3, len(seg) - 1)


def trail_anchor_x(trail_x, fp, fps, guard=None):
    """Where the trail foot stood on the rubber, as an x in the image.

    The median of the quiet window before motion begins, NOT the foot's position
    at foot plant. The trail foot is anchored on the rubber during the delivery
    but is dragged forward by the time the lead foot lands, so measuring stride
    from its foot-plant position measures the drag as well as the stride.
    Validated at r-squared 0.82 at azimuth 0, normalised by stature, against the
    release's stride_length.

    The quiet window ends where the trail foot's |vx| first exceeds QUIET_THR of
    its own peak and stays there for QUIET_GUARD frames (see _quiet_end).
    Calling with guard=1 restores the earlier definition.
    """
    seg = trail_x[:fp + 1]
    if len(seg) < 5 or np.all(np.isnan(seg)):
        return float(np.nanmedian(seg[:5])) if len(seg) else float(trail_x[fp])
    m = _quiet_end(seg, fps, guard=guard)
    if m is None:
        return float(np.nanmedian(seg))
    return float(np.nanmedian(seg[:m]))


def trail_quiet_len(trail_x, fp, fps, guard=None):
    """Length (seconds) of the trail foot's pre-motion quiet window -- the same
    window trail_anchor_x averages over. Exposed so callers can DEFER when the
    setup is not still (walk-in / crow-hop deliveries): the anchor is only as
    good as this window, and trail_anchor_x always returns a number, so without
    this check a bad anchor is indistinguishable from a good one.

    CAVEAT for the paper: this gate is validated on OBP, where every delivery is a
    stationary set. Its rejection behaviour on continuous / walk-in deliveries is
    NOT evaluated -- and against a 3D 1 cm-displacement reference the 2D rule calls
    97.2 % of pitches still where the truth says 90.9 %, i.e. it errs toward
    accepting."""
    seg = trail_x[:fp + 1]
    if len(seg) < 5 or np.all(np.isnan(seg)):
        return 0.0
    m = _quiet_end(seg, fps, guard=guard)
    return (len(seg) / fps) if m is None else (m / fps)


def stride_settled_2d(df, lead, rel, fps, J, win_s=0.08):
    """Stride length = distance from the trail foot's pre-motion anchor to the lead
    ankle's SETTLED position, stature-normalised. Truth = OBP stride_length column.

    Adopted 2026-07-24, replacing the read at a single foot-plant frame. The lead
    ankle does NOT stop at foot plant: scored against the OBP fp_100 landmark the
    reading keeps improving for at least 80 ms past 100% load (r2 0.391 at the
    landmark itself, rising monotonically to 0.517 at +30 frames), because the ankle
    joint centre travels forward as the shank rotates over the planted foot. Reading
    the settled position scores 0.844 vs 0.823 for the old at-foot-plant definition,
    and -- the reason for the change -- it removes the foot-plant dependency
    entirely. The only event left is release, which we detect at SD 0.9 frames
    against the OBP BR landmark; our foot plant is SD ~15 frames against fp_100.
    Evidence: docs/legacy_pre_dedup/GT_EVENT_MAP_HANDOFF.md 4.1.

    The window is release-anchored and short (default 80 ms) so it stays inside the
    plateau for every OBP delivery: foot-plant-to-release is ~130 ms at the median
    and ~110 ms at the 5th percentile. Do NOT widen it back toward foot plant -- the
    early frames are the still-travelling ones, which is exactly what the superseded
    definition read.

    The anchor is taken at release rather than at foot plant. trail_anchor_x finds
    the pre-motion quiet window at the START of the clip, so extending its search
    segment changes nothing measurable: anchor(rel) - anchor(fp) is median 0.0000
    statures, SD 0.0020, |.|>0.02 in 0.0% of pitches (stride_anchor_sensitivity.py).
    """
    lan = "l_an" if lead == "left" else "r_an"
    tan = "r_an" if lead == "left" else "l_an"
    la = _xy(df, lan, J)[0]
    ta = _xy(df, tan, J)[0]
    rel = int(rel)
    w = max(1, int(round(win_s * fps)))
    seg = la[max(0, rel - w):rel + 1]
    if seg.size == 0 or np.all(np.isnan(seg)):
        return np.nan
    pos = float(np.nanmedian(seg))
    return abs(pos - trail_anchor_x(ta, rel, fps)) / pixel_stature(df, J)


def release_extension(df, arm, rel, fp, fps, J, height_m=None, min_quiet_s=0.10):
    """Release EXTENSION: forward distance from the pitcher's setup foot position
    to the ball-release point, stature-normalised (metres when height_m given).

    This is the orthogonal partner of release height, and the baseball/Statcast
    sense of "extension". NOTE it does NOT require the rubber to be visible --
    or to exist: the reference is the trail foot's PRE-MOTION position
    (trail_anchor_x), i.e. where the pitcher was standing, which on a mound
    happens to be the rubber. It therefore works on flat ground too.

    What it does require is a STILL setup. Walk-in / crow-hop deliveries have no
    quiet window, the anchor degrades silently, and the metric must be deferred
    -- hence min_quiet_s (returns NaN below it). Substituting a trail-ankle-at-
    release reference is NOT a valid fallback: that is a different quantity
    (r=0.71, offset SD 0.106 m vs a 0.148 m between-pitch SD, because trail-foot
    drag varies by pitcher), so callers must defer rather than swap definitions.

    Sagittal metric: validated only from a pure side view (az0/az180); r2 falls
    to ~0.01-0.18 by az45-az135. Robust to camera elevation at the side.
    Adopted 2026-07-20; revalidated at n=411 after the 2026-07-23 dataset
    recovery and LHP reflection (r2 0.949 unchanged at the az0/el0 anchor)."""
    a = arm[0]
    trail = "r_an" if arm == "right" else "l_an"   # pivot foot = throwing side
    wx = np.asarray(_xy(df, f"{a}_wr", J)[0], float)
    tx = np.asarray(_xy(df, trail, J)[0], float)
    if trail_quiet_len(tx, fp, fps) < min_quiet_s:
        return np.nan
    anc = trail_anchor_x(tx, fp, fps)
    val = abs(wx[int(rel)] - anc) / pixel_stature(df, J)
    return val * height_m if height_m else val


def stride_angle_2d(df, arm, fp, J):
    """Stride ANGLE: image-plane orientation (deg) of the lead-to-trail ankle
    line at foot plant. Captures a closed vs open stride direction. Truth = OBP
    stride_angle column (independent, not self-referential).

    A horizontal/transverse quantity, so its sign flips with handedness. The OBP
    loader reflects every LHP into an equivalent RHP (obp_project.reflect_to_rhp),
    which is what makes this measurable at all: pooling raw handedness cancelled
    the correlation (r2 0.40) whereas the handedness-normalised map recovers it
    (r2 0.888 at the FRONT, az90). The image-plane convention differs from OBP by
    a near-constant ~90 deg offset, so the raw value is biased (raw CCC ~0.04) but
    calibrates cleanly to new pitchers (LOCO CCC 0.94, MAE ~1.5 deg): a CALIBRATE
    metric, like wrist speed and HSS.

    Best from the FRONT (az90); collapses toward the side. Adopted 2026-07-23
    after handedness normalization revived it -- it had been excluded with the
    transverse/geometric-wall columns, and its recovery is direct evidence that
    part of that "wall" was a handedness-pooling artifact, not geometry."""
    lead = "left" if arm == "right" else "right"
    lan = "l_an" if lead == "left" else "r_an"
    tan = "r_an" if arm == "right" else "l_an"
    lax, lay = _xy(df, lan, J)
    tax, tay = _xy(df, tan, J)
    return float(np.degrees(np.arctan2(lay[fp] - tay[fp], lax[fp] - tax[fp])))


# ── metrics adopted 2026-07-24 from the GT-event rescreen ───────────────────────
# All three were REJECTED under our own detected events and recovered once read at
# the correct GT event (and, for torso rotation, from overhead). They take the event
# FRAME directly -- the caller supplies it (GT landmark on OBP; a detector in
# deployment) -- exactly like stride_angle_2d. Full adoption evidence (LOCO / absacc
# / within-pitcher / independence at a pre-specified anchor) is in
# analysis/tier1_adoption_probe.py; summary in docs/legacy_pre_dedup/GT_EVENT_MAP_HANDOFF.md 6d.

def elbow_flexion_2d(df, arm, frame, J):
    """Elbow flexion (deg, 0 = straight) at a given frame: 180 - the included
    shoulder-elbow-wrist angle on the throwing arm.

    Truth = OBP elbow_flexion_mer (elbow angle at max external rotation, the
    "lay-back"). A genuinely new, injury-relevant axis: independent of the adopted
    map (est vs adopted-residual r2 = 0.83) because elbow flexion at lay-back is not
    a re-expression of any trunk / stride / arm-slot quantity. Adopted 2026-07-24 at
    the az330/el60 zone; DIRECT (raw CCC 0.84, LOCO CCC 0.94, MAE 3.95->2.15 deg).
    MER is a rotation event 2D cannot locate, which is why this column was untestable
    before the GT events -- see docs/legacy_pre_dedup/GT_EVENT_MAP_HANDOFF.md 6b."""
    a = arm[0]
    sx, sy = _xy(df, f"{a}_sh", J)
    ex, ey = _xy(df, f"{a}_el", J)
    wx, wy = _xy(df, f"{a}_wr", J)
    return float(180.0 - _angle(sx, sy, ex, ey, wx, wy)[int(frame)])


def shoulder_abduction_2d(df, side, frame, J):
    """Shoulder abduction (deg) at a given frame = the elbow-shoulder-hip angle on
    the requested side (the arm's elevation away from the trunk).

    Truth = OBP glove_shoulder_abduction_mer with side = the GLOVE (lead-leg) arm.
    Adopted 2026-07-24 at the az75/el15 zone; CALIBRATE (raw CCC 0.71 -> LOCO CCC
    0.90 with an offset fit, MAE 7.19->3.32 deg), independent (est vs adopted-
    residual r2 = 0.68). The glove-arm angle is not otherwise on the map."""
    s = "l" if side in ("left", "l", "glove") else "r"
    ex, ey = _xy(df, f"{s}_el", J)
    sx, sy = _xy(df, f"{s}_sh", J)
    hx, hy = _xy(df, f"{s}_hip", J)
    return float(_angle(ex, ey, sx, sy, hx, hy)[int(frame)])


def torso_rotation_2d(df, frame, J):
    """Trunk transverse rotation proxy (deg) at a given frame = the image-plane
    orientation of the shoulder line (right- minus left-shoulder), unwrapped.

    Truth = OBP torso_rotation_br (or _mer). A transverse quantity: degenerate on the
    ground (the shoulder line is ~horizontal for every rotation) and recovered from
    OVERHEAD, the same revival that unlocked HSS and pelvis rotation. Adopted
    2026-07-24 at az90/el85; CALIBRATE (raw CCC ~0.02 -- a large near-constant image-
    convention offset, like stride angle -- LOCO CCC 0.89, MAE ~89->3.4 deg). Partly
    overlaps the adopted transverse metrics (est vs adopted-residual r2 = 0.31), so it
    is a distinct named column but not a fully independent axis; read the caveat in
    docs/legacy_pre_dedup/GT_EVENT_MAP_HANDOFF.md 6d before treating it as new information. Best at
    az>=75 overhead; az0-45 at el85 is dead (shoulder line edge-on)."""
    rsx, rsy = _xy(df, "r_sh", J)
    lsx, lsy = _xy(df, "l_sh", J)
    ang = np.degrees(np.unwrap(np.arctan2(rsy - lsy, rsx - lsx)))
    return float(ang[int(frame)])


def trunk_lean_2d(df, frame, J):
    """Image-plane trunk lean (deg) at a given frame: orientation of the hip-midpoint
    -> shoulder-midpoint line vs vertical. Which anatomical lean it captures depends
    on the view -- sagittal (anterior) from the side, coronal (lateral) from the
    front -- so the SAME observable underlies both the adopted anterior-tilt metric
    (read at release from the side) and the lateral-tilt one below.

    Adopted 2026-07-24 as torso LATERAL tilt at MER (truth = OBP torso_lateral_tilt_mer,
    anchor az90/el30): a coronal quantity, anatomically distinct from the adopted
    anterior tilt, DIRECT (raw & LOCO CCC 0.91, MAE ~3.3 deg). Its low independence
    (est vs adopted-residual r2 = 0.24) means it adds little PREDICTIVE information
    over the existing map, but the paper's question is measurability, not
    independence: it is a distinct biomechanical axis that 2D recovers. See
    docs/legacy_pre_dedup/GT_EVENT_MAP_HANDOFF.md 6d."""
    lsx, lsy = _xy(df, "l_sh", J); rsx, rsy = _xy(df, "r_sh", J)
    lhx, lhy = _xy(df, "l_hip", J); rhx, rhy = _xy(df, "r_hip", J)
    msx, msy = (lsx + rsx) / 2, (lsy + rsy) / 2
    mhx, mhy = (lhx + rhx) / 2, (lhy + rhy) / 2
    return float(np.degrees(np.arctan2(msx - mhx, -(msy - mhy)))[int(frame)])


# ── HSS: hip-shoulder separation (overhead-only) ────────
# Adopted 2026-07-04 after OBP validation (now n=411, el>=85): the full recipe
# below reaches r2=0.62-0.63 vs max_rotation_hip_shoulder_separation —
# parity with the release-anchored recipe, without needing release/arm
# detection for the fine anchor. Degenerate from any ground-level camera.

def hss_sep_series(df, J):
    """Signed per-frame angle (deg) between the shoulder line and the hip
    line in the image plane. Scale-free (it is an angle). Only meaningful
    from elevated views (full ring at el=85; partial usable arcs from el=60,
    see angle_zone_table 2026-07-10); our 2D chord angle reads ~2x the 3D truth
    on OBP elite — never compare raw values to literature 3D numbers."""
    lsx, lsy = _xy(df, "l_sh", J); rsx, rsy = _xy(df, "r_sh", J)
    lhx, lhy = _xy(df, "l_hip", J); rhx, rhy = _xy(df, "r_hip", J)
    shx, shy = rsx - lsx, rsy - lsy
    hx, hy = rhx - lhx, rhy - lhy
    cross = shx * hy - shy * hx
    dot = shx * hx + shy * hy
    return np.degrees(np.arctan2(cross, dot))


def hss_chord_valid(df, J, min_frac=0.45):
    """Geometric validity mask for hss_sep_series: the angle between two
    chords is meaningless when either projected chord collapses (bent-over
    posture seen overhead puts shoulders/hips end-on; this — not landmark
    noise — caused every real-video false anchor). Valid = both chords
    >= min_frac of their clip median length. No-op on OBP (clean posture)."""
    lsx, lsy = _xy(df, "l_sh", J); rsx, rsy = _xy(df, "r_sh", J)
    lhx, lhy = _xy(df, "l_hip", J); rhx, rhy = _xy(df, "r_hip", J)
    sh = np.hypot(rsx - lsx, rsy - lsy)
    hp = np.hypot(rhx - lhx, rhy - lhy)
    return ((sh >= min_frac * np.nanmedian(sh)) &
            (hp >= min_frac * np.nanmedian(hp)))


def hss_transition(sep, fps, span_s=0.25, bounds=None, n_cand=8):
    """Release-free anchor: start frame and direction of the largest
    SUSTAINED sep swing (~span_s of motion time) that PERSISTS (a real whip
    leaves the trunk rotated for >= one more span; transient glitch humps
    revert to baseline). Single-frame diff is glitch-bait; heavy pre-filter
    blunts the 0.15-0.25 s elite dip (OBP r2 0.63 -> 0.58) — this candidate
    + persistence-check formulation keeps OBP r2 at 0.63.
    bounds=(lo,hi) restricts the search (coarse pitch localization)."""
    w = max(1, int(span_s * fps))
    if len(sep) <= w:
        return None, 0
    swing = sep[w:] - sep[:-w]
    lo, hi = 0, len(swing) - 1
    if bounds is not None:
        lo, hi = max(0, bounds[0]), min(len(swing) - 1, bounds[1])
        if hi <= lo:
            return None, 0
    order = lo + np.argsort(np.abs(swing[lo:hi + 1]))[::-1]
    cands = []
    for i in order:
        if all(abs(i - c) >= w for c in cands):
            cands.append(int(i))
        if len(cands) == n_cand:
            break
    for i in cands:
        s = np.sign(swing[i])
        tail = sep[i + w: min(len(sep), i + 2 * w)]
        if len(tail) == 0:
            continue
        if (np.nanmean(tail) - sep[i]) * s >= 0.5 * abs(swing[i]):
            return i, s
    # nothing persistent -> fall back to the raw largest swing
    return (cands[0], np.sign(swing[cands[0]])) if cands else (None, 0)


def hss_peak_overhead(df, fps, J, arm=None, slomo=1.0,
                      span_s=0.25, win_pre_s=0.45, medfilt_s=0.09):
    """Full adopted overhead-HSS recipe (OBP r2 0.629 at n=411). FULLY
    RELEASE-FREE: chord-validity gate -> 90 ms medfilt -> persistence-checked
    sustained-swing anchor (torso-only) -> windowed |max| over
    [t*-win_pre, t*+span/4] motion time. HSS peaks near foot plant, unrelated
    to release; release is NOT used (dropping the old +/-1.5 s wrist-speed
    localization is a verified no-op: identical OBP r2=0.63 and identical
    frames on all 4 real clips — the chord gate handles pre-pitch motion).
    `arm` is accepted for interface compatibility but unused.
    slomo = slow-motion factor of the video (4 for a 4x clip).
    Returns a dict incl. the filtered series for rendering, or None."""
    sep = hss_sep_series(df, J)
    valid = hss_chord_valid(df, J)
    k = max(3, int(medfilt_s * fps) // 2 * 2 + 1)
    sep_f = _medfilt(np.where(valid, np.nan_to_num(sep, nan=0.0), 0.0),
                     kernel_size=k)
    fps_m = fps * slomo
    t0, sgn = hss_transition(sep_f, fps_m, span_s)
    if t0 is None or sgn == 0:
        return None
    w = max(1, int(span_s * fps_m))
    lo = max(0, t0 - int(win_pre_s * fps_m))
    hi = min(len(sep_f) - 1, t0 + w // 4)
    # COIL-direction extremum, NOT sign-blind |max|: HSS max separation is the
    # coiled dip (opposite sign to the unwind swing sgn). |max| can grab an
    # in-window follow-through rebound (+) on clean/occlusion-refined coords
    # (verified: refined oh02 |max|=+22.8 rebound vs coil=-14.8). No-op on OBP
    # (coil == |max|, r2 unchanged; 0.629 at n=411) and on standard real clips
    # (13.2/18.0/11.7 unchanged); handedness-robust (sep and sgn co-flip).
    seg = sep_f[lo:hi + 1]
    pk = lo + int(np.nanargmax(-sgn * seg))
    return {"hss": float(abs(sep_f[pk])), "peak_f": int(pk),
            "anchor_f": int(t0),
            "window": (int(lo), int(hi)),
            "gated_frac": float(1.0 - valid.mean()),
            "sep_f": sep_f, "valid": valid}


def pelvis_rot_velo_overhead(df, fps, J, conf=0.4, chord_frac=0.5,
                             medfilt_s=0.042):
    """Pelvis transverse rotational velocity (deg/s) from an OVERHEAD 2D view.
    The hip-line yaw rate reproduces OBP max_pelvis_rotational_velo (r=0.90,
    n=411, el>=75): overhead looks straight at the transverse plane, so the axial
    rotation is seen face-on. DEGENERATE from side/front (the hip line does not
    sweep in-plane there) -- overhead only.

    On REAL overhead pose the hips break only during the brief release occlusion
    (arm/torso crossing over the pelvis) while the shoulders stay reliable, so
    this recovers them:
      1. resolve the hip-line 180 flip by CONTINUITY, re-anchored to the shoulder
         line -- only true ~180 jumps flip, not noise (a raw derivative explodes).
      2. gate occluded hip frames (low confidence if _v columns exist, OR a
         collapsed chord < chord_frac of median) and INTERPOLATE the resolved
         hip angle across them (bridges the occlusion).
      3. SG derivative -> |vel| -> medfilt to drop residual jitter spikes.
    Returns (rot_velo deg/s array, occluded-frame mask).

    Absolute scale on real video is pose-jitter-inflated ~2x (residual jitter on
    ~0.65-conf hips, NOT geometry -- verified: elevation/perspective give <=1.4x);
    the shape / timing / gate crossings are correct, so gate / sequencing /
    relative use are valid. See docs/MEASURABILITY_SURVEY.md."""
    lhx, lhy = _xy(df, "l_hip", J); rhx, rhy = _xy(df, "r_hip", J)
    lsx, lsy = _xy(df, "l_sh", J);  rsx, rsy = _xy(df, "r_sh", J)
    hip_raw = np.degrees(np.arctan2(rhy - lhy, rhx - lhx))
    sh = np.degrees(np.arctan2(rsy - lsy, rsx - lsx))
    chord = np.hypot(rhx - lhx, rhy - lhy)
    occl = chord < chord_frac * np.nanmedian(chord)
    lhv, rhv = f"{J['l_hip']}_v", f"{J['r_hip']}_v"
    if lhv in df.columns and rhv in df.columns:   # real video has scores; OBP proj. does not
        hv = np.minimum(df[lhv].to_numpy(float), df[rhv].to_numpy(float))
        occl = occl | (hv < conf)

    def _wrap180(a):
        return (a + 180.0) % 360.0 - 180.0

    n = len(hip_raw)
    resolved = np.full(n, np.nan)
    prev = None
    for t in range(n):
        if occl[t] or not np.isfinite(hip_raw[t]):
            continue
        c0, c1 = hip_raw[t], hip_raw[t] + 180.0
        anchor = prev if prev is not None else sh[t]
        best = c0 if abs(_wrap180(c0 - anchor)) <= abs(_wrap180(c1 - anchor)) else c1
        resolved[t] = prev + _wrap180(best - prev) if prev is not None else best
        prev = resolved[t]

    good = np.isfinite(resolved)
    if good.sum() < 3:
        return np.zeros(n), occl
    idx = np.arange(n)
    resolved = np.interp(idx, idx[good], resolved[good])   # bridge gaps, hold edges

    win = max(5, int(round(0.05 * fps)))
    win += (win % 2 == 0)
    win = min(win, n if n % 2 == 1 else n - 1)
    if win <= 3:
        vel = np.gradient(resolved) * fps
    else:
        vel = _savgol(resolved, win, 3, deriv=1, delta=1.0 / fps, mode="interp")
    k = max(3, int(medfilt_s * fps) // 2 * 2 + 1)
    return _medfilt(np.abs(vel), k), occl


# The estimators themselves.
def compute_candidates(df, fps=60, arm="right", view="side", raw_df=None,
                       height_m=None, rel=None, az=None, el=None):
    """One pitch of coordinates to {quantity: (value, tag)}.

    raw_df: pass UNSMOOTHED coordinates from video and the release frame is
    refined on them locally (see release_frame), after which every
    release-anchored quantity is read at that frame. Smoothing shifts an
    asymmetric speed peak by a frame, which is why the raw series is wanted for
    the instant even though the smoothed one is wanted for the value. The
    projected path calls without raw_df and is unaffected.
    height_m: subject standing height in meters. When provided, the adopted
    wrist_speed_abs (absolute m/s) is populated; otherwise it is NaN. A
    smartphone user enters this once (tape measure, no mocap needed).
    rel: optional externally detected release frame (plumbing only — the
    deployment path passes deploy/release_offset.corrected_release_frame's
    gated viewpoint-corrected frame). None = detect here as before.
    az, el: estimated camera viewpoint in degrees. When BOTH are given, the
    foot-plant detector is chosen per viewpoint by the fitted FP routing rule
    (deploy/fp_routing.fp_view, adopted 2026-07-27) instead of always using the
    side detector. This moves ONLY the foot-plant-dependent outputs — lead knee
    at fp / extension, stride length and %height, cog_velo_pkh (via pkh),
    release_ext, stride_angle — because `fp` feeds nothing else. Release-anchored
    outputs are untouched by construction. Leaving az/el as None reproduces the
    previous behaviour exactly, which is what every validation caller does."""
    J = JOINTS
    scale = body_scale_px(df, J)
    lead = "left" if arm == "right" else "right"   # the leg opposite the throwing arm
    if rel is None:
        rel = release_frame(df, arm, fps, J, view=view, raw_df=raw_df)
    rel = int(rel)
    # Foot-plant detector: routed by viewpoint when one is known, else the
    # incumbent side detector. Imported lazily — deploy/ imports this module, so a
    # top-level import would be circular.
    fp_strategy = "side"
    if az is not None and el is not None:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))
            from fp_routing import fp_view
            fp_strategy = fp_view(az, el)
        except Exception:
            fp_strategy = "side"
    fp = foot_plant_frame(df, lead, fps, J, rel, view=fp_strategy)
    out = {}

    # 1) Release height: the throwing wrist's y, turned into a height above the
    #    ankles and divided by the scale. Image y grows downward. CLEAN, it is a
    #    vertical quantity and a side view carries it whole.
    wkey = "r_wr" if arm == "right" else "l_wr"
    wx, wy = _xy(df, wkey, J)
    lanx, lany = _xy(df, "l_an", J); ranx, rany = _xy(df, "r_an", J)
    ground = np.nanmax(np.concatenate([lany, rany]))  # lowest ankle in the image
    out["release_height"] = ((ground - wy[rel]) / scale, CLEAN)

    # 2) Throwing-wrist peak speed (arm-speed proxy), validated vs 3D-direct
    #    wrist speed. Loss decomposition (2026-07-09, then n=408): geometry ceiling
    #    r²=0.93; dividing px speed by a body-relative scale is what costs r².
    #    ADOPTED = wrist_speed_abs: recover ABSOLUTE m/s = (px_peak / px_stature)
    #    * height_m. r²=0.82 @0 (vs 0.60 for /stature, 0.38 for /scale). Needs
    #    the subject's height; NaN when height_m is not supplied.
    #    The /stature and /scale outputs are kept as backward-compat aliases.
    wrist_pk = float(np.nanmax(_speed(wx, wy, fps)))
    _stat0 = pixel_stature(df, J)
    wrist_abs = (wrist_pk / _stat0) * height_m if height_m else np.nan
    out["wrist_speed_abs"] = (float(wrist_abs), CLEAN)       # ADOPTED (absolute m/s)
    out["wrist_speed"] = (wrist_pk / _stat0, CLEAN)          # alias, /stature, r2=0.60
    out["wrist_peak_speed"] = (wrist_pk / scale, CLEAN)      # alias, /scale, r2=0.38

    # 3) Lead knee angle at release: angle(hip, knee, ankle). A lead-leg block
    #    quantity. Validated at r2=0.94 at azimuth 0 against a 3D direct truth.
    #    Best from the side, collapses from the front. The ABSOLUTE angle beats
    #    the fp-to-release DIFFERENCE (lead_knee_extension below) by a wide
    #    margin, which is why the absolute angle is the adopted one.
    hk = "l_hip" if lead == "left" else "r_hip"
    kk = "l_kn"  if lead == "left" else "r_kn"
    ak = "l_an"  if lead == "left" else "r_an"
    hx, hy = _xy(df, hk, J); kx, ky = _xy(df, kk, J); ax, ay = _xy(df, ak, J)
    knee = _angle(hx, hy, kx, ky, ax, ay)
    out["lead_knee_angle"] = (float(knee[rel]), CLEAN)        # adopted: absolute angle at release
    out["lead_knee_at_release"] = (float(knee[rel]), CLEAN)   # backwards-compatible alias
    out["lead_knee_at_fp"] = (float(knee[fp]), CLEAN)
    out["lead_knee_extension"] = (float(knee[rel] - knee[fp]), CLEAN)  # reference: the fp->br difference

    # 3b) Lead knee extension angular velocity at release: the time derivative of
    #     the knee angle. Validated at r2=0.65 at azimuth 0 against an OBP column.
    #     A derivative, so it is noise-sensitive and assumes the SG smoothing is
    #     in place.
    kvel = np.gradient(knee) * fps
    out["knee_ext_velo_br"] = (float(kvel[rel]), CLEAN)

    # 4) Trunk anterior tilt at release: the angle between the hip-mid to
    #    shoulder-mid vector and the vertical. Validated against the release's
    #    torso_anterior_tilt_br at r2=0.70; against the LATERAL column it is 0.00,
    #    which is what settled the naming. What a side view carries is the forward
    #    lean, not the lateral one. CLEAN, best at azimuth 0.
    #    The sign is normalised to the pitch direction, positive leaning towards
    #    home, so a delivery that runs -x across the image does not come out
    #    mirrored. The direction is read from how the lead ankle travels.
    midhx = ((_xy(df,"l_hip",J)[0]+_xy(df,"r_hip",J)[0])/2)
    midhy = (_xy(df,"l_hip",J)[1]+_xy(df,"r_hip",J)[1])/2
    midsx = (_xy(df,"l_sh",J)[0]+_xy(df,"r_sh",J)[0])/2
    midsy = (_xy(df,"l_sh",J)[1]+_xy(df,"r_sh",J)[1])/2
    trunk_tilt = np.degrees(np.arctan2(midsx-midhx, -(midsy-midhy)))  # from vertical
    _d = ax[fp] - np.nanmedian(ax[:max(3, fp // 4)])   # which way the lead ankle went
    pitch_dir = 1.0 if _d >= 0 else -1.0
    out["trunk_anterior_tilt"] = (float(trunk_tilt[rel]) * pitch_dir, CLEAN)  # adopted
    out["lateral_trunk_tilt"] = (float(trunk_tilt[rel]) * pitch_dir, CLEAN)   # alias kept from before the renaming

    # 5) Arm slot at release: the angle between the shoulder-to-hand vector and
    #    the vertical (Escamilla & Fleisig 2018). Overhand is small, 0 to 40;
    #    three-quarters 50 to 60; sidearm above 70. It is a coronal-plane
    #    quantity, so it is most accurate from the front, near azimuth 90, and
    #    cannot be read from a pure side view.
    skey = "r_sh" if arm == "right" else "l_sh"
    shx, shy = _xy(df, skey, J)
    run  = abs(wx[rel] - shx[rel])          # horizontal component
    rise = (shy[rel] - wy[rel])             # vertical component; image y grows down, so a raised hand is positive
    arm_slot = np.degrees(np.arctan2(run, rise))
    out["arm_slot"] = (float(arm_slot), CLEAN)

    # 6) Stride: trail-foot rubber anchor -> lead ankle at fp, horizontal px.
    #    NOTE (2026-07-09): the validated r2=0.82 @0 (OBP-column) belongs to the
    #    /stature normalization = stride_pct_height (6b). This /body_scale
    #    output reads r2=0.59 — kept for backward compat; quote 0.82 only for
    #    stride_pct_height. Anchor recipe beats fixed-sides 0.44 / both-feet@fp.
    lead_ax = lanx if lead == "left" else ranx          # lead ankle x
    trail_ax = ranx if lead == "left" else lanx         # trail ankle x
    trail_anchor = trail_anchor_x(trail_ax, fp, fps)    # where it stood on the rubber
    stride_px = abs(lead_ax[fp] - trail_anchor)
    out["stride_length"] = (float(stride_px / scale), CLEAN)

    # 6b) Stride as a percentage of height: stride_px over pixel stature, which
    #     is how the release defines stride_length. A ratio of pixels, so it needs
    #     neither a stature input nor a known mound and is invariant to camera
    #     distance and zoom. Any calibration constant is applied downstream.
    #  DIVERGENCE, 2026-07-24 -- READ BEFORE TRUSTING THIS AGAINST A PAPER NUMBER.
    #  The VALIDATION path now reads the settled lead-ankle position over a
    #  release-anchored window (stride_settled_2d, r2 0.844) instead of the single
    #  foot-plant frame used here (0.823): the ankle is still travelling at fp.
    #  This deployment output is deliberately left on the old definition until the
    #  real-video layer (holdout-60 scoring, arc zones, measure_auto) is re-validated
    #  against the new one -- switching it silently would invalidate those results.
    #  Do NOT quote the 0.844 figure for this output. Follow-up is tracked in
    #  docs/legacy_pre_dedup/GT_EVENT_MAP_HANDOFF.md 6.
    stat = pixel_stature(df, J)
    out["stride_pct_height"] = (float(stride_px / stat), CLEAN)

    # 6) Hip-shoulder separation, as a width proxy only. It is a rotation about
    #    the vertical, so a ground-level camera carries no component of it and it
    #    is flagged DEGENERATE. The overhead estimator is hss_peak_overhead.
    lsx, lsy = _xy(df, "l_sh", J); rsx, rsy = _xy(df, "r_sh", J)
    lhx, lhy = _xy(df, "l_hip", J); rhx, rhy = _xy(df, "r_hip", J)
    sw = np.abs(rsx-lsx); hw = np.abs(rhx-lhx)
    out["hss_width_proxy"] = (float(np.nanmax(np.abs(sw-hw))) / scale, DEGEN)

    # 7) COG velocity (Winter whole-body COM), side/ground point metrics (m/s).
    #    Both need the subject height (like wrist_speed_abs) -> NaN without it.
    #    cog_fwd_velo uses [0, rel]; cog_velo_pkh is release-FREE (peak-knee-
    #    height event only). Shared helpers = single source of truth with the
    #    validation estimators. Adopted 2026-07-18 (r2~0.75 / ~0.80 @side).
    pkh = peak_knee_height_frame(df, lead, fp, J)
    out["cog_fwd_velo"] = (float(cog_fwd_velo(df, fps, rel, J, height_m))
                           if height_m else np.nan, CLEAN)
    out["cog_velo_pkh"] = (float(cog_velo_at_pkh(df, fps, pkh, J, height_m))
                           if height_m else np.nan, CLEAN)

    # 8) Release extension (side/ground). Returns NaN when the delivery has no
    #    still setup -- the trail-foot anchor needs a pre-motion quiet window, so
    #    walk-in deliveries DEFER rather than report a silently bad anchor. The
    #    rubber itself is never required (or detected). Adopted 2026-07-20.
    out["release_ext"] = (float(release_extension(df, arm, rel, fp, fps, J,
                                                  height_m))
                          if height_m else np.nan, CLEAN)

    # 8) The 2026-07-24 adoptions that need no GT event. All four were in the map
    #    but not in this candidate set, which is why the deployment driver
    #    reported 11 of the map's 15 (analysis/deployability_audit.py).
    #    Definitions are imported, never re-implemented (CLAUDE.md).
    #
    #    stride angle reads the ankle line AT FOOT PLANT, and from the front the
    #    side detector cannot fire (the stride runs along the camera axis), so it
    #    uses the frontal detector when the view is frontal.
    #    2026-07-27: when az/el are supplied, `fp` above is ALREADY routed by the
    #    fitted FP rule, so this fallback is only for the older view= path. The
    #    earlier note that `fp` is "deliberately left alone" no longer holds --
    #    routing it is the adopted policy, and stride length / release extension
    #    move with it on purpose.
    fp_angle = fp
    if az is None and _is_frontal(view):
        try:
            # |v|+guard speed detector, adopted 2026-07-26 (delivery-invariant).
            fp_angle = foot_plant_frame_front_speed(df, lead, fps, J, rel)
        except Exception:
            fp_angle = fp
    out["stride_angle"] = (float(stride_angle_2d(df, arm, fp_angle, J)), CLEAN)
    out["torso_rot_br"] = (float(torso_rotation_2d(df, rel, J)), CLEAN)

    #    MER is a rotation instant 2D cannot locate, but it sits 11 frames before
    #    release at 360 Hz with SD 1.1 (scratch/mer_timing_probe.py). Metrics that
    #    are FLAT in time there ride that proxy: trunk lateral tilt LOCO CCC 0.884,
    #    glove abduction 0.891 (analysis/mer_proxy_score.py). Elbow flexion is NOT
    #    flat (4.32 deg/frame) and is deliberately absent here -- it stays a
    #    GT-only, measurable-but-not-deployable metric (GT_EVENT_MAP_HANDOFF 6g).
    mer_px = int(round(rel - (11.0 / 360.0) * fps))
    if mer_px < 0:
        mer_px = 0
    out["torso_lat_tilt_mer"] = (float(trunk_lean_2d(df, mer_px, J)), CLEAN)
    out["glove_sh_abd_mer"] = (float(shoulder_abduction_2d(df, "glove", mer_px,
                                                           J)), CLEAN)

    return out


if __name__ == "__main__":
    # A synthetic full-body delivery, enough to exercise every estimator.
    np.random.seed(1); N=90; cx,cy=320,240
    t=np.linspace(0,1,N)
    def col(base, x, y): 
        return {f"{base}_x": x+np.random.normal(0,1,N), f"{base}_y": y+np.random.normal(0,1,N)}
    import pandas as pd
    data={}
    data.update(col("left_shoulder",  cx-35, cy-60))
    data.update(col("right_shoulder", cx+35, cy-60))
    data.update(col("left_hip",  cx-25, cy+10))
    data.update(col("right_hip", cx+25, cy+10))
    data.update(col("left_knee",  cx-30, cy+70))
    data.update(col("right_knee", cx+20+40*t, cy+70))   # lead leg travelling
    data.update(col("left_ankle",  cx-30, cy+130))
    data.update(col("right_ankle", cx+20+80*t, cy+130)) # the stride
    wy=cy-60-40*np.sin(np.pi*t)                          # wrist accelerating through an arc
    data.update(col("right_wrist", cx+30+60*t, wy))
    data.update(col("left_wrist",  cx-40, cy+20))
    data.update(col("right_elbow", cx+30, cy-30)); data.update(col("left_elbow", cx-30, cy-30))
    df=pd.DataFrame(data)
    res=compute_candidates(df, fps=60, arm="right")
    print("quantity                value     tag")
    print("-"*46)
    for k,(v,c) in res.items():
        print(f"{k:24s}{v:8.2f}   {c}")


# Static projection-reliability tag per quantity, imported by the analysers and
# the interface. The r-squared in each note is the value measured against the
# OpenBiomechanics gold standard; (O) means the truth was an OBP column and (D)
# that it was computed directly on the 3D markers.
CANDIDATE_GRADES = {
    # Validated, adopted quantities.
    "lead_knee_angle":      CLEAN,   # r2=0.94 (D) @0deg, absolute knee angle at release
    "stride_length":        CLEAN,   # r²=0.59 (O) @0°  /body_scale (compat only)
    "stride_pct_height":    CLEAN,   # r²=0.82 (O) @0°  /stature — ADOPTED headline
    "trunk_anterior_tilt":  CLEAN,   # r2=0.70 (O) @0deg, forward lean
    "knee_ext_velo_br":     CLEAN,   # r2=0.65 (O) @0deg, angular velocity at
                                     # release, assumes the smoothing. The 3D
                                     # ceiling is 0.67, so this is effectively the
                                     # limit rather than an implementation gap.
    "wrist_speed_abs":      CLEAN,   # r²=0.82 (D) @0°  ABSOLUTE m/s (needs height_m)
    "arm_slot":             CLEAN,   # best from the front (90deg); see armslot_validate
    "release_height":       CLEAN,   # r2=0.999 (D) @0deg, remeasured 2026-07-03
                                     # by angle_map_2d; the earlier 0.51 is stale.
                                     # Near 1.0 because it is almost purely
                                     # vertical, so a side projection loses none of
                                     # it, and elite stature varies little, so the
                                     # stature normalisation is nearly a constant.
                                     # Holds at el=0 only; it collapses above it.
    # Backwards-compatible aliases; the values are the same or closely related.
    "wrist_speed":          CLEAN,   # wrist_speed_abs divided by stature (r2=0.60)
    "wrist_peak_speed":     CLEAN,   # wrist_speed divided by scale (r2=0.38)
    "lead_knee_at_release": CLEAN,   # = lead_knee_angle
    "lead_knee_at_fp":      CLEAN,
    "lead_knee_extension":  CLEAN,   # reference: the fp->br difference, r2=0.62
    "lateral_trunk_tilt":   CLEAN,   # = trunk_anterior_tilt, kept from before the renaming
    # Not measurable from a ground-level camera: rotations about the vertical.
    "hss_width_proxy":      DEGEN,
    # Overhead only. A full ring at el=85; a partial arc between el 60 and 75,
    # for which angle_zone_table gives the extent.
    "hss_peak_overhead":    CLEAN,   # r²=0.63 (O) @el=85  signature anchor recipe
}