"""OBP ground-truth PEAK FRAMES for the window-extremum poi columns.

Sibling of obp_gt_events.py. That module gives the six landmark EVENTS; this one
gives the instant at which each `max_*` poi column actually attains its value.

Why this exists (2026-07-27). Task 5 of the event rebuild asks whether a 2D argmax
lands on the same extremum the 3D signal has. Answering it needs the truth's own
peak FRAME, and the earlier HSS attempt used a marker-line transverse-angle proxy
instead -- which is precisely why its result was uninterpretable
(docs/legacy_pre_dedup/EVENT_SYSTEM_HANDOFF_2026-07-27.md 5). OBP ships the underlying series in
full_sig, so the truth frame can be had exactly.

THE METHOD IS VALUE-MATCHING, NOT WINDOW RECONSTRUCTION. The first attempt tried to
reproduce each poi scalar by brute-forcing (axis x window) over the series, which
pinned the CHANNEL for all five columns but only reproduced the value exactly for
three of them -- `max_elbow_flexion` reached 92.5 % and
`lead_knee_extension_angular_velo_max` stalled at 52 %, and the handoff recorded the
latter as unresolved. Reconstructing the window is unnecessary: locating the frame
whose series value EQUALS the poi scalar recovers the instant directly, and it
succeeds for **all five columns on all 411 pitches** (|poi - series| < 0.01 in the
column's own units). The unresolved row is therefore resolved.

    column                                  channel                 exact  unique
    max_pelvis_rotational_velo              pelvis_velo_z           1.000   1.000
    max_torso_rotational_velo               torso_velo_z            1.000   1.000
    max_rotation_hip_shoulder_separation    |torso_pelvis_angle_z|  1.000   0.983
    max_elbow_flexion                       elbow_angle_x           1.000   0.866
    lead_knee_extension_angular_velo_max    lead_knee_velo_x        1.000   0.998

`unique` = the frames within tolerance of the match form ONE contiguous run, i.e.
the truth instant is a single peak rather than a plateau. Elbow flexion is the one
that plateaus (0.866): near max external rotation the elbow angle is flat, so its
truth instant carries a few frames of its own ambiguity. `plateau` (the width of
that run, in frames) is returned per pitch so a frame gap can be scored against the
truth's own tolerance instead of against a point.

So `z` is the transverse/axial axis for pelvis, torso and torso-pelvis, and
`torso_pelvis_angle_z` IS hip-shoulder separation -- no reconstruction needed.

NOT COVERED, and do not fake it:
  - `max_cog_velo_x` has no COM channel in full_sig at all. A 3D Winter COM
    reproduces it at only r2 0.71, so it is a proxy, not the truth -- callers that
    want a COG peak must label it as the 3D counterpart of OUR definition, never as
    OBP's instant.
  - Wrist speed's truth is 3D-direct by adoption (master_angle_table.t3_wrist), so
    its peak frame comes from the c3d series and needs nothing from here.

Cached to OBP_VALIDATION_DIR/obp_gt_peaks.csv because the two full_sig zips take
~40 s to parse; pass refresh=True to rebuild.

    from obp_gt_peaks import load_gt_peaks
    pk = load_gt_peaks()               # {session_pitch: {column: frame}}
    pk_full = load_gt_peaks(full=True) # DataFrame incl. err / unique / plateau
"""
import os, sys, zipfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# poi column -> (full_sig file, channel, transform)
SPEC = {
    "max_pelvis_rotational_velo":           ("joint_velos",  "pelvis_velo_z",        None),
    "max_torso_rotational_velo":            ("joint_velos",  "torso_velo_z",         None),
    "max_rotation_hip_shoulder_separation": ("joint_angles", "torso_pelvis_angle_z", "abs"),
    "max_elbow_flexion":                    ("joint_angles", "elbow_angle_x",        None),
    "lead_knee_extension_angular_velo_max": ("joint_velos",  "lead_knee_velo_x",     None),
}
TOL = 0.01
CACHE = os.path.join(config.OBP_VALIDATION_DIR, "obp_gt_peaks.csv")


def _load_sig(name):
    zp = os.path.join(config.OBP_DATA_DIR, "full_sig", f"{name}.zip")
    with zipfile.ZipFile(zp) as z:
        with z.open(f"{name}.csv") as f:
            return pd.read_csv(f)


def build():
    """Locate every column's peak frame by value-matching. Returns a long DataFrame
    with one row per (session_pitch, column)."""
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    need = sorted({s[0] for s in SPEC.values()})
    grp = {n: {sp: g for sp, g in _load_sig(n).groupby("session_pitch", sort=False)}
           for n in need}

    recs = []
    for col, (src, ch, tf) in SPEC.items():
        if col not in poi.columns:
            continue
        for sp, g in grp[src].items():
            if sp not in poi.index:
                continue
            t = float(poi.loc[sp, col])
            s = g[ch].to_numpy(float)
            if tf == "abs":
                s = np.abs(s)
            d = np.abs(s - t)
            if not np.isfinite(d).any():
                continue
            k = int(np.nanargmin(d))
            # the truth's own ambiguity: frames indistinguishable from the match
            near = np.flatnonzero(d < max(TOL, 5 * d[k]))
            contiguous = bool(near.size and
                              (near.max() - near.min()) == near.size - 1)
            recs.append(dict(session_pitch=sp, column=col, frame=k,
                             value=float(s[k]), truth=t, err=float(d[k]),
                             unique=contiguous, plateau=int(near.size),
                             n_frames=len(s)))
    return pd.DataFrame(recs)


def load_gt_peaks(full=False, refresh=False):
    """{session_pitch: {column: frame}}, or the full DataFrame when full=True.

    Only matches within TOL are returned in the dict form -- a column that failed
    to value-match is absent rather than silently wrong."""
    if refresh or not os.path.exists(CACHE):
        d = build()
        d.to_csv(CACHE, index=False, float_format="%.6g")
    else:
        d = pd.read_csv(CACHE)
    if full:
        return d
    out = {}
    for r in d[d.err < TOL].itertuples(index=False):
        out.setdefault(r.session_pitch, {})[r.column] = int(r.frame)
    return out


if __name__ == "__main__":
    d = load_gt_peaks(full=True, refresh="--refresh" in sys.argv)
    print(f"cache -> {CACHE}\n")
    hdr = (f"{'poi column':<40}{'n':>5}{'exact':>8}{'unique':>8}"
           f"{'plateau p90':>13}")
    print(hdr); print("-" * len(hdr))
    for col, g in d.groupby("column"):
        print(f"{col:<40}{len(g):>5}{float((g.err < TOL).mean()):>8.3f}"
              f"{float(g.unique.mean()):>8.3f}{g.plateau.quantile(.9):>13.0f}")
    print(f"\nexact = |poi - series| < {TOL} in the column's own units")
    print("unique = the tolerance-matching frames form one contiguous run")
    print("plateau = width of that run, in frames @360Hz (the truth's own "
          "instant ambiguity)")
