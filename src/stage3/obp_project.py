"""Project an OpenBiomechanics pitch to a virtual 2D camera.

This is where a 3D marker recording becomes something a 2D estimator can read. The
c3d markers are reduced to joint centres, the joint centres are projected to one
(azimuth, elevation) of the sweep, and the projected coordinates go to
metrics.compute_candidates in the same schema the pose extractor writes from video.
The same estimator therefore reads a projected pitch and a real recording.

World axes, verified against the release rather than assumed: X is the pitch
direction, Z is vertical, Y is the remaining horizontal axis, which is what a side
camera looks along.
"""
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
import metrics as M   # stage2/metrics.py

import ezc3d

# Marker names to joint centres. A joint centre is the centroid of the markers
# listed, so a two-marker entry is a midpoint and a one-marker entry is that marker.
MARKER_MAP = {
    "left_shoulder": ["LSHO"], "right_shoulder": ["RSHO"],
    "left_hip": ["LASI", "LPSI"], "right_hip": ["RASI", "RPSI"],
    "left_knee": ["LKNE", "LMKNE"], "right_knee": ["RKNE", "RMKNE"],
    "left_ankle": ["LANK", "LMANK"], "right_ankle": ["RANK", "RMANK"],
    "left_elbow": ["LELB", "LMELB"], "right_elbow": ["RELB", "RMELB"],
    "left_wrist": ["LWRA", "LWRB"], "right_wrist": ["RWRA", "RWRB"],
    "head": ["LFHD", "RFHD", "LBHD", "RBHD"],
}


def load_c3d_joints(path):
    """c3d -> (joints3d: {name:(3,nframes)}, fps)"""
    c = ezc3d.c3d(path)
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    fps = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    pts = c["data"]["points"][:3]            # (3, nmarker, nframes), metres
    idx = {l: i for i, l in enumerate(labels)}

    def col(mk):
        # tolerate a namespace prefix such as "Skeleton_001_LSHO" (user 1017)
        if mk in idx:
            return idx[mk]
        hits = [l for l in labels if l.endswith("_" + mk) or l.endswith(":" + mk)]
        return idx[hits[0]] if len(hits) == 1 else None

    joints = {}
    for name, mks in MARKER_MAP.items():
        cs = [col(m) for m in mks]
        cs = [ci for ci in cs if ci is not None]
        if not cs:
            raise KeyError(f"marker not in this recording: {name}")
        joints[name] = np.nanmean([pts[:, ci, :] for ci in cs], axis=0)
    # Handedness normalization (adopted 2026-07-23): reflect every LHP into an
    # equivalent RHP so the azimuth frame is open-side-relative (az0 = open side
    # for all). Unifies the map, accuracy layers and the viewpoint classifier.
    # detect_throwing_arm on the returned joints always reports "right"; read
    # true handedness from poi p_throws. Set DIAMOND_NO_REFLECT=1 to disable
    # (diagnostics only). See obp_project.reflect_to_rhp.
    if os.environ.get("DIAMOND_NO_REFLECT") != "1" \
            and detect_throwing_arm(joints, fps) == "left":
        joints = reflect_to_rhp(joints)
    return joints, fps


def detect_throwing_arm(joints, fps):
    """The throwing arm is the wrist that reaches the higher peak speed in X-Z.

    On joints loaded through this module it always answers `right', because
    left-handed deliveries are reflected at load time. True handedness comes from
    the release's p_throws field, never from here.
    """
    def pk(w):
        v = np.linalg.norm(np.diff(w[[0, 2]], axis=1), axis=0)
        return np.nanmax(v)
    return "right" if pk(joints["right_wrist"]) >= pk(joints["left_wrist"]) else "left"


def reflect_to_rhp(joints):
    """Mirror a LHP delivery into an equivalent RHP by reflecting across the
    pitch sagittal plane. Handedness normalization so the azimuth map is
    open-side-relative (az0 = open side for every pitcher); without it, pooling
    a LHP at field azimuth mixes its glove-side view with RHP open-side views
    and corrupts every off-cardinal cell (cardinal cells are self-symmetric and
    unaffected). At az0 the side view has image-x = world X and image-y = world
    Z, so world Y is the lateral axis. The reflection negates world Y and swaps
    left_* <-> right_* labels. Idempotent only in pairs, so call exactly once
    (the analysis loader applies it for LHP right after loading)."""
    out = {}
    for name, arr in joints.items():
        a = arr.copy()
        a[1] = -a[1]
        if name.startswith("left_"):
            out["right_" + name[len("left_"):]] = a
        elif name.startswith("right_"):
            out["left_" + name[len("right_"):]] = a
        else:
            out[name] = a
    return out


def project_view(joints, azimuth_deg=0.0, elevation_deg=0.0, ppm=300.0,
                 perspective=False, cam_dist=8.0, noise_px=0.0, seed=0):
    """3D joint centres to 2D image points, for a camera anywhere on the sweep.

    The camera is rotated by `azimuth' in the horizontal plane and raised by
    `elevation'. The projection is orthographic, so a joint's image position does
    not depend on its distance from the camera.

    OFFICIAL az convention (user decision 2026-07-15; physically anchored by
    scratch/obp_axis_probe.py depth tests on 3D-confirmed RHP/LHP pitches.
    az increases 3B -> home; the original code had az increasing 3B -> 2B,
    i.e. the mirror, and its docstring wrongly called az90 "front"):
      azimuth=0   : SIDE view from 3B (RHP OPEN side; throwing arm
                    near-camera at release), image u runs along X
      azimuth=90  : FRONT view from home/catcher (arm-slot view)
      azimuth=180 : side view from 1B (RHP glove/closed side)
      azimuth=270 : REAR view from 2B (pitcher's back)
      in between  : an oblique view
      elevation   : how far the camera is raised, measured from the ground
    Image axes:
      u = the camera's right, the horizontal axis rotated by azimuth
      v = the camera's up, world Z carried through the elevation
    NOTE: projection at az under this convention == pre-2026-07-15
    projection at (360-az) EXACTLY (view_dir x-sign flip only), so every
    stored OBP table was relabeled az -> (360-az)%360, not recomputed.
    """
    rng = np.random.default_rng(seed)
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    # The viewing direction: azimuth in the horizontal plane, elevation above it.
    # Azimuth increases from third base towards home, per the convention above.
    view_dir = np.array([-np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)])
    world_up = np.array([0, 0, 1.0])
    right = np.cross(view_dir, world_up); right /= (np.linalg.norm(right)+1e-9)
    up = np.cross(right, view_dir); up /= (np.linalg.norm(up)+1e-9)

    allZ = np.concatenate([j[2] for j in joints.values()])
    z_top = np.nanmax(allZ)
    # el != 0: image-up reference must be GLOBAL, shared by all joints. Using a
    # per-joint vv.max() offsets each joint independently and corrupts inter-joint
    # geometry at elevation (verified: pelvis rot-velo gate collapses 0.90 -> 0.56,
    # scale -34%). Mirror the el==0 path, which already uses a global z_top.
    if abs(el) < 1e-6:
        v_top = None
    else:
        v_top = np.nanmax(np.concatenate([(j.T @ up) for j in joints.values()]))
    # Depth is measured from the subject's centroid, so it is centred on the body
    # rather than on the capture volume's origin.
    center = np.nanmean(np.concatenate([j for j in joints.values()], axis=1), axis=1)
    cols = {}
    for name, j in joints.items():
        P = j.T  # (nframes,3)
        u = P @ right
        vv = P @ up
        v = (z_top - (P @ world_up)) if abs(el) < 1e-6 else (v_top - vv)
        if perspective:
            depth = (P - center) @ view_dir
            f = cam_dist / np.clip(cam_dist - depth, 0.5, None)
            u = u * f; v = v * f
        u = u * ppm; v = v * ppm
        if noise_px > 0:
            u = u + rng.normal(0, noise_px, u.shape)
            v = v + rng.normal(0, noise_px, v.shape)
        cols[f"{name}_x"] = u
        cols[f"{name}_y"] = v
    return pd.DataFrame(cols)


# backwards-compatible wrapper
def project_lateral(joints, ppm=300.0, perspective=False, cam_y=6.0,
                    noise_px=0.0, seed=0, view="lateral"):
    az = 90.0 if view == "frontal" else 0.0
    return project_view(joints, azimuth_deg=az, ppm=ppm,
                        perspective=perspective, noise_px=noise_px, seed=seed)


def process_file(path, perspective=False, noise_px=0.0, view="lateral", height_m=None):
    joints, fps = load_c3d_joints(path)
    arm = detect_throwing_arm(joints, fps)
    df2d = project_lateral(joints, perspective=perspective, noise_px=noise_px, view=view)
    cand = M.compute_candidates(df2d, fps=fps, arm=arm, height_m=height_m)
    return arm, fps, {k: v for k, (v, _) in cand.items()}


# Batch: every c3d named in metadata.csv, one feature row each.
def run_batch(obp_data_dir, out_csv, perspective=False, noise_px=0.0, limit=None, view="lateral"):
    md = pd.read_csv(os.path.join(obp_data_dir, "metadata.csv"))
    c3d_root = os.path.join(obp_data_dir, "c3d")
    rows, fail = [], 0
    items = md.itertuples(index=False)
    for i, r in enumerate(items):
        if limit and i >= limit:
            break
        user_dir = f"{int(r.user):06d}"
        path = os.path.join(c3d_root, user_dir, r.filename_new)
        if not os.path.exists(path):
            fail += 1; continue
        height_m = getattr(r, "session_height_m", None)   # for wrist_speed_abs
        try:
            arm, fps, feats = process_file(path, perspective, noise_px, view, height_m)
        except Exception as e:
            fail += 1; continue
        row = {"session_pitch": r.session_pitch, "throws": arm}
        row.update(feats)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"done {len(out)}, failed {fail}, view={view} -> {out_csv}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3d", help="one c3d path; only that file is processed")
    ap.add_argument("--obp", default=config.OBP_DATA_DIR,
                    help="OBP data folder, holding metadata.csv and c3d/")
    ap.add_argument("--out", default=os.path.join(config.OBP_VALIDATION_DIR,
                    "candidate_features_obp.csv"))
    ap.add_argument("--perspective", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0, help="keypoint noise to add, in pixels")
    ap.add_argument("--view", choices=["lateral", "frontal"], default="lateral")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    if a.c3d:
        arm, fps, feats = process_file(a.c3d, a.perspective, a.noise, a.view)
        print(f"throwing_arm={arm}  fps={fps}  view={a.view}")
        for k, v in feats.items():
            print(f"  {k:24s} {v:8.3f}")
    else:
        run_batch(a.obp, a.out, a.perspective, a.noise, a.limit, a.view)