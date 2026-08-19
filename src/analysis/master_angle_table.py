"""
Master table of every quantity against camera azimuth.

The adopted quantities and the ones tried and rejected, measured under one
protocol at seven camera azimuths from 0 to 90 degrees, giving an r-squared
table. It is what shows on one page why 0 and 90 were adopted and what was
rejected and why.

The truth basis is mixed and is marked per row:
  OBP-column  compared against a column of poi_metrics.csv
  3D direct   computed directly from the c3d 3D coordinates
Both use the same azimuth projection and the same event detection from
metrics.py.

Rows are quantities, adopted or rejected; columns are azimuths. An adopted row
is marked [O] and a rejected one [x].

Usage:
    python master_angle_table.py --limit 60
    python master_angle_table.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(_HERE, "..", "stage3"))

import config
import obp_project as O
import metrics as M

AZ = [0, 15, 30, 45, 60, 75, 90]
HEEL = {"left_heel": ["LHEE"], "right_heel": ["RHEE"]}


def _marker_col(labels, idx, mk):
    """Resolve one marker label to its column index.

    A few sessions namespace every label (for example "Skeleton_001_LSHO"
    instead of "LSHO"), which otherwise drops the whole pitch. Fall back to a
    suffix match, and only accept it when it is unambiguous.
    """
    if mk in idx:
        return idx[mk]
    hits = [l for l in labels if l.endswith("_" + mk) or l.endswith(":" + mk)]
    return idx[hits[0]] if len(hits) == 1 else None


def load_feet(path):
    import ezc3d
    c = ezc3d.c3d(path)
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    fps = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    pts = c["data"]["points"][:3]
    idx = {l: i for i, l in enumerate(labels)}
    j = {}
    for name, mks in {**O.MARKER_MAP, **HEEL}.items():
        cols = [_marker_col(labels, idx, m) for m in mks]
        cols = [ci for ci in cols if ci is not None]
        if not cols:
            if name in HEEL:
                continue
            raise KeyError(name)
        j[name] = np.nanmean([pts[:, ci, :] for ci in cols], axis=0)
    # Handedness normalization (adopted 2026-07-23, see obp_project.reflect_to_rhp):
    # reflect every LHP into an equivalent RHP so the azimuth frame is
    # open-side-relative. Unified with obp_project.load_c3d_joints (independent
    # loaders, no double reflection). DIAMOND_NO_REFLECT=1 disables (diagnostics).
    if os.environ.get("DIAMOND_NO_REFLECT") != "1" \
            and O.detect_throwing_arm(j, fps) == "left":
        j = O.reflect_to_rhp(j)
    return j, fps


def ang3d(a, b, c):
    v1, v2 = a - b, c - b
    cs = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cs, -1, 1))))


def _xy(df, j):
    return df[f"{j}_x"].to_numpy(float), df[f"{j}_y"].to_numpy(float)


# One row is (label, adopted?, truth basis, estimator(df, az, ctx) -> value,
# truth key). The truth key is either a poi column name or a ("3d", fn) pair.
def build_rows():
    def est_knee_abs(df, ctx):
        l = ctx["lead"]; r = ctx["rel"]
        hx, hy = _xy(df, f"{l}_hip"); kx, ky = _xy(df, f"{l}_knee"); ax, ay = _xy(df, f"{l}_ankle")
        return M._angle(hx, hy, kx, ky, ax, ay)[r]

    def est_knee_diff(df, ctx):
        l = ctx["lead"]; r = ctx["rel"]; f = ctx["fp"]
        hx, hy = _xy(df, f"{l}_hip"); kx, ky = _xy(df, f"{l}_knee"); ax, ay = _xy(df, f"{l}_ankle")
        k = M._angle(hx, hy, kx, ky, ax, ay)
        return k[r] - k[f]

    def est_knee_velo(df, ctx):
        l = ctx["lead"]; r = ctx["rel"]
        hx, hy = _xy(df, f"{l}_hip"); kx, ky = _xy(df, f"{l}_knee"); ax, ay = _xy(df, f"{l}_ankle")
        k = M._angle(hx, hy, kx, ky, ax, ay)
        return (np.gradient(k) * ctx["fps"])[r]

    def est_trunk(df, ctx):
        r = ctx["rel"]
        mhx = (_xy(df, "left_hip")[0] + _xy(df, "right_hip")[0]) / 2
        mhy = (_xy(df, "left_hip")[1] + _xy(df, "right_hip")[1]) / 2
        msx = (_xy(df, "left_shoulder")[0] + _xy(df, "right_shoulder")[0]) / 2
        msy = (_xy(df, "left_shoulder")[1] + _xy(df, "right_shoulder")[1]) / 2
        return np.degrees(np.arctan2(msx - mhx, -(msy - mhy)))[r]

    def est_stride_anchor(df, ctx):
        """Settled-position stride (metrics.stride_settled_2d, adopted 2026-07-24).
        Release-anchored: no foot-plant input. Supersedes est_stride_at_fp below."""
        return M.stride_settled_2d(df, ctx["lead"], ctx["rel"], ctx["fps"], M.JOINTS)

    def est_stride_at_fp(df, ctx):
        """SUPERSEDED definition, kept as a rejected row so the change stays visible:
        the lead-ankle position at the detected foot plant. It reads a frame at which
        the ankle is still travelling, which is why it collapses (0.823 -> 0.391) when
        handed the OBP fp_100 landmark instead of our plateau-seeking detector."""
        l, t = ctx["lead"], ctx["trail"]; f = ctx["fp"]
        la = _xy(df, f"{l}_ankle")[0]; ta = _xy(df, f"{t}_ankle")[0]
        anc = M.trail_anchor_x(ta, f, ctx["fps"])
        return abs(la[f] - anc) / M.pixel_stature(df, M.JOINTS)

    def est_stride_fp(df, ctx):
        l, t = ctx["lead"], ctx["trail"]; f = ctx["fp"]
        la = _xy(df, f"{l}_ankle")[0]; ta = _xy(df, f"{t}_ankle")[0]
        return abs(la[f] - ta[f]) / M.pixel_stature(df, M.JOINTS)

    def est_wrist(df, ctx):
        a = ctx["arm"]
        wx, wy = _xy(df, f"{a}_wrist")
        return float(np.nanmax(M._speed(wx, wy, ctx["fps"]))) / M.pixel_stature(df, M.JOINTS)

    def est_relh(df, ctx):
        a = ctx["arm"]; r = ctx["rel"]
        wy = _xy(df, f"{a}_wrist")[1]
        g = np.nanmax(np.concatenate([_xy(df, "left_ankle")[1], _xy(df, "right_ankle")[1]]))
        return (g - wy[r]) / M.body_scale_px(df, M.JOINTS)

    def est_armslot(df, ctx):
        a = ctx["arm"]; r = ctx["rel"]
        sx, sy = _xy(df, f"{a}_shoulder"); wx, wy = _xy(df, f"{a}_wrist")
        return float(np.degrees(np.arctan2(abs(wx[r] - sx[r]), (sy[r] - wy[r]))))

    def est_elbow_fp(df, ctx):
        a = ctx["arm"]; f = ctx["fp"]
        sx, sy = _xy(df, f"{a}_shoulder"); ex, ey = _xy(df, f"{a}_elbow"); wx, wy = _xy(df, f"{a}_wrist")
        return (180.0 - M._angle(sx, sy, ex, ey, wx, wy))[f]

    def est_knee_toe(df, ctx):
        l = ctx["lead"]; r = ctx["rel"]; f = ctx["fp"]
        if f"{l}_toe_x" not in df.columns:
            return np.nan
        hx, hy = _xy(df, f"{l}_hip"); kx, ky = _xy(df, f"{l}_knee"); tx, ty = _xy(df, f"{l}_toe")
        k = M._angle(hx, hy, kx, ky, tx, ty)
        return k[r] - k[f]

    # 3D truth fns
    def t3_knee_abs(j, ctx):
        l = ctx["lead"]; r = ctx["rel"]
        return ang3d(j[f"{l}_hip"][:, r], j[f"{l}_knee"][:, r], j[f"{l}_ankle"][:, r])

    def t3_knee_diff(j, ctx):
        l = ctx["lead"]; r = ctx["rel"]; f = ctx["fp"]
        return (ang3d(j[f"{l}_hip"][:, r], j[f"{l}_knee"][:, r], j[f"{l}_ankle"][:, r]) -
                ang3d(j[f"{l}_hip"][:, f], j[f"{l}_knee"][:, f], j[f"{l}_ankle"][:, f]))

    def t3_wrist(j, ctx):
        w = j[f"{ctx['arm']}_wrist"]
        return float(np.nanmax(np.linalg.norm(np.diff(w, axis=1), axis=0) * ctx["fps"]))

    def t3_relh(j, ctx):
        r = ctx["rel"]
        return float(j[f"{ctx['arm']}_wrist"][2, r] -
                     np.nanmin([j["left_ankle"][2, r], j["right_ankle"][2, r]]))

    def t3_armslot(j, ctx):
        r = ctx["rel"]; a = ctx["arm"]
        v = j[f"{a}_wrist"][:, r] - j[f"{a}_shoulder"][:, r]
        return float(np.degrees(np.arctan2(abs(v[1]), v[2])))

    R = []
    # adopted
    R.append(("Lead Knee Angle [O]", est_knee_abs, ("3d", t3_knee_abs)))
    R.append(("Stride (anchor) [O]", est_stride_anchor, "stride_length"))
    R.append(("Trunk Tilt (ant) [O]", est_trunk, "torso_anterior_tilt_br"))
    R.append(("Wrist Speed [O]", est_wrist, ("3d", t3_wrist)))
    R.append(("Arm Slot [O]", est_armslot, ("3d", t3_armslot)))
    R.append(("Release Height [O]", est_relh, ("3d", t3_relh)))
    # rejected, or an alternative
    # Knee Ext Velo BR: DE-ADOPTED 2026-07-24. Under GT (exact) events it does not
    # clear 0.60 at ANY view (best 0.185); its detected-event score of 0.651 came
    # from reading the 2D knee-velocity curve ~11 ms before the OBP BR instant, where
    # the curve happens to track the column. It is reclassified from viewpoint-
    # measurable to the worked example of the event-precision wall (a 5.5 ms timing
    # error costs r2 0.13). See docs/legacy_pre_dedup/GT_EVENT_MAP_HANDOFF.md 5c. Kept as a computed
    # rejected row so its numbers stay reproducible.
    R.append(("  Knee Ext Velo BR [x]", est_knee_velo, "lead_knee_extension_angular_velo_br"))
    R.append(("  Knee Diff (fp-br) [x]", est_knee_diff, ("3d", t3_knee_diff)))
    R.append(("  Stride both@fp [x]", est_stride_fp, "stride_length"))
    R.append(("  Stride @fp anchor [x]", est_stride_at_fp, "stride_length"))
    R.append(("  Elbow Flexion fp [x]", est_elbow_fp, "elbow_flexion_fp"))
    R.append(("  Trunk vs lateral [x]", est_trunk, "torso_lateral_tilt_br"))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")
    rows = build_rows()

    # Collect: the estimates per (row, azimuth), and the truth per row.
    est = {(ri, az): [] for ri in range(len(rows)) for az in AZ}
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
            ctx = {"arm": arm, "lead": lead, "trail": trail,
                   "rel": rel, "fp": fp, "fps": fps}
            sp = r.session_pitch

            for ri, (label, estfn, truth) in enumerate(rows):
                # truth
                if isinstance(truth, tuple):
                    tval = truth[1](joints, ctx)
                else:
                    tval = poi.loc[sp, truth] if (sp in poi.index and truth in poi.columns) else np.nan
                tru[ri].append(tval)
                # estimate per az
                for az in AZ:
                    df = O.project_view(joints, azimuth_deg=az)
                    try:
                        est[(ri, az)].append(estfn(df, ctx))
                    except Exception:
                        est[(ri, az)].append(np.nan)
            done += 1
        except Exception:
            fail += 1
        if done and done % 100 == 0:
            print(f"  ...{done} processed")
    print(f"processed {done} / failed {fail}\n")

    def r2(e, t):
        e = np.asarray(e, float); t = np.asarray(t, float)
        m = np.isfinite(e) & np.isfinite(t)
        if m.sum() <= 2:
            return np.nan
        return np.corrcoef(e[m], t[m])[0, 1] ** 2

    print("=" * 96)
    print("[MASTER] r2 by camera azimuth   ([O]=adopted, [x]=rejected)")
    print("=" * 96)
    hdr = f"{'metric':24s}" + "".join(f"{az:>7d}" for az in AZ) + "    best"
    print(hdr); print("-" * len(hdr))
    out_rows = []
    for ri, (label, _, truth) in enumerate(rows):
        line = f"{label:24s}"; best_az, best = None, -1
        for az in AZ:
            v = r2(est[(ri, az)], tru[ri])
            if pd.notna(v) and v > best:
                best, best_az = v, az
            line += f"{v:7.2f}"
        line += f"   {best_az}({best:.2f})"
        print(line)
        for az in AZ:
            out_rows.append({"metric": label.strip(), "azimuth": az,
                             "r2": r2(est[(ri, az)], tru[ri]),
                             "best_az": best_az})

    out = os.path.join(config.OBP_VALIDATION_DIR, "master_angle_table.csv")
    pd.DataFrame(out_rows).to_csv(out, index=False)
    print(f"\nsaved -> {out}")
    print("\nNote: arm slot noise robustness and classification accuracy are in armslot_robust_test.py.")


if __name__ == "__main__":
    main()