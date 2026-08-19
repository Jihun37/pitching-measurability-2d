"""Release extension + arm-slot decomposition probe (paper-prep, exploratory).

TWO families, one pitch loop.

(1) RELEASE EXTENSION -- the forward distance from the rubber to the release
    point. Release HEIGHT is adopted (r2 0.999); its orthogonal component has
    never been tested. Note a side view images the forward axis in-plane, so a
    naive forward read would be a self-reference (the trap cog_curve_probe fell
    into). It is NOT one here because extension needs the RUBBER, which is not
    visible at release: it must be inferred from the trail foot's pre-motion
    quiet window (metrics.trail_anchor_x). That anchor is the same machinery
    stride length uses, and stride scores 0.824 -- not 1.0 -- precisely because
    the anchor carries real error. Variants differ in the reference point.

(2) ARM-SLOT DECOMPOSITION -- the adopted arm slot is shoulder->wrist vs
    vertical, validated 3D-direct (CCC 1.000 = a synthetic clean-projection
    identity, per the handoff). CLAUDE.md notes OBP's `arm_slot` COLUMN is
    FOREARM-based and forbids validating our shoulder->wrist definition against
    it. The untested question is the converse: does a FOREARM-based 2D estimator
    recover that column? If it does, this ADDS an OBP-column-validated metric
    rather than merely refining one. Also tested: humerus-relative-to-trunk (the
    component closest to true shoulder abduction) against 3D truth.

Both families report pooled r2 AND the within-pitcher decomposition, and an
ORACLE variant anchored on OBP's own release frame, so a shortfall can be
attributed to event timing vs the measurement itself.

GT hygiene (learned the hard way): OBP event times of 0/NaN mean the event is
missing; scoring against them manufactures huge fake errors. Excluded here.

EXPLORATORY: touches no adopted definition, no official table.
"""
import argparse
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(HERE, "..", "stage3"))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))

import config                                    # noqa: E402
import metrics as M                              # noqa: E402
import obp_project as O                          # noqa: E402
from master_angle_table import load_feet         # noqa: E402
from hss_elevation_test import project_cam       # noqa: E402

AZ = [0, 45, 90, 135, 180, 225, 270, 315]
EL = [0, 15, 30]
LM_ZIP = os.path.join(config.OBP_DATA_DIR, "full_sig", "landmarks.zip")


# ── estimators (2D, from a projected df) ──────────────────────────────────
def _ank_keys(lead):
    trail = "right" if lead == "left" else "left"
    return ("l_an" if lead == "left" else "r_an",
            "l_an" if trail == "left" else "r_an")


def est_extension(df, ctx):
    """Forward release distance from the inferred rubber (trail-foot pre-motion
    anchor) to the wrist at release, stature-normalised -> metres."""
    a = ctx["arm"]; rel = ctx["rel"]
    _, tk = _ank_keys(ctx["lead"])
    wx = np.asarray(M._xy(df, f"{a[0]}_wr", M.JOINTS)[0], float)
    tx = np.asarray(M._xy(df, tk, M.JOINTS)[0], float)
    anc = M.trail_anchor_x(tx, ctx["fp"], ctx["fps"])
    return abs(wx[rel] - anc) / M.pixel_stature(df, M.JOINTS) * ctx["height_m"]


def est_ext_rear(df, ctx):
    """Same, but referenced to the trail ankle AT RELEASE (no anchor inference)
    -- isolates how much of any error the rubber inference contributes."""
    a = ctx["arm"]; rel = ctx["rel"]
    _, tk = _ank_keys(ctx["lead"])
    wx = np.asarray(M._xy(df, f"{a[0]}_wr", M.JOINTS)[0], float)
    tx = np.asarray(M._xy(df, tk, M.JOINTS)[0], float)
    return abs(wx[rel] - tx[rel]) / M.pixel_stature(df, M.JOINTS) * ctx["height_m"]


def est_ext_lead(df, ctx):
    """Release point relative to the LANDING foot (how far past the front foot
    the ball is released) -- the coaching-facing form of extension."""
    a = ctx["arm"]; rel = ctx["rel"]
    lk, _ = _ank_keys(ctx["lead"])
    wx = np.asarray(M._xy(df, f"{a[0]}_wr", M.JOINTS)[0], float)
    lx = np.asarray(M._xy(df, lk, M.JOINTS)[0], float)
    return (wx[rel] - lx[rel]) / M.pixel_stature(df, M.JOINTS) * ctx["height_m"]


def est_slot_global(df, ctx):
    """Adopted definition: shoulder->wrist vs vertical (degrees)."""
    a = ctx["arm"][0]; r = ctx["rel"]
    sx, sy = M._xy(df, f"{a}_sh", M.JOINTS)
    wx, wy = M._xy(df, f"{a}_wr", M.JOINTS)
    return float(np.degrees(np.arctan2(abs(wx[r] - sx[r]), (sy[r] - wy[r]))))


def est_slot_forearm(df, ctx):
    """Forearm segment (elbow->wrist) vs vertical -- the definition OBP's
    `arm_slot` column actually uses."""
    a = ctx["arm"][0]; r = ctx["rel"]
    ex, ey = M._xy(df, f"{a}_el", M.JOINTS)
    wx, wy = M._xy(df, f"{a}_wr", M.JOINTS)
    return float(np.degrees(np.arctan2(abs(wx[r] - ex[r]), (ey[r] - wy[r]))))


def est_slot_humerus_trunk(df, ctx):
    """Upper arm (shoulder->elbow) measured against the TRUNK axis rather than
    the world vertical -- removes the trunk-lateral-tilt contribution that is
    baked into a global slot."""
    a = ctx["arm"][0]; r = ctx["rel"]
    sx, sy = M._xy(df, f"{a}_sh", M.JOINTS)
    ex, ey = M._xy(df, f"{a}_el", M.JOINTS)
    hx = (np.asarray(M._xy(df, "l_hip", M.JOINTS)[0], float)
          + np.asarray(M._xy(df, "r_hip", M.JOINTS)[0], float)) / 2
    hy = (np.asarray(M._xy(df, "l_hip", M.JOINTS)[1], float)
          + np.asarray(M._xy(df, "r_hip", M.JOINTS)[1], float)) / 2
    sxm = (np.asarray(M._xy(df, "l_sh", M.JOINTS)[0], float)
           + np.asarray(M._xy(df, "r_sh", M.JOINTS)[0], float)) / 2
    sym = (np.asarray(M._xy(df, "l_sh", M.JOINTS)[1], float)
           + np.asarray(M._xy(df, "r_sh", M.JOINTS)[1], float)) / 2
    tr = np.array([sxm[r] - hx[r], sym[r] - hy[r]])          # hip -> shoulder
    ua = np.array([ex[r] - sx[r], ey[r] - sy[r]])            # shoulder -> elbow
    n1 = np.linalg.norm(tr); n2 = np.linalg.norm(ua)
    if n1 < 1e-9 or n2 < 1e-9:
        return np.nan
    cs = float(np.clip(np.dot(tr, ua) / (n1 * n2), -1, 1))
    return float(np.degrees(np.arccos(cs)))


# ── 3D-direct truths ──────────────────────────────────────────────────────
def t3_extension(j, ctx):
    tk = "right_ankle" if ctx["lead"] == "left" else "left_ankle"
    wx = j[f"{ctx['arm']}_wrist"][0]
    tx = j[tk][0]
    anc = M.trail_anchor_x(tx, ctx["fp"], ctx["fps"])
    return float(abs(wx[ctx["rel"]] - anc))


def t3_ext_rear(j, ctx):
    tk = "right_ankle" if ctx["lead"] == "left" else "left_ankle"
    return float(abs(j[f"{ctx['arm']}_wrist"][0, ctx["rel"]] - j[tk][0, ctx["rel"]]))


def t3_ext_lead(j, ctx):
    lk = f"{ctx['lead']}_ankle"
    v = j[f"{ctx['arm']}_wrist"][0, ctx["rel"]] - j[lk][0, ctx["rel"]]
    return float(v if abs(v) > 0 else 0.0)


def t3_humerus_trunk(j, ctx):
    r = ctx["rel"]; a = ctx["arm"]
    tr = ((j["left_shoulder"][:, r] + j["right_shoulder"][:, r]) / 2
          - (j["left_hip"][:, r] + j["right_hip"][:, r]) / 2)
    ua = j[f"{a}_elbow"][:, r] - j[f"{a}_shoulder"][:, r]
    n1 = np.linalg.norm(tr); n2 = np.linalg.norm(ua)
    if n1 < 1e-9 or n2 < 1e-9:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(tr, ua) / (n1 * n2), -1, 1))))


# label -> (estimator, truth)   truth: ("3d", fn) or an OBP column name
ROWS = [
    ("Ext (rubber anchor)", est_extension,          ("3d", t3_extension)),
    ("Ext (rear ankle@BR)", est_ext_rear,           ("3d", t3_ext_rear)),
    ("Ext (past lead foot)", est_ext_lead,          ("3d", t3_ext_lead)),
    ("Slot global (adopted)", est_slot_global,      "arm_slot"),
    ("Slot FOREARM", est_slot_forearm,              "arm_slot"),
    ("Slot humerus-vs-trunk", est_slot_humerus_trunk, ("3d", t3_humerus_trunk)),
]


def r2(e, t):
    e = np.asarray(e, float); t = np.asarray(t, float)
    m = np.isfinite(e) & np.isfinite(t)
    if m.sum() < 3 or e[m].std() < 1e-12 or t[m].std() < 1e-12:
        return np.nan
    return float(np.corrcoef(e[m], t[m])[0, 1] ** 2)


def decompose(d):
    d = d.dropna(subset=["est", "truth"])
    if len(d) < 10:
        return dict(pooled=np.nan, within=np.nan, icc=np.nan)
    cen_e = d.est - d.groupby("user").est.transform("mean")
    cen_t = d.truth - d.groupby("user").truth.transform("mean")
    cnt = d.groupby("user").size()
    keep = d.user.isin(cnt[cnt >= 2].index)
    grand = d.truth.mean(); gt = d.groupby("user").truth
    ssb = (gt.count() * (gt.mean() - grand) ** 2).sum()
    ssw = ((d.truth - d.groupby("user").truth.transform("mean")) ** 2).sum()
    return dict(pooled=r2(d.est, d.truth), within=r2(cen_e[keep], cen_t[keep]),
                icc=ssb / (ssb + ssw) if (ssb + ssw) > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv")
                      ).set_index("session_pitch")
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    print("loading OBP release-frame GT (for the oracle variant) ...")
    with zipfile.ZipFile(LM_ZIP) as z:
        with z.open("landmarks.csv") as f:
            lm = pd.read_csv(f, usecols=["session_pitch", "time", "BR_time"])
    obp_rel, bad = {}, 0
    for sp, g in lm.groupby("session_pitch"):
        g = g.sort_values("time"); t = g.time.to_numpy(float)
        br = float(g.BR_time.iloc[0])
        if not np.isfinite(br) or br <= 0:
            bad += 1
            continue
        obp_rel[sp] = int(np.argmin(np.abs(t - br)))
    print(f"  {len(obp_rel)} pitches ({bad} excluded: BR time missing/zero)\n")

    recs = []
    done = fail = 0
    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        sp = r.session_pitch
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path) or sp not in obp_rel:
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
            ctx = {"arm": arm, "lead": lead, "rel": rel, "fp": fp,
                   "fps": fps, "height_m": float(r.session_height_m)}
            ctx_or = dict(ctx, rel=obp_rel[sp])

            for label, estfn, truth in ROWS:
                if isinstance(truth, tuple):
                    tval = truth[1](joints, ctx_or)     # truth at OBP's release
                else:
                    tval = (float(poi.loc[sp, truth])
                            if (sp in poi.index and truth in poi.columns) else np.nan)
                for az in AZ:
                    for el in EL:
                        df = project_cam(joints, az, el)
                        try:
                            e1 = estfn(df, ctx)
                        except Exception:
                            e1 = np.nan
                        try:
                            e2 = estfn(df, ctx_or)
                        except Exception:
                            e2 = np.nan
                        recs.append((label, az, el, int(r.user), sp, e1, e2, tval))
            done += 1
        except Exception:
            fail += 1
        if done and done % 100 == 0:
            print(f"  ...{done} processed")

    print(f"processed {done} / failed {fail}\n")
    df = pd.DataFrame(recs, columns=["metric", "az", "el", "user",
                                     "session_pitch", "est", "est_oracle", "truth"])
    out = []
    for label, _, truth in ROWS:
        g = df[df.metric == label]
        cells, cells_or = [], []
        for (az, el), d in g.groupby(["az", "el"]):
            m = decompose(d); m.update(az=az, el=el); cells.append(m)
            mo = decompose(d.rename(columns={"est": "_e", "est_oracle": "est"}))
            mo.update(az=az, el=el); cells_or.append(mo)
        cells = pd.DataFrame(cells); cells_or = pd.DataFrame(cells_or)
        if cells.pooled.isna().all():
            continue
        b = cells.loc[cells.pooled.idxmax()]
        bo = cells_or.loc[cells_or.pooled.idxmax()]
        out.append({
            "metric": label,
            "truth": truth if isinstance(truth, str) else "3D-direct",
            "best_cell": f"az{int(b.az)}/el{int(b.el)}",
            "best_r2": b.pooled, "within_r2": b.within, "ICC": b.icc,
            "oracle_r2": bo.pooled,
        })

    o = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print("=" * 112)
    print(f"RELEASE EXTENSION + ARM-SLOT DECOMPOSITION   (n pitches = {done})")
    print("  adoption floor r2 = 0.50 | within = pitcher-centered | "
          "oracle = OBP's own release frame")
    print("=" * 112)
    fmt = {c: "{:.3f}".format for c in
           ["best_r2", "within_r2", "ICC", "oracle_r2"]}
    print(o.to_string(index=False, formatters=fmt))

    dst = os.path.join(config.ROOT, "data", "outputs", "obp_validation",
                       "ext_armslot_probe.csv")
    df.to_csv(dst.replace(".csv", "_pairs.csv"), index=False)
    o.to_csv(dst, index=False)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
