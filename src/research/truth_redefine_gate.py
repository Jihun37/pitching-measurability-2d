"""
Diamond - Redefine 3D pelvis/torso rotational velocity to match OBP, re-gate.
truth_redefine_gate.py

The sequencing-timing probe found overhead 2D localizes rotation peaks to ~1 frame,
but scoring the interval vs OBP's official column stayed low because our 2-marker
hip-line axial angle is NOT the same quantity as OBP's segment rotational velocity
(gate r=0.27, our intervals ~3x longer). This rebuilds the 3D truth from proper
rigid-body segment frames and asks WHICH definition matches OBP.

Segment frames (Plug-in-Gait markers, all present in the c3d):
  pelvis  : LASI,RASI,LPSI,RPSI    torso : C7,CLAV,STRN,T10
Rigid-body angular velocity omega from R-dot @ R^T. Candidate scalar signals:
  |omega|   full 3D angular speed
  omega_z   world vertical component (transverse-plane rotation rate)
  omega_ax  component about the segment's own superior axis (axial rotation)
  yaw       d/dt of the segment ML-axis heading in the horizontal plane (cheap)
  line      OLD baseline: d/dt of the 2-marker hip/shoulder line azimuth

Gates (Pearson r across pitches, n~408):
  MAGNITUDE: peak |signal| vs OBP max_pelvis/torso_rotational_velo  (is it the
             same physical quantity?)
  TIMING   : dt = t_torso_peak - t_pelvis_peak vs OBP
             timing_peak_torso_to_peak_pelvis_rot_velo
Only signals that pass the magnitude gate are worth the 2D re-scan.
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3"))
import config
import obp_project as O
import metrics as M
import ezc3d

PEL_MK = ["LASI", "RASI", "LPSI", "RPSI"]
TOR_MK = ["C7", "CLAV", "STRN", "T10"]
SIGNALS = ["mag", "omega_z", "omega_ax", "yaw", "line"]


def load_markers(path, names):
    c = ezc3d.c3d(path)
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    fps = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    pts = c["data"]["points"][:3]
    idx = {l: i for i, l in enumerate(labels)}
    return {n: pts[:, idx[n], :] for n in names if n in idx}, fps


def unit(v):
    n = np.linalg.norm(v, axis=0, keepdims=True)
    return v / (n + 1e-9)


def pelvis_frame(mk):
    """(3,3,n) rotation matrices + superior axis (3,n) for the pelvis."""
    asis = (mk["LASI"] + mk["RASI"]) / 2
    psis = (mk["LPSI"] + mk["RPSI"]) / 2
    ml = unit(mk["RASI"] - mk["LASI"])            # medio-lateral
    ant0 = asis - psis                            # anterior (provisional)
    sup = unit(np.cross(ml.T, ant0.T).T)          # superior
    ant = unit(np.cross(sup.T, ml.T).T)           # re-orthogonalized anterior
    R = np.stack([ml, ant, sup], axis=1)          # (3,3,n): cols = axes
    return R, sup


def torso_frame(mk):
    up0 = (mk["C7"] + mk["CLAV"]) / 2 - (mk["T10"] + mk["STRN"]) / 2   # superior
    ant0 = (mk["CLAV"] + mk["STRN"]) / 2 - (mk["C7"] + mk["T10"]) / 2  # anterior
    sup = unit(up0)
    ml = unit(np.cross(sup.T, ant0.T).T)          # medio-lateral
    ant = unit(np.cross(ml.T, sup.T).T)
    R = np.stack([ml, ant, sup], axis=1)
    return R, sup


def omega_world(R, fps):
    """World-frame angular velocity (3,n) from R-dot @ R^T."""
    Rd = np.gradient(R, axis=2) * fps             # (3,3,n)
    n = R.shape[2]
    w = np.zeros((3, n))
    for t in range(n):
        W = Rd[:, :, t] @ R[:, :, t].T
        w[:, t] = [(W[2, 1] - W[1, 2]) / 2, (W[0, 2] - W[2, 0]) / 2,
                   (W[1, 0] - W[0, 1]) / 2]
    return w


def sg(sig, fps):
    n = len(sig)
    win = max(5, int(round(0.05 * fps)))
    win += (win % 2 == 0)
    win = min(win, n - (n % 2 == 0))
    if win <= 3:
        return sig
    return savgol_filter(sig, win, 3, mode="interp")


def yaw_rate(axis_ml, fps):
    """Heading of the ML axis in the horizontal X-Y plane, differentiated."""
    ang = np.unwrap(np.arctan2(axis_ml[1], axis_ml[0]))
    return np.gradient(ang) * fps


def line_rate(a, b, fps):
    """OLD baseline: azimuth rate of the 2-marker line (a,b are (3,n))."""
    ang = np.unwrap(np.arctan2((b[1] - a[1]), (b[0] - a[0])))
    return np.gradient(ang) * fps


def peak(sig, fps, rel):
    n = len(sig)
    lo, hi = max(0, rel - int(0.40 * fps)), min(n - 1, rel + int(0.05 * fps))
    seg = np.abs(sig[lo:hi + 1])
    if seg.size == 0 or np.all(np.isnan(seg)):
        return None, np.nan
    k = lo + int(np.nanargmax(seg))
    return k, float(np.abs(sig[k]))


def signals_for(R, sup, ml, a, b, fps):
    w = omega_world(R, fps)
    return {
        "mag": np.degrees(np.linalg.norm(w, axis=0)),
        "omega_z": np.degrees(w[2]),
        "omega_ax": np.degrees(np.sum(w * sup, axis=0)),
        "yaw": np.degrees(yaw_rate(ml, fps)),
        "line": np.degrees(line_rate(a, b, fps)),
    }


def process(path):
    mk, fps = load_markers(path, PEL_MK + TOR_MK + ["LSHO", "RSHO"])
    if any(m not in mk for m in PEL_MK + TOR_MK):
        return None
    joints, _ = O.load_c3d_joints(path)
    arm = O.detect_throwing_arm(joints, fps)
    rel = M.release_frame(O.project_view(joints, azimuth_deg=0.0), arm, fps, M.JOINTS)

    Rp, sp = pelvis_frame(mk)
    Rt, st = torso_frame(mk)
    pel = signals_for(Rp, sp, unit(mk["RASI"] - mk["LASI"]),
                      (mk["LASI"] + mk["LPSI"]) / 2, (mk["RASI"] + mk["RPSI"]) / 2, fps)
    tor = signals_for(Rt, st, unit(mk["RSHO"] - mk["LSHO"]), mk["LSHO"], mk["RSHO"], fps)

    out = {}
    for s in SIGNALS:
        kp, vp = peak(sg(pel[s], fps), fps, rel)
        kt, vt = peak(sg(tor[s], fps), fps, rel)
        if kp is None or kt is None:
            out[s] = None
        else:
            out[s] = dict(pel_max=vp, tor_max=vt, dt=(kt - kp) / fps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    poi = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "poi", "poi_metrics.csv"))
    c3d_root = os.path.join(config.OBP_DATA_DIR, "c3d")

    rec = {s: {"sp": [], "pel_max": [], "tor_max": [], "dt": []} for s in SIGNALS}
    n = fail = 0
    for i, r in enumerate(md.itertuples(index=False)):
        if a.limit and i >= a.limit:
            break
        path = os.path.join(c3d_root, f"{int(r.user):06d}", r.filename_new)
        if not os.path.exists(path):
            fail += 1
            continue
        try:
            o = process(path)
            if o is None:
                fail += 1
                continue
            for s in SIGNALS:
                if o[s] is not None:
                    rec[s]["sp"].append(r.session_pitch)
                    rec[s]["pel_max"].append(o[s]["pel_max"])
                    rec[s]["tor_max"].append(o[s]["tor_max"])
                    rec[s]["dt"].append(o[s]["dt"])
            n += 1
        except Exception:
            fail += 1
    print(f"processed {n} / fail {fail}\n")

    def gate(sp, vals, col):
        d = pd.DataFrame({"session_pitch": sp, "v": vals}).merge(
            poi[["session_pitch", col]], on="session_pitch", how="inner").dropna()
        if len(d) < 20:
            return np.nan, len(d)
        return float(np.corrcoef(d["v"], d[col])[0, 1]), len(d)

    print(f"{'signal':>9} {'pel_mag_r':>10} {'tor_mag_r':>10} {'timing_r':>9} "
          f"{'dt_mean_ms':>11} {'pel_max_med':>12}")
    print("-" * 66)
    for s in SIGNALS:
        pr, _ = gate(rec[s]["sp"], rec[s]["pel_max"], "max_pelvis_rotational_velo")
        tr, _ = gate(rec[s]["sp"], rec[s]["tor_max"], "max_torso_rotational_velo")
        dr, nd = gate(rec[s]["sp"], rec[s]["dt"],
                      "timing_peak_torso_to_peak_pelvis_rot_velo")
        dtm = np.nanmean(rec[s]["dt"]) * 1000 if rec[s]["dt"] else np.nan
        pmm = np.nanmedian(rec[s]["pel_max"]) if rec[s]["pel_max"] else np.nan
        print(f"{s:>9} {pr:>10.3f} {tr:>10.3f} {dr:>9.3f} {dtm:>11.1f} {pmm:>12.1f}")

    print("\nOBP columns for reference:")
    print(f"  max_pelvis_rotational_velo med={poi['max_pelvis_rotational_velo'].median():.0f} deg/s")
    print(f"  max_torso_rotational_velo  med={poi['max_torso_rotational_velo'].median():.0f} deg/s")
    print(f"  timing col mean={poi['timing_peak_torso_to_peak_pelvis_rot_velo'].mean()*1000:+.1f} ms")


if __name__ == "__main__":
    main()
