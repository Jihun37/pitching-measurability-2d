"""Figure (patent 도5): view- and metric-dependent EVENT / ANCHOR detection.

PATENT-ONLY copy of fig_event_detection.py. Identical panels (a)-(d) PLUS a
fifth OBP panel (e) for the peak-knee-height (PKH) event that the adopted
COG Velo @PKH metric reads. Kept as a SEPARATE script/PNG so the paper's
fig_event_detection.png (Fig in PAPER_DRAFT) is never touched.

  (a) side   release  = wrist-speed argmax (ball travels in-plane).
  (b) front  release  = arm-extension argmax within a 200 ms window ending at
                        the speed peak (the speed peak lands in follow-through).
  (c) foot plant      = lead ankle carried forward, vertical speed settling;
                        stride reference = TRAIL foot pre-motion quiet window.
  (d) overhead HSS anchor = signature anchor (start of the sustained hip-shoulder
                        separation swing) -> windowed coil extremum (= HSS).
  (e) peak knee height= lead knee at its highest (min image y) before foot plant;
                        azimuth-invariant, one ground-view detection reused.
  (g) MER proxy       = release - 11 frames (@360 Hz, SD 1.1 f). The two MER
                        metrics (torso lateral tilt, glove-arm abduction) are
                        temporally flat there, so a fixed offset from release
                        reads them with no MER detector (2D cannot locate MER).

The image-domain ball-separation release detector (ballsep, panel (f) in the
patent caption) is angle-invariant and lives on real video; it is shown separately
(ball_separation_probe --debug), not as an OBP signal.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, os.path.join(HERE, "..", "stage3"))
import config
import obp_project as O
import metrics as M

INK = "#0E1B33"; GREEN = "#16A34A"; AMBER = "#F2A900"; RED = "#E0533D"
PURPLE = "#7C3AED"; CYAN = "#0FA3B1"


def wrist_ext(df, arm):
    wk = "r_wr" if arm == "right" else "l_wr"
    sk = "r_sh" if arm == "right" else "l_sh"
    wx, wy = M._xy(df, wk, M.JOINTS); sx, sy = M._xy(df, sk, M.JOINTS)
    return wx, wy, np.hypot(wx - sx, wy - sy)


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    md = pd.read_csv(os.path.join(config.OBP_DATA_DIR, "metadata.csv"))
    r = md.iloc[0]
    path = os.path.join(config.OBP_DATA_DIR, "c3d",
                        f"{int(r.user):06d}", r.filename_new)
    joints, fps = O.load_c3d_joints(path)
    arm = O.detect_throwing_arm(joints, fps)
    lead = "left" if arm == "right" else "right"
    trail = "right" if lead == "left" else "left"

    fig, axes = plt.subplots(4, 2, figsize=(11.5, 14.2),
                             gridspec_kw={"height_ratios": [1, 1, 1, 0.42]})
    win = int(0.20 * fps)

    # ---- (a) side release : wrist-speed peak ----
    ax = axes[0, 0]
    df = O.project_view(joints, azimuth_deg=0)
    wx, wy, ext = wrist_ext(df, arm)
    spd = M._speed(wx, wy, fps)
    t = np.arange(len(spd)) / fps * 1000.0
    pk = int(np.nanargmax(spd))
    ax.plot(t, spd / np.nanmax(spd), color=GREEN, lw=1.8, label="wrist speed (norm.)")
    ax.plot(t, ext / np.nanmax(ext), color=AMBER, lw=1.6, label="arm extension (norm.)")
    ax.axvline(t[pk], color=RED, lw=1.8)
    ax.text(t[pk] - 3, 1.03, "release = speed peak", color=RED, fontsize=9, ha="right")
    ax.set_title("(a) Side (0°): wrist-speed peak", fontsize=12, color=INK, weight="bold")
    ax.set_ylabel("normalized"); ax.set_xlabel("time (ms)"); ax.set_ylim(0, 1.14)
    ax.set_xlim(t[max(0, pk - int(0.5 * fps))], t[min(len(t) - 1, pk + int(0.25 * fps))])
    despine(ax)

    # ---- (b) front release : windowed arm extension ----
    ax = axes[0, 1]
    df = O.project_view(joints, azimuth_deg=90)
    wx, wy, ext = wrist_ext(df, arm)
    spd = M._speed(wx, wy, fps)
    t = np.arange(len(spd)) / fps * 1000.0
    pk = int(np.nanargmax(spd))
    lo = max(0, pk - win)
    rel = lo + int(np.nanargmax(ext[lo:pk + 1]))
    ax.plot(t, spd / np.nanmax(spd), color=GREEN, lw=1.8)
    ax.plot(t, ext / np.nanmax(ext), color=AMBER, lw=1.6)
    ax.axvspan(t[lo], t[pk], color="0.9", zorder=0)
    ax.text((t[lo] + t[pk]) / 2, 0.06, "200 ms window", color="0.4", fontsize=8, ha="center")
    ax.axvline(t[pk], color=GREEN, ls="--", lw=1.4)
    ax.text(t[pk] + 3, 1.03, "speed peak\n(follow-through)", color=GREEN, fontsize=8, ha="left")
    ax.axvline(t[rel], color=RED, lw=1.8)
    ax.text(t[rel] - 3, 1.03, "release = ext. max in window", color=RED, fontsize=9, ha="right")
    ax.set_title("(b) Front (90°): windowed arm extension", fontsize=12, color=INK, weight="bold")
    ax.set_xlabel("time (ms)"); ax.set_ylim(0, 1.14)
    ax.set_xlim(t[max(0, pk - int(0.5 * fps))], t[min(len(t) - 1, pk + int(0.25 * fps))])
    despine(ax)

    # ---- (c) foot plant : lead ankle forward + vertical settle ----
    ax = axes[1, 0]
    df = O.project_view(joints, azimuth_deg=0)
    wxs, wys, _ = wrist_ext(df, arm)
    rel_s = int(np.nanargmax(M._speed(wxs, wys, fps)))
    fp = M.foot_plant_frame(df, lead, fps, M.JOINTS, rel_s)
    ak = "l_an" if lead == "left" else "r_an"
    x, y = M._xy(df, ak, M.JOINTS)
    fwd = x - x[0]
    if abs(np.nanmin(fwd)) > abs(np.nanmax(fwd)):
        fwd = -fwd
    fwd_n = fwd / (np.nanmax(np.abs(fwd)) + 1e-9)
    vy = np.abs(np.concatenate([[0.0], np.diff(y)])) * fps
    vy_n = vy / (np.nanmax(vy[:rel_s + 1]) + 1e-9)
    t = np.arange(len(fwd)) / fps * 1000.0
    tx, _ = M._xy(df, "l_an" if trail == "left" else "r_an", M.JOINTS)
    vtx = np.abs(np.concatenate([[0.0], np.diff(tx[:fp + 1])])) * fps
    if np.nanmax(vtx) > 0:
        idx = np.where(vtx > 0.20 * np.nanmax(vtx))[0]
        qend = int(idx[0]) if len(idx) else fp
    else:
        qend = fp
    ax.plot(t, fwd_n, color=AMBER, lw=1.8, label="lead ankle forward (norm.)")
    ax.plot(t, vy_n, color=GREEN, lw=1.6, label="lead ankle |vertical vel| (norm.)")
    ax.axvspan(t[0], t[max(1, qend)], color=CYAN, alpha=0.10, zorder=0)
    ax.text(t[max(1, qend // 2)] if qend > 1 else t[1], 0.5,
            "trail-foot quiet window\n(stride anchor ref)", color=CYAN,
            fontsize=8, ha="center")
    ax.axvline(t[fp], color=RED, lw=1.8)
    ax.text(t[fp] + 4, 1.03, "foot plant", color=RED, fontsize=9, ha="left")
    ax.set_title("(c) Foot plant (side): forward + vertical settle",
                 fontsize=12, color=INK, weight="bold")
    ax.set_ylabel("normalized"); ax.set_xlabel("time (ms)"); ax.set_ylim(0, 1.14)
    ax.set_xlim(t[0], t[min(len(t) - 1, rel_s + int(0.1 * fps))])
    despine(ax)

    # ---- (d) overhead HSS anchor : signature anchor + windowed extremum ----
    ax = axes[1, 1]
    dfo = O.project_view(joints, azimuth_deg=0, elevation_deg=85)
    res = M.hss_peak_overhead(dfo, fps, M.JOINTS, arm=arm)
    if res is not None:
        sep_f = res["sep_f"]
        t = np.arange(len(sep_f)) / fps * 1000.0
        lo, hi = res["window"]; a0 = res["anchor_f"]; pkf = res["peak_f"]
        ax.plot(t, sep_f, color=PURPLE, lw=1.8, label="hip-shoulder sep angle (deg)")
        ax.axvspan(t[lo], t[hi], color=AMBER, alpha=0.15, zorder=0)
        ax.text((t[lo] + t[hi]) / 2, np.nanmin(sep_f), "search window",
                color="0.4", fontsize=8, ha="center", va="bottom")
        smin, smax = np.nanmin(sep_f), np.nanmax(sep_f)
        ax.axvline(t[a0], color=CYAN, ls="--", lw=1.4)
        ax.text(t[a0] - 6, smin + 0.30 * (smax - smin), "signature\nanchor",
                color=CYAN, fontsize=8.5, ha="right", va="center")
        ax.plot(t[pkf], sep_f[pkf], "o", color=RED, ms=9)
        ax.text(t[pkf] + 8, sep_f[pkf] + 0.06 * (smax - smin),
                f"HSS = {res['hss']:.0f}°", color=RED, fontsize=9,
                ha="left", va="bottom")
        ax.set_xlim(t[max(0, a0 - int(0.6 * fps))], t[min(len(t) - 1, hi + int(0.2 * fps))])
        ax.set_ylim(smin - 0.08 * (smax - smin), smax + 0.20 * (smax - smin))
        ax.set_ylabel("degrees")
    ax.set_title("(d) Overhead: HSS signature anchor → windowed extremum",
                 fontsize=12, color=INK, weight="bold")
    ax.set_xlabel("time (ms)")
    despine(ax)

    # ---- (e) peak knee height : lead-knee vertical max before foot plant ----
    ax = axes[2, 0]
    df = O.project_view(joints, azimuth_deg=0)
    wxs, wys, _ = wrist_ext(df, arm)
    rel_s = int(np.nanargmax(M._speed(wxs, wys, fps)))
    fp = M.foot_plant_frame(df, lead, fps, M.JOINTS, rel_s)
    pkh = M.peak_knee_height_frame(df, lead, fp, M.JOINTS)
    _, ky = M._xy(df, "l_kn" if lead == "left" else "r_kn", M.JOINTS)
    height = -ky                       # invert image-y so higher knee = higher curve
    hn = (height - np.nanmin(height)) / (np.nanmax(height) - np.nanmin(height) + 1e-9)
    t = np.arange(len(ky)) / fps * 1000.0
    ax.plot(t, hn, color=GREEN, lw=1.8, label="lead knee height (norm.)")
    ax.axvspan(t[0], t[max(1, fp)], color=CYAN, alpha=0.10, zorder=0)
    ax.text(t[max(1, fp // 2)], 0.08, "search window\n[0, foot plant]",
            color=CYAN, fontsize=8, ha="center")
    ax.axvline(t[pkh], color=RED, lw=1.8)
    ax.text(t[pkh] + 4, 1.03, "peak knee height", color=RED, fontsize=9, ha="left")
    ax.set_title("(e) Peak knee height (azimuth-invariant)",
                 fontsize=12, color=INK, weight="bold")
    ax.set_ylabel("normalized"); ax.set_xlabel("time (ms)"); ax.set_ylim(0, 1.14)
    ax.set_xlim(t[0], t[min(len(t) - 1, fp + int(0.15 * fps))])
    despine(ax)

    # ---- (g) MER proxy : release - 11 frames, metric flat in time ----
    ax = axes[2, 1]
    df0 = O.project_view(joints, azimuth_deg=0)                 # side view -> true release
    wx0, wy0, _ = wrist_ext(df0, arm)
    rel_g = int(np.nanargmax(M._speed(wx0, wy0, fps)))
    mer_px = int(round(rel_g - (11.0 / 360.0) * fps))           # deployment proxy offset
    dfv = O.project_view(joints, azimuth_deg=90, elevation_deg=30)   # torso lat-tilt view
    f0 = max(0, rel_g - int(0.12 * fps)); f1 = min(len(dfv) - 1, rel_g + int(0.04 * fps))
    frames = np.arange(f0, f1 + 1)
    tilt = np.array([M.trunk_lean_2d(dfv, int(f), M.JOINTS) for f in frames], float)
    tg = frames / fps * 1000.0
    smin, smax = np.nanmin(tilt), np.nanmax(tilt); rng = smax - smin + 1e-9
    band = 0.012 * fps                                          # ~12 ms flat-neighbourhood
    ax.axvspan((mer_px - band) / fps * 1000.0, (mer_px + band) / fps * 1000.0,
               color=CYAN, alpha=0.14, zorder=0)
    ax.text(mer_px / fps * 1000.0, smin - 0.02 * rng, "flat near MER",
            color=CYAN, fontsize=8, ha="center", va="top")
    ax.plot(tg, tilt, color=PURPLE, lw=1.8, label="torso lateral tilt @MER (deg)")
    ax.axvline(rel_g / fps * 1000.0, color=RED, lw=1.8)
    ax.text(rel_g / fps * 1000.0 + 3, smax + 0.05 * rng, "release",
            color=RED, fontsize=9, ha="left", va="bottom")
    ax.axvline(mer_px / fps * 1000.0, color=PURPLE, ls="--", lw=1.6)
    ax.text(mer_px / fps * 1000.0 - 3, smax + 0.05 * rng, "MER proxy\n(release - 11 f)",
            color=PURPLE, fontsize=8.5, ha="right", va="bottom")
    ax.set_title("(g) MER proxy: fixed release-11f offset",
                 fontsize=12, color=INK, weight="bold")
    ax.set_ylabel("degrees"); ax.set_xlabel("time (ms)")
    ax.set_ylim(smin - 0.14 * rng, smax + 0.26 * rng)
    ax.set_xlim(tg[0], tg[-1])
    despine(ax)

    axes[3, 0].axis("off"); axes[3, 1].axis("off")     # legend area

    handles, labels = [], []
    for a in (axes[0, 0], axes[1, 0], axes[2, 0], axes[2, 1]):
        h, l = a.get_legend_handles_labels(); handles += h; labels += l
    axes[3, 0].legend(handles, labels, frameon=False, fontsize=9.5,
                      loc="center", title="signals")
    out = os.path.join(config.OBP_VALIDATION_DIR, "fig_event_detection_patent.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved ->", out)


if __name__ == "__main__":
    main()
