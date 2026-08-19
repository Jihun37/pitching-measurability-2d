"""
Diamond - BALL-SEPARATION release detection probe (read-only).

The side/frontal strategies were designed for the canonical az0/az90 views;
rotating them around the compass is what produces the +2..+10 residuals.
This probe detects release the way the GT labels define it: the frame the
ball visibly leaves the hand. Angle-invariant wherever the ball is visible.

Pipeline per clip:
  1. kinematic candidate rel0 = deployed corrected_release_frame (fallback:
     side wrist-speed peak) -> window rel0 +- WIN frames.
  2. exact sequential frame grab; per-frame gray crop centered on the
     (sanitized) wrist track.
  3. consecutive-frame abs-diff (small-shift camera compensation via
     phaseCorrelate on the crop), threshold + morphology -> moving blobs.
  4. ball candidates = blobs with area ~ ball size (ball diameter estimated
     from pixel stature: 7.3 cm / ~175 cm ~ 0.042 stature) and compact bbox.
  5. separation frame = first t whose ball blob sits beyond SEP_D ball
     diameters from the wrist AND recedes at t+1 (consistent direction).
     Reported label convention: sep-1 (GT labels the transition frame).
  6. --debug saves a montage with blobs/wrist drawn per frame.

Run:  cd src\tests
      python ball_separation_probe.py --clips angle03_02 --debug
      python ball_separation_probe.py --all        (score vs holdout GT)
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("..", "../stage2", "../stage3", "../analysis", "../deploy",
          "../scratch"):
    sys.path.insert(0, os.path.join(_HERE, p))

import config
import metrics as M
from real_station_test import detect_arm_2d
from measure_auto import load_clip
from release_offset import corrected_release_frame
from hss_overhead_real import grab_frames_exact

WIN = 10                 # candidate window half-width (frames). v0 FROZEN
                         # value - the full 60-clip + dev-15 evaluation ran
                         # at 10. WIN=15 (to cover the +-12f kinematic bias
                         # range) fixed the window-edge misses (dev 15
                         # +14 -> -2, dev 08 +5 -> +1) but let junk lines
                         # into other clips (angle09_01 +1 -> +16): the
                         # 10-vs-15 choice needs NEW clips, not this set.
BALL_D_STAT = 0.042      # ball diameter in statures (7.3 cm / ~175 cm)
AREA_LO, AREA_HI = 0.2, 5.0   # blob area tolerance x nominal ball area
DIFF_THR = 25            # abs-diff gray threshold (after blur)
SEP_D = 1.2              # separation distance in ball diameters
RECEDE_D = 0.5           # required distance growth per frame (ball diam)
BRIGHT_MIN = 140         # min mean gray of a ball blob (white baseball)


def video_path(name):
    for e in (".mov", ".MOV", ".mp4"):
        p = os.path.join(config.ROOT, "data", "videos", name + e)
        if os.path.exists(p):
            return p
    return None


def sanitized_wrist(df, arm, frames, stat):
    """Wrist track over the window; frames where the wrist jumps > 0.5
    stature from the window median are replaced by the median (teleport
    protection - the crop must not follow a background grab)."""
    wx = df[f"{arm}_wrist_x"].to_numpy(float)
    wy = df[f"{arm}_wrist_y"].to_numpy(float)
    xs = np.array([wx[f] if f < len(wx) else np.nan for f in frames])
    ys = np.array([wy[f] if f < len(wy) else np.nan for f in frames])
    mx, my = np.nanmedian(xs), np.nanmedian(ys)
    bad = ~np.isfinite(xs) | (np.hypot(xs - mx, ys - my) > 0.5 * stat)
    xs[bad], ys[bad] = mx, my
    return xs, ys


def ball_blobs(g0, g1, ball_d):
    """Moving, ball-sized, BRIGHT blobs between two gray crops. The
    brightness gate (white baseball vs grass/trees/dark clothing) is what
    thins the body-motion cloud enough for the spacetime RANSAC."""
    # sub-pixel global shift compensation (handheld camera)
    try:
        (dx, dy), _ = cv2.phaseCorrelate(g0.astype(np.float32),
                                         g1.astype(np.float32))
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        g1c = cv2.warpAffine(g1, m, (g1.shape[1], g1.shape[0]))
    except cv2.error:
        g1c = g1
    d = cv2.absdiff(cv2.GaussianBlur(g0, (5, 5), 0),
                    cv2.GaussianBlur(g1c, (5, 5), 0))
    _, bw = cv2.threshold(d, DIFF_THR, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(bw)
    nominal = np.pi * (ball_d / 2) ** 2
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if not (AREA_LO * nominal <= a <= AREA_HI * nominal):
            continue
        if max(w, h) > 3.0 * ball_d or max(w, h) / max(min(w, h), 1) > 3.0:
            continue
        # the blob must be BRIGHT in the current frame (white ball)
        mask = lab == i
        if float(np.mean(g1[mask])) < BRIGHT_MIN:
            continue
        out.append((float(cents[i][0]), float(cents[i][1]), float(a)))
    return out


def detect_separation(name, df, raw, arm, fps, rel0, debug=False):
    stat = M.pixel_stature(df, M.JOINTS)
    if not np.isfinite(stat) or stat <= 0:
        return None, "no stature"
    ball_d = BALL_D_STAT * stat
    frames_idx = list(range(max(0, rel0 - WIN), rel0 + WIN + 1))
    frames = grab_frames_exact(video_path(name), frames_idx)
    if len(frames) < 5:
        return None, "frame grab failed"
    wx, wy = sanitized_wrist(df, arm, frames_idx, stat)

    box = int(np.clip(1.5 * stat, 400, 1600))
    crops, origins = {}, {}
    for k, f in enumerate(frames_idx):
        if f not in frames:
            continue
        img = frames[f]
        H, W = img.shape[:2]
        x0 = int(np.clip(wx[k] - box / 2, 0, max(0, W - box)))
        y0 = int(np.clip(wy[k] - box / 2, 0, max(0, H - box)))
        crops[f] = cv2.cvtColor(img[y0:y0 + box, x0:x0 + box],
                                cv2.COLOR_BGR2GRAY)
        origins[f] = (x0, y0)

    # per-frame ball candidates with wrist distance (image coords)
    cand = {}
    for k in range(1, len(frames_idx)):
        f0, f1 = frames_idx[k - 1], frames_idx[k]
        if f0 not in crops or f1 not in crops:
            continue
        # diff in f1's crop frame: shift f0 crop onto f1's origin
        ox1, oy1 = origins[f1]
        g0 = crops[f0]
        ox0, oy0 = origins[f0]
        if (ox0, oy0) != (ox1, oy1):
            m = np.float32([[1, 0, ox0 - ox1], [0, 1, oy0 - oy1]])
            g0 = cv2.warpAffine(g0, m, (g0.shape[1], g0.shape[0]))
        blobs = ball_blobs(g0, crops[f1], ball_d)
        rows = []
        for cx, cy, a in blobs:
            gx, gy = cx + ox1, cy + oy1
            dist = float(np.hypot(gx - wx[k], gy - wy[k]))
            rows.append((gx, gy, dist, a))
        cand[f1] = rows

    # ---- spacetime RANSAC: the flying ball is a constant-velocity LINE in
    # (x, y, t); body-motion blobs are dense but temporally incoherent, so
    # they cannot assemble a multi-frame straight consensus ----
    pts = [(f, np.array(b[:2])) for f in sorted(cand) for b in cand[f]]
    rng = np.random.default_rng(0)
    best_fit = None          # (n_inliers, v, p_ref, t_ref, inlier_list)
    if len(pts) >= 4:
        idx_pairs = [(i, j) for i in range(len(pts)) for j in range(len(pts))
                     if 1 <= pts[j][0] - pts[i][0] <= 3]
        if len(idx_pairs) > 4000:
            sel = rng.choice(len(idx_pairs), 4000, replace=False)
            idx_pairs = [idx_pairs[s] for s in sel]
        for i, j in idx_pairs:
            t1, p1 = pts[i]
            t2, p2 = pts[j]
            v = (p2 - p1) / (t2 - t1)
            sp = float(np.hypot(*v))
            if not (1.0 * ball_d <= sp <= 8.0 * ball_d):
                continue
            inl, seen = [], {}
            for t, p in pts:
                d = float(np.hypot(*(p - (p1 + v * (t - t1)))))
                if d <= 1.0 * ball_d and (t not in seen or d < seen[t][0]):
                    seen[t] = (d, p)
            if len(seen) < 4 or max(seen) - min(seen) < 4:
                continue
            inl = sorted((t, p) for t, (d, p) in seen.items())
            # receding from the wrist over the consensus
            k0 = frames_idx.index(inl[0][0])
            k1 = frames_idx.index(inl[-1][0])
            d0 = float(np.hypot(inl[0][1][0] - wx[k0],
                                inl[0][1][1] - wy[k0]))
            d1 = float(np.hypot(inl[-1][1][0] - wx[k1],
                                inl[-1][1][1] - wy[k1]))
            if d1 - d0 < 2.0 * ball_d:
                continue
            # LAUNCH ANCHOR: the ball line must originate at the hand -
            # its backward extrapolation has to pass within 5 bd of the
            # wrist track at some frame before the first inlier (chimera
            # lines through glove/background blobs fail this)
            t_first = inl[0][0]
            d_launch = min(
                float(np.hypot(*(p1 + v * (t0 - t1)
                                 - np.array([wx[frames_idx.index(t0)],
                                             wy[frames_idx.index(t0)]]))))
                for t0 in range(frames_idx[0], t_first + 1))
            if d_launch > 5.0 * ball_d:
                continue
            # among launch-anchored receding lines the ball is the FASTEST
            # coherent one (slow body-drift lines can collect more inliers)
            score = len(seen) * sp
            if best_fit is None or score > best_fit[0]:
                best_fit = (score, v, p1, t1, inl)

    if debug:
        nb = {f: len(cand[f]) for f in sorted(cand)}
        print(f"  blobs/frame: min {min(nb.values())} max {max(nb.values())}"
              f"  total {sum(nb.values())}  ball_d={ball_d:.0f}px")
        if best_fit is not None:
            _, v, _, _, inl = best_fit
            print(f"  best line: {len(inl)} inliers over f{inl[0][0]}-"
                  f"f{inl[-1][0]}  speed {np.hypot(*v) / ball_d:.1f} bd/f")
        else:
            print("  best line: NONE")
    sep, ball_tr = None, None
    if best_fit is not None:
        _, v, p_ref, t_ref, inl = best_fit
        # least-squares refit on the inliers (per-axis linear in t)
        ts = np.array([t for t, _ in inl], float)
        xs_i = np.array([p[0] for _, p in inl])
        ys_i = np.array([p[1] for _, p in inl])
        vx, x0 = np.polyfit(ts, xs_i, 1)
        vy, y0 = np.polyfit(ts, ys_i, 1)
        ball_tr = [(int(t), (float(vx * t + x0), float(vy * t + y0), 0.0))
                   for t, _ in inl]
        # BACKWARD extrapolation: separation = frame where the fitted ball
        # line meets the hand (closest approach to the wrist track)
        best = None
        for t0 in range(frames_idx[0], int(ts.min()) + 1):
            k = frames_idx.index(t0)
            px, py = vx * t0 + x0, vy * t0 + y0
            d = float(np.hypot(px - wx[k], py - wy[k]))
            if best is None or d < best[0]:
                best = (d, t0)
        if debug and best is not None:
            print(f"  backtrack: closest approach {best[0] / ball_d:.1f} bd"
                  f" at f{best[1]}")
        # accept threshold: fingertip-to-wrist alone is ~0.11 stature
        # (~2.7 ball diameters) + wrist detection error -> 5 bd
        if best is not None and best[0] <= 5.0 * ball_d:
            sep = best[1]

    if debug:
        tiles = []
        for k, f in enumerate(frames_idx):
            if f not in crops:
                continue
            t = cv2.cvtColor(crops[f], cv2.COLOR_GRAY2BGR)
            ox, oy = origins[f]
            cv2.circle(t, (int(wx[k] - ox), int(wy[k] - oy)), 8,
                       (0, 255, 255), -1)
            for gx, gy, dist, a in cand.get(f, []):
                col = (0, 0, 255) if dist > SEP_D * ball_d else (0, 255, 0)
                cv2.circle(t, (int(gx - ox), int(gy - oy)),
                           int(ball_d / 2), col, 2)
            if ball_tr is not None:
                for tf, tb in ball_tr:
                    if tf == f:
                        cv2.circle(t, (int(tb[0] - ox), int(tb[1] - oy)),
                                   int(ball_d / 2) + 6, (255, 0, 255), 3)
            lab = f"f{f}" + (" SEP" if sep == f else "")
            t = cv2.resize(t, (320, 320))
            cv2.putText(t, lab, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)
            tiles.append(t)
        rows_img = [np.hstack(tiles[i:i + 7])
                    for i in range(0, len(tiles), 7)]
        w = max(r.shape[1] for r in rows_img)
        rows_img = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)))
                    for r in rows_img]
        dbg = os.path.join(config.ROOT, "data", "outputs", name,
                           f"{name}_ballsep_debug.png")
        cv2.imwrite(dbg, np.vstack(rows_img))
        print(f"  debug montage -> {dbg}")

    if sep is None:
        return None, "no ball track"
    _, v, _, _, inl = best_fit
    stats = (f"n={len(inl)} sp={np.hypot(*v) / ball_d:.1f} "
             f"span={inl[-1][0] - inl[0][0]} start={inl[0][0] - sep}")
    return sep, stats  # backtracked hand-meet frame; convention checked vs GT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--gt", default=None,
                    help="alternate GT csv (release_gt_real15 schema)")
    a = ap.parse_args()

    gt = pd.read_csv(a.gt or os.path.join(config.ROOT, "data", "outputs",
                                          "holdout60_scoring_inputs.csv"))
    votes = gt.set_index("clip")
    clips = a.clips or (list(votes.index) if a.all else [])

    print(f"{'clip':<14}{'true_az':>8}{'GT':>6}{'kin':>10}{'ballsep':>10}"
          f"  note")
    errs_k, errs_b, nosep = [], [], []
    for name in clips:
        r = votes.loc[name]
        df, raw = load_clip(name, "_rtmp")
        arm = detect_arm_2d(df)
        cap = cv2.VideoCapture(video_path(name))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        true = int(r.true_release)
        rel_k, _, _, _ = corrected_release_frame(
            df, arm, fps, M.JOINTS, int(r.voted_az), int(r.voted_el),
            raw_df=raw, vote_share=float(r.vote_share))
        if rel_k is None:
            rel_k = M.release_frame(df, arm, fps, M.JOINTS, view="side",
                                    raw_df=raw)
            kin_note = "(defer->side)"
        else:
            kin_note = ""
        sep, note = detect_separation(name, df, raw, arm, fps, int(rel_k),
                                      debug=a.debug)
        ktxt = f"f{rel_k} {rel_k - true:+d}"
        if sep is None:
            nosep.append(name)
            btxt = "--"
        else:
            errs_b.append(sep - true)
            btxt = f"f{sep} {sep - true:+d}"
        errs_k.append(rel_k - true)
        print(f"{name:<14}{int(r.nominal_az):>8}{true:>6}{ktxt:>10}"
              f"{btxt:>10}  {kin_note}{note}")

    if errs_b:
        ab = np.abs(errs_b)
        print(f"\n[ballsep] detected {len(ab)}/{len(clips)}"
              f"   <=1f {int((ab <= 1).sum())}/{len(ab)}"
              f"   <=2f {int((ab <= 2).sum())}/{len(ab)}"
              f"   >3f {int((ab > 3).sum())}")
        if nosep:
            print(f"no ball track: {', '.join(nosep)}")


if __name__ == "__main__":
    main()
