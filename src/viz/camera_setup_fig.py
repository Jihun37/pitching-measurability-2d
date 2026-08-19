# camera_setup_fig.py — top-view cameras + mound + pitcher (circle, in front of rubber) + home plate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon

center = np.array([0.0, 0.0])

# Original fig: figsize 3.8 across x-range width 2.5 -> 1.52 in/unit.
# Keep that scale so circles/text render the same physical size.
_xlo, _xhi = -0.7, 1.8
_ylo, _yhi = -0.7, 2.05   # taller now (home plate added above front label)
_in_per_unit = 3.8 / (_xhi - _xlo)
fig, ax = plt.subplots(
    figsize=(_in_per_unit * (_xhi - _xlo), _in_per_unit * (_yhi - _ylo)),
    dpi=300)

# ── Pitcher = circle, slightly in front of rubber (+Y, toward front) ──
pitcher = np.array([0.0, 0.08])

# ── Mound (top view) ──
ax.add_patch(Circle((0, 0), 0.46, facecolor="0.93", edgecolor="0.65",
                    lw=1.0, zorder=1))
# Rubber: behind mound center (-Y)
ax.add_patch(Rectangle((-0.13, -0.10), 0.26, 0.045, facecolor="white",
                       edgecolor="black", lw=1.2, zorder=2))

# ── Pitcher circle ──
ax.add_patch(Circle(pitcher, 0.11, facecolor="white", edgecolor="black",
                    lw=1.6, zorder=6))
ax.text(pitcher[0], pitcher[1], "pitcher", fontsize=6, ha="center",
        va="center", zorder=7)

# ── Camera arc: centered on the PITCHER, not the mound origin. ──
#    0 = right (true side view), 90 = up (true front view).
Rc = 1.2
arc = np.radians(np.linspace(0, 90, 60))
ax.plot(pitcher[0] + Rc*np.cos(arc), pitcher[1] + Rc*np.sin(arc),
        ":", color="0.6", lw=0.8, zorder=2)
for az in [0, 15, 30, 45, 60, 75, 90]:
    th = np.radians(az)
    d = np.array([np.cos(th), np.sin(th)])
    cam = pitcher + Rc*d
    ax.plot(cam[0], cam[1], "s", color="0.30", ms=5.5, zorder=4)
    # aim straight back toward the pitcher, stopping just outside the circle
    tip = pitcher + 0.135 * d
    ax.annotate("", xy=tip, xytext=cam,
                arrowprops=dict(arrowstyle="->", color="0.6", lw=0.7), zorder=3)

# ── Home plate: in front of pitcher, beyond the 90° (front) label ──
hp_y = pitcher[1] + Rc + 0.42
hp_w = 0.10
home = np.array([
    [-hp_w,  hp_y - 0.05],
    [ hp_w,  hp_y - 0.05],
    [ hp_w,  hp_y + 0.02],
    [ 0.0,   hp_y + 0.09],
    [-hp_w,  hp_y + 0.02],
])
ax.add_patch(Polygon(home, closed=True, facecolor="white",
                    edgecolor="black", lw=1.2, zorder=5))
ax.text(0, hp_y + 0.12, "home", fontsize=7, ha="center", va="bottom",
        color="0.25", zorder=5)

ax.text(pitcher[0]+Rc+0.10, pitcher[1], "0\u00b0\n(side)",
        fontsize=8.5, ha="left", va="center", color="0.25")
ax.text(pitcher[0], pitcher[1]+Rc+0.12, "90\u00b0\n(front)",
        fontsize=8.5, ha="center", va="bottom", color="0.25")

ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(_xlo, _xhi); ax.set_ylim(_ylo, _yhi)
plt.tight_layout(pad=0.2)
plt.savefig("fig_camera_setup.png", bbox_inches="tight", dpi=300)
plt.savefig("fig_camera_setup.pdf", bbox_inches="tight")
print("saved")