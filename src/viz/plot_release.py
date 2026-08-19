# plot_release.py — Fig.2 (frontal release detection) 작도
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import savgol_filter

d = pd.read_csv("release_curve.csv")

t_release   = 1.592
t_speakpeak = 1.617
w           = 0.200
win_lo      = t_speakpeak - w

# 스무딩: 점선(speed)은 더 강하게, 실선(ext)은 약하게
ext_s = savgol_filter(d.ext.to_numpy(), 21, 3)
spd_s = savgol_filter(d.spd.to_numpy(), 51, 3)   # ← 윈도우 키워 매끈하게

lo, hi = win_lo - 0.05, t_speakpeak + 0.08
m = (d.t >= lo) & (d.t <= hi)
t   = d.t[m].to_numpy()
ext = ext_s[m.to_numpy()]
spd = spd_s[m.to_numpy()]

tms    = (t - t_release) * 1000.0
rel_ms = 0.0
pk_ms  = (t_speakpeak - t_release) * 1000
lo_ms  = (win_lo - t_release) * 1000

fig, ax1 = plt.subplots(figsize=(3.4, 2.5), dpi=300)
ax2 = ax1.twinx()

# ★ ax1을 ax2 위로 올리고 ax1 배경 투명화 → ax1의 텍스트가 ax2 곡선 위로
ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)

# 음영은 가장 아래 (ax2에 그려 맨 뒤로)
ax2.axvspan(lo_ms, pk_ms, color="0.90", zorder=0)

# 곡선
l2, = ax2.plot(tms, spd, color="0.45", lw=1.5, ls="--", label="Wrist speed", zorder=2)
l1, = ax1.plot(tms, ext, color="black", lw=1.8, label="Arm extension", zorder=3)

ax1.set_xlabel("Time relative to release (ms)", fontsize=8)
ax1.set_ylabel("Arm extension (px)", fontsize=8)
ax2.set_ylabel("Wrist speed (px/s)", fontsize=8)
ax1.tick_params(labelsize=7)
ax2.tick_params(labelsize=7)

ax1.axvline(rel_ms, color="black", lw=1.0, ls=":", zorder=2)
ax1.axvline(pk_ms,  color="0.45",  lw=1.0, ls=":", zorder=2)

ymax = ax1.get_ylim()[1]
bbox = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9)

ax1.annotate("release\n(max extension)", xy=(rel_ms, ext.max()),
             xytext=(rel_ms - 110, ymax * 0.62), fontsize=6.5, ha="left",
             arrowprops=dict(arrowstyle="->", lw=0.7), zorder=10, bbox=bbox)
ax1.annotate("wrist-speed peak\n(follow-through)", xy=(pk_ms, ymax * 0.30),
             xytext=(pk_ms + 8, ymax * 0.42), fontsize=6.5, ha="left",
             zorder=10, bbox=bbox)
ax1.text((lo_ms + pk_ms) / 2, ymax * 0.97, "200 ms search window",
         fontsize=6.5, ha="center", va="top", color="0.30", zorder=10, bbox=bbox)

ax1.set_xlim(lo_ms - 5, (hi - t_release) * 1000)
ax1.xaxis.set_major_locator(MaxNLocator(6))
leg = ax1.legend(handles=[l1, l2], fontsize=6.5, loc="lower left", framealpha=0.9)
leg.set_zorder(10)

plt.tight_layout(pad=0.3)
plt.savefig("fig_release.png", bbox_inches="tight", dpi=300)
plt.savefig("fig_release.pdf", bbox_inches="tight")
print("saved -> fig_release.png / fig_release.pdf")