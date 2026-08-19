# plot_camera_setup.py — Fig (camera azimuth setup, top view)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("topview_joints.csv")
P = {r.joint: np.array([r.x, r.y]) for _, r in df.iterrows()}

# 스켈레톤 연결 (top view)
BONES = [
    ("head", "left_shoulder"), ("head", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]

# 스켈레톤 중심 (엉덩이 중앙)
center = (P["left_hip"] + P["right_hip"]) / 2

fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=300)

# --- 스켈레톤 ---
for a, b in BONES:
    xs = [P[a][0], P[b][0]]
    ys = [P[a][1], P[b][1]]
    ax.plot(xs, ys, color="black", lw=1.4, zorder=3, solid_capstyle="round")
for name, p in P.items():
    ax.plot(p[0], p[1], "o", color="black", ms=2.5, zorder=4)

# --- 카메라 호 ---
# azimuth=0 (side, +X=투구방향 쪽 측면) ~ 90 (front, 포수쪽)
# 카메라를 중심에서 반경 R 호에 배치
R = 1.1
azs = [0, 15, 30, 45, 60, 75, 90]
for az in azs:
    th = np.radians(az)
    # az=0: 측면(+Y 쪽에서 봄), az=90: 정면(+X 투구방향 앞에서 봄)
    # 시선 방향 d=(sin, cos); 카메라는 중심에서 -d 방향(즉 반대편)에 위치
    d = np.array([np.sin(th), np.cos(th)])
    cam = center + R * d
    # 카메라 마커
    ax.plot(cam[0], cam[1], "s", color="0.30", ms=4, zorder=5)
    # 시선 화살표 (카메라 -> 중심)
    ax.annotate("", xy=center, xytext=cam,
                arrowprops=dict(arrowstyle="->", color="0.55", lw=0.7), zorder=2)
    # 각도 라벨 (0, 90만 표시해 깔끔하게)
    if az in (0, 90):
        lbl = f"{az}°"
        ax.text(cam[0] + 0.06*np.sign(d[0]+0.01), cam[1] + 0.06,
                lbl, fontsize=7, ha="center", va="center")

# 0/90 의미 주석
ax.text(center[0] + R*0.55, center[1] + R*1.18, "front (90°)",
        fontsize=6.5, ha="center", color="0.30")
ax.text(center[0] + R*1.15, center[1] - 0.0, "side (0°)",
        fontsize=6.5, ha="center", color="0.30", rotation=90)

ax.set_aspect("equal")
ax.axis("off")
plt.tight_layout(pad=0.2)
plt.savefig("fig_camera_setup.png", bbox_inches="tight", dpi=300)
plt.savefig("fig_camera_setup.pdf", bbox_inches="tight")
print("saved -> fig_camera_setup.png / .pdf")