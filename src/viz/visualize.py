"""
Diamond - 분석 결과 시각화
visualize.py
보정된 좌표로 계산한 지표들을 그래프로 그려서 눈으로 확인한다.
스로잉 영상 검증용.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

# stage2 폴더의 calculator를 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
from calculator import (
    angle_series, joint_speed, hip_shoulder_separation,
    peak_frame, motion_start_frame
)

# config import (src 폴더 기준)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ── 설정 ────────────────────────────────────────────────
CSV = config.SMOOTHED_CSV
FPS = config.FPS_DEFAULT
OUT = config.GRAPH_PNG
ARM = config.THROWING_ARM

df = pd.read_csv(CSV)

# ── 지표 계산 ───────────────────────────────────────────
# 던지는 팔 (config에서 설정)
elbow = angle_series(df, f"{ARM}_shoulder", f"{ARM}_elbow", f"{ARM}_wrist")
wrist_spd = joint_speed(df, f"{ARM}_wrist", FPS)
hss = hip_shoulder_separation(df)

release = peak_frame(wrist_spd)   # 손목 최고 속도 = 릴리즈 추정
start   = motion_start_frame(wrist_spd)

print(f"동작 시작 프레임: {start}")
print(f"릴리즈 추정 프레임: {release}")

# ── 그래프 ──────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

# 1) 손목 속도
axes[0].plot(wrist_spd, color="tab:blue")
axes[0].axvline(release, color="red", ls="--", label=f"Release (frame {release})")
axes[0].axvline(start, color="green", ls="--", label=f"Start (frame {start})")
axes[0].set_ylabel("Wrist speed (px/s)")
axes[0].set_title("Throwing analysis - test_03")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2) 팔꿈치 각도
axes[1].plot(elbow, color="tab:orange")
axes[1].axvline(release, color="red", ls="--")
axes[1].set_ylabel("Elbow angle (deg)")
axes[1].grid(alpha=0.3)

# 3) 힙-어깨 분리
axes[2].plot(hss, color="tab:green")
axes[2].axvline(release, color="red", ls="--")
axes[2].set_ylabel("Hip-shoulder sep (deg)")
axes[2].set_xlabel("Frame")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=120)
print(f"그래프 저장 완료: {OUT}")
plt.show()