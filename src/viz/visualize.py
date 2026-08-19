"""
Plot the quantities computed from smoothed coordinates.

A visual check on a throwing clip, not a measurement: the numbers that matter
come from the validation, not from reading these curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

# calculator lives in stage2/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
from calculator import (
    angle_series, joint_speed, hip_shoulder_separation,
    peak_frame, motion_start_frame
)

# config lives one level up, in src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Settings
CSV = config.SMOOTHED_CSV
FPS = config.FPS_DEFAULT
OUT = config.GRAPH_PNG
ARM = config.THROWING_ARM

df = pd.read_csv(CSV)

# Quantities
# The throwing arm, from config
elbow = angle_series(df, f"{ARM}_shoulder", f"{ARM}_elbow", f"{ARM}_wrist")
wrist_spd = joint_speed(df, f"{ARM}_wrist", FPS)
hss = hip_shoulder_separation(df)

release = peak_frame(wrist_spd)   # peak wrist speed stands in for release
start   = motion_start_frame(wrist_spd)

print(f"motion starts at frame : {start}")
print(f"release estimated at   : {release}")

# Plots
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

# 1) wrist speed
axes[0].plot(wrist_spd, color="tab:blue")
axes[0].axvline(release, color="red", ls="--", label=f"Release (frame {release})")
axes[0].axvline(start, color="green", ls="--", label=f"Start (frame {start})")
axes[0].set_ylabel("Wrist speed (px/s)")
axes[0].set_title("Throwing analysis - test_03")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2) elbow angle
axes[1].plot(elbow, color="tab:orange")
axes[1].axvline(release, color="red", ls="--")
axes[1].set_ylabel("Elbow angle (deg)")
axes[1].grid(alpha=0.3)

# 3) hip-shoulder separation
axes[2].plot(hss, color="tab:green")
axes[2].axvline(release, color="red", ls="--")
axes[2].set_ylabel("Hip-shoulder sep (deg)")
axes[2].set_xlabel("Frame")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=120)
print(f"graph written: {OUT}")
plt.show()