"""
Diamond - Dataset inspection
inspect_dataset.py
Check the columns and structure of the PitcherMotion dataset.
(Large files, so only reads part of them.)
"""

import pandas as pd
import os

DATASET_DIR  = r"D:\project\diamond\data\datasets\pitcher_motion"
MOTION_CSV   = os.path.join(DATASET_DIR, "Pitcher_Motion_Data.csv")
STATCAST_CSV = os.path.join(DATASET_DIR, "Pitcher_Motion_Data_Statcast_Companion.csv")

print("=" * 60)
print("1) Motion data (pose coordinates)")
print("=" * 60)

# Large file, read only first 1000 rows
motion = pd.read_csv(MOTION_CSV, nrows=1000)
print(f"Number of columns: {len(motion.columns)}")
print("Column list:")
print(list(motion.columns))
print()
print("First 3 rows:")
print(motion.head(3).to_string())
print()

print("=" * 60)
print("2) Statcast data (labels such as pitch speed)")
print("=" * 60)

# Try utf-8, fall back to latin-1 on failure
try:
    statcast = pd.read_csv(STATCAST_CSV, nrows=1000)
except UnicodeDecodeError:
    statcast = pd.read_csv(STATCAST_CSV, nrows=1000, encoding="latin-1")

print(f"Number of columns: {len(statcast.columns)}")
print("Column list:")
print(list(statcast.columns))
print()
print("First 3 rows:")
print(statcast.head(3).to_string())