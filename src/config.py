"""Shared paths for the whole project.

Every path used anywhere is built here from one root and one clip name, so a script
never spells out a directory of its own. Set VIDEO_NAME and the per-clip outputs follow;
the OpenBiomechanics locations do not depend on it.
"""

import os

# The single-clip settings. Only these change from one recording to the next; the
# OpenBiomechanics work below ignores them entirely.
VIDEO_NAME = "frontier_test"   # clip name, without the extension
VIDEO_EXT = ".MOV"
FPS_DEFAULT = 120              # fallback only; the real rate is read from the file
THROWING_ARM = "left"          # "right" or "left"

# The parent of the directory holding this file, so the tree can be moved or
# cloned anywhere. DIAMOND_ROOT overrides it when the data lives elsewhere.
ROOT = os.environ.get(
    "DIAMOND_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VIDEO_PATH = os.path.join(ROOT, "data", "videos", VIDEO_NAME + VIDEO_EXT)
MODEL_PATH = os.path.join(ROOT, "models", "pose_landmarker_full.task")

OUTPUT_DIR = os.path.join(ROOT, "data", "outputs", VIDEO_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_output.mp4")
COORDS_CSV = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_coords.csv")
SMOOTHED_CSV = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_smoothed.csv")
GRAPH_PNG = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_graph.png")

# The OpenBiomechanics download, holding metadata.csv, poi/ and c3d/. Not
# redistributed with this code; see the README for where to get it.
OBP_DATA_DIR = os.path.join(ROOT, "data", "datasets", "OBP",
                            "openbiomechanics", "baseball_pitching", "data")
# Everything the validation writes lands here, canonical tables included.
OBP_VALIDATION_DIR = os.path.join(ROOT, "data", "outputs", "obp_validation")
os.makedirs(OBP_VALIDATION_DIR, exist_ok=True)


def refined_or(path, raw=False):
    """Prefer the occlusion-recovery-refined sibling of a coords/smoothed CSV
    when it exists (the refined pose is the project's validated best pose; on
    overhead it removes confidently-wrong wrist/elbow jumps, on side/front it
    is a no-op). raw=True, an already-refined path, or a missing sibling all
    return the path unchanged (safe fallback)."""
    if raw or path.endswith("_refined.csv") or not path.endswith(".csv"):
        return path
    cand = path[:-4] + "_refined.csv"
    return cand if os.path.exists(cand) else path


def show_config():
    print(f"clip name       : {VIDEO_NAME}")
    print(f"clip path       : {VIDEO_PATH}")
    print(f"output folder   : {OUTPUT_DIR}")
    print(f"OBP data        : {OBP_DATA_DIR}")
    print(f"OBP output      : {OBP_VALIDATION_DIR}")


if __name__ == "__main__":
    show_config()
