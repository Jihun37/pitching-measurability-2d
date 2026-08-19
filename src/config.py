"""
Diamond - 공통 설정
config.py
영상 이름만 여기서 바꾸면 모든 경로가 자동으로 만들어진다.
"""

import os

# ── 여기만 바꾸면 됨 ────────────────────────────────────
VIDEO_NAME = "frontier_test"          # 분석할 영상 이름 (확장자 빼고)
VIDEO_EXT  = ".MOV"             # 영상 확장자
FPS_DEFAULT = 120                # 기본 fps (영상에서 자동으로 읽지만 fallback용)
THROWING_ARM = "left"          # 던지는 팔 / 스윙 방향 ("right" 또는 "left")

# ── 프로젝트 루트 ───────────────────────────────────────
# The parent of the directory holding this file, so the tree can be moved or
# cloned anywhere. DIAMOND_ROOT overrides it when the data lives elsewhere.
ROOT = os.environ.get(
    "DIAMOND_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 자동 생성 경로 ──────────────────────────────────────
VIDEO_PATH = os.path.join(ROOT, "data", "videos", VIDEO_NAME + VIDEO_EXT)
MODEL_PATH = os.path.join(ROOT, "models", "pose_landmarker_full.task")

# 영상별 출력 폴더 (없으면 자동 생성)
OUTPUT_DIR = os.path.join(ROOT, "data", "outputs", VIDEO_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 출력 파일들
OUTPUT_VIDEO    = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_output.mp4")
COORDS_CSV      = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_coords.csv")
SMOOTHED_CSV    = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_smoothed.csv")
GRAPH_PNG       = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_graph.png")

# ── OBP 검증 (benchmark) 경로 ───────────────────────────
# OBP 데이터셋 위치 (metadata.csv, poi/, c3d/ 포함)
OBP_DATA_DIR = os.path.join(ROOT, "data", "datasets", "OBP",
                            "openbiomechanics", "baseball_pitching", "data")
# OBP 검증 분석 출력 폴더 (모든 *_results.csv, feature csv 가 여기로)
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
    print(f"영상 이름:   {VIDEO_NAME}")
    print(f"영상 경로:   {VIDEO_PATH}")
    print(f"출력 폴더:   {OUTPUT_DIR}")
    print(f"OBP 데이터:  {OBP_DATA_DIR}")
    print(f"OBP 출력:    {OBP_VALIDATION_DIR}")


if __name__ == "__main__":
    show_config()