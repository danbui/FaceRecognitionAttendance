"""
Centralized configuration for the Edge Attendance system.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
CAPTURE_DIR = BASE_DIR / "captures"
DB_PATH = BASE_DIR / "attendance.db"

CAPTURE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Model files ────────────────────────────────────────
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# ── Face Detection (YuNet) ─────────────────────────────
DETECTION_INPUT_SIZE = (320, 320)
DETECTION_SCORE_THRESHOLD = 0.9
DETECTION_NMS_THRESHOLD = 0.3
DETECTION_TOP_K = 5000

# ── Face Recognition (SFace) ──────────────────────────
# SFace cosine similarity threshold (recommended: 0.363)
# Higher = stricter matching, lower = more lenient
RECOGNITION_COSINE_THRESHOLD = 0.363

# ── Kiosk ──────────────────────────────────────────────
# Time (seconds) face must be stable inside guide box before action
STABLE_FACE_SECONDS = 1.5
# Cooldown (seconds) between two scans of the SAME person
DUPLICATE_COOLDOWN_MINUTES = 5
# Camera resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ── Web / Auth ─────────────────────────────────────────
SECRET_KEY = "edge-attendance-secret-change-in-production"
SESSION_MAX_AGE = 3600 * 8  # 8 hours
