"""
Business logic for enrollment and attendance recognition.

Supports three kiosk modes:
  - ENROLL:    Register a new employee face
  - CHECK_IN:  Record arrival
  - CHECK_OUT: Record departure

Optimizations for Raspberry Pi 4:
  - Sliding-window in-memory cooldown cache: duplicate check in O(1) RAM
    instead of querying SQLite on every frame.
  - EmbeddingCache: matcher loads DB embeddings once, kept in RAM.
  - DB is still the source of truth; cache is just a fast-path guard.
"""
from pathlib import Path
from datetime import datetime, timedelta
import time
import cv2
import numpy as np
from typing import Optional, Dict, Tuple

from .config import CAPTURE_DIR, DUPLICATE_COOLDOWN_MINUTES, CACHE_CLEANUP_INTERVAL
from .database import (
    create_employee, save_embedding,
    add_attendance, get_last_attendance_today,
)
from .matcher import match_embedding, embedding_cache


# ═══════════════════════════════════════════════════════════
#  Sliding-window cooldown cache (RAM)
# ═══════════════════════════════════════════════════════════

class CooldownCache:
    """
    In-memory dict that tracks the last attendance timestamp per
    (employee_id, check_type) pair.

    - is_duplicate() checks in O(1) without touching the SD card.
    - After a successful attendance write, record() updates the cache.
    - Stale entries older than cooldown are purged periodically.
    """

    def __init__(self, cooldown_minutes: int = DUPLICATE_COOLDOWN_MINUTES):
        self._cache: Dict[Tuple[int, str], datetime] = {}
        self._cooldown = timedelta(minutes=cooldown_minutes)
        self._last_cleanup: float = time.time()

    def is_duplicate(self, employee_id: int, check_type: str) -> Optional[dict]:
        """
        Returns info dict if this (employee, type) was recorded within cooldown,
        else None.  Pure RAM lookup – no DB hit.
        """
        key = (employee_id, check_type)
        last_time = self._cache.get(key)
        if last_time is None:
            return None

        elapsed = datetime.now() - last_time
        if elapsed < self._cooldown:
            remaining = self._cooldown.total_seconds() / 60 - elapsed.total_seconds() / 60
            return {
                "last_time": last_time.strftime("%Y-%m-%d %H:%M:%S"),
                "remaining_minutes": round(remaining, 1),
            }
        return None

    def record(self, employee_id: int, check_type: str):
        """Mark this (employee, type) as just recorded."""
        self._cache[(employee_id, check_type)] = datetime.now()
        self._maybe_cleanup()

    def _maybe_cleanup(self):
        """Purge entries older than cooldown, runs at most once per CACHE_CLEANUP_INTERVAL."""
        now_ts = time.time()
        if now_ts - self._last_cleanup < CACHE_CLEANUP_INTERVAL:
            return
        self._last_cleanup = now_ts
        cutoff = datetime.now() - self._cooldown
        expired = [k for k, v in self._cache.items() if v < cutoff]
        for k in expired:
            del self._cache[k]


# Module-level singleton
_cooldown = CooldownCache()


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def save_face_image(face_crop: np.ndarray, prefix: str) -> str:
    """Save cropped face image and return the file path."""
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = CAPTURE_DIR / filename
    cv2.imwrite(str(path), face_crop)
    return str(path)


# ═══════════════════════════════════════════════════════════
#  Enrollment
# ═══════════════════════════════════════════════════════════

def enroll_employee(
    employee_code: str,
    full_name: str,
    department: str,
    frame: np.ndarray,
    face_detection: np.ndarray,
    embedder,
    force_new: bool = False,
) -> dict:
    """
    Enroll a new employee: save their face embedding + photo to database.

    If a similar face already exists in the DB (and force_new is False),
    returns status 'confirm_duplicate' so the UI can ask the user whether
    to update the existing employee or create a new one.

    Args:
        employee_code: Unique employee code (e.g., 'E001')
        full_name: Employee full name
        department: Department name
        frame: Original full frame (BGR)
        face_detection: Raw YuNet detection row (bbox + landmarks)
        embedder: FaceEmbedder instance
        force_new: If True, skip duplicate check and create new employee
    """
    # Extract embedding FIRST (before creating employee)
    embedding = embedder.get_embedding(frame, face_detection)

    # Crop face for saving image
    x, y, w, h = int(face_detection[0]), int(face_detection[1]), int(face_detection[2]), int(face_detection[3])
    face_crop = frame[y:y+h, x:x+w].copy()
    image_path = save_face_image(face_crop, f"enroll_{employee_code}")

    # ── Check for similar existing face ──
    if not force_new:
        match = match_embedding(embedding)
        if match is not None:
            return {
                "status": "confirm_duplicate",
                "existing_employee_id": match["employee_id"],
                "existing_employee_code": match["employee_code"],
                "existing_full_name": match["full_name"],
                "confidence": match["confidence"],
                # Carry forward for later use by confirm actions
                "embedding": embedding,
                "image_path": image_path,
                "new_employee_code": employee_code,
                "new_full_name": full_name,
                "new_department": department,
            }

    # ── No match (or forced) → create new employee ──
    employee_id = create_employee(employee_code, full_name, department)
    save_embedding(employee_id, embedding, image_path)

    # ── Invalidate embedding cache so new enrollment is picked up ──
    embedding_cache.invalidate()

    return {
        "status": "enrolled",
        "employee_id": employee_id,
        "employee_code": employee_code,
        "full_name": full_name,
        "image_path": image_path,
    }


def confirm_enroll_update(employee_id: int, embedding: np.ndarray, image_path: str) -> dict:
    """Add a new embedding to an existing employee (user confirmed Y)."""
    save_embedding(employee_id, embedding, image_path)
    embedding_cache.invalidate()


def enroll_new_with_embedding(
    employee_code: str, full_name: str, department: str,
    embedding: np.ndarray, image_path: str,
) -> dict:
    """Create a new employee with a pre-computed embedding (user confirmed N)."""
    employee_id = create_employee(employee_code, full_name, department)
    save_embedding(employee_id, embedding, image_path)
    embedding_cache.invalidate()
    return {
        "status": "enrolled",
        "employee_id": employee_id,
        "employee_code": employee_code,
        "full_name": full_name,
        "image_path": image_path,
    }


# ═══════════════════════════════════════════════════════════
#  Recognition + Attendance
# ═══════════════════════════════════════════════════════════

def recognize_and_attend(
    frame: np.ndarray,
    face_detection: np.ndarray,
    embedder,
    check_type: str = "CHECK_IN",
    threshold: Optional[float] = None,
) -> dict:
    """
    Recognize a face and record attendance (CHECK_IN or CHECK_OUT).

    Pipeline (optimized for Pi 4):
      1. Embed face → 128-dim vector           (~40ms on Pi 4)
      2. Vectorized match against RAM cache     (~0.3ms)
      3. Sliding-window duplicate check in RAM  (~0us)
      4. Write DB only if not duplicate          (~5ms)

    Returns dict with status:
      - 'success':    matched and recorded
      - 'duplicate':  already recorded within DUPLICATE_COOLDOWN_MINUTES
      - 'not_found':  no matching face in database
    """
    embedding = embedder.get_embedding(frame, face_detection)

    kwargs = {}
    if threshold is not None:
        kwargs["threshold"] = threshold

    match = match_embedding(embedding, **kwargs)

    if match is None:
        return {
            "status": "not_found",
            "message": "Không tìm thấy nhân viên trong cơ sở dữ liệu",
        }

    # ── Fast duplicate check: RAM cache first (O(1)) ──
    dup_info = _cooldown.is_duplicate(match["employee_id"], check_type)
    if dup_info:
        return {
            "status": "duplicate",
            "employee_id": match["employee_id"],
            "employee_code": match["employee_code"],
            "full_name": match["full_name"],
            "check_type": check_type,
            "last_time": dup_info["last_time"],
            "message": f"Đã ghi nhận {check_type} lúc {dup_info['last_time']}",
        }

    # ── Record attendance (DB write) ──
    x, y, w, h = int(face_detection[0]), int(face_detection[1]), int(face_detection[2]), int(face_detection[3])
    face_crop = frame[y:y+h, x:x+w].copy()
    image_path = save_face_image(face_crop, f"attend_{match['employee_code']}")
    add_attendance(match["employee_id"], match["confidence"], check_type, image_path)

    # ── Update RAM cache ──
    _cooldown.record(match["employee_id"], check_type)

    return {
        "status": "success",
        "employee_id": match["employee_id"],
        "employee_code": match["employee_code"],
        "full_name": match["full_name"],
        "confidence": match["confidence"],
        "check_type": check_type,
        "image_path": image_path,
    }