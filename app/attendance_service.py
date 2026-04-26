"""
Business logic for enrollment and attendance recognition.

Supports three kiosk modes:
  - ENROLL:    Register a new employee face
  - CHECK_IN:  Record arrival
  - CHECK_OUT: Record departure

Duplicate protection: if the same person already has a record of the same type
within the last 5 minutes, the action is skipped and a 'duplicate' status is returned.
"""
from pathlib import Path
from datetime import datetime, timedelta
import cv2
import numpy as np
from typing import Optional

from .config import CAPTURE_DIR, DUPLICATE_COOLDOWN_MINUTES
from .database import (
    create_employee, save_embedding, load_embeddings,
    add_attendance, get_last_attendance_today,
)
from .matcher import match_embedding


def save_face_image(face_crop: np.ndarray, prefix: str) -> str:
    """Save cropped face image and return the file path."""
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = CAPTURE_DIR / filename
    cv2.imwrite(str(path), face_crop)
    return str(path)


def enroll_employee(
    employee_code: str,
    full_name: str,
    department: str,
    frame: np.ndarray,
    face_detection: np.ndarray,
    embedder,
) -> dict:
    """
    Enroll a new employee: save their face embedding + photo to database.

    Args:
        employee_code: Unique employee code (e.g., 'E001')
        full_name: Employee full name
        department: Department name
        frame: Original full frame (BGR)
        face_detection: Raw YuNet detection row (bbox + landmarks)
        embedder: FaceEmbedder instance
    """
    employee_id = create_employee(employee_code, full_name, department)
    embedding = embedder.get_embedding(frame, face_detection)

    # Crop face for saving image
    x, y, w, h = int(face_detection[0]), int(face_detection[1]), int(face_detection[2]), int(face_detection[3])
    face_crop = frame[y:y+h, x:x+w].copy()
    image_path = save_face_image(face_crop, f"enroll_{employee_code}")

    save_embedding(employee_id, embedding, image_path)

    return {
        "status": "enrolled",
        "employee_id": employee_id,
        "employee_code": employee_code,
        "full_name": full_name,
        "image_path": image_path,
    }


def recognize_and_attend(
    frame: np.ndarray,
    face_detection: np.ndarray,
    embedder,
    check_type: str = "CHECK_IN",
    threshold: Optional[float] = None,
) -> dict:
    """
    Recognize a face and record attendance (CHECK_IN or CHECK_OUT).

    Returns dict with status:
      - 'success':    matched and recorded
      - 'duplicate':  already recorded within DUPLICATE_COOLDOWN_MINUTES
      - 'not_found':  no matching face in database
    """
    embedding = embedder.get_embedding(frame, face_detection)
    db_embeddings = load_embeddings()

    kwargs = {}
    if threshold is not None:
        kwargs["threshold"] = threshold

    match = match_embedding(embedding, db_embeddings, **kwargs)

    if match is None:
        return {
            "status": "not_found",
            "message": "Không tìm thấy nhân viên trong cơ sở dữ liệu",
        }

    # ── Duplicate check: already attended within N minutes? ──
    last_log = get_last_attendance_today(match["employee_id"], check_type)
    if last_log:
        last_time = datetime.fromisoformat(last_log["check_time"])
        elapsed = datetime.now() - last_time
        if elapsed < timedelta(minutes=DUPLICATE_COOLDOWN_MINUTES):
            remaining = DUPLICATE_COOLDOWN_MINUTES - (elapsed.total_seconds() / 60)
            return {
                "status": "duplicate",
                "employee_id": match["employee_id"],
                "employee_code": match["employee_code"],
                "full_name": match["full_name"],
                "check_type": check_type,
                "last_time": last_log["check_time"],
                "message": f"Đã ghi nhận {check_type} lúc {last_log['check_time']}",
            }

    # ── Record attendance ──
    x, y, w, h = int(face_detection[0]), int(face_detection[1]), int(face_detection[2]), int(face_detection[3])
    face_crop = frame[y:y+h, x:x+w].copy()
    image_path = save_face_image(face_crop, f"attend_{match['employee_code']}")
    add_attendance(match["employee_id"], match["confidence"], check_type, image_path)

    return {
        "status": "success",
        "employee_id": match["employee_id"],
        "employee_code": match["employee_code"],
        "full_name": match["full_name"],
        "confidence": match["confidence"],
        "check_type": check_type,
        "image_path": image_path,
    }