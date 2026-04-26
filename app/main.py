"""
Kiosk main loop for the Edge Attendance system.

The kiosk has two screens:
  1. MODE SELECTION: User chooses one of three modes:
     - [1] Ghi danh (Enroll)
     - [2] Diem danh vao (Check-in)
     - [3] Diem danh ra (Check-out)

  2. CAMERA SCAN: Camera runs face detection + recognition/enrollment.
     Press ESC to return to mode selection screen.
     Press Q to quit entirely.

For enrollment mode, employee info is passed via CLI arguments.
"""
import argparse
import time
import cv2
import numpy as np

from .camera_service import CameraService
from .face_detector import FaceDetector, is_face_inside_guide
from .face_embedder import FaceEmbedder
from .attendance_service import enroll_employee, recognize_and_attend
from .database import init_db
from .config import STABLE_FACE_SECONDS


# ── Colors ──────────────────────────────────────────────
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (180, 180, 180)
COLOR_DARK_BG = (30, 30, 30)
COLOR_BLUE = (255, 160, 50)
COLOR_ORANGE = (0, 140, 255)

WINDOW_NAME = "Edge Attendance Kiosk"


def guide_box(frame):
    """Calculate the guide frame rectangle (center of screen)."""
    h, w, _ = frame.shape
    gw, gh = int(w * 0.35), int(h * 0.50)
    gx, gy = (w - gw) // 2, (h - gh) // 2
    return gx, gy, gw, gh


def draw_text_centered(frame, text, y, font_scale=0.8, color=COLOR_WHITE, thickness=2):
    """Draw text centered horizontally on the frame."""
    h, w = frame.shape[:2]
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    x = (w - text_size[0]) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def draw_mode_selection(frame):
    """Draw the mode selection screen on the frame."""
    h, w = frame.shape[:2]
    # Dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), COLOR_DARK_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Title
    draw_text_centered(frame, "EDGE ATTENDANCE SYSTEM", h // 6, 1.0, COLOR_WHITE, 2)
    draw_text_centered(frame, "Chon che do:", h // 6 + 50, 0.7, COLOR_GRAY, 1)

    # Mode options - draw as boxes
    box_w, box_h = 350, 55
    start_y = h // 3
    gap = 75

    modes = [
        ("1", "GHI DANH (Enroll)", COLOR_BLUE),
        ("2", "DIEM DANH VAO (Check-in)", COLOR_GREEN),
        ("3", "DIEM DANH RA (Check-out)", COLOR_ORANGE),
    ]

    for i, (key, label, color) in enumerate(modes):
        by = start_y + i * gap
        bx = (w - box_w) // 2

        # Box background
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), color, -1)
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), COLOR_WHITE, 2)

        # Key circle
        cx, cy = bx + 30, by + box_h // 2
        cv2.circle(frame, (cx, cy), 18, COLOR_DARK_BG, -1)
        cv2.putText(frame, key, (cx - 7, cy + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

        # Label
        cv2.putText(frame, label, (bx + 60, by + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_DARK_BG, 2)

    # Footer
    draw_text_centered(frame, "Nhan Q de thoat", h - 40, 0.5, COLOR_GRAY, 1)


def draw_status_bar(frame, text, color, bg_alpha=0.7):
    """Draw a status bar at the top of the screen."""
    h, w = frame.shape[:2]
    bar_h = 60

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)

    # Status text
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def draw_result_overlay(frame, text, color, sub_text=""):
    """Draw a large centered result overlay (for success/error/duplicate)."""
    h, w = frame.shape[:2]
    # Dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h // 3), (w, 2 * h // 3), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    draw_text_centered(frame, text, h // 2 - 10, 0.9, color, 2)
    if sub_text:
        draw_text_centered(frame, sub_text, h // 2 + 30, 0.6, COLOR_GRAY, 1)


def run_kiosk(args):
    init_db()
    cam = CameraService(args.camera, demo=args.demo).start()
    detector = FaceDetector()
    embedder = FaceEmbedder()

    current_mode = None  # None = mode selection screen
    stable_start = None
    last_result = None
    result_display_until = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    mode_labels = {
        "enroll": "GHI DANH",
        "check_in": "DIEM DANH VAO",
        "check_out": "DIEM DANH RA",
    }

    for frame in cam.frames():
        now = time.time()
        key = cv2.waitKey(1) & 0xFF

        # ── Q = Quit ──
        if key == ord("q"):
            break

        # ═══════════════════════════════════════════════════
        # MODE SELECTION SCREEN
        # ═══════════════════════════════════════════════════
        if current_mode is None:
            draw_mode_selection(frame)
            cv2.imshow(WINDOW_NAME, frame)

            if key == ord("1"):
                current_mode = "enroll"
                stable_start = None
                last_result = None
            elif key == ord("2"):
                current_mode = "check_in"
                stable_start = None
                last_result = None
            elif key == ord("3"):
                current_mode = "check_out"
                stable_start = None
                last_result = None
            continue

        # ═══════════════════════════════════════════════════
        # CAMERA SCAN SCREEN
        # ═══════════════════════════════════════════════════

        # ESC = Back to mode selection
        if key == 27:  # ESC
            current_mode = None
            stable_start = None
            last_result = None
            continue

        guide = guide_box(frame)
        gx, gy, gw, gh = guide

        guide_color = COLOR_YELLOW
        status = "Dua mat vao khung vang"

        # ── Block re-scan while showing result (4s) ──
        showing_result = last_result and now < result_display_until
        if showing_result:
            face_box, face_raw = None, None
            stable_start = None
        else:
            # Only detect face when NOT showing a result
            face_box, face_raw = detector.detect_largest_with_raw(frame)
            # Clear expired result
            if last_result and now >= result_display_until:
                last_result = None

        if face_box:
            fx, fy, fw, fh = face_box
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)

            if is_face_inside_guide(face_box, guide):
                guide_color = COLOR_GREEN

                if stable_start is None:
                    stable_start = now

                elapsed = now - stable_start

                if elapsed >= STABLE_FACE_SECONDS:
                    # ── Perform action ──
                    if current_mode == "enroll":
                        result = enroll_employee(
                            args.employee_code,
                            args.full_name,
                            args.department,
                            frame, face_raw, embedder,
                        )
                        last_result = result
                    else:
                        check_type = "CHECK_IN" if current_mode == "check_in" else "CHECK_OUT"
                        result = recognize_and_attend(
                            frame, face_raw, embedder,
                            check_type=check_type,
                            threshold=args.threshold,
                        )
                        last_result = result

                    result_display_until = now + 4.0  # Show result for 4 seconds
                    stable_start = None
                else:
                    # Progress indicator
                    progress = elapsed / STABLE_FACE_SECONDS
                    status = f"Giu yen... {progress * 100:.0f}%"

                    # Draw progress bar
                    bar_x = gx
                    bar_y = gy + gh + 10
                    bar_w = int(gw * progress)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + gw, bar_y + 8), (50, 50, 50), -1)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), COLOR_GREEN, -1)
            else:
                stable_start = None
                status = "Can chinh mat vao giua khung"
        else:
            stable_start = None

        # Draw guide box
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), guide_color, 3)

        # Draw guide corner decorations
        corner_len = 20
        corners = [
            ((gx, gy), (gx + corner_len, gy), (gx, gy + corner_len)),
            ((gx + gw, gy), (gx + gw - corner_len, gy), (gx + gw, gy + corner_len)),
            ((gx, gy + gh), (gx + corner_len, gy + gh), (gx, gy + gh - corner_len)),
            ((gx + gw, gy + gh), (gx + gw - corner_len, gy + gh), (gx + gw, gy + gh - corner_len)),
        ]
        for pt, pt_h, pt_v in corners:
            cv2.line(frame, pt, pt_h, guide_color, 4)
            cv2.line(frame, pt, pt_v, guide_color, 4)

        # ── Show result overlay (4 seconds) ──
        if last_result and now < result_display_until:
            r = last_result

            if r["status"] == "enrolled":
                draw_result_overlay(
                    frame,
                    f"Da ghi danh: {r['full_name']}",
                    COLOR_GREEN,
                    f"Ma NV: {r['employee_code']}",
                )
            elif r["status"] == "success":
                draw_result_overlay(
                    frame,
                    f"{r['full_name']}",
                    COLOR_GREEN,
                    f"{r['check_type']} - Do tin cay: {r['confidence']:.3f}",
                )
            elif r["status"] == "duplicate":
                draw_result_overlay(
                    frame,
                    f"Da ghi nhan: {r['full_name']}",
                    COLOR_ORANGE,
                    r["message"],
                )
            elif r["status"] == "not_found":
                draw_result_overlay(
                    frame,
                    "Khong tim thay nhan vien",
                    COLOR_RED,
                    "Vui long ghi danh truoc",
                )


            status = ""

        # Mode indicator (top-right)
        mode_text = mode_labels.get(current_mode, "")
        draw_status_bar(frame, f"[{mode_text}]  {status}", guide_color)

        # Footer: ESC instruction
        h, w = frame.shape[:2]
        cv2.putText(
            frame, "ESC: Quay lai | Q: Thoat", (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1,
        )

        cv2.imshow(WINDOW_NAME, frame)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Attendance Kiosk")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--demo", action="store_true", help="Run without camera")
    parser.add_argument("--employee-code", default="E001", help="Employee code for enrollment")
    parser.add_argument("--full-name", default="Demo Employee", help="Full name for enrollment")
    parser.add_argument("--department", default="Demo", help="Department for enrollment")
    parser.add_argument("--threshold", type=float, default=None, help="Recognition threshold override")
    args = parser.parse_args()
    run_kiosk(args)