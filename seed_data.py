"""
Seed sample data for testing the Edge Attendance dashboard.

Creates:
  - 10 employees across 3 departments
  - Fake face embeddings (random vectors, not real faces)
  - Attendance logs for the past 5 days
  - Employee user accounts (each employee can login)

Usage:
    cd FaceRecognitionAttendance
    python seed_data.py
"""
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.database import (
    init_db, create_employee, save_embedding, add_attendance,
    create_user, get_conn,
)

# ── Sample employees ──────────────────────────────────
EMPLOYEES = [
    ("NV001", "Nguyen Van An",     "Ky Thuat"),
    ("NV002", "Tran Thi Bich",     "Ky Thuat"),
    ("NV003", "Le Hoang Cuong",    "Ky Thuat"),
    ("NV004", "Pham Minh Duc",     "Kinh Doanh"),
    ("NV005", "Vo Thi Em",         "Kinh Doanh"),
    ("NV006", "Hoang Van Phuc",    "Kinh Doanh"),
    ("NV007", "Dang Thi Giang",    "Nhan Su"),
    ("NV008", "Bui Quoc Hung",     "Nhan Su"),
    ("NV009", "Ngo Thanh Inh",     "Ky Thuat"),
    ("NV010", "Ly Thi Kim",        "Kinh Doanh"),
]


def create_fake_embedding() -> np.ndarray:
    """Create a random 128-dim L2-normalized embedding (for demo only)."""
    vec = np.random.randn(128).astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec.reshape(1, -1)


def seed():
    print("=" * 60)
    print("  Seeding sample data...")
    print("=" * 60)

    init_db()

    # ── Create employees + embeddings ──
    employee_ids = {}
    for code, name, dept in EMPLOYEES:
        eid = create_employee(code, name, dept)
        employee_ids[code] = eid

        # Save a fake embedding for each
        emb = create_fake_embedding()
        save_embedding(eid, emb, f"captures/fake_{code}.jpg")
        print(f"  [+] {code} - {name} ({dept})")

    # ── Create user accounts for employees ──
    print("\n  Creating user accounts...")
    for code, name, dept in EMPLOYEES:
        username = code.lower()  # e.g., "nv001"
        try:
            create_user(username, "123456", role="employee", employee_id=employee_ids[code])
            print(f"  [+] User: {username} / 123456 (role: employee)")
        except Exception:
            print(f"  [SKIP] User {username} already exists")

    # ── Generate attendance logs for past 5 days ──
    print("\n  Generating attendance logs...")
    today = datetime.now().date()
    log_count = 0

    for day_offset in range(5):
        log_date = today - timedelta(days=day_offset)

        # Skip weekends
        if log_date.weekday() >= 5:
            continue

        # Random subset of employees checked in each day
        checked_in = random.sample(EMPLOYEES, k=random.randint(7, 10))

        for code, name, dept in checked_in:
            eid = employee_ids[code]

            # CHECK_IN: between 7:30 and 9:00
            check_in_hour = random.randint(7, 8)
            check_in_min = random.randint(0, 59)
            check_in_time = datetime(
                log_date.year, log_date.month, log_date.day,
                check_in_hour, check_in_min, random.randint(0, 59)
            )

            confidence = round(random.uniform(0.45, 0.95), 4)

            conn = get_conn()
            conn.execute(
                "INSERT INTO attendance_logs(employee_id, check_time, check_type, confidence, image_path) VALUES (?, ?, ?, ?, ?)",
                (eid, check_in_time.isoformat(), "CHECK_IN", confidence, ""),
            )
            conn.commit()
            log_count += 1

            # 80% chance of CHECK_OUT
            if random.random() < 0.8:
                check_out_hour = random.randint(17, 18)
                check_out_min = random.randint(0, 59)
                check_out_time = datetime(
                    log_date.year, log_date.month, log_date.day,
                    check_out_hour, check_out_min, random.randint(0, 59)
                )

                conn.execute(
                    "INSERT INTO attendance_logs(employee_id, check_time, check_type, confidence, image_path) VALUES (?, ?, ?, ?, ?)",
                    (eid, check_out_time.isoformat(), "CHECK_OUT", confidence, ""),
                )
                conn.commit()
                log_count += 1

            conn.close()

    print(f"  [+] Created {log_count} attendance logs")

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"  Done! Created:")
    print(f"    - {len(EMPLOYEES)} employees")
    print(f"    - {len(EMPLOYEES)} face embeddings (fake)")
    print(f"    - {len(EMPLOYEES)} employee accounts")
    print(f"    - {log_count} attendance logs")
    print()
    print("  Login accounts:")
    print("    Admin:    admin / admin123")
    print("    Employee: nv001 / 123456")
    print("    Employee: nv002 / 123456")
    print("    ...etc (nv001-nv010, all password: 123456)")
    print("=" * 60)


if __name__ == "__main__":
    seed()
