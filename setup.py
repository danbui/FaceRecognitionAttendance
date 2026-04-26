"""
One-time setup script for Edge Attendance system.
Run this ONCE after cloning or updating the codebase.

Usage:
    python setup.py
"""
import subprocess
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def step(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def main():
    # ── Step 1: Delete old database ──
    step("Step 1: Removing old database (schema changed)")
    db_path = BASE_DIR / "attendance.db"
    if db_path.exists():
        os.remove(db_path)
        print(f"  Deleted: {db_path}")
    else:
        print("  No old database found, skipping.")

    # ── Step 2: Install Python dependencies ──
    step("Step 2: Installing Python dependencies")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r",
        str(BASE_DIR / "requirements.txt"),
    ])

    # ── Step 3: Download AI models ──
    step("Step 3: Downloading AI models (YuNet + SFace)")
    subprocess.check_call([sys.executable, str(BASE_DIR / "download_models.py")])

    # ── Step 4: Initialize new database ──
    step("Step 4: Initializing new database")
    # Add project root to path so we can import app modules
    sys.path.insert(0, str(BASE_DIR))
    from app.database import init_db
    init_db()
    print(f"  Created: {db_path}")
    print("  Default admin: username='admin', password='admin123'")

    # ── Done ──
    step("Setup complete!")
    print("""
  You can now run:

  1. Kiosk mode (with camera):
     python -m app.main

  2. Kiosk mode (demo, no camera):
     python -m app.main --demo

  3. Web dashboard:
     uvicorn app.web_api:api --host 0.0.0.0 --port 8000
     Then open: http://localhost:8000
     Login: admin / admin123
""")


if __name__ == "__main__":
    main()
