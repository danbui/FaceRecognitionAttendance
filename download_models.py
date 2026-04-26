"""
Download YuNet (face detection) and SFace (face recognition) ONNX models
from OpenCV Zoo / HuggingFace.
"""
import urllib.request
import os
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}


def download_file(url: str, dest: Path):
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  [SKIP] {dest.name} already exists ({size_mb:.2f} MB)")
        return

    print(f"  [DOWNLOAD] {dest.name} ...")
    print(f"    URL: {url}")
    urllib.request.urlretrieve(url, str(dest))
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  [OK] {dest.name} ({size_mb:.2f} MB)")


def main():
    print("=" * 60)
    print("  Edge Attendance – Model Downloader")
    print("=" * 60)
    print(f"  Target: {MODELS_DIR}\n")

    for filename, url in MODELS.items():
        dest = MODELS_DIR / filename
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"  [ERROR] Failed to download {filename}: {e}")
            print(f"  Please download manually from: {url}")
            print(f"  And place it in: {dest}")

    print("\n  Done! You can now run the attendance system.")


if __name__ == "__main__":
    main()
