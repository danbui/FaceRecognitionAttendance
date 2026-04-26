# 🏢 Edge Attendance System – Face Recognition

Hệ thống điểm danh bằng nhận diện khuôn mặt, thiết kế chạy trên **Raspberry Pi 5** (edge deployment).  
Không cần internet, không cần cloud – mọi thứ chạy local trên thiết bị.

---

## 📋 Mục Lục

1. [Pipeline Tổng Quan](#1-pipeline-tổng-quan)
2. [Kiến Trúc Hệ Thống](#2-kiến-trúc-hệ-thống)
3. [Cấu Trúc Source Code](#3-cấu-trúc-source-code)
4. [Hướng Dẫn Cài Đặt](#4-hướng-dẫn-cài-đặt)
5. [Hướng Dẫn Sử Dụng](#5-hướng-dẫn-sử-dụng)
6. [Tài Khoản Mặc Định](#6-tài-khoản-mặc-định)
7. [Cấu Hình Hệ Thống](#7-cấu-hình-hệ-thống)
8. [Tech Stack](#8-tech-stack)
9. [Phase 2 Roadmap](#9-phase-2-roadmap)

---

## 1. Pipeline Tổng Quan

### 1.1 Quy Trình Ghi Danh (Enrollment)

```
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│  Nhân viên  │───▶│ Chọn chế độ │───▶│ Mở Camera    │───▶│ Đưa mặt vào │───▶│ Hệ thống    │
│  đến kiosk  │    │ "1: Ghi danh"│    │ Live video   │    │ khung vàng   │    │ detect mặt  │
└─────────────┘    └─────────────┘    └──────────────┘    └──────┬───────┘    └──────┬──────┘
                                                                  │                   │
                                                                  ▼                   ▼
                                                          ┌──────────────┐    ┌──────────────┐
                                                          │ Khung chuyển │    │ Giữ yên 1.5s │
                                                          │ XANH (OK)    │◀───│ Progress bar │
                                                          └──────┬───────┘    └──────────────┘
                                                                  │
                                                                  ▼
                                                          ┌──────────────┐
                                                          │ Chụp frame   │
                                                          │    ▼         │
                                                          │ YuNet detect │
                                                          │ + landmarks  │
                                                          │    ▼         │
                                                          │ SFace align  │
                                                          │ + embedding  │
                                                          │    ▼         │
                                                          │ Lưu vào DB:  │
                                                          │ • ảnh mặt    │
                                                          │ • embedding  │
                                                          │ • mã NV      │
                                                          └──────────────┘
```

### 1.2 Quy Trình Điểm Danh (Check-in / Check-out)

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Nhân viên  │───▶│ Chọn chế độ  │───▶│ Mở Camera    │───▶│ Đưa mặt vào │
│  đến kiosk  │    │ "2: Vào"     │    │ Live video   │    │ khung vàng   │
│             │    │ "3: Ra"      │    │              │    │              │
└─────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                  │
                              ┌────────────────────────────────────┘
                              ▼
                      ┌──────────────┐
                      │ Khung XANH   │
                      │ Giữ yên 1.5s │
                      └──────┬───────┘
                              │
                              ▼
                      ┌──────────────┐
                      │ YuNet detect │──▶ Crop + Align face (5 landmarks)
                      └──────┬───────┘
                              │
                              ▼
                      ┌──────────────┐
                      │ SFace embed  │──▶ Trích xuất 128-dim vector
                      └──────┬───────┘
                              │
                              ▼
                      ┌──────────────────┐
                      │ So sánh Cosine   │──▶ với TẤT CẢ embeddings trong DB
                      │ Similarity       │
                      └──────┬───────────┘
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
             ┌──────────┐ ┌──────────┐ ┌──────────────┐
             │ MATCH    │ │ DUPLICATE│ │ NOT FOUND    │
             │ score≥   │ │ Đã điểm  │ │ Không có     │
             │ 0.363    │ │ danh     │ │ trong DB     │
             │          │ │ <5 phút  │ │              │
             └────┬─────┘ └────┬─────┘ └──────┬───────┘
                  │            │               │
                  ▼            ▼               ▼
             ┌──────────┐ ┌──────────┐ ┌──────────────┐
             │ ✅ Ghi   │ │ 🟠 Báo  │ │ ❌ Báo lỗi  │
             │ nhận     │ │ "Đã ghi │ │ "Vui lòng   │
             │ điểm danh│ │ nhận"   │ │ ghi danh"   │
             │ vào DB   │ │ (4 giây)│ │ (4 giây)    │
             └──────────┘ └──────────┘ └──────────────┘
```

### 1.3 Quy Trình Xem Kết Quả (Web Dashboard)

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐
│ Mở trình     │───▶│ Trang Login  │───▶│ Nhập username / password     │
│ duyệt web    │    │              │    │                              │
│ :8000        │    │              │    │                              │
└──────────────┘    └──────────────┘    └──────────────┬───────────────┘
                                                       │
                                        ┌──────────────┼──────────────┐
                                        ▼                             ▼
                                 ┌──────────────┐             ┌──────────────┐
                                 │   ADMIN      │             │  EMPLOYEE    │
                                 │              │             │              │
                                 │ • Stat cards │             │ • Xem lịch   │
                                 │ • Tất cả NV  │             │   sử điểm   │
                                 │ • Tất cả log │             │   danh CÁ    │
                                 │ • Filter     │             │   NHÂN       │
                                 │   ngày/phòng │             │ • Filter     │
                                 │ • Thêm NV    │             │   theo ngày  │
                                 │ • Export CSV │             │              │
                                 └──────────────┘             └──────────────┘
```

---

## 2. Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RASPBERRY PI 5 (hoặc PC)                        │
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐               │
│  │              KIOSK (OpenCV GUI – trên Screen)        │               │
│  │                                                      │               │
│  │  ┌────────────┐  ┌──────────┐  ┌──────────────────┐ │               │
│  │  │ Pi Camera  │  │  YuNet   │  │     SFace        │ │               │
│  │  │ hoặc USB   │─▶│ Face     │─▶│ Face Recognition │ │               │
│  │  │ Camera     │  │ Detect   │  │ 128-dim embed    │ │               │
│  │  │            │  │ 240KB    │  │ 37MB             │ │               │
│  │  └────────────┘  └──────────┘  └────────┬─────────┘ │               │
│  │                                          │           │               │
│  │  ┌──────────────────────────────────────┐│           │               │
│  │  │        Attendance Service            ││           │               │
│  │  │  • Enrollment                        ││           │               │
│  │  │  • Match (cosine similarity)         │◀           │               │
│  │  │  • Duplicate check (5 min)           │            │               │
│  │  └────────────────┬─────────────────────┘            │               │
│  └───────────────────┼──────────────────────────────────┘               │
│                      │                                                   │
│                      ▼                                                   │
│  ┌──────────────────────────────────────┐                               │
│  │         SQLite Database              │                               │
│  │         attendance.db                │                               │
│  │                                      │                               │
│  │  ┌───────────┐  ┌────────────────┐   │                               │
│  │  │ employees │  │ face_embeddings│   │                               │
│  │  │ • code    │  │ • employee_id  │   │                               │
│  │  │ • name    │  │ • embedding    │   │                               │
│  │  │ • dept    │  │   (BLOB 512B)  │   │                               │
│  │  └───────────┘  └────────────────┘   │                               │
│  │  ┌───────────────┐  ┌────────┐       │                               │
│  │  │attendance_logs│  │ users  │       │                               │
│  │  │ • check_time  │  │ • user │       │                               │
│  │  │ • check_type  │  │ • hash │       │                               │
│  │  │ • confidence  │  │ • role │       │                               │
│  │  └───────────────┘  └────────┘       │                               │
│  └──────────────────────────────────────┘                               │
│                      │                                                   │
│                      ▼                                                   │
│  ┌──────────────────────────────────────────────────────┐               │
│  │          WEB DASHBOARD (FastAPI + Jinja2)            │               │
│  │          http://localhost:8000                        │               │
│  │                                                      │               │
│  │  • Login (bcrypt + signed session cookies)           │               │
│  │  • Admin: thống kê, quản lý NV, filter, CSV export  │               │
│  │  • Employee: xem điểm danh cá nhân                  │               │
│  └──────────────────────────────────────────────────────┘               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Cấu Trúc Source Code

```
FaceRecognitionAttendance/
│
├── setup.py                  # 🔧 Script setup tự động (chạy 1 lần)
├── download_models.py        # 📥 Download AI models từ OpenCV Zoo
├── seed_data.py              # 🌱 Import dữ liệu mẫu để test
├── requirements.txt          # 📦 Python dependencies
├── README.md                 # 📖 File này
│
├── models/                   # 🧠 AI Models (auto-downloaded)
│   ├── face_detection_yunet_2023mar.onnx    (~240 KB)
│   └── face_recognition_sface_2021dec.onnx  (~37 MB)
│
├── captures/                 # 📸 Ảnh khuôn mặt đã chụp
│   ├── enroll_E001_20260426_094012.jpg
│   └── attend_E001_20260426_094215.jpg
│
├── attendance.db             # 💾 SQLite database
│
└── app/                      # 📁 Application modules
    ├── __init__.py
    ├── config.py             # ⚙️ Cấu hình tập trung (thresholds, paths)
    │
    ├── main.py               # 🖥️ Kiosk main loop
    │                         #    Màn hình chọn chế độ (1/2/3)
    │                         #    Camera → Detect → Embed → Action
    │
    ├── camera_service.py     # 📷 Camera (auto-detect Pi / USB / demo)
    ├── face_detector.py      # 🔍 YuNet face detection + 5 landmarks
    ├── face_embedder.py      # 🧠 SFace face embedding (128-dim)
    ├── matcher.py            # 🎯 Cosine similarity matching
    ├── attendance_service.py # ⚙️ Business logic (enroll, check-in/out)
    ├── database.py           # 💾 SQLite CRUD + auth
    │
    ├── web_api.py            # 🌐 FastAPI web server
    └── web_ui/               # 🎨 HTML templates
        ├── login.html        #    Trang đăng nhập
        └── index.html        #    Dashboard (admin + employee)
```

### Module Dependency Flow

```
main.py (Kiosk)
 ├── camera_service.py       → Capture frames
 ├── face_detector.py        → YuNet detection
 ├── face_embedder.py        → SFace embedding
 ├── attendance_service.py   → Business logic
 │    ├── database.py        → SQLite CRUD
 │    └── matcher.py         → Cosine similarity
 └── config.py               → All settings

web_api.py (Web Dashboard)
 ├── database.py             → SQLite CRUD
 └── config.py               → All settings
```

---

## 4. Hướng Dẫn Cài Đặt

### 4.1 Yêu Cầu Hệ Thống

| Yêu cầu | Chi tiết |
|----------|----------|
| **Python** | 3.10 trở lên (đã test trên 3.13) |
| **OS** | Windows / Linux / Raspberry Pi OS |
| **Camera** | USB webcam, laptop camera, hoặc Pi Camera |
| **RAM** | Tối thiểu 2GB (khuyến nghị 4GB+) |
| **Disk** | ~100 MB (code + models) |

### 4.2 Cài Đặt Tự Động (Khuyến Nghị)

```bash
# Clone hoặc copy project
cd FaceRecognitionAttendance

# Chạy setup tự động
python setup.py
```

Script `setup.py` sẽ:
1. ✅ Xóa database cũ (nếu có)
2. ✅ Cài đặt dependencies (`pip install -r requirements.txt`)
3. ✅ Download AI models từ OpenCV Zoo (~37 MB)
4. ✅ Tạo database mới + tài khoản admin mặc định

### 4.3 Cài Đặt Thủ Công

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Download AI models
python download_models.py

# 3. (Tùy chọn) Import dữ liệu mẫu để test
python seed_data.py
```

### 4.4 Cài Đặt Trên Raspberry Pi 5

```bash
# Cài dependencies hệ thống
sudo apt update && sudo apt install -y \
    python3-pip python3-venv python3-opencv python3-picamera2

# Tạo virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Tiếp tục như bước 4.2
python setup.py
```

---

## 5. Hướng Dẫn Sử Dụng

### 5.1 Chạy Kiosk (Màn Hình Điểm Danh)

```bash
cd FaceRecognitionAttendance

# Với camera thật
python -m app.main

# Demo mode (không camera)
python -m app.main --demo
```

#### Màn Hình Chọn Chế Độ

Khi kiosk khởi động, màn hình hiển thị 3 lựa chọn:

| Phím | Chế độ | Mô tả |
|------|--------|--------|
| `1` | **GHI DANH** | Đăng ký khuôn mặt nhân viên mới |
| `2` | **ĐIỂM DANH VÀO** | Check-in đầu ngày |
| `3` | **ĐIỂM DANH RA** | Check-out cuối ngày |
| `ESC` | – | Quay lại màn hình chọn chế độ |
| `Q` | – | Thoát chương trình |

#### Quy Trình Ghi Danh

1. Nhấn `1` trên màn hình chọn chế độ
2. Đưa mặt vào **khung vàng** ở giữa màn hình
3. Khi mặt được detect → khung chuyển **xanh lá**
4. **Giữ yên** ~1.5 giây (có progress bar)
5. Hệ thống chụp ảnh + lưu embedding → hiển thị **"Đã ghi danh"**

> **Lưu ý**: Cần truyền thông tin nhân viên qua CLI:
> ```bash
> python -m app.main --employee-code NV001 --full-name "Nguyen Van An" --department "Ky Thuat"
> ```

#### Quy Trình Điểm Danh

1. Nhấn `2` (vào) hoặc `3` (ra)
2. Đưa mặt vào khung vàng → chờ khung xanh → giữ yên
3. Kết quả:
   - ✅ **Xanh**: Nhận diện thành công – hiển thị tên + độ tin cậy
   - 🟠 **Cam**: Đã ghi nhận trong vòng 5 phút – không ghi lại
   - ❌ **Đỏ**: Không tìm thấy – chưa ghi danh

### 5.2 Chạy Web Dashboard

```bash
cd FaceRecognitionAttendance
uvicorn app.web_api:api --host 0.0.0.0 --port 8000
```

Mở trình duyệt: **http://localhost:8000**

#### Admin Dashboard

Đăng nhập với tài khoản admin để:
- 📊 Xem **thống kê** tổng quan (check-in, check-out, chưa điểm danh)
- 👥 **Quản lý nhân viên** (thêm, xem danh sách)
- 📋 **Xem lịch sử điểm danh** tất cả nhân viên
- 🔍 **Lọc** theo ngày, phòng ban
- 📥 **Xuất CSV** báo cáo điểm danh

#### Employee Self-Service

Nhân viên đăng nhập bằng tài khoản cá nhân để:
- 📋 Xem **lịch sử điểm danh** của riêng mình
- 🔍 Lọc theo ngày

### 5.3 Import Dữ Liệu Mẫu

```bash
python seed_data.py
```

Tạo: 10 nhân viên, 3 phòng ban, ~50-80 bản ghi điểm danh, tài khoản employee.

---

## 6. Tài Khoản Mặc Định

| Loại | Username | Password | Quyền |
|------|----------|----------|-------|
| **Admin** | `admin` | `admin123` | Xem tất cả, quản lý, export |
| **Nhân viên** | `nv001` ~ `nv010` | `123456` | Xem điểm danh cá nhân |

> ⚠️ Tài khoản nhân viên chỉ có sau khi chạy `python seed_data.py`

---

## 7. Cấu Hình Hệ Thống

Tất cả config nằm trong file `app/config.py`:

| Tham số | Giá trị mặc định | Mô tả |
|---------|-------------------|--------|
| `DETECTION_SCORE_THRESHOLD` | `0.9` | Ngưỡng confidence detect mặt (YuNet) |
| `RECOGNITION_COSINE_THRESHOLD` | `0.363` | Ngưỡng nhận diện (SFace khuyến nghị) |
| `STABLE_FACE_SECONDS` | `1.5` | Thời gian giữ yên trước khi chụp (giây) |
| `DUPLICATE_COOLDOWN_MINUTES` | `5` | Chặn duplicate trong N phút |
| `CAMERA_WIDTH` | `640` | Độ phân giải camera (width) |
| `CAMERA_HEIGHT` | `480` | Độ phân giải camera (height) |
| `SESSION_MAX_AGE` | `28800` | Phiên đăng nhập web (giây, mặc định 8h) |

---

## 8. Tech Stack

| Layer | Công nghệ | Chi tiết |
|-------|-----------|----------|
| **Face Detection** | OpenCV YuNet | `cv2.FaceDetectorYN`, ONNX, ~240 KB, ~10-15ms |
| **Face Recognition** | OpenCV SFace | `cv2.FaceRecognizerSF`, ONNX, ~37 MB, 128-dim |
| **Face Matching** | Cosine Similarity | Linear scan, đủ nhanh cho <1000 người |
| **Database** | SQLite | WAL mode, BLOB embeddings, zero config |
| **Auth** | bcrypt + itsdangerous | Hash passwords, signed session cookies |
| **Web Server** | FastAPI + Uvicorn | REST API + Jinja2 templates |
| **Camera** | OpenCV / picamera2 | Tự detect Pi Camera vs USB camera |
| **UI Kiosk** | OpenCV highgui | Fullscreen camera display |

### Tại sao chọn YuNet + SFace?

- **Zero extra dependencies** – cả 2 đều built-in OpenCV
- **Nhẹ** – tổng ~37 MB models, chạy tốt trên Pi 5
- **YuNet** cho 5 face landmarks → giúp SFace align mặt trước embedding
- **SFace** accuracy 99%+ trên LFW benchmark
- **Inference** ~50-80ms/frame trên Pi 5 → 12-20 FPS real-time

---

## 9. Phase 2 Roadmap

| Feature | Mô tả | Trạng thái |
|---------|--------|------------|
| 🔒 Liveness Detection | Chống giả mạo bằng ảnh/video | Chưa bắt đầu |
| ☁️ Cloud Sync | Đẩy dữ liệu lên server | Chưa bắt đầu |
| 📱 Multi-angle Enrollment | Chụp nhiều góc mặt tăng accuracy | Chưa bắt đầu |
| 🚀 FAISS Matching | Vector search nếu >5000 người | Chưa bắt đầu |
| 📊 Advanced Analytics | Biểu đồ tỷ lệ đi muộn, thống kê | Chưa bắt đầu |
| 🖥️ Kiosk Touchscreen UI | Thay OpenCV GUI bằng web-based kiosk | Chưa bắt đầu |
| 🔄 Auto-start (systemd) | Tự khởi động trên Pi khi bật nguồn | Chưa bắt đầu |

---

## ❓ FAQ / Troubleshooting

**Q: Lỗi `ModuleNotFoundError: No module named 'app'`**  
A: Đảm bảo chạy lệnh từ thư mục gốc project (`cd FaceRecognitionAttendance`)

**Q: Lỗi `YuNet model not found` hoặc `SFace model not found`**  
A: Chạy `python download_models.py` để download models

**Q: Lỗi `Internal Server Error` khi mở web**  
A: Có thể database schema cũ. Xóa `attendance.db` rồi chạy lại `python setup.py`

**Q: Camera không mở được**  
A: Thử `python -m app.main --demo` để test không camera. Hoặc đổi camera index: `python -m app.main --camera 1`

**Q: Nhận diện sai người / không nhận ra**  
A: Điều chỉnh `RECOGNITION_COSINE_THRESHOLD` trong `app/config.py`. Giảm giá trị = dễ match hơn (nhưng có thể nhận nhầm).