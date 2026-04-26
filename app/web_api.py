"""
FastAPI web server for the Edge Attendance system.

Provides:
  - Login/Logout (session-based auth via signed cookies)
  - Admin dashboard: view all employees, attendance logs, filters, export CSV
  - Employee self-service: view own attendance records
  - REST API endpoints for programmatic access
"""
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import csv
import io
from typing import Optional

from .config import SECRET_KEY, SESSION_MAX_AGE
from .database import (
    init_db, list_attendance, list_employees, create_employee,
    verify_user, get_attendance_summary, get_departments,
)

api = FastAPI(title="Edge Attendance API")
templates = Jinja2Templates(directory="app/web_ui")
serializer = URLSafeTimedSerializer(SECRET_KEY)


# ── Session helpers ─────────────────────────────────────

def create_session_token(user: dict) -> str:
    """Create a signed session token from user data."""
    return serializer.dumps({
        "user_id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "employee_id": user.get("employee_id"),
    })


def get_current_user(request: Request) -> Optional[dict]:
    """Extract user from session cookie. Returns None if not authenticated."""
    token = request.cookies.get("session")
    if not token:
        return None
    try:
        data = serializer.loads(token, max_age=SESSION_MAX_AGE)
        return data
    except (BadSignature, SignatureExpired):
        return None


# ── Startup ─────────────────────────────────────────────

@api.on_event("startup")
def startup():
    init_db()


# ── Auth routes ─────────────────────────────────────────

@api.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    user = get_current_user(request)
    if user:
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "login.html", context={
        "error": None,
    })


@api.post("/login")
def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    user = verify_user(username, password)
    if not user:
        return templates.TemplateResponse(request, "login.html", context={
            "error": "Sai tên đăng nhập hoặc mật khẩu",
        })
    token = create_session_token(user)
    response = RedirectResponse("/", status_code=302)
    response.set_cookie("session", token, httponly=True, max_age=SESSION_MAX_AGE)
    return response


@api.get("/logout")
def logout():
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("session")
    return response


# ── Main dashboard ──────────────────────────────────────

@api.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    department: Optional[str] = None,
    employee_code: Optional[str] = None,
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    # Build filter params
    filter_kwargs = {}
    if date_from:
        filter_kwargs["date_from"] = date_from
    if date_to:
        filter_kwargs["date_to"] = date_to
    if department:
        filter_kwargs["department"] = department

    # Employee can only see their own records
    if user["role"] == "employee" and user.get("employee_id"):
        filter_kwargs["employee_id"] = user["employee_id"]

    logs = list_attendance(limit=200, **filter_kwargs)
    employees = list_employees() if user["role"] == "admin" else []
    summary = get_attendance_summary()
    departments = get_departments()

    return templates.TemplateResponse(request, "index.html", context={
        "user": user,
        "employees": employees,
        "logs": logs,
        "summary": summary,
        "departments": departments,
        "filters": {
            "date_from": date_from or "",
            "date_to": date_to or "",
            "department": department or "",
            "employee_code": employee_code or "",
        },
    })


# ── Employee management (admin only) ───────────────────

@api.post("/employees")
def add_employee(
    request: Request,
    employee_code: str = Form(...),
    full_name: str = Form(...),
    department: str = Form(""),
):
    user = get_current_user(request)
    if not user or user["role"] != "admin":
        return RedirectResponse("/login", status_code=302)
    create_employee(employee_code, full_name, department)
    return RedirectResponse("/", status_code=303)


# ── API endpoints ──────────────────────────────────────

@api.get("/api/attendance")
def api_attendance(
    request: Request,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    department: Optional[str] = None,
):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    kwargs = {}
    if date_from:
        kwargs["date_from"] = date_from
    if date_to:
        kwargs["date_to"] = date_to
    if department:
        kwargs["department"] = department
    if user["role"] == "employee" and user.get("employee_id"):
        kwargs["employee_id"] = user["employee_id"]

    return list_attendance(200, **kwargs)


@api.get("/api/employees")
def api_employees(request: Request):
    user = get_current_user(request)
    if not user or user["role"] != "admin":
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return list_employees()


@api.get("/api/summary")
def api_summary(request: Request, date: Optional[str] = None):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return get_attendance_summary(date)


# ── CSV Export (admin only) ─────────────────────────────

@api.get("/export/attendance.csv")
def export_attendance(
    request: Request,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    department: Optional[str] = None,
):
    user = get_current_user(request)
    if not user or user["role"] != "admin":
        return RedirectResponse("/login", status_code=302)

    kwargs = {}
    if date_from:
        kwargs["date_from"] = date_from
    if date_to:
        kwargs["date_to"] = date_to
    if department:
        kwargs["department"] = department

    rows = list_attendance(10000, **kwargs)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=[
        "id", "employee_code", "full_name", "department",
        "check_time", "check_type", "confidence", "image_path"
    ])
    writer.writeheader()
    writer.writerows(rows)
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=attendance.csv"},
    )