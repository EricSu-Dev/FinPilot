"""Deep health checks shared by HTTP endpoint and server-side script."""

from __future__ import annotations

import importlib.util
import os
import socket
import time
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, get_settings

CHROMA_ROOT = PROJECT_ROOT / "data" / "chroma_db"

Status = str


def _mask(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _check(
    name: str,
    status: Status,
    message: str,
    *,
    duration_ms: int | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {"name": name, "status": status, "message": message}
    if duration_ms is not None:
        item["duration_ms"] = duration_ms
    if data:
        item["data"] = data
    return item


def _timed(fn):
    started_at = time.perf_counter()
    try:
        status, message, data = fn()
    except Exception as exc:  # noqa: BLE001
        status, message, data = "error", str(exc), {}
    return status, message, int((time.perf_counter() - started_at) * 1000), data


def _is_writable_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and os.access(path, os.W_OK)


def _tcp_connect(host: str, port: int = 443, timeout: float = 3.0) -> tuple[Status, str, dict[str, Any]]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return "ok", f"{host}:{port} reachable", {"host": host, "port": port}
    except Exception as exc:  # noqa: BLE001
        return "warn", f"{host}:{port} unreachable: {exc}", {"host": host, "port": port}


def _overall_status(checks: list[dict[str, Any]]) -> str:
    statuses = {item["status"] for item in checks}
    if "error" in statuses:
        return "unhealthy"
    if "warn" in statuses:
        return "degraded"
    return "healthy"


def run_deep_health_check(*, external: bool = False) -> dict[str, Any]:
    """Run deep health checks without calling paid LLM/search APIs by default."""
    settings = get_settings()
    checks: list[dict[str, Any]] = []

    def config_check():
        warnings: list[str] = []
        required = {
            "DEEPSEEK_API_KEY": settings.DEEPSEEK_API_KEY,
            "TAVILY_API_KEY": settings.TAVILY_API_KEY,
            "DASHSCOPE_API_KEY": settings.DASHSCOPE_API_KEY,
            "MYSQL_HOST": settings.MYSQL_HOST,
            "MYSQL_USER": settings.MYSQL_USER,
            "MYSQL_DATABASE": settings.MYSQL_DATABASE,
        }
        missing = [key for key, value in required.items() if not str(value or "").strip()]
        if missing:
            warnings.append(f"missing: {', '.join(missing)}")
        if settings.SECRET_KEY == "change-me-in-production":
            warnings.append("SECRET_KEY still uses default value")
        status = "warn" if warnings else "ok"
        data = {
            "model": settings.MODEL_NAME,
            "mysql_host": settings.MYSQL_HOST,
            "mysql_port": settings.MYSQL_PORT,
            "mysql_database": settings.MYSQL_DATABASE,
            "keys": {
                "deepseek": bool(settings.DEEPSEEK_API_KEY.strip()),
                "tavily": bool(settings.TAVILY_API_KEY.strip()),
                "dashscope": bool(settings.DASHSCOPE_API_KEY.strip()),
                "secret_key": "default" if settings.SECRET_KEY == "change-me-in-production" else "custom",
            },
            "masked": {
                "DEEPSEEK_API_KEY": _mask(settings.DEEPSEEK_API_KEY),
                "TAVILY_API_KEY": _mask(settings.TAVILY_API_KEY),
                "DASHSCOPE_API_KEY": _mask(settings.DASHSCOPE_API_KEY),
            },
        }
        return status, "; ".join(warnings) if warnings else "configuration looks usable", data

    status, message, duration_ms, data = _timed(config_check)
    checks.append(_check("configuration", status, message, duration_ms=duration_ms, data=data))

    def mysql_check():
        import pymysql

        conn = pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DATABASE,
            connect_timeout=3,
            read_timeout=3,
            write_timeout=3,
            charset="utf8mb4",
            ssl_disabled=True,
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                value = cursor.fetchone()[0]
            if value != 1:
                return "error", f"unexpected SELECT 1 result: {value!r}", {}
        finally:
            conn.close()
        return "ok", "MySQL connection succeeded", {
            "host": settings.MYSQL_HOST,
            "port": settings.MYSQL_PORT,
            "database": settings.MYSQL_DATABASE,
        }

    status, message, duration_ms, data = _timed(mysql_check)
    checks.append(_check("mysql", status, message, duration_ms=duration_ms, data=data))

    def storage_check():
        data_dir = PROJECT_ROOT / "data"
        upload_dir = data_dir / "uploads"
        details = {
            "project_root": str(PROJECT_ROOT),
            "data_dir": str(data_dir),
            "data_exists": data_dir.exists(),
            "data_writable": _is_writable_dir(data_dir),
            "upload_dir": str(upload_dir),
            "upload_exists": upload_dir.exists(),
            "upload_writable": _is_writable_dir(upload_dir) if upload_dir.exists() else None,
        }
        if not data_dir.exists():
            return "error", "data directory is missing", details
        if not details["data_writable"]:
            return "error", "data directory is not writable", details
        if upload_dir.exists() and not details["upload_writable"]:
            return "warn", "uploads directory exists but is not writable", details
        return "ok", "storage directories look usable", details

    status, message, duration_ms, data = _timed(storage_check)
    checks.append(_check("storage", status, message, duration_ms=duration_ms, data=data))

    def chroma_check():
        building_markers = list(CHROMA_ROOT.glob("*/.building")) if CHROMA_ROOT.exists() else []
        now = time.time()
        stale = [
            str(path)
            for path in building_markers
            if now - path.stat().st_mtime > 900
        ]
        details = {
            "chroma_root": str(CHROMA_ROOT),
            "exists": CHROMA_ROOT.exists(),
            "writable": _is_writable_dir(CHROMA_ROOT) if CHROMA_ROOT.exists() else None,
            "building_markers": len(building_markers),
            "stale_building_markers": stale,
        }
        if not CHROMA_ROOT.exists():
            return "warn", "Chroma directory does not exist yet; upload a report to create it", details
        if not details["writable"]:
            return "error", "Chroma directory is not writable", details
        if stale:
            return "warn", "stale Chroma .building markers found", details
        if building_markers:
            return "warn", "Chroma corpus build is currently in progress", details
        return "ok", "Chroma directory looks usable", details

    status, message, duration_ms, data = _timed(chroma_check)
    checks.append(_check("chroma", status, message, duration_ms=duration_ms, data=data))

    dependency_modules = [
        "fastapi",
        "sqlalchemy",
        "pymysql",
        "langchain",
        "langgraph",
        "langchain_chroma",
        "chromadb",
        "akshare",
        "baostock",
        "dashscope",
        "fitz",
    ]
    missing_modules = [name for name in dependency_modules if importlib.util.find_spec(name) is None]
    checks.append(
        _check(
            "dependencies",
            "error" if missing_modules else "ok",
            f"missing modules: {', '.join(missing_modules)}" if missing_modules else "required modules are installed",
            data={"missing": missing_modules, "checked": dependency_modules},
        )
    )

    if external:
        for name, host in [
            ("deepseek_network", "api.deepseek.com"),
            ("tavily_network", "api.tavily.com"),
            ("dashscope_network", "dashscope.aliyuncs.com"),
            ("eastmoney_network", "push2.eastmoney.com"),
        ]:
            status, message, duration_ms, data = _timed(lambda host=host: _tcp_connect(host))
            checks.append(_check(name, status, message, duration_ms=duration_ms, data=data))
    else:
        checks.append(
            _check(
                "external_network",
                "skipped",
                "external network checks skipped; use external=true or --external",
            )
        )

    return {
        "status": _overall_status(checks),
        "external": external,
        "checks": checks,
    }
