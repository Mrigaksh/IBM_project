"""
config.py — Application configuration

MODEL PATH FIXES (why "Detection Failed" was happening):
─────────────────────────────────────────────────────────
  PROBLEM 1 — Path points to a FOLDER, not a file
    User set:  MODEL_PATH=C:/Users/Admin/runs/detect/train8/weights
    Code needs: .../weights/best.pt   (the actual file)
    Result: YOLO(model_path) tries to load a DIRECTORY → crashes
    FIX: _resolve_model_path() auto-appends /best.pt when path is a directory.

  PROBLEM 2 — Windows backslashes in .env cause escape sequences
    backslash-U, backslash-r, backslash-t in paths get interpreted as Unicode/control escapes
    by some dotenv parsers — corrupting the path string.
    FIX: documented to use forward slashes in .env (C:/Users/...)
         Path() on Windows accepts forward slashes perfectly.

  PROBLEM 3 — load_dotenv() must run before any os.getenv()
    Already fixed — load_dotenv() is called at the top of this module.
"""

import os
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env before ANY os.getenv() call ─────────────────────────────────────
_here = Path(__file__).resolve().parent   # backend/
_root = _here.parent                      # project root (where .env lives)
load_dotenv(dotenv_path=_root / ".env", override=True)


def _resolve_model_path(raw: str) -> str:
    """
    Normalise the model path regardless of how the user typed it.

    Case 1:  C:/Users/Admin/runs/detect/train8/weights          (folder — no best.pt)
             → C:/Users/Admin/runs/detect/train8/weights/best.pt

    Case 2:  C:/Users/Admin/runs/detect/train8/weights/best.pt  (correct)
             → returned as-is

    Case 3:  Path with backslashes from .env
             → Path() normalises separators automatically
    """
    p = Path(raw.strip())

    # If it resolves to an existing directory, user forgot /best.pt
    if p.is_dir():
        p = p / "best.pt"
    # If it has no .pt extension at all (e.g. just a partial name)
    elif p.suffix.lower() != ".pt":
        p = p / "best.pt"

    return str(p)


class BaseConfig:

    # ── Flask ──────────────────────────────────────────────────────────────
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    DEBUG      = False

    # ── JWT ────────────────────────────────────────────────────────────────
    JWT_SECRET_KEY           = os.getenv("JWT_SECRET_KEY", "change-jwt-secret")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    JWT_TOKEN_LOCATION       = ["headers"]
    JWT_HEADER_NAME          = "Authorization"
    JWT_HEADER_TYPE          = "Bearer"

    # ── Database ───────────────────────────────────────────────────────────
    # Priority: DATABASE_URL from .env -> MySQL/TiDB fallback
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")

    if not SQLALCHEMY_DATABASE_URI:
        from urllib.parse import quote_plus
        _DB_HOST = os.getenv("DB_HOST",     "localhost")
        _DB_PORT = os.getenv("DB_PORT",     "4000")       # TiDB default port
        _DB_NAME = os.getenv("DB_NAME",     "anpr_db")
        _DB_USER = os.getenv("DB_USER",     "root")
        _DB_PASS = os.getenv("DB_PASSWORD", "")
        SQLALCHEMY_DATABASE_URI = (
            f"mysql+pymysql://{_DB_USER}:{quote_plus(_DB_PASS)}"
            f"@{_DB_HOST}:{_DB_PORT}/{_DB_NAME}"
            "?charset=utf8mb4"
        )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # TiDB requires SSL — verify_cert=False skips CA validation (safe for TiDB Cloud)
    if "mysql" in (SQLALCHEMY_DATABASE_URI or ""):
        SQLALCHEMY_ENGINE_OPTIONS = {
            "pool_recycle":  280,
            "pool_pre_ping": True,
            "pool_size":     5,
            "max_overflow":  10,
            "connect_args": {
                "ssl": {
                    "verify_cert":     False,
                    "verify_identity": False,
                }
            },
        }
    else:
        SQLALCHEMY_ENGINE_OPTIONS = {}

    # ── File Uploads ───────────────────────────────────────────────────────
    UPLOAD_FOLDER = os.getenv(
        "UPLOAD_FOLDER",
        str(_here / "uploads")
    )
    ALLOWED_EXTENSIONS        = {"png", "jpg", "jpeg", "bmp", "webp"}
    MAX_CONTENT_LENGTH_NORMAL = int(os.getenv("MAX_CONTENT_LENGTH_NORMAL", 16 * 1024 * 1024))
    MAX_CONTENT_LENGTH_HIGH   = int(os.getenv("MAX_CONTENT_LENGTH_HIGH",   64 * 1024 * 1024))
    MAX_CONTENT_LENGTH        = MAX_CONTENT_LENGTH_HIGH

    # ── YOLO Model ─────────────────────────────────────────────────────────
    MODEL_PATH = _resolve_model_path(
        os.getenv(
            "MODEL_PATH",
            str(_root / "runs" / "detect" / "train8" / "weights" / "best.pt")
        )
    )

    # ── CORS ───────────────────────────────────────────────────────────────
    # Add your Render frontend URL here once deployed, e.g.:
    # "https://your-streamlit-app.onrender.com"
CORS_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "https://ibmproj.streamlit.app",
]

class DevelopmentConfig(BaseConfig):
    DEBUG = True


class TestingConfig(BaseConfig):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


class ProductionConfig(BaseConfig):
    DEBUG = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        **getattr(BaseConfig, "SQLALCHEMY_ENGINE_OPTIONS", {}),
        "pool_size":    20,
        "max_overflow": 40,
    }


def get_config():
    env = os.getenv("FLASK_ENV", "development")
    return {
        "development": DevelopmentConfig,
        "testing":     TestingConfig,
        "production":  ProductionConfig,
    }.get(env, DevelopmentConfig)