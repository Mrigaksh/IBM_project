"""
app.py — Flask Application Factory (RBAC edition)
FIX 1: os.environ KMP fix set BEFORE any imports that load torch/numpy
FIX 2: load_dotenv() now happens inside config.py (at module level),
        so by the time we import config here, env vars are already loaded.
        Removed the broken late load_dotenv() call from original app.py.
"""

import os
# ── MUST be set before importing torch, ultralytics, or numpy ─────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from pathlib import Path

from flask import Flask, jsonify
from flask_jwt_extended import JWTManager
from flask_cors import CORS

# config.py calls load_dotenv() at import time — env vars are loaded now
from config import get_config
from models import db
from routes.auth import auth_bp
from routes.detection import detect_bp
from routes.admin import admin_bp
from routes.chat import chat_bp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(config_class=None) -> Flask:
    app = Flask(__name__)

    cfg = config_class or get_config()
    app.config.from_object(cfg)

    # Ensure upload folder exists at startup
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    # ── Extensions ─────────────────────────────────────────────────────────
    db.init_app(app)
    jwt = JWTManager(app)
    CORS(app, origins=app.config.get("CORS_ORIGINS", "*"), supports_credentials=True)

    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        print(f"[JWT DEBUG] Invalid token: {error}")
        return jsonify({"success": False, "message": "Invalid token.", "error": error}), 422

    @jwt.unauthorized_loader
    def missing_token_callback(error):
        print(f"[JWT DEBUG] Missing token: {error}")
        return jsonify({"success": False, "message": "Authorization header missing.", "error": error}), 401
    
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        print(f"[JWT DEBUG] Expired token")
        return jsonify({"success": False, "message": "Token expired."}), 401

    # ── Blueprints ──────────────────────────────────────────────────────────
    app.register_blueprint(auth_bp)    # /auth/register  /auth/login  /auth/me
    app.register_blueprint(detect_bp)  # /detect  /history  /stats  /export  /image
    app.register_blueprint(admin_bp)   # /admin/*
    app.register_blueprint(chat_bp)    # /chat

    # ── Create DB tables + seed admin ──────────────────────────────────────
    with app.app_context():
        try:
            db.create_all()
            _seed_first_admin()
            logger.info("Database ready.")
        except Exception as e:
            logger.error(
                "DB connection failed: %s\n"
                "Check: MySQL running? DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD in .env correct?",
                e
            )
            raise  # Fail fast — don't silently swallow DB errors

    # ── Error handlers ──────────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"success": False, "message": "Endpoint not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"success": False, "message": "Method not allowed."}), 405

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"success": False, "message": "File too large for your role."}), 413

    @app.errorhandler(500)
    def server_error(e):
        db.session.rollback()
        return jsonify({"success": False, "message": "Internal server error."}), 500

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "service": "ANPR API (RBAC)"}), 200

    @app.route("/roles")
    def roles_info():
        from models import ROLE_PERMISSIONS
        return jsonify({
            "roles": {r: sorted(perms) for r, perms in ROLE_PERMISSIONS.items()}
        }), 200

    logger.info("App ready  [env=%s]  model=%s",
                os.getenv("FLASK_ENV", "development"),
                app.config.get("MODEL_PATH", "NOT SET"))
    return app


def _seed_first_admin():
    from models import User, Role
    if User.query.filter_by(role=Role.ADMIN).first():
        return  # already have an admin, skip

    admin = User(
        username = os.getenv("ADMIN_USERNAME", "admin"),
        email    = os.getenv("ADMIN_EMAIL",    "admin@anpr.local"),
        role     = Role.ADMIN,
    )
    admin.set_password(os.getenv("ADMIN_PASSWORD", "Admin@1234"))
    db.session.add(admin)
    db.session.commit()
    logger.warning("Seeded default admin '%s' — CHANGE PASSWORD IMMEDIATELY.",
                   admin.username)


if __name__ == "__main__":
    application = create_app()
    application.run(
        host  = "0.0.0.0",
        port  = int(os.getenv("PORT", 5000)),
        debug = application.config.get("DEBUG", False),
        use_reloader = False  # DO NOT RESTART if torch modifies its own cache
    )
