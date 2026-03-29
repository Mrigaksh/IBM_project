"""
middleware/rbac.py
Reusable decorators for role and permission enforcement.
Use these on top of @jwt_required() to gate routes.
"""

from typing import Optional
from functools import wraps
from flask import jsonify
from flask_jwt_extended import get_jwt_identity
from models import User, Role, Permission


def _current_user() -> Optional[User]:
    uid = get_jwt_identity()
    if not uid: return None
    try:
        return User.query.get(int(uid))
    except (ValueError, TypeError):
        return None


def require_permission(*perms: str):
    """
    Decorator: user must hold ALL listed permissions.
    Must be placed BELOW @jwt_required().

    Usage:
        @jwt_required()
        @require_permission(Permission.VIEW_ALL)
        def my_route(): ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = _current_user()
            if not user:
                return jsonify({"success": False, "message": "User not found."}), 404
            if not user.is_active:
                return jsonify({"success": False, "message": "Account is deactivated."}), 403
            missing = [p for p in perms if not user.has_permission(p)]
            if missing:
                return jsonify({
                    "success": False,
                    "message": f"Access denied. Required permission(s): {', '.join(missing)}",
                    "your_role": user.role,
                }), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def require_role(*roles: str):
    """
    Decorator: user's role must be one of the listed roles.
    Must be placed BELOW @jwt_required().

    Usage:
        @jwt_required()
        @require_role(Role.ADMIN)
        def admin_only(): ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = _current_user()
            if not user:
                return jsonify({"success": False, "message": "User not found."}), 404
            if not user.is_active:
                return jsonify({"success": False, "message": "Account is deactivated."}), 403
            if user.role not in roles:
                return jsonify({
                    "success": False,
                    "message": f"Access denied. Required role(s): {', '.join(roles)}",
                    "your_role": user.role,
                }), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def require_active(fn):
    """
    Decorator: blocks deactivated accounts.
    Must be placed BELOW @jwt_required().
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = _current_user()
        if not user:
            return jsonify({"success": False, "message": "User not found."}), 404
        if not user.is_active:
            return jsonify({"success": False, "message": "Your account has been deactivated."}), 403
        return fn(*args, **kwargs)
    return wrapper


def inject_user(fn):
    """
    Convenience decorator: injects `current_user` as keyword argument.
    Must be placed BELOW @jwt_required().
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = _current_user()
        if not user:
            return jsonify({"success": False, "message": "User not found."}), 404
        kwargs["current_user"] = user
        return fn(*args, **kwargs)
    return wrapper