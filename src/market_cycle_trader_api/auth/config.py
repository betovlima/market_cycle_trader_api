from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _boolean(name: str, default: bool) -> bool:
    raw = str(os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean value.")


def _positive_integer(name: str, default: int, minimum: int = 1) -> int:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}.")
    return value


def _normalized_email(value: str) -> str:
    return str(value or "").strip().casefold()


@dataclass(frozen=True)
class AuthSettings:
    admin_password: str
    admin_google_email: str
    session_secret: str
    session_max_age_seconds: int
    viewer_session_max_age_seconds: int
    viewer_session_idle_seconds: int
    trader_session_max_age_seconds: int
    trader_session_idle_seconds: int
    admin_session_max_age_seconds: int
    admin_session_idle_seconds: int
    cookie_secure: bool
    cookie_samesite: str
    auth_storage: str
    mongo_url: str
    mongo_database: str
    frontend_base_url: str
    google_client_id: str = ""

    def session_max_age_for_role(self, role: str) -> int:
        return {
            "viewer": self.viewer_session_max_age_seconds,
            "trader": self.trader_session_max_age_seconds,
            "admin": self.admin_session_max_age_seconds,
        }.get(role, self.session_max_age_seconds)

    def session_idle_for_role(self, role: str) -> int:
        return {
            "viewer": self.viewer_session_idle_seconds,
            "trader": self.trader_session_idle_seconds,
            "admin": self.admin_session_idle_seconds,
        }.get(role, self.session_max_age_seconds)

    def validate_runtime(self) -> None:
        missing: list[str] = []
        if not self.admin_password:
            missing.append("TRADER_ADMIN_PASSWORD")
        if not self.admin_google_email:
            missing.append("TRADER_ADMIN_GOOGLE_EMAIL")
        if not self.session_secret:
            missing.append("TRADER_SESSION_SECRET")
        if self.auth_storage == "mongodb" and not self.mongo_url:
            missing.append("MONGO_URL")
        if self.auth_storage == "mongodb" and not self.mongo_database:
            missing.append("MONGO_DATABASE")
        if not self.frontend_base_url:
            missing.append("TRADER_FRONTEND_BASE_URL")
        if not self.google_client_id:
            missing.append("GOOGLE_CLIENT_ID")
        if missing:
            raise RuntimeError("Missing required authentication variables: " + ", ".join(missing))


@lru_cache(maxsize=1)
def get_auth_settings() -> AuthSettings:
    secure = _boolean("TRADER_COOKIE_SECURE", True)
    same_site = str(os.getenv("TRADER_COOKIE_SAMESITE") or ("none" if secure else "lax")).strip().lower()
    if same_site not in {"lax", "strict", "none"}:
        raise RuntimeError("TRADER_COOKIE_SAMESITE must be lax, strict or none.")
    if same_site == "none" and not secure:
        raise RuntimeError("TRADER_COOKIE_SECURE must be true when SameSite is none.")
    storage = str(os.getenv("TRADER_AUTH_STORAGE") or "mongodb").strip().lower()
    if storage not in {"mongodb", "memory"}:
        raise RuntimeError("TRADER_AUTH_STORAGE must be mongodb or memory.")
    return AuthSettings(
        admin_password=str(os.getenv("TRADER_ADMIN_PASSWORD") or ""),
        admin_google_email=_normalized_email(os.getenv("TRADER_ADMIN_GOOGLE_EMAIL") or ""),
        session_secret=str(os.getenv("TRADER_SESSION_SECRET") or ""),
        session_max_age_seconds=_positive_integer("TRADER_SESSION_MAX_AGE_SECONDS", 28_800, 300),
        viewer_session_max_age_seconds=_positive_integer("TRADER_VIEWER_SESSION_MAX_AGE_SECONDS", 43_200, 300),
        viewer_session_idle_seconds=_positive_integer("TRADER_VIEWER_SESSION_IDLE_SECONDS", 7_200, 300),
        trader_session_max_age_seconds=_positive_integer("TRADER_TRADER_SESSION_MAX_AGE_SECONDS", 28_800, 300),
        trader_session_idle_seconds=_positive_integer("TRADER_TRADER_SESSION_IDLE_SECONDS", 3_600, 300),
        admin_session_max_age_seconds=_positive_integer("TRADER_ADMIN_SESSION_MAX_AGE_SECONDS", 7_200, 300),
        admin_session_idle_seconds=_positive_integer("TRADER_ADMIN_SESSION_IDLE_SECONDS", 1_800, 300),
        cookie_secure=secure,
        cookie_samesite=same_site,
        auth_storage=storage,
        mongo_url=str(os.getenv("MONGO_URL") or os.getenv("MONGO_URI") or "").strip(),
        mongo_database=str(os.getenv("MONGO_DATABASE") or "").strip(),
        frontend_base_url=str(os.getenv("TRADER_FRONTEND_BASE_URL") or "").strip().rstrip("/"),
        google_client_id=str(os.getenv("GOOGLE_CLIENT_ID") or "").strip(),
    )
