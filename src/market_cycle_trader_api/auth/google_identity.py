from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import HTTPException

from market_cycle_trader_api.auth.config import get_auth_settings


@dataclass(frozen=True)
class VerifiedGoogleIdentity:
    subject: str
    email: str
    display_name: str | None = None
    picture_url: str | None = None
    hosted_domain: str | None = None


class GoogleIdentityVerifier(Protocol):
    def verify(self, credential: str) -> VerifiedGoogleIdentity: ...


def normalize_email(value: str) -> str:
    return str(value or "").strip().casefold()


class ProductionGoogleIdentityVerifier:
    





    def verify(self, credential: str) -> VerifiedGoogleIdentity:
        token = str(credential or "").strip()
        if not token:
            raise HTTPException(status_code=401, detail="Google identity verification is required.")

        settings = get_auth_settings()
        try:
            from google.auth.transport import requests as google_requests
            from google.oauth2 import id_token
        except ImportError as exc:  # pragma: no cover - deployment dependency guard
            raise RuntimeError("google-auth is required for Google identity verification.") from exc

        try:
            payload: dict[str, Any] = id_token.verify_oauth2_token(
                token,
                google_requests.Request(),
                settings.google_client_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=401, detail="Invalid Google identity credential.") from exc

        subject = str(payload.get("sub") or "").strip()
        email = normalize_email(str(payload.get("email") or ""))
        verified = payload.get("email_verified")
        email_verified = verified is True or str(verified).strip().lower() == "true"
        if not subject or not email or not email_verified:
            raise HTTPException(status_code=401, detail="A verified Google email is required.")

        return VerifiedGoogleIdentity(
            subject=subject,
            email=email,
            display_name=str(payload.get("name") or "").strip() or None,
            picture_url=str(payload.get("picture") or "").strip() or None,
            hosted_domain=str(payload.get("hd") or "").strip() or None,
        )
