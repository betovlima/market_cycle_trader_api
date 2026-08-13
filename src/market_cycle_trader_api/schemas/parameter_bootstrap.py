from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class BootstrapParametersRequest(BaseModel):
    

    model_config = ConfigDict(extra="forbid")

    confirm_insert_missing_only: Literal[True]


class ParameterizationItemResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    collection: str
    document_id: str
    status: Literal[
        "inserted",
        "migrated_existing",
        "repaired_invalid",
        "skipped_existing_valid",
        "skipped_existing_invalid",
        "missing",
    ]
    valid: bool
    message: str

    @field_validator("key", "collection", "document_id", "message")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return str(value).strip()
