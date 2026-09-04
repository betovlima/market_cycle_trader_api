from __future__ import annotations

from typing import Any


_RELAXED_SOURCE_CHANGE_ERRORS = frozenset({
    "The Strategy source revision changed after Full Strategy validation. Validate again.",
    "The Strategy source configuration changed after Full Strategy validation. Validate again.",
    "The selected Strategy Research source revision changed after Full Strategy validation. Validate again.",
    "The selected Strategy Research source configuration changed after Full Strategy validation. Validate again.",
})


def install_asset_discovery_validation_revision_policy() -> None:
    from . import asset_discovery as service

    original = service._validated_creation_source
    if getattr(original, "_asset_discovery_allows_source_revision_drift", False):
        return

    def relaxed_source_validation(
        db: Any,
        document: dict[str, Any],
        requested_symbols: list[str],
    ) -> tuple[dict[str, Any], Any, dict[str, Any]]:
        try:
            return original(db, document, requested_symbols)
        except service.AssetDiscoveryConflict as exc:
            if str(exc) not in _RELAXED_SOURCE_CHANGE_ERRORS:
                raise

        validation = (
            document.get("full_strategy_validation")
            if isinstance(document.get("full_strategy_validation"), dict)
            else {}
        )
        if (
            str(validation.get("status") or "").lower() != "completed"
            or str(validation.get("decision") or "").upper() != "PASS"
        ):
            raise service.AssetDiscoveryConflict(
                "Run Full Strategy validation and obtain PASS before creating a Research Strategy."
            )
        if not service._selection_matches(validation.get("selected_assets"), requested_symbols):
            raise service.AssetDiscoveryConflict(
                "The selected assets changed after Full Strategy validation. Validate the exact selection again."
            )

        source_id = str(validation.get("source_strategy_id") or "").strip()
        source = service._raw_strategy(db, source_id)
        if source is None:
            raise service.AssetDiscoveryConflict(
                "The Strategy source used by Full Strategy validation is no longer available."
            )

        current_source, _current_config = service._current_research_source(db)
        if str(current_source.get("_id") or "") != source_id:
            raise service.AssetDiscoveryConflict(
                "The selected Strategy Research source changed after Full Strategy validation. Validate again."
            )

        current_model_snapshot = service.get_strategy_model_snapshot(db, source_id)
        if str(current_model_snapshot.get("family") or "") != str(validation.get("source_model_family") or ""):
            raise service.AssetDiscoveryConflict(
                "The selected Strategy Research model changed after Full Strategy validation. Validate again."
            )
        if str(current_model_snapshot.get("settings_hash") or "") != str(validation.get("source_model_settings_hash") or ""):
            raise service.AssetDiscoveryConflict(
                "The selected Strategy Research model settings changed after Full Strategy validation. Validate again."
            )

        config = service.BacktestRequest.model_validate(source.get("configuration") or {})
        return source, config, validation

    setattr(relaxed_source_validation, "_asset_discovery_allows_source_revision_drift", True)
    service._validated_creation_source = relaxed_source_validation
