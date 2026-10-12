"""Regression checks for the native endpoint search in Swagger UI (/docs)."""

from __future__ import annotations

import unittest

from market_cycle_trader_api.main import app


class SwaggerDocsSearchTests(unittest.TestCase):
    def test_swagger_filter_is_enabled_without_changing_api_schema(self) -> None:
        settings = app.swagger_ui_parameters or {}
        self.assertIs(settings.get("filter"), True)
        self.assertEqual(settings.get("docExpansion"), "none")
        self.assertEqual(settings.get("tagsSorter"), "alpha")
        self.assertEqual(settings.get("operationsSorter"), "alpha")
        self.assertIs(settings.get("deepLinking"), True)
        self.assertEqual(app.docs_url, "/docs")
        self.assertEqual(app.openapi_url, "/openapi.json")

    def test_challenger_endpoints_keep_searchable_tag(self) -> None:
        schema = app.openapi()
        path = "/api/research/tcc-u77-fixed/strategy"
        self.assertIn(path, schema["paths"])
        tags = schema["paths"][path]["post"].get("tags", [])
        self.assertIn("tcc-u77-fixed-challenger", tags)


if __name__ == "__main__":
    unittest.main()
