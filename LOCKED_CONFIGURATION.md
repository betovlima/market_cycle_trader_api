# MongoDB configuration administration

All MongoDB configuration changes are performed through protected HTTP endpoints.
The API startup creates storage indexes and validates state, but does not install or replace configuration documents.

Administrative authentication uses the `X-Parameter-Bootstrap-Token` header and the server-side `PARAMETER_BOOTSTRAP_API_TOKEN` environment variable.

## Initial configuration

1. `GET /api/admin/parameters/status`
2. `POST /api/admin/parameters/bootstrap`
3. `GET /api/admin/strategy-configuration`
4. `POST /api/admin/setup/initialize`
5. `GET /api/admin/setup/status`

Strategy configuration changes use the protected `/api/admin/strategy-configuration` endpoints.
No MongoDB configuration script or Railway pre-deploy database command is required.
