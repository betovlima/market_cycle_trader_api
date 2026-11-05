"""Public-safe error details for authenticated Live brokerage HTTP routes.

Provider exception strings can include credential headers, token fragments,
account identities, internal URLs or raw requests. Never serialize them into
a client response, even for an administrator session. Detailed diagnostics
must use separately controlled observability with its own secret redaction.
"""


def public_live_broker_error() -> str:
    """Static response; intentionally accepts no provider exception text."""
    return "Live brokerage operation is unavailable. Check the restricted operational audit."
