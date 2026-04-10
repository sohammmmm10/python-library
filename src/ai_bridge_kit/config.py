from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: str | None, *, default: float | None) -> float | None:
    if value is None or value.strip() == "":
        return default
    return float(value)


def _as_int(value: str | None, *, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    return int(value)


@dataclass
class AIBridgeSettings:
    default_provider: str = "local"
    request_timeout: float | None = 30.0
    retries: int = 2
    backoff_seconds: float = 0.4
    max_backoff_seconds: float = 4.0
    use_jitter: bool = True

    @classmethod
    def from_env(cls) -> "AIBridgeSettings":
        return cls(
            default_provider=os.getenv("AIBRIDGE_DEFAULT_PROVIDER", "local"),
            request_timeout=_as_float(
                os.getenv("AIBRIDGE_REQUEST_TIMEOUT_SECONDS"), default=30.0
            ),
            retries=_as_int(os.getenv("AIBRIDGE_RETRIES"), default=2),
            backoff_seconds=_as_float(
                os.getenv("AIBRIDGE_BACKOFF_SECONDS"), default=0.4
            )
            or 0.4,
            max_backoff_seconds=_as_float(
                os.getenv("AIBRIDGE_MAX_BACKOFF_SECONDS"), default=4.0
            )
            or 4.0,
            use_jitter=_as_bool(os.getenv("AIBRIDGE_USE_JITTER"), default=True),
        )
